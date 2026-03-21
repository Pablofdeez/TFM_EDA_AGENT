"""
profiling.py — Funciones de perfilado básico de DataFrames.

Aquí están las funciones "puras" que analizan un DataFrame y devuelven
diccionarios serializables. No saben nada de MCP ni del agente: solo
reciben un DataFrame y devuelven datos.

Las tools del server.py llaman a estas funciones. Esta separación nos permite
testear la lógica analítica sin levantar el servidor MCP.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Convierte valores de pandas/numpy a tipos nativos de Python.

    Esto es necesario porque pandas usa tipos propios (np.int64, pd.Timestamp, etc.)
    que no son serializables directamente a JSON. El servidor MCP necesita devolver
    JSON válido al agente, así que pasamos todo por esta función.
    """
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    # np.int64, np.float64, etc. tienen .item() para convertir a int/float nativos
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def describe_schema_df(df: pd.DataFrame) -> dict[str, Any]:
    """Genera un resumen del esquema: tipo, nulos, únicos y ejemplos por columna.

    Esto es lo que el agente recibe cuando el usuario pregunta "describe el esquema".
    Le damos solo la información justa para que entienda la tabla sin ver todas las filas.
    """
    columns: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        # Cogemos 3 ejemplos no nulos para que el LLM tenga una idea del contenido
        examples = [to_jsonable(v) for v in series.dropna().head(3).tolist()]

        columns.append(
            {
                "name": col,
                "dtype": str(series.dtype),
                "n_null": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
                "examples": examples,
            }
        )

    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": columns,
    }


def detect_datetime_columns_df(
    df: pd.DataFrame,
    candidates: list[str] | None = None,
    sample_size: int = 50,
    threshold: float = 0.85,
) -> dict[str, Any]:
    """Detecta qué columnas tienen semántica temporal.

    Estrategia en dos pasos:
    1. Si la columna ya es datetime64 en pandas → confianza 1.0
    2. Si no, cogemos una muestra e intentamos parsearla con pd.to_datetime.
       Si el ratio de valores parseados correctamente supera el threshold (85%),
       la consideramos temporal.

    El threshold del 85% es deliberadamente permisivo: preferimos detectar
    una columna de más (falso positivo) que perdernos una fecha real.
    El agente siempre puede verificar después.
    """
    cols = candidates or list(df.columns)
    detected: list[dict[str, Any]] = []

    for col in cols:
        if col not in df.columns:
            continue

        series = df[col]

        # Caso 1: ya es datetime nativo de pandas
        if pd.api.types.is_datetime64_any_dtype(series):
            detected.append(
                {
                    "column": col,
                    "confidence": 1.0,
                    "reason": "dtype datetime",
                }
            )
            continue

        # Caso 2: intentamos parsear una muestra como fecha
        sample = series.dropna().head(sample_size)
        if sample.empty:
            continue

        parsed = pd.to_datetime(sample.astype(str), errors="coerce")
        ratio = float(parsed.notna().mean())

        if ratio >= threshold:
            detected.append(
                {
                    "column": col,
                    "confidence": round(ratio, 3),
                    "reason": f"parse ratio {ratio:.2%} >= {threshold:.0%}",
                }
            )

    # La columna con mayor confianza se marca como "principal"
    primary_datetime = None
    if detected:
        primary_datetime = max(detected, key=lambda x: x["confidence"])["column"]

    return {
        "datetime_columns": detected,
        "primary_datetime": primary_datetime,
    }
