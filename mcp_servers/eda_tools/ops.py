"""
ops.py — Operaciones analíticas sobre DataFrames.

Contiene las funciones de agregación y reporte de nulos que ejecutan
los cálculos reales. Al igual que profiling.py, estas funciones son puras:
reciben un DataFrame y devuelven un diccionario.

Principio clave del TFM: el LLM nunca calcula estadísticas por sí mismo.
Toda respuesta numérica sale de estas funciones.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

# Las operaciones soportadas en la v1. Se define como Literal
# para que pydantic y el LLM sepan exactamente qué valores son válidos.
Operation = Literal["mean", "max", "min", "sum", "count"]


def apply_time_filter(
    df: pd.DataFrame, time_filter: dict[str, Any] | None
) -> pd.DataFrame:
    """Filtra un DataFrame por rango temporal.

    El time_filter es un diccionario con:
    - date_column: nombre de la columna que contiene las fechas
    - start (opcional): fecha ISO de inicio del rango
    - end (opcional): fecha ISO de fin del rango

    Si no hay filtro, devuelve el DataFrame original sin copiar (eficiente).
    Si hay filtro, parsea la columna temporal, descarta filas no parseables
    y aplica el rango.
    """
    if not time_filter:
        return df

    date_column = time_filter["date_column"]
    if date_column not in df.columns:
        raise ValueError(f"Columna temporal no existe: {date_column}")

    # Trabajamos sobre una copia para no modificar el DataFrame original
    temp = df.copy()
    temp["__dt__"] = pd.to_datetime(temp[date_column], errors="coerce")
    temp = temp.dropna(subset=["__dt__"])

    start = time_filter.get("start")
    end = time_filter.get("end")

    if start:
        temp = temp[temp["__dt__"] >= pd.to_datetime(start)]
    if end:
        temp = temp[temp["__dt__"] <= pd.to_datetime(end)]

    # Eliminamos la columna auxiliar antes de devolver
    return temp.drop(columns=["__dt__"])


def aggregate_metric_df(
    df: pd.DataFrame,
    column: str,
    operation: Operation,
    time_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta una agregación sobre una columna, con filtro temporal opcional.

    Operaciones soportadas: mean, max, min, sum, count.
    Para mean y sum se fuerza conversión numérica (por si la columna es string).
    Para max/min se deja el tipo original (puede ser fecha, string, etc.).
    """
    if column not in df.columns:
        raise ValueError(f"Columna no existe: {column}")

    # Aplicar filtro temporal si lo hay
    dff = apply_time_filter(df, time_filter)
    series = dff[column]

    # Cada operación tiene su propia lógica de conversión
    if operation == "mean":
        value = float(pd.to_numeric(series, errors="coerce").mean())
    elif operation == "sum":
        value = float(pd.to_numeric(series, errors="coerce").sum())
    elif operation == "max":
        raw = series.max()
        value = raw.item() if hasattr(raw, "item") else raw
    elif operation == "min":
        raw = series.min()
        value = raw.item() if hasattr(raw, "item") else raw
    elif operation == "count":
        # count() excluye nulos automáticamente en pandas
        value = int(series.count())
    else:
        raise ValueError(f"Operación no soportada: {operation}")

    return {
        "column": column,
        "operation": operation,
        "value": value,
        "n_rows_used": int(len(dff)),
        "time_filter": time_filter,
    }


def null_report_df(
    df: pd.DataFrame, column: str | None = None
) -> dict[str, Any]:
    """Genera un informe de valores nulos.

    Si se especifica una columna, devuelve nulos solo para esa columna.
    Si no, devuelve el informe completo para todas las columnas.
    Incluye tanto el conteo absoluto como el porcentaje.
    """
    n_total = len(df)

    if column:
        if column not in df.columns:
            raise ValueError(f"Columna no existe: {column}")
        n_null = int(df[column].isna().sum())
        return {
            "column": column,
            "n_null": n_null,
            "pct_null": round(n_null / n_total, 4) if n_total else 0.0,
        }

    # Informe completo: una entrada por columna
    report = []
    for col in df.columns:
        n_null = int(df[col].isna().sum())
        report.append(
            {
                "column": col,
                "n_null": n_null,
                "pct_null": round(n_null / n_total, 4) if n_total else 0.0,
            }
        )

    return {"columns": report}
