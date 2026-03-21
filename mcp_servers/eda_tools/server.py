"""
server.py — Servidor MCP que expone las tools de EDA.

Este es el punto de entrada del servidor MCP. Se ejecuta como un proceso
independiente que se comunica con el agente (PydanticAI) por stdio.

Cada función decorada con @mcp.tool() queda expuesta como una tool MCP
que el agente puede invocar. Las tools son "finas": reciben parámetros,
delegan la lógica pesada a profiling.py u ops.py, y devuelven un diccionario
serializable.

Para arrancarlo manualmente (útil para debug):
    python -m mcp_servers.eda_tools.server

En producción lo arranca automáticamente el agente a través de MCPServerStdio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastmcp import FastMCP

# Imports relativos dentro del paquete mcp_servers.eda_tools
from .ops import aggregate_metric_df, null_report_df
from .profiling import describe_schema_df, detect_datetime_columns_df, to_jsonable
from .store import STORE

# Creamos la instancia de FastMCP que registra las tools
mcp = FastMCP("EDA Tools Server")


def _infer_format(path: str, file_format: str | None) -> str:
    """Infiere el formato del fichero por extensión si no se indica explícitamente."""
    if file_format:
        return file_format
    suffix = Path(path).suffix.lower()
    return "parquet" if suffix == ".parquet" else "csv"


# --- Tools MCP ---
# Cada tool tiene un docstring que el LLM lee para saber cuándo y cómo usarla.
# Es importante que las descripciones sean claras porque el modelo decide
# qué tool llamar basándose en ellas.


@mcp.tool()
def load_table(
    path: str,
    table_name: str = "table",
    file_format: Literal["csv", "parquet"] | None = None,
    csv_delimiter: str = ",",
) -> dict[str, Any]:
    """Carga un CSV o Parquet y lo registra como tabla activa.

    Devuelve metadatos básicos y una preview de 5 filas para que el agente
    pueda describir la tabla al usuario sin necesidad de ver todo el dataset.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el fichero: {p}")

    fmt = _infer_format(path, file_format)

    if fmt == "csv":
        df = pd.read_csv(p, sep=csv_delimiter)
    elif fmt == "parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Formato no soportado: {fmt}")

    # Registramos la tabla en el store (queda como activa)
    handle = STORE.register(table_name=table_name, path=str(p), df=df)

    # Preparamos una preview serializable de las primeras 5 filas
    preview_df = df.head(5).copy()
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    preview = [
        {k: to_jsonable(v) for k, v in row.items()}
        for row in preview_df.to_dict(orient="records")
    ]

    return {
        "table_id": handle.table_id,
        "table_name": handle.table_name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "preview": preview,
    }


@mcp.tool()
def describe_schema(table_id: str | None = None) -> dict[str, Any]:
    """Describe el esquema de la tabla activa o de una tabla concreta.

    Devuelve tipo de dato, nulos, valores únicos y ejemplos por columna.
    El agente usa esta información para entender la estructura de los datos.
    """
    handle = STORE.resolve(table_id)
    schema = describe_schema_df(handle.df)
    return {
        "table_id": handle.table_id,
        "table_name": handle.table_name,
        **schema,
    }


@mcp.tool()
def detect_datetime_columns(
    table_id: str | None = None,
    candidates: list[str] | None = None,
) -> dict[str, Any]:
    """Detecta columnas con semántica temporal en la tabla activa.

    Importante: el agente debe llamar a esta tool ANTES de aplicar
    filtros temporales, para saber qué columna usar como fecha.
    """
    handle = STORE.resolve(table_id)
    result = detect_datetime_columns_df(handle.df, candidates=candidates)
    return {
        "table_id": handle.table_id,
        "table_name": handle.table_name,
        **result,
    }


@mcp.tool()
def aggregate_metric(
    column: str,
    operation: Literal["mean", "max", "min", "sum", "count"],
    table_id: str | None = None,
    time_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ejecuta una agregación (mean, max, min, sum, count) sobre una columna.

    Opcionalmente acepta un time_filter para acotar por rango de fechas.
    El time_filter debe incluir: date_column, y opcionalmente start y end (ISO).
    """
    handle = STORE.resolve(table_id)
    result = aggregate_metric_df(
        handle.df,
        column=column,
        operation=operation,
        time_filter=time_filter,
    )
    return {
        "table_id": handle.table_id,
        "table_name": handle.table_name,
        **result,
    }


@mcp.tool()
def null_report(
    table_id: str | None = None,
    column: str | None = None,
) -> dict[str, Any]:
    """Informe de nulos: para una columna concreta o para todas.

    Devuelve conteo absoluto y porcentaje. Si no se indica columna,
    devuelve el informe completo (útil para una vista general de calidad).
    """
    handle = STORE.resolve(table_id)
    result = null_report_df(handle.df, column=column)
    return {
        "table_id": handle.table_id,
        "table_name": handle.table_name,
        **result,
    }


# Punto de entrada: FastMCP arranca el servidor en modo stdio
# (lee JSON-RPC por stdin, responde por stdout)
if __name__ == "__main__":
    mcp.run()
