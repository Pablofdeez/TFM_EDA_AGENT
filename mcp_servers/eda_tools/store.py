"""
store.py — Almacén de tablas en memoria para el servidor MCP.

Este módulo gestiona los DataFrames que el usuario va cargando a través del agente.
Funciona como un diccionario de tablas indexado por UUID, donde siempre hay una
"tabla activa" que es la que se usa por defecto cuando el usuario no especifica cuál.

La idea es que el LLM no necesite recordar UUIDs: basta con que diga "la tabla activa"
o el nombre que le dio el usuario al cargarla (ej: "sales").
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TableHandle:
    """Wrapper que asocia un DataFrame con sus metadatos de carga.

    Cada vez que el usuario carga un CSV/Parquet, se crea un TableHandle
    con un UUID único, el nombre que le puso, la ruta de origen y el DataFrame.
    """

    table_id: str
    table_name: str
    path: str
    df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class TableStore:
    """Almacén en memoria de tablas cargadas.

    En la v1 solo soportamos una tabla activa a la vez, pero internamente
    se guardan todas las que se hayan cargado por si se necesitan después.
    """

    def __init__(self) -> None:
        # Diccionario {table_id: TableHandle} con todas las tablas cargadas
        self._tables: dict[str, TableHandle] = {}
        # ID de la tabla que se usa por defecto en las tools
        self._active_table_id: str | None = None

    def register(
        self, *, table_name: str, path: str, df: pd.DataFrame
    ) -> TableHandle:
        """Registra un nuevo DataFrame y lo marca como tabla activa."""
        table_id = str(uuid.uuid4())
        handle = TableHandle(
            table_id=table_id,
            table_name=table_name,
            path=path,
            df=df,
        )
        self._tables[table_id] = handle
        # La última tabla cargada pasa a ser la activa automáticamente
        self._active_table_id = table_id
        return handle

    def get(self, table_id: str) -> TableHandle:
        """Obtiene una tabla por su UUID. Lanza KeyError si no existe."""
        if table_id not in self._tables:
            raise KeyError(f"Table id no encontrado: {table_id}")
        return self._tables[table_id]

    def get_active(self) -> TableHandle:
        """Devuelve la tabla activa. Lanza ValueError si no hay ninguna cargada."""
        if self._active_table_id is None:
            raise ValueError("No hay ninguna tabla activa. Usa load_table primero.")
        return self.get(self._active_table_id)

    # Palabras que el LLM puede enviar queriendo decir "usa la tabla activa".
    # Esto es un guardarraíl: un modelo pequeño a veces pasa "active" o "none"
    # en vez de omitir el parámetro.
    _ACTIVE_ALIASES = {"active", "current", "none", "null", ""}

    def get_by_name(self, name: str) -> TableHandle | None:
        """Busca una tabla por el nombre que le puso el usuario (case-insensitive)."""
        for handle in self._tables.values():
            if handle.table_name.lower() == name.lower():
                return handle
        return None

    def resolve(self, table_id: str | None) -> TableHandle:
        """Punto de entrada principal para las tools: resuelve qué tabla usar.

        Lógica de resolución:
        1. Si table_id es None o un alias como "active" → tabla activa
        2. Si coincide con un UUID registrado → esa tabla
        3. Si coincide con un nombre de tabla → esa tabla
        4. Si no → error

        Este método es defensivo a propósito: los modelos locales pequeños
        no siempre manejan bien los parámetros opcionales, así que preferimos
        ser flexibles aquí antes que fallar por un argumento mal formateado.
        """
        if not table_id or table_id.lower().strip() in self._ACTIVE_ALIASES:
            return self.get_active()
        # Intenta por UUID primero
        if table_id in self._tables:
            return self._tables[table_id]
        # Intenta por nombre (ej: el LLM pasa "sales" en vez del UUID)
        by_name = self.get_by_name(table_id)
        if by_name:
            return by_name
        raise KeyError(f"Tabla no encontrada (ni por id ni por nombre): {table_id}")

    def set_active(self, table_id: str) -> None:
        """Cambia la tabla activa (valida que exista antes)."""
        _ = self.get(table_id)
        self._active_table_id = table_id

    @property
    def active_table_id(self) -> str | None:
        return self._active_table_id


# Instancia global (singleton) compartida por todas las tools del server MCP.
# Como el servidor corre en un único proceso, esto mantiene el estado entre llamadas.
STORE = TableStore()
