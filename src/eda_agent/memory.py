"""
memory.py — Memoria de sesión del agente (v1 simple).

En la v1 usamos un diccionario en memoria para mantener contexto entre turnos:
qué tabla hay cargada, qué columnas datetime se detectaron, etc.

Esto NO persiste entre ejecuciones: si cierras el agente, se pierde.
Para la v1 es suficiente porque cada sesión de benchmark empieza de cero.

En v2 podría persistir a disco (JSON/SQLite) o integrarse con el
system_prompt dinámico de PydanticAI para inyectar contexto automáticamente.
"""

from __future__ import annotations

from typing import Any


class SessionMemory:
    """Almacén clave-valor para el contexto de la sesión actual."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Guarda un valor en la memoria de sesión."""
        self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Recupera un valor, o default si no existe."""
        return self._state.get(key, default)

    def clear(self) -> None:
        """Limpia toda la memoria (útil entre tests del benchmark)."""
        self._state.clear()

    def summary(self) -> dict[str, Any]:
        """Devuelve una copia del estado actual (para debug o inyección en prompt)."""
        return dict(self._state)
