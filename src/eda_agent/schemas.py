"""
schemas.py — Modelos Pydantic para structured outputs y tipos auxiliares.

Definimos aquí los esquemas que usa el agente para:
- Validar la respuesta final (EDAAnswer): toda respuesta del agente se ajusta a este modelo
- Representar el contexto del dataset (DatasetContext)
- Describir pasos de un plan (Plan, ToolStep)

El uso de Pydantic como output_type en PydanticAI es clave:
obliga al LLM a devolver JSON válido que se parsea y valida automáticamente.
Si el LLM devuelve algo que no encaja, PydanticAI lanza un error en vez de
dejar pasar una respuesta mal formada.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class TimeFilter(BaseModel):
    """Filtro temporal para acotar consultas por rango de fechas.

    El agente construye este objeto cuando el usuario pregunta algo como
    "ventas de los últimos 3 años" o "revenue en 2024".
    """

    date_column: str = Field(..., description="Nombre de la columna temporal")
    start: Optional[str] = Field(default=None, description="Fecha ISO de inicio")
    end: Optional[str] = Field(default=None, description="Fecha ISO de fin")


class ToolStep(BaseModel):
    """Un paso individual dentro de un plan de ejecución.

    Describe qué tool llamar y con qué argumentos. Se usa cuando el agente
    necesita encadenar varias tools (ej: detectar fechas → luego agregar).
    """

    tool: Literal[
        "load_table",
        "describe_schema",
        "detect_datetime_columns",
        "aggregate_metric",
        "null_report",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    """Plan de ejecución generado por el agente antes de actuar.

    En la v1 no se usa explícitamente (el agente actúa de forma reactiva),
    pero lo dejamos definido para poder trazar el razonamiento del agente
    en el benchmark y en futuras versiones.
    """

    intent: str
    steps: list[ToolStep] = Field(default_factory=list)
    expected_output: str


class DatasetContext(BaseModel):
    """Contexto de la tabla activa que se puede inyectar en el prompt.

    Sirve para que el agente "recuerde" entre turnos qué tabla tiene cargada,
    qué columnas son temporales, y si hay algún warning pendiente.
    """

    active_table_id: Optional[str] = None
    active_table_name: Optional[str] = None
    datetime_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EDAAnswer(BaseModel):
    """Respuesta estructurada del agente. TODAS las respuestas deben usar este formato.

    Campos:
    - answer: explicación en lenguaje natural para el usuario
    - tools_used: qué tools se invocaron (para trazabilidad)
    - evidence: datos crudos devueltos por las tools (para verificación)
    - warnings: problemas detectados (columna no existe, dato ambiguo, etc.)

    Este esquema es el output_type del Agent de PydanticAI, lo que significa
    que el LLM está obligado a devolver JSON que se ajuste a esta estructura.
    """

    answer: str = Field(..., description="Respuesta en lenguaje natural")
    tools_used: list[str] = Field(
        default_factory=list,
        description="Lista de tools invocadas para producir la respuesta",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Datos crudos devueltos por las tools como evidencia",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Advertencias: columna no encontrada, ambiguedad, etc.",
    )
