"""
ollama.py — Adaptador para conectar PydanticAI con Ollama.

Ollama expone una API compatible con OpenAI en http://localhost:11434/v1.
Esto significa que podemos usar el OpenAIModel de PydanticAI apuntando
a esa URL, sin que haga falta una cuenta de OpenAI ni gastar dinero.

La api_key="ollama" es un placeholder: Ollama no la valida, pero
PydanticAI la requiere como parámetro obligatorio.

Si en el futuro queremos comparar con un modelo remoto (ej: GPT-4o para
el benchmark), bastaría con cambiar base_url y api_key en el .env.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


def build_ollama_model() -> OpenAIModel:
    """Construye un modelo PydanticAI que apunta a Ollama local.

    Lee la configuración del .env:
    - OLLAMA_BASE_URL: URL del servidor Ollama (por defecto localhost:11434/v1)
    - OLLAMA_MODEL: nombre del modelo (por defecto qwen3:8b)
    - OLLAMA_API_KEY: placeholder (por defecto "ollama")
    """
    load_dotenv()

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    api_key = os.getenv("OLLAMA_API_KEY", "ollama")

    # OpenAIProvider habla el protocolo de OpenAI, pero contra nuestro Ollama local
    return OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )
