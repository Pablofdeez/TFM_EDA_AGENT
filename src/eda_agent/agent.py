"""
agent.py — Construcción del agente y REPL interactivo.

Este es el punto de entrada principal del sistema. Hace dos cosas:
1. Construye el agente PydanticAI con el modelo Ollama y el servidor MCP
2. Arranca un REPL (Read-Eval-Print Loop) para interactuar con el agente

El flujo cuando el usuario escribe una pregunta es:
  usuario → Agent.run() → LLM decide qué tool llamar → MCP server ejecuta la tool
  → resultado vuelve al LLM → LLM genera EDAAnswer → se muestra al usuario

El servidor MCP se arranca como subproceso (stdio) y se mantiene vivo
durante toda la sesión del REPL. Esto es importante porque el TableStore
vive en la memoria de ese subproceso: si se reiniciara en cada pregunta,
perdería las tablas cargadas.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from eda_agent.adapters.ollama import build_ollama_model
from eda_agent.prompts import SYSTEM_PROMPT
from eda_agent.schemas import EDAAnswer

console = Console()


def build_agent() -> Agent:
    """Construye el agente con el modelo Ollama y el server MCP de tools.

    MCPServerStdio lanza el servidor MCP como un proceso hijo que se comunica
    por stdin/stdout (protocolo JSON-RPC). El parámetro timeout=60 da margen
    para que modelos lentos completen la respuesta.
    """
    model = build_ollama_model()

    # El servidor MCP se lanza como módulo Python (-m) para que los imports
    # relativos funcionen correctamente
    mcp_server = MCPServerStdio(
        sys.executable,
        args=["-m", "mcp_servers.eda_tools.server"],
        timeout=60,
    )

    return Agent(
        model=model,
        output_type=EDAAnswer,       # Fuerza structured output validado con Pydantic
        system_prompt=SYSTEM_PROMPT,  # Instrucciones de comportamiento del agente
        mcp_servers=[mcp_server],     # Tools disponibles vía MCP
    )


async def repl() -> None:
    """REPL interactivo para probar el agente desde terminal.

    Arranca el servidor MCP, mantiene la sesión abierta (tablas persistentes
    entre preguntas) y muestra las respuestas como JSON formateado.
    """
    agent = build_agent()

    console.print("[bold green]EDA Agent listo.[/bold green]")
    console.print("Ejemplos de uso:")
    console.print("  Carga ./data/samples/sales.csv como sales")
    console.print("  Describe el esquema")
    console.print("  Que columnas parecen fecha")
    console.print("  Cual es la media de revenue")
    console.print("  Cuantos nulos hay en region")
    console.print("Escribe 'exit' para salir.\n")

    # async with agent: arranca los servidores MCP y los cierra al salir
    async with agent:
        while True:
            try:
                user_input = input("eda> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nSaliendo.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "salir"}:
                break

            try:
                result = await agent.run(user_input)
                # Mostramos la respuesta como JSON bonito con rich
                console.print_json(
                    json.dumps(result.output.model_dump(), ensure_ascii=False)
                )
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")


def main() -> None:
    """Punto de entrada para ejecutar como script."""
    # Forzar UTF-8 en Windows para evitar problemas con caracteres especiales
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(repl())


if __name__ == "__main__":
    main()
