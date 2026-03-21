"""
app.py — Interfaz web con Streamlit para el agente EDA.

En vez de interactuar por consola (REPL), este módulo levanta una interfaz
web con un chat interactivo. El usuario puede subir un CSV desde el navegador,
cargarlo como tabla y hacerle preguntas en lenguaje natural.

El reto principal es que Streamlit re-ejecuta el script COMPLETO cada vez
que el usuario interactúa (pulsa un botón, envía un mensaje, etc.).
Pero nuestro agente necesita mantener el servidor MCP vivo entre preguntas
(porque el TableStore con las tablas cargadas vive en ese subproceso).

La solución: un hilo background con su propio event loop de asyncio.
- El hilo arranca el agente una sola vez (async with agent → MCP server vivo)
- @st.cache_resource evita que se recree en cada re-ejecución de Streamlit
- ask() envía preguntas desde el hilo de Streamlit al event loop del agente

Para ejecutar (desde la raíz del proyecto):
    pip install -e .
    streamlit run src/eda_agent/app.py
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from eda_agent.agent import build_agent
from eda_agent.schemas import EDAAnswer

load_dotenv()

# Ruta raíz del proyecto: app.py está en src/eda_agent/, subimos 3 niveles
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


# ---------------------------------------------------------------------------
# AgentRunner — mantiene el agente + MCP server vivos en un hilo background
# ---------------------------------------------------------------------------

class AgentRunner:
    """Ejecuta el agente PydanticAI en un hilo dedicado con su event loop.

    ¿Por qué un hilo separado?
    Streamlit es síncrono y re-ejecuta el script en cada interacción.
    El agente necesita un event loop asyncio PERSISTENTE para mantener
    el servidor MCP (subproceso stdio) vivo entre preguntas.
    Si no hiciéramos esto, cada pregunta arrancaría y mataría el MCP server,
    perdiendo las tablas cargadas en el TableStore.

    Flujo:
    1. __init__ crea un hilo con un event loop nuevo
    2. El hilo arranca el agente (async with → MCP server vivo)
    3. ask() envía preguntas al loop del hilo vía run_coroutine_threadsafe
    4. El TableStore mantiene las tablas en memoria entre preguntas
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._agent = build_agent()
        self._ready = threading.Event()

        # daemon=True: el hilo muere automáticamente al cerrar Streamlit
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

        if not self._ready.wait(timeout=30):
            raise RuntimeError("Timeout: el agente no arrancó en 30s")

    def _run(self) -> None:
        """Entry point del hilo background."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self) -> None:
        """Mantiene el contexto del agente (y MCP server) abierto.

        El asyncio.Event() bloquea _main sin consumir CPU.
        Mientras esperamos, el event loop sigue vivo y puede procesar
        las queries que llegan vía run_coroutine_threadsafe.
        """
        self._stop = asyncio.Event()
        async with self._agent:
            self._ready.set()
            await self._stop.wait()

    def ask(self, query: str) -> EDAAnswer:
        """Envía una pregunta al agente y espera la respuesta (síncrono).

        run_coroutine_threadsafe es thread-safe: permite enviar una coroutine
        al event loop del hilo background desde el hilo principal de Streamlit.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._agent.run(query), self._loop
        )
        # Timeout alto porque Ollama local puede ser lento
        return future.result(timeout=120).output


@st.cache_resource
def get_runner() -> AgentRunner:
    """Singleton del AgentRunner — se crea una vez y persiste entre re-runs."""
    return AgentRunner()


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def render_answer(answer: EDAAnswer) -> None:
    """Renderiza una respuesta del agente en el chat de Streamlit."""
    st.markdown(answer.answer)

    if answer.tools_used:
        st.caption(f"Tools: {', '.join(answer.tools_used)}")

    if answer.evidence:
        with st.expander("Evidencia"):
            st.json(answer.evidence)

    for w in answer.warnings:
        st.warning(w)


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EDA Agent", page_icon="📊")
st.title("📊 EDA Agent")
st.caption("Análisis exploratorio de datos con lenguaje natural")

# --- Sidebar: carga de datos e info ---
with st.sidebar:
    st.header("Cargar dataset")

    uploaded = st.file_uploader("CSV o Parquet", type=["csv", "parquet"])
    table_name = st.text_input("Nombre de la tabla", value="datos")

    if uploaded and st.button("Cargar tabla"):
        # Guardamos el fichero en data/uploads/ para que el MCP server lo lea
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        save_path = UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())

        with st.spinner("Cargando tabla..."):
            try:
                runner = get_runner()
                answer = runner.ask(f"Carga {save_path} como {table_name}")
                st.session_state.loaded_table = {
                    "name": table_name,
                    "evidence": answer.evidence,
                }
                st.success(f"Tabla '{table_name}' cargada")
            except Exception as e:
                st.error(f"Error cargando tabla: {e}")

    # Mostrar info de la tabla activa
    if "loaded_table" in st.session_state:
        info = st.session_state.loaded_table
        st.divider()
        st.subheader(f"Tabla activa: {info['name']}")
        if info.get("evidence"):
            st.json(info["evidence"])

    st.divider()
    model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    st.caption(f"Modelo: {model}")

# --- Chat ---
if "chat" not in st.session_state:
    st.session_state.chat = []

# Renderizar historial de mensajes
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_answer(msg["data"])

# Input del usuario
if prompt := st.chat_input("Pregunta sobre tus datos..."):
    # Mostrar mensaje del usuario
    st.session_state.chat.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Obtener respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                runner = get_runner()
                answer = runner.ask(prompt)
            except Exception as e:
                st.error(f"Error: {e}")
                answer = None

        if answer:
            render_answer(answer)
            st.session_state.chat.append({"role": "assistant", "data": answer})
