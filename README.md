# Agente EDA con LLMs

**Trabajo Fin de Master — Master en Data Science y Mathematical Computation**  
Universidad de Almeria · 2026  
Autor: Pablo Fernandez Ibanez

---

## Descripcion

Sistema agentico que automatiza consultas habituales de Analisis Exploratorio de Datos (EDA) sobre datos tabulares. El usuario escribe preguntas en lenguaje natural y el agente interpreta la intencion, invoca la herramienta correcta y devuelve una respuesta estructurada y trazable.

**Principio fundamental:** el LLM no calcula — solo orquesta. Toda respuesta numerica proviene de funciones deterministas implementadas en pandas.

---

## Stack tecnologico

| Componente | Tecnologia |
|---|---|
| Orquestacion del agente | PydanticAI |
| Inferencia LLM | Ollama local (qwen3:8b) |
| Protocolo de tools | MCP / FastMCP |
| Operaciones analiticas | pandas |
| Interfaz web | Streamlit |
| Validacion de salidas | Pydantic (EDAAnswer) |

---

## Requisitos

- Python 3.11+
- [Ollama](https://ollama.com) instalado y corriendo con el modelo `qwen3:8b`:

```bash
ollama pull qwen3:8b
ollama serve
```

---

## Instalacion

```bash
git clone https://github.com/Pablofdeez/TFM_EDA_AGENT.git
cd TFM_EDA_AGENT
pip install -e .
cp .env.example .env
```

El fichero `.env` contiene la configuracion de Ollama (URL, modelo, api key). Por defecto apunta a `localhost:11434`.

---

## Uso

### Interfaz web (Streamlit)

```bash
streamlit run src/eda_agent/app.py
```

Abre `http://localhost:8501`, sube un CSV o Parquet desde el sidebar y escribe preguntas en el chat.

Ejemplos de preguntas:
- `describe el esquema`
- `cual es la media de revenue`
- `cuantos nulos hay`
- `que columnas parecen fechas`
- `cual fue el revenue maximo en 2024`

### REPL interactivo (terminal)

```bash
python -m eda_agent.agent
```

---

## Benchmark

El benchmark evalua el sistema con 15 preguntas sobre 3 tipologias de datasets y compara el agente con tools frente al LLM directo sin tools.

```bash
# Solo agente con tools
python -m benchmark.runner

# Solo LLM directo sin tools (baseline)
python -m benchmark.runner --baseline

# Comparativa completa (recomendado)
python -m benchmark.runner --compare
```

### Resultados obtenidos (qwen3:8b, local)

| | Agente con tools | LLM directo |
|---|---|---|
| Corrección numerica | **12/12 (100%)** | 8/12 (66%) |
| Salidas estructurales validas | **15/15 (100%)** | — |
| Tool correcta seleccionada | **15/15 (100%)** | — |
| Latencia media | 6.31s | ~16s |

---

## Estructura del proyecto

```
tfm_eda_agent/
├── src/eda_agent/
│   ├── agent.py          # Construccion del agente y REPL
│   ├── app.py            # Interfaz Streamlit (AgentRunner)
│   ├── prompts.py        # System prompt
│   ├── schemas.py        # EDAAnswer y modelos Pydantic
│   ├── memory.py         # Memoria de sesion
│   └── adapters/
│       └── ollama.py     # Adaptador Ollama via API OpenAI-compatible
├── mcp_servers/eda_tools/
│   ├── server.py         # Servidor MCP con 5 tools
│   ├── store.py          # TableStore (DataFrames en memoria)
│   ├── profiling.py      # Perfilado de esquema y deteccion de fechas
│   └── ops.py            # Agregaciones y reporte de nulos
├── benchmark/
│   ├── runner.py         # Runner con modo agente, baseline y compare
│   ├── datasets/         # 3 CSVs de evaluacion
│   ├── questions/        # Preguntas con ground truth por tipologia
│   └── results/          # Resultados JSON de las ejecuciones
├── data/samples/         # Dataset de ejemplo (sales.csv)
├── doc/                  # Memoria del TFM
└── pyproject.toml
```

---

## Tools MCP disponibles

| Tool | Descripcion |
|---|---|
| `load_table` | Carga CSV o Parquet como tabla activa |
| `describe_schema` | Tipo, nulos, unicos y ejemplos por columna |
| `detect_datetime_columns` | Deteccion heuristica de columnas temporales (umbral 85%) |
| `aggregate_metric` | mean / max / min / sum / count con filtro temporal opcional |
| `null_report` | Conteo y porcentaje de nulos por columna |
