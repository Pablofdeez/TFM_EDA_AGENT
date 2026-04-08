"""
benchmark/runner.py

Script de evaluacion del agente EDA. Lanza las preguntas del benchmark
contra el agente real (con tools MCP) y opcionalmente contra el LLM directo
sin tools (baseline), para poder comparar los resultados.

Modos de uso:
    python -m benchmark.runner              # solo agente con tools
    python -m benchmark.runner --baseline   # solo LLM directo sin tools
    python -m benchmark.runner --compare    # ambos + tabla comparativa

Los resultados se guardan automaticamente en benchmark/results/ como JSON.
"""

from __future__ import annotations

import asyncio
import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

# Necesitamos añadir src/ al path para poder importar eda_agent
# ya que este script corre desde benchmark/, fuera del paquete principal
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eda_agent.agent import build_agent  # noqa: E402

console = Console()

# Rutas de los directorios del benchmark
QUESTIONS_DIR = Path(__file__).parent / "questions"   # JSONs con preguntas y valores esperados
DATASETS_DIR  = Path(__file__).parent / "datasets"    # CSVs de evaluacion
RESULTS_DIR   = Path(__file__).parent / "results"     # donde se guardan los JSONs de resultados

# Crear la carpeta de resultados si no existe
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Funciones de evaluacion
# Cada una mide una dimension distinta de la calidad del agente
# ---------------------------------------------------------------------------

def is_structural_valid(answer) -> bool:
    """
    Comprueba que la respuesta tiene la estructura correcta de EDAAnswer.

    El agente esta obligado a devolver siempre un JSON con estos 4 campos.
    Si alguno falta o el texto esta vacio, la respuesta se considera invalida.
    Esto detecta casos en los que el LLM 'se rompe' y no sigue el formato.
    """
    return (
        hasattr(answer, "answer")
        and hasattr(answer, "tools_used")
        and hasattr(answer, "evidence")
        and hasattr(answer, "warnings")
        and isinstance(answer.answer, str)
        and len(answer.answer) > 0
    )


def check_correct_tool(answer, expected_tools: list[str]) -> bool:
    """
    Comprueba que el agente invoco la tool adecuada para la pregunta.

    Por ejemplo, para 'cuantos nulos hay' esperamos null_report.
    Si el agente usa aggregate_metric en su lugar, es un fallo de interpretacion.
    Usamos 'any' porque algunas preguntas encadenan varias tools (ej: fecha + agregacion).
    """
    if not expected_tools:
        return True
    used = set(answer.tools_used)
    return any(t in used for t in expected_tools)


def check_value_match(answer, check: dict | None) -> bool | None:
    """
    Comprueba si el valor numerico en 'evidence' coincide con el esperado.

    Devuelve None si la pregunta no tiene valor esperado (ej: describe_schema).
    Usa una tolerancia configurable por pregunta para evitar fallos por redondeo.
    El valor viene de pandas (determinista), no del LLM, por eso deberia ser exacto.
    """
    if check is None:
        return None  # esta pregunta no tiene valor numerico que verificar

    key       = check["evidence_key"]
    expected  = check["expected"]
    tolerance = check.get("tolerance", 0)

    value = answer.evidence.get(key)
    if value is None:
        return False  # el agente no devolvio el campo esperado en evidence

    try:
        diff = abs(float(value) - float(expected))
        return diff <= tolerance
    except (TypeError, ValueError):
        # Si no es numerico, comparamos como string (ej: nombres de columnas)
        return str(value) == str(expected)


# ---------------------------------------------------------------------------
# Runner principal: agente con tools MCP
# ---------------------------------------------------------------------------

async def run_benchmark() -> list[dict]:
    """
    Ejecuta las 15 preguntas del benchmark contra el agente con tools MCP.

    Para cada pregunta:
    1. Carga el dataset en el agente (load_table)
    2. Inyecta el contexto de la tabla activa en la pregunta
    3. Lanza la pregunta y mide el tiempo de respuesta
    4. Evalua la respuesta con las 3 metricas

    El agente se mantiene vivo durante todo el benchmark (async with agent:)
    para que el TableStore conserve las tablas entre preguntas del mismo dataset.
    """

    # Cargamos todas las preguntas de los 3 ficheros JSON ordenados por tipologia
    question_files = sorted(QUESTIONS_DIR.glob("tipologia_*.json"))
    all_questions: list[dict] = []
    for qf in question_files:
        all_questions.extend(json.loads(qf.read_text(encoding="utf-8")))

    console.print(f"\n[bold]Benchmark EDA Agent[/bold] — {len(all_questions)} preguntas, 3 tipologias\n")

    agent = build_agent()
    results: list[dict] = []

    # async with agent: arranca el servidor MCP y lo mantiene vivo durante todo el benchmark
    async with agent:
        for q in all_questions:
            dataset   = q["dataset"]
            question  = q["question"]
            qid       = q["id"]
            tipologia = q["tipologia"]

            dataset_path = DATASETS_DIR / f"{dataset}.csv"

            console.print(f"[cyan]{qid}[/cyan] [{tipologia}] {question[:60]}")

            try:
                # Paso 1: cargar la tabla en el TableStore del servidor MCP
                load_msg = f"Carga {dataset_path} como {dataset}"
                await agent.run(load_msg)

                # Paso 2: construir la pregunta con contexto de la tabla activa
                # Esto es necesario porque cada agent.run() es una conversacion nueva
                # y el LLM no recuerda que tabla esta cargada entre preguntas
                import pandas as pd
                df_cols = list(pd.read_csv(dataset_path).columns)
                ctx = (
                    f"[Contexto: la tabla '{dataset}' ya esta cargada y activa "
                    f"con columnas: {', '.join(df_cols)}. "
                    f"NO necesitas usar load_table, usa directamente las otras tools.]"
                )
                full_query = f"{ctx} {question}"

                # Paso 3: lanzar la pregunta y medir latencia
                t0 = time.perf_counter()
                result = await agent.run(full_query)
                latency = round(time.perf_counter() - t0, 2)

                answer = result.output

                # Paso 4: evaluar la respuesta con las 3 metricas
                structural = is_structural_valid(answer)
                tool_ok    = check_correct_tool(answer, q.get("expected_tools", []))
                val_match  = check_value_match(answer, q.get("check"))

                # Una pregunta es OK si pasa las 3 metricas (val_match puede ser None = no aplica)
                status = "OK" if (structural and tool_ok and val_match is not False) else "FAIL"
                color  = "green" if status == "OK" else "red"
                console.print(
                    f"  [{color}]{status}[/{color}] "
                    f"struct={structural} tool={tool_ok} val={val_match} "
                    f"lat={latency}s"
                )

                results.append({
                    "id": qid,
                    "tipologia": tipologia,
                    "dataset": dataset,
                    "question": question,
                    "status": status,
                    "structural_valid": structural,
                    "correct_tool": tool_ok,
                    "value_match": val_match,
                    "latency_s": latency,
                    "tools_used": answer.tools_used,
                    "answer_preview": answer.answer[:120],
                    "warnings": answer.warnings,
                })

            except Exception as exc:
                # Si el agente lanza una excepcion, la registramos como ERROR
                console.print(f"  [red]ERROR[/red] {exc}")
                results.append({
                    "id": qid,
                    "tipologia": tipologia,
                    "dataset": dataset,
                    "question": question,
                    "status": "ERROR",
                    "structural_valid": False,
                    "correct_tool": False,
                    "value_match": False,
                    "latency_s": None,
                    "tools_used": [],
                    "answer_preview": str(exc)[:120],
                    "warnings": [],
                })

    return results


# ---------------------------------------------------------------------------
# Resumen e informe
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    """
    Imprime tabla detallada y metricas globales por consola.
    Incluye desglose por tipologia para ver donde falla el agente.
    """
    console.print("\n")

    # Tabla con el resultado de cada pregunta individual
    table = Table(title="Resultados del Benchmark", show_lines=True)
    table.add_column("ID",         style="cyan",  no_wrap=True)
    table.add_column("Tip.",       justify="center")
    table.add_column("Pregunta",   max_width=40)
    table.add_column("Status",     justify="center")
    table.add_column("Struct",     justify="center")
    table.add_column("Tool",       justify="center")
    table.add_column("Valor",      justify="center")
    table.add_column("Lat (s)",    justify="right")

    for r in results:
        status_fmt = f"[green]{r['status']}[/green]" if r["status"] == "OK" else f"[red]{r['status']}[/red]"
        def fmt_bool(v):
            if v is None:  return "[dim]N/A[/dim]"
            return "[green]si[/green]" if v else "[red]no[/red]"
        table.add_row(
            r["id"],
            str(r["tipologia"]),
            r["question"][:40],
            status_fmt,
            fmt_bool(r["structural_valid"]),
            fmt_bool(r["correct_tool"]),
            fmt_bool(r["value_match"]),
            str(r["latency_s"]) if r["latency_s"] else "-",
        )

    console.print(table)

    # Calcular metricas agregadas
    total      = len(results)
    ok_count   = sum(1 for r in results if r["status"] == "OK")
    struct_count = sum(1 for r in results if r["structural_valid"])
    tool_count   = sum(1 for r in results if r["correct_tool"])
    val_results  = [r for r in results if r["value_match"] is not None]
    val_count    = sum(1 for r in val_results if r["value_match"])
    latencies    = [r["latency_s"] for r in results if r["latency_s"] is not None]
    avg_lat      = round(sum(latencies) / len(latencies), 2) if latencies else None

    console.print(f"\n[bold]Metricas globales ({total} preguntas)[/bold]")
    console.print(f"  Tasa de exito total:             {ok_count}/{total} ({100*ok_count//total}%)")
    console.print(f"  Salidas estructurales validas:   {struct_count}/{total} ({100*struct_count//total}%)")
    console.print(f"  Tool correcta seleccionada:      {tool_count}/{total} ({100*tool_count//total}%)")
    if val_results:
        console.print(f"  Valor correcto (con tolerancia): {val_count}/{len(val_results)} ({100*val_count//len(val_results)}%)")
    if avg_lat:
        console.print(f"  Latencia media:                  {avg_lat}s")

    # Desglose por tipologia: util para ver si el agente falla mas en datos sucios
    console.print(f"\n[bold]Desglose por tipologia[/bold]")
    for tip in [1, 2, 3]:
        tip_results = [r for r in results if r["tipologia"] == tip]
        tip_ok  = sum(1 for r in tip_results if r["status"] == "OK")
        tip_lat = [r["latency_s"] for r in tip_results if r["latency_s"]]
        avg_tip_lat = round(sum(tip_lat) / len(tip_lat), 2) if tip_lat else "-"
        console.print(f"  Tipologia {tip}: {tip_ok}/{len(tip_results)} OK — latencia media {avg_tip_lat}s")


def save_results(results: list[dict], suffix: str = "") -> Path:
    """Guarda los resultados en un JSON con timestamp para tener historico de ejecuciones."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"benchmark_{suffix}_{ts}.json"
    out_path.write_text(
        json.dumps({"timestamp": ts, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# Baseline: LLM directo sin tools
# Mismas preguntas pero el LLM recibe el CSV en el prompt y calcula el solo.
# Sirve como comparativa para demostrar el valor de usar tools deterministas.
# ---------------------------------------------------------------------------

async def run_baseline() -> list[dict]:
    """
    Ejecuta las preguntas numericas contra el LLM directo, sin tools ni MCP.

    Para cada pregunta:
    - Se incluyen las primeras 25 filas del CSV en el prompt como texto plano
    - Se pide al modelo que responda SOLO con el numero
    - Se extrae el numero de la respuesta con una expresion regular

    El objetivo es demostrar que el LLM comete errores cuando calcula por si mismo,
    especialmente en medias con valores nulos y en conteos exactos.
    Esto justifica la decision de arquitectura: el LLM orquesta, pandas calcula.
    """
    import httpx
    from dotenv import load_dotenv
    load_dotenv()

    base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    api_key    = os.getenv("OLLAMA_API_KEY", "ollama")

    # Cargar solo las preguntas con valor numerico esperado
    # (las de describe_schema no tienen numero que verificar)
    question_files = sorted(QUESTIONS_DIR.glob("tipologia_*.json"))
    all_questions: list[dict] = []
    for qf in question_files:
        all_questions.extend(json.loads(qf.read_text(encoding="utf-8")))
    numeric_questions = [q for q in all_questions if q.get("check") is not None]

    console.print(f"\n[bold]Baseline LLM directo (sin tools)[/bold] — {len(numeric_questions)} preguntas numericas\n")

    results: list[dict] = []

    async with httpx.AsyncClient(timeout=120) as client:
        for q in numeric_questions:
            dataset  = q["dataset"]
            question = q["question"]
            qid      = q["id"]
            check    = q["check"]

            # Preparar el CSV como texto plano para incluirlo en el prompt
            dataset_path = DATASETS_DIR / f"{dataset}.csv"
            import pandas as pd
            df = pd.read_csv(dataset_path)
            csv_text = df.head(25).to_csv(index=False)

            # El prompt incluye los datos directamente y pide solo el numero
            # Sin tools ni instrucciones adicionales: el LLM tiene que calcular solo
            prompt = (
                f"Tienes los siguientes datos en formato CSV:\n\n{csv_text}\n\n"
                f"Pregunta: {question}\n\n"
                f"Responde UNICAMENTE con el valor numerico, sin texto adicional. "
                f"Solo el numero, nada mas."
            )

            console.print(f"[cyan]{qid}[/cyan] {question[:55]}")

            t0 = time.perf_counter()
            try:
                # Llamada directa a la API de Ollama, sin pasar por PydanticAI ni MCP
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,  # temperatura 0 para respuestas deterministas
                    },
                )
                resp.raise_for_status()
                latency = round(time.perf_counter() - t0, 2)

                raw_answer = resp.json()["choices"][0]["message"]["content"].strip()

                # Extraer el numero de la respuesta del LLM
                # Primero eliminamos el bloque <think>...</think> que genera qwen3
                import re
                clean = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                # Buscamos el primer numero (entero o decimal, positivo o negativo)
                match = re.search(r"-?\d+(?:[.,]\d+)?", clean)
                parsed_value = float(match.group().replace(",", ".")) if match else None

                # Comparar con el valor esperado usando la misma tolerancia que el agente
                expected  = check["expected"]
                tolerance = check.get("tolerance", 0)
                if parsed_value is not None:
                    val_match = abs(parsed_value - float(expected)) <= tolerance
                else:
                    val_match = False  # no se pudo extraer ningun numero

                color = "green" if val_match else "red"
                console.print(
                    f"  [{color}]{'OK' if val_match else 'FAIL'}[/{color}] "
                    f"esperado={expected} obtenido={parsed_value} lat={latency}s"
                )

                results.append({
                    "id": qid,
                    "tipologia": q["tipologia"],
                    "dataset": dataset,
                    "question": question,
                    "expected": expected,
                    "llm_raw": raw_answer[:120],
                    "llm_parsed": parsed_value,
                    "value_match": val_match,
                    "latency_s": latency,
                })

            except Exception as exc:
                latency = round(time.perf_counter() - t0, 2)
                console.print(f"  [red]ERROR[/red] {exc}")
                results.append({
                    "id": qid,
                    "tipologia": q["tipologia"],
                    "dataset": dataset,
                    "question": question,
                    "expected": check["expected"],
                    "llm_raw": str(exc)[:120],
                    "llm_parsed": None,
                    "value_match": False,
                    "latency_s": latency,
                })

    return results


def print_comparison(agent_results: list[dict], baseline_results: list[dict]) -> None:
    """
    Imprime la tabla comparativa entre el agente con tools y el LLM directo.

    Esta es la tabla mas importante del TFM: muestra cuantas preguntas
    responde correctamente cada enfoque y donde falla el LLM sin tools.
    """

    # Indexamos los resultados del baseline por id para cruzarlos con el agente
    baseline_idx = {r["id"]: r for r in baseline_results}

    # Solo comparamos las preguntas con valor numerico esperado
    agent_numeric = [r for r in agent_results if r["value_match"] is not None]

    table = Table(title="Comparativa: Agente con Tools vs LLM Directo", show_lines=True)
    table.add_column("ID",          style="cyan", no_wrap=True)
    table.add_column("Pregunta",    max_width=35)
    table.add_column("Esperado",    justify="right")
    table.add_column("Agente",      justify="center")
    table.add_column("LLM directo", justify="center")
    table.add_column("LLM obtuvo",  justify="right")

    agent_ok    = 0
    baseline_ok = 0

    for r in agent_numeric:
        qid = r["id"]
        b   = baseline_idx.get(qid)

        agent_val = "[green]OK[/green]" if r["value_match"] else "[red]FAIL[/red]"
        if r["value_match"]: agent_ok += 1

        if b:
            base_val = "[green]OK[/green]" if b["value_match"] else "[red]FAIL[/red]"
            llm_got  = str(b["llm_parsed"]) if b["llm_parsed"] is not None else "[dim]no parseable[/dim]"
            if b["value_match"]: baseline_ok += 1
        else:
            base_val = "[dim]-[/dim]"
            llm_got  = "[dim]-[/dim]"

        # Recuperamos el valor esperado del fichero JSON de preguntas
        tip_name = "clean" if r["tipologia"] == 1 else "messy" if r["tipologia"] == 2 else "numeric"
        q_file   = QUESTIONS_DIR / f"tipologia_{r['tipologia']}_{tip_name}.json"
        check    = next(
            (q["check"] for q in json.loads(q_file.read_text(encoding="utf-8")) if q["id"] == qid),
            {}
        )
        expected = str(check.get("expected", "?"))

        table.add_row(qid, r["question"][:35], expected, agent_val, base_val, llm_got)

    console.print(table)

    # Resumen final con la mejora del sistema agentico respecto al baseline
    total = len(agent_numeric)
    console.print(f"\n[bold]Resumen comparativo ({total} preguntas numericas)[/bold]")
    console.print(f"  Agente con tools:  {agent_ok}/{total} correctas ({100*agent_ok//total}%)")
    console.print(f"  LLM directo:       {baseline_ok}/{total} correctas ({100*baseline_ok//total}%)")
    console.print(
        f"\n  [bold]Mejora del sistema agentico: "
        f"+{agent_ok - baseline_ok} preguntas correctas[/bold]"
    )


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    # En Windows forzamos UTF-8 para evitar errores de encoding en la consola
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Seleccionar modo segun argumento de linea de comandos
    mode = sys.argv[1] if len(sys.argv) > 1 else "agent"

    if mode == "--baseline":
        # Solo ejecutar el LLM directo sin tools
        baseline_results = asyncio.run(run_baseline())
        save_results(baseline_results, suffix="baseline")
        result_files = sorted(RESULTS_DIR.glob("benchmark_agent_*.json"))
        if result_files:
            agent_results = json.loads(result_files[-1].read_text(encoding="utf-8"))["results"]
            print_comparison(agent_results, baseline_results)
        else:
            console.print("[yellow]No hay resultados del agente guardados. Ejecuta primero sin --baseline.[/yellow]")

    elif mode == "--compare":
        # Ejecutar ambos y mostrar la tabla comparativa
        console.print("[bold]Modo comparativo: ejecutando agente y baseline...[/bold]\n")
        agent_results = asyncio.run(run_benchmark())
        save_results(agent_results, suffix="agent")
        print_summary(agent_results)
        baseline_results = asyncio.run(run_baseline())
        save_results(baseline_results, suffix="baseline")
        print_comparison(agent_results, baseline_results)

    else:
        # Modo por defecto: solo el agente con tools MCP
        results  = asyncio.run(run_benchmark())
        out_path = save_results(results, suffix="agent")
        print_summary(results)
        console.print(f"\n[dim]Resultados guardados en: {out_path}[/dim]")


if __name__ == "__main__":
    main()
