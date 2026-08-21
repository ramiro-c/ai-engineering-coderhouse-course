"""Demo async del agente ReAct con LangGraph (pre-entrega-5).

CLI:
    python demo.py                         # REPL: varias preguntas, salir con 'salir'
    python demo.py --thread-id mi-sesion   # REPL con thread_id fijo
    python demo.py --trace                 # demo scriptada cliente 102 + dump de trazas

Imports lazy: importar este módulo NO construye el grafo ni el chat model (todo
se arma dentro de main() async). Debe ser un script en archivo para que
load_dotenv() en config.py resuelva pre-entrega-5/.env (no usar python -c).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

TRACES_DIR = Path(__file__).resolve().parent / "traces"

TRACE_TURN_1 = "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
TRACE_TURN_2 = "¿y el último?"

_COMANDOS_SALIDA = {"salir", "exit", "quit", "q", "s"}


def _es_comando_salida(texto: str) -> bool:
    """¿El texto es un comando de salida del REPL? (salir/exit/q/s...)."""
    return texto.strip().lower() in _COMANDOS_SALIDA


def _content_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes: list[str] = []
        for block in content:
            if isinstance(block, str):
                partes.append(block)
            elif isinstance(block, dict):
                texto = block.get("text")
                if texto:
                    partes.append(str(texto))
            else:
                texto = getattr(block, "text", None)
                if texto:
                    partes.append(_content_str(texto))
        if partes:
            return "\n".join(partes)
    return json.dumps(content, ensure_ascii=False)


def _verificar_credenciales_vertex() -> None:
    """Falla con mensaje claro si Vertex/ADC no está configurado para --trace."""
    from config import (
        GOOGLE_APPLICATION_CREDENTIALS,
        GOOGLE_CLOUD_LOCATION,
        GOOGLE_CLOUD_PROJECT,
        LLM_PROVIDER,
    )

    if LLM_PROVIDER != "gemini":
        return

    faltantes: list[str] = []
    if not GOOGLE_APPLICATION_CREDENTIALS:
        faltantes.append("GOOGLE_APPLICATION_CREDENTIALS")
    elif not Path(GOOGLE_APPLICATION_CREDENTIALS).is_file():
        faltantes.append(
            "GOOGLE_APPLICATION_CREDENTIALS (el archivo del service account no existe)"
        )
    if not GOOGLE_CLOUD_PROJECT:
        faltantes.append("GOOGLE_CLOUD_PROJECT")
    if not GOOGLE_CLOUD_LOCATION:
        faltantes.append("GOOGLE_CLOUD_LOCATION")

    if faltantes:
        detalle = ", ".join(faltantes)
        raise SystemExit(
            "No se puede ejecutar --trace: faltan credenciales Vertex AI "
            f"({detalle}). Copiá .env.example a .env y configurá ADC como en P4."
        )


def _serializar_mensaje(msg: Any) -> dict[str, Any]:
    """JSON de entrega: tipo, name, content, tool_calls."""
    from langchain_core.messages import AIMessage

    payload: dict[str, Any] = {
        "tipo": msg.type,
        "content": _content_str(msg.content),
    }
    name = getattr(msg, "name", None)
    if name:
        payload["name"] = name
    if isinstance(msg, AIMessage) and msg.tool_calls:
        payload["tool_calls"] = msg.tool_calls
    return payload


def _formatear_mensaje_log(msg: Any) -> str:
    """Línea legible estilo ReAct para react-trace.log."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    if isinstance(msg, HumanMessage):
        return f"Usuario: {_content_str(msg.content)}"
    if isinstance(msg, SystemMessage):
        return f"Sistema: {_content_str(msg.content)}"
    if isinstance(msg, ToolMessage):
        tool_name = msg.name or "tool"
        return f"Observación ({tool_name}): {_content_str(msg.content)}"
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            acciones = []
            for call in msg.tool_calls:
                args = call.get("args") or {}
                args_fmt = ", ".join(f"{k}={v!r}" for k, v in args.items())
                acciones.append(f"{call['name']}({args_fmt})")
            linea = f"Asistente → Acción: {'; '.join(acciones)}"
            if msg.content:
                linea = f"Pensamiento: {_content_str(msg.content)}\n{linea}"
            return linea
        return f"Asistente: {_content_str(msg.content)}"
    return f"{msg.type}: {_content_str(msg.content)}"


def _guardar_trazas(messages: list[Any]) -> None:
    """Escribe traces/react-trace.json y traces/react-trace.log."""
    TRACES_DIR.mkdir(parents=True, exist_ok=True)

    serializados = [_serializar_mensaje(m) for m in messages]
    json_path = TRACES_DIR / "react-trace.json"
    json_path.write_text(
        json.dumps(serializados, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    log_lines = [_formatear_mensaje_log(m) for m in messages]
    log_path = TRACES_DIR / "react-trace.log"
    log_path.write_text("\n\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"\nTrazas guardadas en {json_path} y {log_path}")


async def _correr_turno(
    graph: Any,
    pregunta: str,
    config: dict,
    *,
    mostrar_pregunta: bool = True,
    continuar_si_error: bool = True,
) -> None:
    """Un turno del agente: ainvoke + imprime la última respuesta del asistente."""
    from langchain_core.messages import AIMessage, HumanMessage

    if mostrar_pregunta:
        print(f"\nPregunta: {pregunta}\n")

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=config,
        )
    except Exception as exc:
        if not continuar_si_error:
            raise
        print(f"\nError al invocar el agente: {exc}")
        return

    last = result["messages"][-1]
    print("\n--- Respuesta ---")
    if isinstance(last, AIMessage) and last.content:
        print(_content_str(last.content))
    elif isinstance(last, AIMessage) and last.tool_calls:
        print("(El agente terminó en una llamada a herramienta; revisá la traza.)")
    else:
        print(_content_str(getattr(last, "content", last)))


async def _modo_interactivo(graph: Any, config: dict) -> int:
    """REPL async: varias preguntas hasta salir/exit/q o Ctrl+C/Ctrl+D."""
    print("Demo ReAct pre-entrega-5 — escribí una pregunta o 'salir' para terminar.")
    while True:
        try:
            texto = (await asyncio.to_thread(input, "\nPregunta: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            return 0
        if not texto:
            print("(Pregunta vacía — escribí algo o 'salir'.)")
            continue
        if _es_comando_salida(texto):
            print("Hasta luego.")
            return 0
        await _correr_turno(graph, texto, config, mostrar_pregunta=False)
    return 0


async def _modo_trace(graph: Any, config: dict) -> int:
    """Demo scriptada de dos turnos (cliente 102) y dump de trazas ReAct."""
    _verificar_credenciales_vertex()

    print(
        "Modo --trace: demo scriptada del cliente 102 "
        "(dos turnos, thread_id efímero salvo --thread-id explícito)."
    )
    await _correr_turno(graph, TRACE_TURN_1, config, continuar_si_error=False)
    await _correr_turno(graph, TRACE_TURN_2, config, continuar_si_error=False)

    state = graph.get_state(config)
    messages = state.values["messages"]
    _guardar_trazas(messages)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo async del agente ReAct con memoria persistente (LangGraph)."
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help=(
            "thread_id del checkpointer (REPL default: demo; "
            "--trace usa id efímero salvo que lo indiques)"
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Corre la demo scriptada del cliente 102 y guarda trazas en traces/",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Construye el grafo (lazy) y corre REPL o --trace."""
    from graph import build_graph, invoke_config, open_checkpointer

    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.trace:
        thread_id = (
            args.thread_id if args.thread_id is not None else f"trace-{uuid.uuid4()}"
        )
    else:
        thread_id = args.thread_id or "demo"

    checkpointer = open_checkpointer()
    graph = build_graph(checkpointer)
    config = invoke_config(thread_id)

    if args.trace:
        return await _modo_trace(graph, config)
    return await _modo_interactivo(graph, config)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
