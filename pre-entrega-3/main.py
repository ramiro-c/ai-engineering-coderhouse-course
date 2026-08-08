from __future__ import annotations

import argparse
import asyncio
import sys

from rag import get_rag_response
from schemas import RagGenerationError

PREGUNTA_RESPONDIBLE = "¿Cómo funciona la matriz de Eisenhower?"
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"

COMANDOS_SALIR = {"q", "quit", "exit", "salir", "salí", "chau", "s"}


async def mostrar(query: str) -> None:
    respuesta = await get_rag_response(query)
    if isinstance(respuesta, RagGenerationError):
        print(respuesta.model_dump_json(indent=2, ensure_ascii=False))
        print("(el pipeline devolvió un error, no una respuesta)")
        return
    print(respuesta.model_dump_json(indent=2, ensure_ascii=False))


async def mostrar_chat(query: str) -> None:
    """Versión amigable para el chat: texto + referencias, sin JSON crudo."""
    respuesta = await get_rag_response(query)
    if isinstance(respuesta, RagGenerationError):
        print(f"\n  ⚠️  Error al generar la respuesta: {respuesta.error}")
        return
    print(f"\n  {respuesta.text}")
    if respuesta.references:
        print("\n  Referencias:")
        for ref in respuesta.references:
            print(f"    - {ref.source}")
    print()


async def chat() -> None:
    print("=== Chat RAG sobre tus apuntes ===")
    print("Hacé cualquier pregunta sobre el corpus. Escribí 'salir' para terminar.\n")
    while True:
        try:
            query = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChau 👋")
            return
        if not query:
            continue
        if query.lower() in COMANDOS_SALIR:
            print("Chau 👋")
            return
        await mostrar_chat(query)


async def demo() -> None:
    print("=== Demo RAG semántico local sobre apuntes ===")
    print(f"\n>>> Pregunta (respondible): {PREGUNTA_RESPONDIBLE}")
    await mostrar(PREGUNTA_RESPONDIBLE)
    print(f"\n>>> Pregunta (trampa, sin respuesta en el corpus): {PREGUNTA_TRAMPA}")
    await mostrar(PREGUNTA_TRAMPA)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG semántico local sobre apuntes (chat interactivo o demo)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="mostrar el demo fijo (pregunta respondible + trampa) en lugar del chat",
    )
    args = parser.parse_args()
    if args.demo:
        await demo()
        return
    await chat()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nChau 👋")
        sys.exit(0)
