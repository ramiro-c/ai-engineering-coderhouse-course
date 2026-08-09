from __future__ import annotations

import argparse
import asyncio
import sys

from ingest import ingest_corpus
from rag import get_rag_response
from schemas import RagGenerationError
from store import IndexNotReadyError

PREGUNTA_RESPONDIBLE = "¿Cómo funciona la matriz de Eisenhower?"
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"

COMANDOS_SALIR = {"q", "quit", "exit", "salir", "salí", "chau"}


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
        print(f"      {respuesta.detalle}\n")
        return
    print(f"\n  {respuesta.text}")
    if respuesta.references:
        print("\n  Referencias:")
        for ref in respuesta.references:
            # Con el archivo solo, un fragmento irrelevante del .md correcto
            # parece una cita válida y esconde un retrieval malo.
            detalle = f"{ref.source} — {ref.section}" if ref.section else ref.source
            print(f"    - {detalle}")
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


def _preparar_indice(force: bool) -> bool:
    """Deja el índice listo antes de responder. False si no se pudo.

    La ingesta es idempotente: llamarla siempre sale gratis con el índice al día
    y evita que la primera corrida explote pidiendo `python -m ingest` a mano.
    """
    try:
        ingest_corpus(force=force)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False
    except IndexNotReadyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return False
    return True


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG semántico local sobre apuntes (chat interactivo o demo)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="mostrar el demo fijo (pregunta respondible + trampa) en lugar del chat",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="reindexar el corpus antes de arrancar",
    )
    args = parser.parse_args()

    if not _preparar_indice(force=args.reindex):
        sys.exit(1)

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
