from __future__ import annotations

import asyncio

from rag import get_rag_response
from schemas import RagGenerationError

PREGUNTA_RESPONDIBLE = "¿Qué es la falacia de la planificación?"
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"


async def mostrar(query: str) -> None:
    print(f"\n>>> Pregunta: {query}")
    respuesta = await get_rag_response(query)
    if isinstance(respuesta, RagGenerationError):
        print(respuesta.model_dump_json(indent=2))
        print("(el pipeline devolvió un error, no una respuesta)")
        return
    print(respuesta.model_dump_json(indent=2, ensure_ascii=False))


async def main() -> None:
    print("=== RAG semántico local sobre apuntes ===")
    await mostrar(PREGUNTA_RESPONDIBLE)
    await mostrar(PREGUNTA_TRAMPA)


if __name__ == "__main__":
    asyncio.run(main())
