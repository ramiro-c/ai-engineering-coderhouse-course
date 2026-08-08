from __future__ import annotations

import logging

from config import SIMILARITY_THRESHOLD, TOP_K
from retriever import _filter_relevant, build_retriever
from schemas import RagGenerationError, RagReference, RagResponse
from chain import generate_response

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NO_SE_RESPONSE = (
    "No lo sé. No encontré información relacionada con tu pregunta en mis apuntes, "
    "así que prefiero no inventar una respuesta."
)


def _formatear_contexto(relevant: list[tuple[object, float]]) -> str:
    """Convierte los fragmentos relevantes en un bloque de texto para el LLM."""
    bloques = []
    for i, (doc, _score) in enumerate(relevant, start=1):
        fuente = doc.metadata.get("source", "desconocida")
        bloques.append(f"[{i}] Fuente: {fuente}\n{doc.page_content}")
    return "\n\n---\n\n".join(bloques)


def _build_references(relevant: list[tuple[object, float]]) -> list[RagReference]:
    """References refleja SOLO los fragmentos que pasaron el gate (RAG-GEN-02)."""
    references = []
    for doc, _score in relevant:
        references.append(
            RagReference(
                source=doc.metadata.get("source", "desconocida"),
                snippet=doc.page_content[:200],
            )
        )
    return references


async def get_rag_response(query: str) -> RagResponse | RagGenerationError:
    """Responde una consulta con grounding en el corpus.

    Flujo: recupera top_k fragmentos -> gate de relevancia -> si 0 relevantes,
    devuelve "No lo sé" sin llamar al LLM; si no, genera con la cadena A y
    cae a la cadena B si el parser falla (RAG-GEN-01).
    """
    retriever = build_retriever()
    scored = retriever.similarity_search_with_relevance_scores(query, k=TOP_K)
    relevant = _filter_relevant(scored, threshold=SIMILARITY_THRESHOLD)

    if not relevant:
        logger.info(
            "0 fragmentos sobre el umbral (%.2f) para la consulta", SIMILARITY_THRESHOLD
        )
        return RagResponse(text=NO_SE_RESPONSE, references=[])

    logger.info("%d fragmentos superaron el gate de relevancia", len(relevant))
    contexto = _formatear_contexto(relevant)
    resultado = await generate_response(pregunta=query, contexto=contexto)

    if isinstance(resultado, RagGenerationError):
        return resultado

    return RagResponse(
        text=resultado.text,
        references=_build_references(relevant),
    )
