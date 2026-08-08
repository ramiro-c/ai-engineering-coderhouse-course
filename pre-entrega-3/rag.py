from __future__ import annotations

import logging
import unicodedata

from langchain_core.documents import Document

from chain import generate_response
from config import SIMILARITY_THRESHOLD, TOP_K
from retriever import build_retriever, filter_relevant
from schemas import LlmAnswer, RagGenerationError, RagReference, RagResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NO_SE_RESPONSE = (
    "No lo sé. No encontré información relacionada con tu pregunta en mis apuntes, "
    "así que prefiero no inventar una respuesta."
)


def _es_rechazo(respuesta: LlmAnswer) -> bool:
    """True si el modelo no pudo responder con el contexto.

    Se confía en el flag `answered` en vez de leer la prosa: el modelo rechaza
    de formas distintas ("No lo sé", "No lo sé. El contexto no contiene...") y
    cualquier regla sobre el texto se rompe con la siguiente variante. El
    chequeo del texto queda solo como red por si el flag viene mal.
    """
    if not respuesta.answered:
        return True
    plano = unicodedata.normalize("NFD", respuesta.text.strip().lower())
    plano = "".join(c for c in plano if unicodedata.category(c) != "Mn")
    return plano.strip(" .!¡\"'") == "no lo se"


def _formatear_contexto(relevant: list[tuple[Document, float]]) -> str:
    """Convierte los fragmentos relevantes en un bloque de texto para el LLM."""
    bloques = []
    for i, (doc, _score) in enumerate(relevant, start=1):
        fuente = doc.metadata.get("source", "desconocida")
        bloques.append(f"[{i}] Fuente: {fuente}\n{doc.page_content}")
    return "\n\n---\n\n".join(bloques)


def _referencia(doc: Document) -> RagReference:
    return RagReference(
        source=doc.metadata.get("source", "desconocida"),
        section=doc.metadata.get("seccion", ""),
        snippet=doc.page_content[:200],
    )


def _build_references(relevant: list[tuple[Document, float]]) -> list[RagReference]:
    """Citas de los fragmentos que pasaron el gate.

    No las pide el LLM: salen del fragmento que efectivamente recibió como
    contexto, así una cita no puede ser inventada. Se deduplican por (archivo,
    sección) porque una sección larga entra al top_k partida en varios chunks.
    """
    referencias: dict[tuple[str, str], RagReference] = {}
    for doc, _score in relevant:
        ref = _referencia(doc)
        referencias.setdefault((ref.source, ref.section), ref)
    return list(referencias.values())


async def get_rag_response(query: str) -> RagResponse | RagGenerationError:
    """Responde una consulta con grounding en el corpus.

    Flujo: recupera top_k fragmentos -> gate de relevancia -> si 0 relevantes,
    devuelve "No lo sé" sin llamar al LLM; si no, genera con la cadena A y
    cae a la cadena B si el parser falla.
    """
    retriever = build_retriever()
    scored = retriever.similarity_search_with_relevance_scores(query, k=TOP_K)
    relevant = filter_relevant(scored, threshold=SIMILARITY_THRESHOLD)

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

    # Citar fragmentos que no fundamentaron ninguna respuesta es contradictorio.
    if _es_rechazo(resultado):
        logger.info("El modelo no encontró sustento en los fragmentos recuperados")
        return RagResponse(text=NO_SE_RESPONSE, references=[])

    return RagResponse(
        text=resultado.text,
        references=_build_references(relevant),
    )
