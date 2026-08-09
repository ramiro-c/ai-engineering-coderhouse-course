from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import COLLECTION_NAME, SIMILARITY_THRESHOLD, VECTORSTORE_DIR
from embeddings import get_embeddings
from store import IndexNotReadyError, index_problem


@lru_cache(maxsize=1)
def build_retriever() -> Chroma:
    """Retriever sobre el índice persistido.

    Usa el MISMO embedder que la ingesta: `get_embeddings()` está cacheada y se
    comparte entre index y query. El retriever también, para no levantar un
    cliente Chroma por consulta.

    `create_collection_if_not_exists=False` es deliberado: el default es True y
    crea una colección vacía en silencio, con lo cual TODA consulta responde
    "No lo sé" como si el gate de relevancia estuviera funcionando bien.
    """
    problema = index_problem()
    if problema is not None:
        raise IndexNotReadyError(f"{problema}. Ejecutá: python -m ingest")
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=get_embeddings(),
        create_collection_if_not_exists=False,
    )


def filter_relevant(
    results: list[tuple[Document, float]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[Document, float]]:
    """Mantiene solo fragmentos con relevance score >= threshold.

    Con espacio `cosine`, langchain_chroma mapea distancia -> relevance como
    `1 - distance`, así que el score es la similitud coseno.
    """
    return [(doc, score) for doc, score in results if score >= threshold]
