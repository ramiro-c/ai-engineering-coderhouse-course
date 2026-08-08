from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import COLLECTION_NAME, SIMILARITY_THRESHOLD, VECTORSTORE_DIR
from embeddings import get_embeddings


def build_retriever() -> Chroma:
    """Retriever sobre el índice persistido.

    Usa el MISMO embedder que la ingesta (RAG-RET-01): la factory
    `get_embeddings()` está cacheada y se comparte entre index y query.
    """
    if not (VECTORSTORE_DIR / "chroma.sqlite3").exists():
        raise RuntimeError(
            f"No hay índice en {VECTORSTORE_DIR}. Ejecutá primero: python -m ingest"
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=get_embeddings(),
    )


def _filter_relevant(
    results: list[tuple[Document, float]],
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[tuple[Document, float]]:
    """Mantiene solo fragmentos con relevance score >= threshold.

    langchain_chroma con espacio `cosine` mapea distancia -> relevance como
    `1 - distance` (similitud coseno). Un score >= SIMILARITY_THRESHOLD indica
    que el fragmento es afín a la consulta; por debajo se descarta (RAG-GEN-02).
    """
    return [(doc, score) for doc, score in results if score >= threshold]
