from __future__ import annotations

import logging

from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EXPECTED_CORPUS,
    VECTORSTORE_DIR,
)
from embeddings import get_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _validate_corpus() -> list[tuple[str, str]]:
    """Valida que los 4 apuntes existan y devuelve (nombre, contenido).

    Si falta alguno, lanza FileNotFoundError SIN persistir nada (RAG-ING-01).
    """
    missing = [name for name in EXPECTED_CORPUS if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos del corpus en {DATA_DIR}: {missing}".format(
                DATA_DIR=DATA_DIR, missing=", ".join(missing)
            )
        )
    return [(name, (DATA_DIR / name).read_text(encoding="utf-8")) for name in EXPECTED_CORPUS]


def _index_ready() -> bool:
    """True si la colección ya existe con al menos un documento (RAG-ING-02)."""
    if not (VECTORSTORE_DIR / "chroma.sqlite3").exists():
        return False
    try:
        client = PersistentClient(path=str(VECTORSTORE_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except NotFoundError:
        return False


def count_chunks() -> int:
    """Cantidad de chunks persistidos en el índice (0 si no existe)."""
    if not _index_ready():
        return 0
    client = PersistentClient(path=str(VECTORSTORE_DIR))
    return client.get_collection(COLLECTION_NAME).count()


def ingest_corpus() -> int:
    """Indexa el corpus en Chroma (colección local persistente).

    Devuelve la cantidad de chunks indexados; 0 si el índice ya existía
    (idempotente: no regenera embeddings ni duplica chunks, RAG-ING-02).
    """
    if _index_ready():
        logger.info("Índice ya existente en %s — se saltea el reindexado", VECTORSTORE_DIR)
        return 0

    corpus = _validate_corpus()
    documents = [
        Document(page_content=contenido, metadata={"source": nombre})
        for nombre, contenido in corpus
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Dividiendo corpus en %d chunks", len(chunks))

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    logger.info("Índice persistido en %s", VECTORSTORE_DIR)
    return len(chunks)


if __name__ == "__main__":
    indexados = ingest_corpus()
    print(f"Chunks indexados: {indexados}")
