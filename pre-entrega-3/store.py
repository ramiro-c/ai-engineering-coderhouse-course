"""Acceso de bajo nivel al índice Chroma persistido.

Único módulo que sabe cómo está guardado el índice. Lo usan tanto la ingesta
(escritura) como el retriever (lectura), para que ambos coincidan en qué
significa exactamente "el índice está listo": si cada uno lo chequeara a su
manera, se puede consultar contra una colección vacía sin que nadie avise.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)

FINGERPRINT_KEY = "corpus_fingerprint"

# Subila a mano al cambiar CÓMO se trocea el corpus: entra en la huella y así
# invalida los índices armados con la estrategia anterior.
CHUNKING_VERSION = 3


class IndexNotReadyError(RuntimeError):
    """El índice no existe, está vacío o no corresponde al corpus actual."""


def corpus_files() -> list[Path]:
    """Apuntes .md del corpus, en orden estable.

    Por glob: sumar un .md a data/ alcanza para que entre en la próxima ingesta.
    """
    return sorted(DATA_DIR.glob("*.md"))


def corpus_fingerprint() -> str:
    """Huella del contenido del corpus + modelo + chunking.

    Se hashea el contenido y no el mtime: así un `git checkout` no dispara
    reindexados falsos.
    """
    huella = hashlib.sha256()
    huella.update(
        f"{EMBEDDING_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}|v{CHUNKING_VERSION}".encode()
    )
    for path in corpus_files():
        huella.update(path.name.encode())
        huella.update(path.read_bytes())
    return huella.hexdigest()[:16]


def get_collection() -> Collection | None:
    """Colección persistida, o None si todavía no hay índice."""
    if not (VECTORSTORE_DIR / "chroma.sqlite3").exists():
        return None
    try:
        client = PersistentClient(path=str(VECTORSTORE_DIR))
        return client.get_collection(COLLECTION_NAME)
    except (NotFoundError, ValueError):
        return None


def drop_collection() -> None:
    """Borra la colección para poder reindexar sin duplicar chunks."""
    if get_collection() is None:
        return
    PersistentClient(path=str(VECTORSTORE_DIR)).delete_collection(COLLECTION_NAME)


def count_chunks() -> int:
    """Cantidad de chunks persistidos (0 si no hay colección)."""
    collection = get_collection()
    return 0 if collection is None else collection.count()


def index_problem() -> str | None:
    """Motivo por el que el índice no sirve, o None si está listo.

    Chequea lo que la existencia de `chroma.sqlite3` no garantiza: que la
    colección exista, tenga documentos y corresponda al corpus actual.
    """
    collection = get_collection()
    if collection is None:
        return f"no hay una colección '{COLLECTION_NAME}' en {VECTORSTORE_DIR}"
    if collection.count() == 0:
        return f"la colección '{COLLECTION_NAME}' está vacía"
    guardada = (collection.metadata or {}).get(FINGERPRINT_KEY)
    if guardada != corpus_fingerprint():
        return (
            "el índice no corresponde al corpus actual: cambió algún apunte, el "
            "modelo de embeddings o el chunking"
        )
    return None
