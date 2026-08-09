from __future__ import annotations

import argparse
import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)
from embeddings import get_embeddings
from store import (
    FINGERPRINT_KEY,
    corpus_files,
    corpus_fingerprint,
    count_chunks,
    drop_collection,
    index_problem,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

__all__ = ["count_chunks", "ingest_corpus"]


ENCABEZADOS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _titulo(metadata: dict[str, str]) -> str:
    """Ruta de encabezados de una sección, p. ej. 'Sesgos > Sesgo de Confirmación'."""
    return " > ".join(v for k, v in metadata.items() if k.startswith("h"))


def build_chunks() -> list[Document]:
    """Trocea el corpus por secciones de markdown.

    El título de la sección se escribe DENTRO del texto que se embebe: sin eso
    la sección "Sesgo de Disponibilidad" no matchea "¿qué es el sesgo de
    disponibilidad?", porque el cuerpo explica el concepto sin nombrarlo nunca.

    Las secciones largas se subdividen para no pasar el tope del embedder.

    Si no hay apuntes, lanza FileNotFoundError SIN persistir nada.
    """
    files = corpus_files()
    if not files:
        raise FileNotFoundError(f"No hay apuntes .md en {DATA_DIR}")

    secciones = MarkdownHeaderTextSplitter(headers_to_split_on=ENCABEZADOS)
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[Document] = []
    for path in files:
        for seccion in secciones.split_text(path.read_text(encoding="utf-8")):
            titulo = _titulo(seccion.metadata)
            for trozo in sub_splitter.split_text(seccion.page_content):
                chunks.append(
                    Document(
                        page_content=f"{titulo}\n{trozo}" if titulo else trozo,
                        metadata={"source": path.name, "seccion": titulo},
                    )
                )
    return chunks


TRUNCADO_ACEPTABLE = 0.10


def _avisar_si_hay_truncado(chunks: list[Document]) -> None:
    """Avisa si los chunks no entran en lo que el embedder puede leer.

    El modelo descarta en silencio lo que pase de `max_seq_length`: ese texto
    queda indexado pero invisible para la búsqueda. No se corrige el CHUNK_SIZE
    del usuario, solo se avisa.
    """
    modelo = get_embeddings()._client
    limite = modelo.max_seq_length
    tokenizer = modelo.tokenizer
    largos = [len(tokenizer.encode(c.page_content)) for c in chunks]
    truncados = [n for n in largos if n > limite]
    if len(truncados) <= TRUNCADO_ACEPTABLE * len(chunks):
        return

    porcentaje = 100 * len(truncados) / len(chunks)
    # Regla empírica para español: ~3 caracteres por word piece.
    sugerido = int(limite * 3)
    logger.warning(
        "%d de %d chunks (%.0f%%) superan los %d word pieces que embebe %s: "
        "el texto sobrante queda indexado pero INVISIBLE para la búsqueda. "
        "Bajá CHUNK_SIZE (ahora %d) a ~%d en tu .env y reindexá.",
        len(truncados),
        len(chunks),
        porcentaje,
        limite,
        EMBEDDING_MODEL,
        CHUNK_SIZE,
        sugerido,
    )


def ingest_corpus(force: bool = False) -> int:
    """Indexa el corpus en Chroma (colección local persistente).

    Devuelve la cantidad de chunks indexados; 0 si el índice ya estaba al día
    (idempotente: no regenera embeddings ni duplica chunks). Reindexa solo si
    cambió el corpus, el modelo de embeddings o el chunking, o si `force`.
    """
    problema = index_problem()
    if problema is None and not force:
        logger.info("Índice al día en %s — se saltea el reindexado", VECTORSTORE_DIR)
        return 0

    logger.info("Reindexando: %s", "forzado por --reindex" if force else problema)

    chunks = build_chunks()
    logger.info(
        "Dividiendo %d apuntes en %d chunks por sección",
        len(corpus_files()),
        len(chunks),
    )
    _avisar_si_hay_truncado(chunks)

    drop_collection()  # si no, el reindexado apila los nuevos sobre los viejos

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
            FINGERPRINT_KEY: corpus_fingerprint(),
        },
    )
    logger.info("Índice persistido en %s", VECTORSTORE_DIR)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indexa los apuntes de data/ en el vectorstore local"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="reindexar aunque el índice figure al día",
    )
    args = parser.parse_args()
    indexados = ingest_corpus(force=args.reindex)
    print(f"Chunks indexados: {indexados} (total en el índice: {count_chunks()})")
