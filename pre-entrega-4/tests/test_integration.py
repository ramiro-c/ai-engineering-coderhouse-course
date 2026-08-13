"""Tests de integración (slow) — índice Pinecone real y embeddings de OpenAI.

Cubren los happy paths de RF-2/RF-3 contra el índice Serverless real: la
ingesta es idempotente (re-indexar no duplica vectores porque los chunk_ids
son deterministas, D5) y el retrieve devuelve top-5 con metadata completa
para citar (texto original vía text_key="texto", D4, y document_id/source).
También verifican el aislamiento por namespace (D7). Se saltean sin
PINECONE_API_KEY/OPENAI_API_KEY (fixtures de conftest).
"""

from __future__ import annotations

import time

import pytest

from config import DATA_DIR, FUENTE_NAMESPACES, INDEX_NAME

pytestmark = pytest.mark.slow


def _conteo_vectores(cliente, namespace: str) -> int:
    """Vectores de un namespace vía describe_index_stats (tolerante al SDK)."""
    stats = cliente.Index(INDEX_NAME).describe_index_stats()
    stats_ns = (stats.namespaces or {}).get(namespace)
    if stats_ns is None:
        return 0
    if hasattr(stats_ns, "vector_count"):
        return stats_ns.vector_count
    return stats_ns.get("vector_count", 0)


def _esperar_conteo(cliente, namespace: str, esperado: int, timeout_segundos: int = 30) -> None:
    """Poll hasta que el namespace refleje el conteo (consistencia eventual)."""
    inicio = time.monotonic()
    while time.monotonic() - inicio < timeout_segundos:
        if _conteo_vectores(cliente, namespace) == esperado:
            return
        time.sleep(2)
    actual = _conteo_vectores(cliente, namespace)
    raise AssertionError(
        f"namespace '{namespace}': {actual} vectores != {esperado} esperados tras "
        f"{timeout_segundos}s (¿ids no deterministas? ¿duplicados?)"
    )


def _buscar_con_reintento(vectorstore, consulta: str, k: int = 5, reintentos: int = 15):
    """Query con reintentos: Pinecone Serverless tiene consistencia eventual."""
    hits = []
    for _ in range(reintentos):
        hits = vectorstore.similarity_search_with_score(consulta, k=k)
        if hits:
            return hits
        time.sleep(2)
    return hits


def test_ingesta_es_idempotente(_pinecone_index):
    """Re-indexar el corpus no duplica vectores (mismo chunk_id, RF-2)."""
    from embeddings import get_embeddings
    from ingest import upsert_corpus

    contexto = _pinecone_index
    totales = contexto["totales"]
    assert totales, "la ingesta no indexó ningún namespace"

    upsert_corpus(contexto["indice"], embeddings=get_embeddings())  # 2da ingesta

    for namespace, esperado in totales.items():
        _esperar_conteo(contexto["cliente"], namespace, esperado)


def test_retrieve_top5_devuelve_metadata_completa(_pinecone_index):
    """Query real: top-5 con metadata de cita (texto original, document_id, RF-3)."""
    from langchain_pinecone import PineconeVectorStore

    from embeddings import get_embeddings

    contexto = _pinecone_index
    vectorstore = PineconeVectorStore(
        index=contexto["indice"],
        embedding=get_embeddings(),
        text_key="texto",
        namespace=FUENTE_NAMESPACES["features"],
    )
    hits = _buscar_con_reintento(
        vectorstore, "¿Cómo defino una ruta con el decorador de APIRouter en FastAPI?"
    )
    assert 0 < len(hits) <= 5
    for documento, _score in hits:
        assert documento.page_content  # texto original (text_key="texto")
        metadata = documento.metadata
        assert metadata.get("document_id")
        assert metadata.get("source") == metadata.get("document_id")
        assert metadata.get("texto") == documento.page_content  # D4
    # Happy path: la query de routing recupera routing.md en el top.
    assert hits[0][0].metadata["document_id"] == "routing.md"


def test_namespace_aísla_fuentes(_pinecone_index):
    """Consultar el namespace de tutorial no recupera features (D7)."""
    from langchain_pinecone import PineconeVectorStore

    from embeddings import get_embeddings

    contexto = _pinecone_index
    ids_tutorial = {p.name for p in sorted((DATA_DIR / "tutorial").glob("*.md"))}
    vectorstore = PineconeVectorStore(
        index=contexto["indice"],
        embedding=get_embeddings(),
        text_key="texto",
        namespace=FUENTE_NAMESPACES["tutorial"],
    )
    hits = _buscar_con_reintento(
        vectorstore, "¿Cómo defino una ruta con APIRouter en FastAPI?", k=3
    )
    assert len(hits) > 0
    for documento, _score in hits:
        assert documento.metadata["document_id"] in ids_tutorial
