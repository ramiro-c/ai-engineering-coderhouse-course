"""Tests unit de helpers de ingesta pura de la pre-entrega 4 (Fase 3, sin red).

Cubren los contratos RF-2/D5/D7: chunk_id() determinista (sha1) para la
idempotencia, build_namespace() con fallback NAMESPACE_DEFAULT y
validate_metadata_size() contra el limite de ~40KB de metadata de Pinecone.
Tambien validan el documento_id unico del corpus de U2 y que get_embeddings()
sea lazy y cacheado (sin llamadas de red).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from config import DATA_DIR, EMBEDDING_MODEL, FUENTE_NAMESPACES, NAMESPACE_DEFAULT
from ingest import build_namespace, chunk_id, validate_metadata_size

_DOCS_CORPUS = (
    sorted((DATA_DIR / "features").glob("*.md"))
    + sorted((DATA_DIR / "tutorial").glob("*.md"))
)


def test_chunk_id_determinista():
    """Mismo namespace y contenido -> mismo id (D5, idempotencia RF-2)."""
    assert chunk_id("fastapi-core", "texto del chunk") == chunk_id(
        "fastapi-core", "texto del chunk"
    )


def test_chunk_id_distinto_contenido_distinto_id():
    assert chunk_id("fastapi-core", "texto A") != chunk_id("fastapi-core", "texto B")


def test_chunk_id_distinto_namespace_distinto_id():
    assert chunk_id("fastapi-core", "mismo texto") != chunk_id(
        "fastapi-tutorial", "mismo texto"
    )


def test_chunk_id_formato_hex():
    assert re.fullmatch(r"[0-9a-f]{16}", chunk_id("docs", "cualquier texto"))


def test_build_namespace_fuentes_conocidas():
    for fuente, ns in FUENTE_NAMESPACES.items():
        assert build_namespace(fuente) == ns


def test_build_namespace_fuente_desconocida():
    assert build_namespace("blog") == NAMESPACE_DEFAULT
    assert NAMESPACE_DEFAULT == "docs"


def test_validate_metadata_size_realista():
    """Metadata realista de un chunk (~2-4KB) entra en el limite de ~40KB."""
    metadata = {
        "texto": "x" * 3_000,
        "source": "routing.md",
        "document_id": "routing.md",
        "seccion": "Routing en FastAPI > Decoradores de metodos HTTP",
        "etiquetas": ["features"],
        "namespace": "fastapi-core",
    }
    assert len(json.dumps(metadata)) < 40_000
    assert validate_metadata_size(metadata) is True


def test_validate_metadata_size_excede_limite():
    """Metadata gigante (texto completo de un doc) queda fuera del limite."""
    metadata = {"texto": "x" * 50_000, "document_id": "routing.md"}
    assert len(json.dumps(metadata)) > 40_000
    assert validate_metadata_size(metadata) is False


def test_document_id_unico_entre_carpetas():
    """Corpus U2: nombres de .md unicos entre features/ y tutorial/ (decision U2)."""
    ids = [p.name for p in _DOCS_CORPUS]
    assert len(ids) == len(set(ids)) >= 8
    assert len(_DOCS_CORPUS) == 12


def test_import_embeddings_no_instancia_sin_key():
    """Importar embeddings.py no instancia el cliente ni exige API key."""
    import embeddings

    embeddings.get_embeddings.cache_clear()
    assert embeddings.get_embeddings.cache_info().currsize == 0


def test_get_embeddings_modelo_y_cache(monkeypatch):
    """get_embeddings() devuelve una unica instancia con el modelo de config."""
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-para-tests-unit")
    from embeddings import get_embeddings

    primera = get_embeddings()
    segunda = get_embeddings()
    assert primera is segunda  # lru_cache: misma instancia indexar/consultar
    assert primera.model == EMBEDDING_MODEL


# --- Fase 4: wiring de upsert real (upsert_corpus) ---


class _VectorStoreStub:
    """Stub de PineconeVectorStore: captura la instanciacion y add_documents."""

    def __init__(self, index, embedding, text_key, namespace):
        self.index = index
        self.embedding = embedding
        self.text_key = text_key
        self.namespace = namespace
        self.add_calls: list[tuple[list, list[str]]] = []

    def add_documents(self, documents, ids=None):
        self.add_calls.append((documents, list(ids or [])))


def _fabrica_de_vectorstores(coleccion: list[_VectorStoreStub]):
    """Devuelve una fabrica que registra cada vectorstore instanciado."""

    def _fabrica(**kwargs):
        vectorstore = _VectorStoreStub(**kwargs)
        coleccion.append(vectorstore)
        return vectorstore

    return _fabrica


def _ids_de(vectorstores: list[_VectorStoreStub]) -> list[str]:
    return [id_ for vs in vectorstores for _, ids in vs.add_calls for id_ in ids]


def _docs_de(vectorstores: list[_VectorStoreStub]) -> list:
    return [doc for vs in vectorstores for docs, _ in vs.add_calls for doc in docs]


def test_upsert_corpus_indexa_por_namespace_de_fuente(monkeypatch):
    """El wiring indexa todo data/ en el namespace de su subcarpeta (D7)."""
    import ingest

    vectorstores: list[_VectorStoreStub] = []
    monkeypatch.setattr(ingest, "PineconeVectorStore", _fabrica_de_vectorstores(vectorstores))

    totales = ingest.upsert_corpus(indice=object(), embeddings=object())

    assert set(totales) == {"fastapi-core", "fastapi-tutorial"}
    assert totales["fastapi-core"] >= 8  # un chunk por doc como minimo
    assert totales["fastapi-tutorial"] >= 4
    assert {vs.text_key for vs in vectorstores} == {"texto"}  # D4
    docs = _docs_de(vectorstores)
    assert len(docs) == sum(totales.values()) >= 12
    for doc in docs:
        assert doc.metadata["document_id"] == doc.metadata["source"]
        assert doc.metadata["document_id"].endswith(".md")
        assert doc.metadata["seccion"]
        assert doc.metadata["namespace"] in {"fastapi-core", "fastapi-tutorial"}
        assert doc.page_content  # texto del chunk


def test_upsert_corpus_es_idempotente(monkeypatch):
    """Re-indexar el corpus produce exactamente los mismos ids (RF-2, D5)."""
    import ingest

    primera: list[_VectorStoreStub] = []
    segunda: list[_VectorStoreStub] = []
    monkeypatch.setattr(ingest, "PineconeVectorStore", _fabrica_de_vectorstores(primera))
    ingest.upsert_corpus(indice=object(), embeddings=object())
    monkeypatch.setattr(ingest, "PineconeVectorStore", _fabrica_de_vectorstores(segunda))
    ingest.upsert_corpus(indice=object(), embeddings=object())

    ids_primera = _ids_de(primera)
    ids_segunda = _ids_de(segunda)
    assert len(ids_primera) == len(ids_segunda) >= 12
    assert ids_primera == ids_segunda  # mismo chunk -> mismo id -> upsert sin duplicar


def test_upsert_corpus_valida_metadata_antes_de_upsert(monkeypatch):
    """Metadata que excede el limite aborta antes de instanciar el vectorstore."""
    import ingest

    vectorstores: list[_VectorStoreStub] = []
    monkeypatch.setattr(ingest, "PineconeVectorStore", _fabrica_de_vectorstores(vectorstores))
    monkeypatch.setattr(ingest, "validate_metadata_size", lambda metadata: False)

    with pytest.raises(ValueError, match="limite"):
        ingest.upsert_corpus(indice=object(), embeddings=object())

    assert vectorstores == []  # nada se instancio ni se hizo upsert
