"""Recuperador híbrido RAGSystem (Fase 5, RF-3/D8).

Cubre la fusión RRF sintética c=60 con pesos 0.5/0.5 (D8), el mapeo
fuente->namespace con fallback "docs" (RF-3 edge, D7) y el retrieve sin
coincidencias -> lista vacía sin excepción (RF-3 edge). El retrieve real se
prueba con stubs de BM25/vectorstore para no tocar red ni API keys; la fusión
RRF la ejecuta el EnsembleRetriever real de langchain_classic.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import rag_system
from config import FUENTE_NAMESPACES, NAMESPACE_DEFAULT
from rag_system import RAGSystem, resolver_namespaces, rrf_combine


def _doc(document_id: str) -> Document:
    """Document con metadata de cita (D9): document_id y texto original."""
    return Document(
        page_content=f"contenido de {document_id}",
        metadata={"document_id": document_id, "texto": f"contenido de {document_id}"},
    )


# --- RRF sintético (D8: c=60, pesos 0.5/0.5) ---


def test_rrf_combine_c60_pesos_5050():
    bm25 = ["a.md", "b.md", "c.md"]
    vectorial = ["a.md", "c.md", "d.md"]
    puntajes = rrf_combine([bm25, vectorial], c=60, pesos=[0.5, 0.5])

    assert puntajes["a.md"] == pytest.approx(0.5 / 61 + 0.5 / 61)
    assert puntajes["c.md"] == pytest.approx(0.5 / 63 + 0.5 / 62)
    assert puntajes["b.md"] == pytest.approx(0.5 / 62)
    assert puntajes["d.md"] == pytest.approx(0.5 / 63)

    orden = sorted(puntajes, key=puntajes.get, reverse=True)
    assert orden == ["a.md", "c.md", "b.md", "d.md"]


def test_rrf_combine_documento_en_ambas_listas_suma_mas():
    """Un documento rankeado por ambos retrievers supera al de uno solo."""
    puntajes = rrf_combine([["x.md"], ["x.md", "y.md"]], c=60, pesos=[0.5, 0.5])
    assert puntajes["x.md"] == pytest.approx(0.5 / 61 + 0.5 / 61)
    assert puntajes["y.md"] == pytest.approx(0.5 / 62)
    assert puntajes["x.md"] > puntajes["y.md"]


def test_rrf_combine_pesos_desbalanceados_respetan_el_peso():
    """Con peso 1.0 a BM25, la segunda lista no aporta al ranking."""
    puntajes = rrf_combine(
        [["a.md", "b.md"], ["b.md", "a.md"]], c=60, pesos=[1.0, 0.0]
    )
    assert puntajes["a.md"] == pytest.approx(1.0 / 61)
    assert puntajes["b.md"] == pytest.approx(1.0 / 62)
    assert puntajes["a.md"] > puntajes["b.md"]


# --- Mapeo fuente->namespace (D7, RF-3 edge) ---


def test_resolver_namespaces_sin_especificar_consulta_las_fuentes():
    """namespace=None -> namespaces de la ingesta real (U4 indexa por fuente)."""
    assert resolver_namespaces(None) == sorted(FUENTE_NAMESPACES.values())


def test_resolver_namespaces_fuente_conocida_mapea_al_namespace():
    assert resolver_namespaces("features") == ["fastapi-core"]
    assert resolver_namespaces("tutorial") == ["fastapi-tutorial"]
    # Un namespace directo ya mapeado se usa tal cual.
    assert resolver_namespaces("fastapi-core") == ["fastapi-core"]


def test_resolver_namespaces_fuente_desconocida_fallback_docs():
    assert resolver_namespaces("desconocida") == [NAMESPACE_DEFAULT]


# --- retrieve() con stubs (sin red ni API keys) ---


class _RetrieverStub(BaseRetriever):
    """Retriever mínimo para inyectar en el EnsembleRetriever real."""

    docs: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self.docs


class _BM25Stub:
    """Stub de BM25Retriever: devuelve los docs del corpus fijados por el test."""

    @classmethod
    def from_documents(cls, documentos, k=5):
        return _RetrieverStub(docs=list(documentos)[:k])


class _VectorstoreStub:
    """Stub de PineconeVectorStore: expone un retriever con hits fijos."""

    def __init__(self, docs, *args, **kwargs):
        self._docs = docs

    def as_retriever(self, **kwargs):
        return _RetrieverStub(docs=self._docs)


class _PineconeStub:
    """Stub del cliente Pinecone: nunca toca la red."""

    def __init__(self, api_key=None):
        pass

    def Index(self, nombre):
        return object()


@pytest.fixture
def sistema_con_stubs(monkeypatch):
    """RAGSystem con BM25/vectorstore/cliente Pinecone stubeados."""

    def _armar(bm25_docs, vs_docs, corpus):
        monkeypatch.setattr(rag_system, "BM25Retriever", _BM25Stub)
        monkeypatch.setattr(
            rag_system,
            "PineconeVectorStore",
            lambda *args, **kwargs: _VectorstoreStub(vs_docs),
        )
        monkeypatch.setattr(rag_system, "Pinecone", _PineconeStub)
        monkeypatch.setattr(rag_system, "get_embeddings", lambda: None)
        monkeypatch.setattr(rag_system, "construir_corpus", lambda: corpus)
        return RAGSystem()

    return _armar


def test_retrieve_sin_coincidencias_devuelve_lista_vacia(sistema_con_stubs):
    """RF-3 edge: sin hits en BM25 ni en el vectorial -> [] sin excepción."""
    sistema = sistema_con_stubs(bm25_docs=[], vs_docs=[], corpus=[])
    assert sistema.retrieve("consulta sin resultados", k=5) == []
    assert sistema.retrieve("también vacía", k=2) == []


def test_retrieve_fusiona_bm25_y_vectorial_con_rrf_y_dedupe(sistema_con_stubs):
    """RRF real (c=60, pesos 0.5/0.5) sobre BM25 + vectorial, dedupe D9."""
    a, b, c, d = _doc("a.md"), _doc("b.md"), _doc("c.md"), _doc("d.md")
    sistema = sistema_con_stubs(
        bm25_docs=[a, b, c], vs_docs=[a, c, d], corpus=[a, b, c]
    )

    hits = sistema.retrieve("consulta sobre a", k=5, namespace="features")

    ids = [h.metadata["document_id"] for h in hits]
    assert ids == ["a.md", "c.md", "b.md", "d.md"]  # fusión RRF c=60 0.5/0.5
    for hit in hits:
        assert hit.metadata["document_id"]
        assert hit.page_content  # texto expuesto para citar (metadata)
    # Dedupe a nivel documento: a.md rankea en BM25 y vectorial, aparece una vez.
    assert len({h.metadata["document_id"] for h in hits}) == len(hits)
