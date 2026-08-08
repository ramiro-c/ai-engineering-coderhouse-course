"""Tests de integración del RAG local.

Los tests de integración usan `asyncio.run` (sin pytest-asyncio) y están
marcados como `slow` porque requieren el modelo de embeddings local.
Los tests unitarios (gate) corren sin modelo ni red.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document

from config import SIMILARITY_THRESHOLD
from ingest import count_chunks, ingest_corpus
from rag import get_rag_response
from retriever import _filter_relevant

PREGUNTA_RESPONDIBLE = "¿Cómo funciona la matriz de Eisenhower?"
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"


@pytest.mark.slow
def test_pregunta_respondible_con_grounding(vectorstore):
    """Texto en español fundamentado y references no vacías."""
    respuesta = asyncio.run(get_rag_response(PREGUNTA_RESPONDIBLE))
    assert respuesta.text.strip(), "la respuesta no puede estar vacía"
    assert "no lo sé" not in respuesta.text.lower(), (
        "una consulta respondible debe responderse con el contexto, no con 'No lo sé'"
    )
    assert len(respuesta.references) > 0, (
        "una consulta respondible debe traer references"
    )


@pytest.mark.slow
def test_pregunta_trampa_no_alucina(vectorstore):
    """Pregunta ajena al corpus -> 'No lo sé' sin references."""
    respuesta = asyncio.run(get_rag_response(PREGUNTA_TRAMPA))
    assert "no lo sé" in respuesta.text.lower()
    assert respuesta.references == []


@pytest.mark.slow
def test_ingesta_idempotente(vectorstore):
    """Re-ingestar no duplica chunks."""
    antes = count_chunks()
    assert antes > 0, "el fixture debe haber indexado chunks"
    reindexados = ingest_corpus()
    despues = count_chunks()
    assert reindexados == 0, "la segunda ingesta debe saltear el reindexado"
    assert despues == antes, "el conteo de chunks no debe cambiar"


def test_gate_filtra_por_umbral():
    """Unit: fragmentos con score >= umbral se mantienen; el resto se descarta."""
    resultados = [
        (Document(page_content="a", metadata={"source": "x.md"}), 0.9),
        (Document(page_content="b", metadata={"source": "x.md"}), SIMILARITY_THRESHOLD),
        (
            Document(page_content="c", metadata={"source": "x.md"}),
            SIMILARITY_THRESHOLD - 0.01,
        ),
        (Document(page_content="d", metadata={"source": "x.md"}), 0.1),
    ]
    filtrados = _filter_relevant(resultados)
    assert [doc.page_content for doc, _ in filtrados] == ["a", "b"]


def test_gate_sin_relevantes_devuelve_vacio():
    """Unit: si ningún fragmento pasa el umbral, el resultado es []."""
    resultados = [
        (Document(page_content="a", metadata={"source": "x.md"}), 0.2),
        (Document(page_content="b", metadata={"source": "x.md"}), 0.05),
    ]
    assert _filter_relevant(resultados) == []
