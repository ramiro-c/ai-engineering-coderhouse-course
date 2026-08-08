"""Tests unitarios de los schemas Pydantic del RAG (sin modelo ni red)."""

from __future__ import annotations

import pytest

from schemas import RagGenerationError, RagReference, RagResponse


def test_rag_response_minimo():
    respuesta = RagResponse(text="No lo sé")
    assert respuesta.text == "No lo sé"
    assert respuesta.references == []


def test_rag_response_con_references():
    respuesta = RagResponse(
        text="Respuesta fundamentada",
        references=[
            RagReference(source="Sesgos.md", snippet="La falacia de planificación..."),
            RagReference(source="Mercados.md", snippet="Externalidades..."),
        ],
    )
    assert len(respuesta.references) == 2
    assert respuesta.references[0].source == "Sesgos.md"


def test_rag_response_requiere_text():
    with pytest.raises(Exception):
        RagResponse()


def test_rag_generation_error():
    error = RagGenerationError(error="ValueError", detalle="falló el parser")
    assert error.error == "ValueError"
    assert error.detalle.startswith("falló")
