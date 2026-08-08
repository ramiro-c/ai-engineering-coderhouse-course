"""Tests de integración del RAG local.

Los tests de integración usan `asyncio.run` (sin pytest-asyncio) y están
marcados como `slow` porque requieren el modelo de embeddings local.
Los tests unitarios (gate, guardas del índice) corren sin modelo ni red.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.documents import Document

import store
from config import SIMILARITY_THRESHOLD
from ingest import ingest_corpus
from rag import _es_rechazo, get_rag_response
from retriever import build_retriever, filter_relevant
from schemas import LlmAnswer, RagGenerationError
from store import IndexNotReadyError, count_chunks

PREGUNTA_RESPONDIBLE = "¿Cómo funciona la matriz de Eisenhower?"
PREGUNTA_TRAMPA = "¿Cuál es la capital de Australia?"


@pytest.mark.slow
def test_pregunta_respondible_con_grounding(vectorstore, llm):
    """Texto en español fundamentado y references no vacías."""
    respuesta = asyncio.run(get_rag_response(PREGUNTA_RESPONDIBLE))
    if isinstance(respuesta, RagGenerationError):
        # Tener credenciales no garantiza poder generar: cuota agotada, 429 o
        # proveedor caído devuelven RagGenerationError. Es un problema de
        # infraestructura, no del pipeline, y se reporta como skip con el motivo
        # a la vista en lugar de un AttributeError sobre `.text`.
        pytest.skip(
            f"el proveedor no pudo generar ({respuesta.error}): {respuesta.detalle}"
        )
    assert respuesta.text.strip(), "la respuesta no puede estar vacía"
    assert "no lo sé" not in respuesta.text.lower(), (
        "una consulta respondible debe responderse con el contexto, no con 'No lo sé'"
    )
    assert len(respuesta.references) > 0, (
        "una consulta respondible debe traer references"
    )


@pytest.mark.slow
def test_pregunta_trampa_no_alucina(vectorstore):
    """Pregunta ajena al corpus -> 'No lo sé' sin references.

    No necesita credenciales: con 0 fragmentos sobre el umbral, el pipeline
    corta antes de llamar al LLM.
    """
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


@pytest.mark.slow
def test_fingerprint_detecta_cambio_de_parametros(vectorstore, monkeypatch):
    """Cambiar el chunking invalida el índice en vez de usarlo viejo."""
    assert store.index_problem() is None, "el fixture deja el índice al día"
    monkeypatch.setattr(store, "CHUNK_SIZE", store.CHUNK_SIZE + 1)
    assert store.index_problem() is not None, (
        "cambiar CHUNK_SIZE debe marcar el índice como desactualizado"
    )


@pytest.mark.parametrize(
    "texto",
    [
        "No lo sé",
        "No lo sé. El contexto provisto no contiene información sobre nginx.",
        "El corpus no habla de eso.",
    ],
)
def test_rechazo_por_flag(texto):
    """Unit: con answered=False es rechazo, diga lo que diga la prosa."""
    assert _es_rechazo(LlmAnswer(text=texto, answered=False))


@pytest.mark.parametrize(
    "texto",
    ["No lo sé", "no lo se", "  No lo sé.  ", "NO LO SÉ!", '"No lo sé"'],
)
def test_rechazo_pelado_aunque_el_flag_venga_mal(texto):
    """Unit: red de seguridad si el modelo dice que no sabe con answered=True."""
    assert _es_rechazo(LlmAnswer(text=texto, answered=True))


@pytest.mark.parametrize(
    "texto",
    [
        "No lo sé con certeza, pero los apuntes mencionan el sesgo de anclaje",
        "El contexto no lo define, pero da ejemplos como el sesgo de confirmación",
        "La matriz de Eisenhower ordena tareas en dos ejes",
    ],
)
def test_respuesta_parcial_no_es_rechazo(texto):
    """Unit: una respuesta parcial SÍ se apoya en el contexto y conserva citas."""
    assert not _es_rechazo(LlmAnswer(text=texto, answered=True))


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
    filtrados = filter_relevant(resultados)
    assert [doc.page_content for doc, _ in filtrados] == ["a", "b"]


def test_gate_sin_relevantes_devuelve_vacio():
    """Unit: si ningún fragmento pasa el umbral, el resultado es []."""
    resultados = [
        (Document(page_content="a", metadata={"source": "x.md"}), 0.2),
        (Document(page_content="b", metadata={"source": "x.md"}), 0.05),
    ]
    assert filter_relevant(resultados) == []


def test_sqlite_presente_sin_coleccion_no_pasa_como_indice_listo(tmp_path, monkeypatch):
    """Regresión: un chroma.sqlite3 suelto no alcanza para dar el índice por bueno.

    Chroma crea la colección vacía en silencio (create_collection_if_not_exists
    default True), así que sin esta guarda toda consulta respondía "No lo sé"
    como si el gate de relevancia estuviera funcionando.
    """
    monkeypatch.setattr(store, "VECTORSTORE_DIR", tmp_path)
    (tmp_path / "chroma.sqlite3").touch()
    assert store.index_problem() is not None


def test_build_retriever_falla_fuerte_sin_indice(tmp_path, monkeypatch):
    """Unit: sin índice usable se lanza IndexNotReadyError, no 0 resultados."""
    monkeypatch.setattr(store, "VECTORSTORE_DIR", tmp_path)
    build_retriever.cache_clear()
    with pytest.raises(IndexNotReadyError, match="python -m ingest"):
        build_retriever()
    build_retriever.cache_clear()
