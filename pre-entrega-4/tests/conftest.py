"""Fixtures compartidos para los tests de la pre-entrega 4.

Los tests de integración (marcados `slow`) necesitan red y el índice Pinecone
real. ENMIENDA 2026-08-12 (U7): los embeddings son locales HuggingFace
(all-MiniLM-L6-v2, 384d) y ya NO dependen de OPENAI_API_KEY; el gate pasa a
exigir que el índice real exista a DIMENSION=384 (el orquestador lo RECREA a
384d en el harness tras U7; init_index.py crea, nunca borra). La verificación
es una llamada describe_index de SOLO LECTURA, cacheada por sesión. Lección
#870: un índice inexistente lanza pinecone.exceptions.NotFoundException, no
devuelve None — el gate la captura y saltea. A diferencia de pre-entrega-3,
los imports de pinecone/langchain se hacen dentro de los fixtures (no al
nivel del módulo), para que la colección de los tests unit no dependa del
entorno real.
"""

from __future__ import annotations

import pytest

from config import DIMENSION, INDEX_NAME, PINECONE_API_KEY

_estado_indice_384d: bool | None = None


def _indice_listo_para_integracion() -> bool:
    """¿El índice real existe a 384d? (gate de los tests slow, cacheado).

    Los embeddings HF son locales, así que la integración ya no exige
    OPENAI_API_KEY: el requisito es que el índice Pinecone `pre-entrega-4-rag`
    esté recreado a DIMENSION=384 (lo hace el orquestador en el harness). La
    comprobación es una sola llamada describe_index de SOLO LECTURA por sesión.
    """
    global _estado_indice_384d
    if _estado_indice_384d is None:
        if not PINECONE_API_KEY:
            _estado_indice_384d = False
        else:
            try:
                import pinecone

                cliente = pinecone.Pinecone(api_key=PINECONE_API_KEY)
                descripcion = cliente.describe_index(INDEX_NAME)
                # NotFoundException (semántica real de la SDK v6.x) cae en el
                # except genérico: el índice aún no existe o no está a 384d.
                _estado_indice_384d = (
                    int(getattr(descripcion, "dimension", 0)) == DIMENSION
                )
            except Exception:  # noqa: BLE001 — sin red o índice inexistente
                _estado_indice_384d = False
    return _estado_indice_384d


@pytest.fixture(scope="session")
def credenciales() -> None:
    """Saltea los tests de integración si el índice real no está a 384d."""
    if not _indice_listo_para_integracion():
        pytest.skip(
            "El índice Pinecone real no existe a 384d: la integración slow "
            "necesita el índice recreado (el orquestador lo recree en el "
            "harness tras U7, con embeddings HF locales all-MiniLM-L6-v2). "
            "Completá PINECONE_API_KEY y recreá el índice a 384d."
        )


@pytest.fixture(autouse=True)
def _skip_slow_sin_credenciales(request: pytest.FixtureRequest) -> None:
    """Saltea automáticamente cualquier test `slow` sin el índice 384d real."""
    if request.node.get_closest_marker("slow") and not _indice_listo_para_integracion():
        pytest.skip(
            "Test marcado `slow`: requiere el índice Pinecone real a 384d "
            "(recreado por el orquestador en el harness tras U7). Corré solo "
            "los tests unit con -m 'not slow'."
        )


@pytest.fixture(scope="session")
def _pinecone_index(credenciales):
    """Índice Pinecone real (init + ingesta) para los tests de integración.

    Se construye una sola vez por sesión: init_index() verifica/crea el índice
    Serverless y espera a READY, y upsert_corpus() sube el corpus de data/ con
    ids deterministas (re-ejecución sin duplicados). El índice debe estar a
    DIMENSION=384 (embeddings HF locales); si quedó a 1536d, init_index()
    advierte la discrepancia sin recrearlo: recréelo el orquestador. Devuelve
    un dict con el objeto índice conectado, el cliente Pinecone y el resumen
    de la ingesta por namespace.
    """
    import pinecone

    from config import INDEX_NAME, PINECONE_API_KEY
    from embeddings import get_embeddings
    from ingest import upsert_corpus
    from init_index import init_index

    init_index()
    cliente = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    indice = cliente.Index(INDEX_NAME)
    totales = upsert_corpus(indice, embeddings=get_embeddings())
    return {"indice": indice, "cliente": cliente, "totales": totales}
