"""Fixtures compartidos para los tests de la pre-entrega 4.

Los tests de integración (marcados `slow`) necesitan red y credenciales reales
de Pinecone/OpenAI. Si faltan PINECONE_API_KEY u OPENAI_API_KEY, se saltean con
un mensaje claro — patrón de pre-entrega-3. A diferencia de pre-entrega-3, los
imports de pinecone/langchain se hacen dentro de los fixtures (no al nivel del
módulo), para que la colección de los tests unit no dependa del entorno real.
"""

from __future__ import annotations

import os

import pytest

from config import OPENAI_API_KEY, PINECONE_API_KEY


def _hay_credenciales() -> bool:
    return bool(PINECONE_API_KEY and OPENAI_API_KEY)


@pytest.fixture(scope="session")
def credenciales() -> None:
    """Saltea los tests de integración si faltan credenciales de Pinecone/OpenAI."""
    if not _hay_credenciales():
        pytest.skip(
            "Faltan PINECONE_API_KEY u OPENAI_API_KEY: este test necesita el "
            "índice real y embeddings de OpenAI. Completá el .env."
        )


@pytest.fixture(autouse=True)
def _skip_slow_sin_credenciales(request: pytest.FixtureRequest) -> None:
    """Saltea automáticamente cualquier test `slow` sin credenciales."""
    if request.node.get_closest_marker("slow") and not _hay_credenciales():
        pytest.skip(
            "Test marcado `slow`: requiere PINECONE_API_KEY y OPENAI_API_KEY "
            "para tocar el índice real. Completá el .env o corré solo tests unit."
        )


@pytest.fixture(scope="session")
def _pinecone_index(credenciales):
    """Índice Pinecone real (init + ingesta) para los tests de integración.

    Se construye una sola vez por sesión: init_index() verifica/crea el índice
    Serverless y espera a READY, y upsert_corpus() sube el corpus de data/ con
    ids deterministas (re-ejecución sin duplicados). Devuelve un dict con el
    objeto índice conectado, el cliente Pinecone y el resumen de la ingesta
    por namespace, para que los tests de la fase 4 los usen.
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
