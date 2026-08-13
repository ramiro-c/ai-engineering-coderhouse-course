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
