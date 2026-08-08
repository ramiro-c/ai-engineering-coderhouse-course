"""Fixtures compartidos para los tests de integración del RAG.

El fixture `vectorstore` es session-scoped: la primera corrida descarga el
modelo (~470MB) y persiste el índice en ./vectorstore; las siguientes reusan
el índice (idempotente). Con `HF_HUB_OFFLINE=1` y sin caché, se saltea con un
mensaje claro.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from config import EMBEDDING_MODEL, VECTORSTORE_DIR
from ingest import ingest_corpus


def _model_cached() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_id = EMBEDDING_MODEL.replace("/", "--")
    return (hf_home / "hub" / f"models--{repo_id}").is_dir()


@pytest.fixture(scope="session")
def vectorstore() -> int:
    """Índice persistido compartido por todos los tests de integración."""
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline and not _model_cached():
        pytest.skip(
            "HF_HUB_OFFLINE=1 y el modelo no está cacheado: no se puede "
            "descargar el modelo de embeddings (~470MB la primera vez). "
            "Corré la ingesta con internet o desactivá HF_HUB_OFFLINE."
        )
    if not (VECTORSTORE_DIR / "chroma.sqlite3").exists() and offline:
        pytest.skip(
            "HF_HUB_OFFLINE=1 y no hay índice persistido: no se puede construir "
            "sin descargar el modelo de embeddings. Corré `python -m ingest` con internet."
        )
    return ingest_corpus()
