"""Fixtures compartidos para los tests de integración del RAG.

El fixture `vectorstore` es session-scoped: la primera corrida descarga el
modelo (~470MB) y persiste el índice en ./vectorstore; las siguientes reusan
el índice (idempotente). Con `HF_HUB_OFFLINE=1` y sin caché, se saltea con un
mensaje claro.

`llm` se saltea cuando no hay credenciales del proveedor configurado, para que
`pytest` no falle por una API key ausente o un 429 ajeno al código.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from config import EMBEDDING_MODEL, LLM_PROVIDER, VECTORSTORE_DIR
from ingest import ingest_corpus

# Variables de entorno que habilitan cada proveedor de generación.
CLAVES_POR_PROVEEDOR = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _model_cached() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_id = EMBEDDING_MODEL.replace("/", "--")
    return (hf_home / "hub" / f"models--{repo_id}").is_dir()


def _llm_configurado() -> bool:
    """True si el proveedor activo tiene con qué autenticarse."""
    vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower()
    if LLM_PROVIDER == "gemini" and vertex in {"1", "true", "yes", "on"}:
        return True  # Vertex usa ADC, no una API key
    claves = CLAVES_POR_PROVEEDOR.get(LLM_PROVIDER, ())
    return any(os.environ.get(nombre) for nombre in claves)


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


@pytest.fixture
def llm() -> None:
    """Saltea el test si el proveedor de generación no está configurado."""
    if not _llm_configurado():
        pytest.skip(
            f"El proveedor '{LLM_PROVIDER}' no tiene credenciales configuradas: "
            "este test necesita generar con el LLM real. Completá el .env."
        )
