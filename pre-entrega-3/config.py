from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


# --- Proveedor LLM (espejo de pre-entrega-2) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Vertex AI backend for Gemini (uses ADC, not GEMINI_API_KEY)
GOOGLE_GENAI_USE_VERTEXAI = _env_flag("GOOGLE_GENAI_USE_VERTEXAI", default=False)
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# --- RAG semántico local ---
CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50)
TOP_K = _env_int("TOP_K", 4)
SIMILARITY_THRESHOLD = _env_float("SIMILARITY_THRESHOLD", 0.5)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "apuntes")
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", str(BASE_DIR / "vectorstore")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

EXPECTED_CORPUS = [
    "Super Thinking - Sesgos y Objetividad.md",
    "Super Thinking - Sistemas y Mercados.md",
    "Super Thinking - Decisiones y Priorización.md",
    "LLM knowledge bases.md",
]
