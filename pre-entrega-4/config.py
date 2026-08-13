"""Configuración de la pre-entrega 4: RAG híbrido Pinecone + BM25.

Patrón de pre-entrega-3: load_dotenv() al importar, helpers _env_int/_env_float
para leer variables con default tipado, y constantes del pipeline. Las claves
de API se leen del entorno (o .env) y las usan directamente las librerías.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


# --- Credenciales y despliegue ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = _env_str("INDEX_NAME", "pre-entrega-4-rag")
PINECONE_CLOUD = _env_str("PINECONE_CLOUD", "aws")
PINECONE_REGION = _env_str("PINECONE_REGION", "us-east-1")

# --- Embeddings ---
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "text-embedding-3-small")
# text-embedding-3-small produce vectores de 1536 dimensiones; el índice DEBE
# crearse con la misma dimensión (regla de validación de la consigna).
DIMENSION = 1536
METRIC = "cosine"

# --- Chunking (medido en tokens, rango 500-800 de la consigna) ---
CHUNK_SIZE = _env_int("CHUNK_SIZE", 700)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 100)
TOP_K = _env_int("TOP_K", 5)

# --- Recuperación híbrida (RRF) ---
# c amortigua el score rank-based de cada lista; pesos 0.5/0.5 entre BM25 y
# coseno porque ambos rankings pesan igual en la fusión.
RRF_C = _env_int("RRF_C", 60)
RRF_WEIGHTS = [0.5, 0.5]

# --- Ingesta ---
NAMESPACE_DEFAULT = "docs"
# Mapeo subcarpeta de data/ -> namespace de Pinecone (demo multi-tenant). Las
# fuentes sin entrada registrada caen en NAMESPACE_DEFAULT (fallback explícito).
FUENTE_NAMESPACES = {
    "features": "fastapi-core",
    "tutorial": "fastapi-tutorial",
}
BATCH_SIZE = _env_int("BATCH_SIZE", 100)
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
