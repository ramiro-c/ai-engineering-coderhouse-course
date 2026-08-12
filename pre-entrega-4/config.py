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

# --- LLM de generación (evolución B, RF-6/D10) ---
# Claves por provider; la factory las pasa al constructor del modelo
# correspondiente. Solo se necesita la del provider activo. El provider
# gemini pasa a autenticar vía Vertex AI (ADC) en la Fase 7 (U7); la clave
# GEMINI_API_KEY (Google AI Studio) deja de usarse al migrar la factory.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Proveedor por defecto de la factory multi-provider: gemini/openai/
# anthropic/openrouter (patrón de pre-entrega-3).
LLM_PROVIDER = _env_str("LLM_PROVIDER", "gemini")
PINECONE_CLOUD = _env_str("PINECONE_CLOUD", "aws")
PINECONE_REGION = _env_str("PINECONE_REGION", "us-east-1")

# --- Credenciales GCP (generación vía Vertex AI, ENMIENDA 2026-08-12) ---
# ChatVertexAI autentica con ADC/service account: GOOGLE_APPLICATION_CREDENTIALS
# apunta al JSON de la service account (rol Vertex AI User) y las otras dos
# fijan el proyecto y la región de Vertex. Sin ellas, la construcción del
# modelo falla y responder() degrada a answered=false (RF-6 edge).
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# --- Embeddings ---
# ENMIENDA 2026-08-12 (U7): embeddings LOCALES HuggingFace
# sentence-transformers/all-MiniLM-L6-v2 (384d, mismo modelo de pre-entrega-3),
# sin API key. El modelo se descarga una vez a disco y luego se cachea.
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# all-MiniLM-L6-v2 produce vectores de 384 dimensiones; el índice DEBE crearse
# con la misma dimensión (regla de validación de la consigna: un mismatch
# embeddings<->índice es el error a evitar; el índice 1536d previo se recrea).
DIMENSION = 384
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
