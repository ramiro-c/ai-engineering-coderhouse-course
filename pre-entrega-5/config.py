"""Configuración de la pre-entrega 5: agente ReAct con LangGraph.

Patrón de pre-entrega-4: load_dotenv() al importar, helpers _env_int/_env_str
para leer variables con default tipado, y constantes del agente. Las claves
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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


# --- Credenciales y despliegue ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Proveedor por defecto de la factory multi-provider: gemini/openai/
# anthropic/openrouter (patrón de pre-entrega-4).
LLM_PROVIDER = _env_str("LLM_PROVIDER", "gemini")

# --- Credenciales GCP (generación vía Vertex AI) ---
# ChatGoogleGenerativeAI (langchain-google-genai) autentica con ADC/service
# account: GOOGLE_APPLICATION_CREDENTIALS apunta al JSON de la service account
# (rol Vertex AI User) y las otras dos fijan el proyecto y la región.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")

# --- Agente ReAct (LangGraph) ---
CHECKPOINT_PATH = Path(
    _env_str("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints.sqlite"))
)
RECURSION_LIMIT = 10
