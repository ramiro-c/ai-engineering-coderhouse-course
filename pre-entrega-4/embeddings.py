"""Cliente de embeddings de la pre-entrega 4 (ENMIENDA 2026-08-12, sin API key).

get_embeddings() devuelve UNA instancia cacheada de HuggingFaceEmbeddings
(sentence-transformers/all-MiniLM-L6-v2, 384d) para que indexar y consultar
usen el mismo objeto (patrón de pre-entrega-3). Los embeddings son LOCALES:
no requieren ninguna API key (OPENAI_API_KEY deja de aplicar; la consigna
menciona text-embedding-3-small como EJEMPLO, no como requisito). La clase se
importa a nivel de módulo pero no se instancia: la carga del modelo (torch,
pesado) queda dentro de la función (lazy) y el import no dispara nada.

NOTA (lección #793): HF_HUB_OFFLINE —si se quiere correr sin descargar el
modelo— se exporta en el ENTORNO DEL PROCESO (shell / .venv/bin/activate),
NO en .env: huggingface_hub lo lee en import time y load_dotenv() llega tarde.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el cliente de embeddings cacheado (misma instancia indexar/consultar)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
