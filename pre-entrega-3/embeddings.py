from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from config import EMBEDDING_MODEL

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Factory única del embedder local.

    Garantiza que indexación y consulta usen la MISMA instancia y configuración:
    embeber con un modelo distinto al que indexó devuelve resultados basura.

    HF_HUB_OFFLINE hay que exportarlo en el entorno, no alcanza el .env:
    `huggingface_hub` lo lee en import time y langchain_core/chromadb lo
    arrastran antes de que corra el `load_dotenv()` de config.py.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
