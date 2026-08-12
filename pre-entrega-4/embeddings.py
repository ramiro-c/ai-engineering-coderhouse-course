"""Cliente de embeddings de la pre-entrega 4 (Fase 3, sin red).

get_embeddings() devuelve UNA instancia cacheada de OpenAIEmbeddings
(text-embedding-3-small, 1536d, DIMENSION en config) para que indexar y
consultar usen el mismo objeto (patron de pre-entrega-3). La clase se importa
a nivel de modulo pero no se instancia: el constructor exige OPENAI_API_KEY,
asi que la creacion queda dentro de la funcion (lazy) y el import no dispara
nada, ni red ni validacion de credenciales.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Devuelve el cliente de embeddings cacheado (misma instancia indexar/consultar)."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
