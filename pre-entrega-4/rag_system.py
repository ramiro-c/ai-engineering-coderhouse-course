"""Recuperador híbrido del RAG (Fase 5, RF-3).

RAGSystem combina BM25Retriever local (langchain_community, rank_bm25) con
PineconeVectorStore (langchain_pinecone) mediante EnsembleRetriever con RRF
c=60 y pesos 0.5/0.5 (D8, langchain_classic.retrievers). retrieve() devuelve
top-k a nivel documento (dedupe por document_id, D9) con metadata completa
para citar (document_id/texto), y mapea el namespace según la fuente (D7):
sin namespace explícito consulta los namespaces de la ingesta real
(FUENTE_NAMESPACES, porque la fase 4 indexa por fuente, no en "docs"), y una
fuente desconocida cae en NAMESPACE_DEFAULT con warning. Sin coincidencias
devuelve lista vacía sin excepción (RF-3 edge).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pinecone import Pinecone
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from config import (
    DATA_DIR,
    FUENTE_NAMESPACES,
    INDEX_NAME,
    NAMESPACE_DEFAULT,
    PINECONE_API_KEY,
    RRF_C,
    RRF_WEIGHTS,
    TOP_K,
)
from embeddings import get_embeddings
from ingest import build_chunks

logger = logging.getLogger(__name__)


def rrf_combine(
    listas_rankeadas: list[list[str]],
    c: int = RRF_C,
    pesos: list[float] | None = None,
) -> dict[str, float]:
    """Fusión rank-based RRF de varias listas de ids (D8).

    score(doc) = suma(peso_i / (c + rango_i)) sobre las listas donde aparece.
    Por defecto todas las listas pesan igual (1/n); c=60 amortigua el rango.
    """
    cantidad = len(listas_rankeadas)
    if cantidad == 0:
        return {}
    if pesos is None:
        pesos = [1.0 / cantidad] * cantidad
    puntajes: dict[str, float] = {}
    for lista, peso in zip(listas_rankeadas, pesos):
        for posicion, doc_id in enumerate(lista, start=1):
            puntajes[doc_id] = puntajes.get(doc_id, 0.0) + peso / (c + posicion)
    return puntajes


def resolver_namespaces(namespace: str | None) -> list[str]:
    """Resuelve los namespaces a consultar (D7, RF-3 edge).

    - None: todos los namespaces de la ingesta (FUENTE_NAMESPACES), porque la
      fase 4 indexa por fuente y el namespace "docs" quedaría vacío.
    - fuente conocida (features/tutorial): su namespace mapeado.
    - namespace directo ya mapeado: se usa tal cual.
    - fuente desconocida: fallback NAMESPACE_DEFAULT ("docs") con warning.
    """
    if namespace is None:
        return sorted(set(FUENTE_NAMESPACES.values()))
    mapeado = FUENTE_NAMESPACES.get(namespace)
    if mapeado is not None:
        return [mapeado]
    if namespace in FUENTE_NAMESPACES.values():
        return [namespace]
    logger.warning(
        "Fuente '%s' sin namespace registrado; se usa el fallback '%s'.",
        namespace,
        NAMESPACE_DEFAULT,
    )
    return [NAMESPACE_DEFAULT]


def construir_corpus() -> list[Document]:
    """Chunks locales de data/ para el retriever BM25 (misma metadata que la ingesta).

    Reusa build_chunks() de ingest.py con la ruta relativa como source para que
    document_id == nombre del .md y el BM25 rankee los mismos ids que Pinecone.
    """
    documentos: list[Document] = []
    for ruta in sorted(DATA_DIR.rglob("*.md")):
        rel = ruta.relative_to(DATA_DIR)
        documentos.extend(build_chunks(ruta.read_text(encoding="utf-8"), str(rel)))
    return documentos


class RAGSystem:
    """Recuperador híbrido: BM25 (corpus local) + vectores Pinecone con RRF.

    Construye BM25 y los vectorstores de forma perezosa y cacheada; los
    imports de las librerías ya están resueltos (requirements.txt). Sin
    PINECONE_API_KEY no se puede consultar el vectorial: retrieve() levanta
    el error de la librería al construir el cliente.
    """

    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k
        self._corpus: list[Document] | None = None
        self._bm25 = None
        self._vectorstores: dict[str, PineconeVectorStore] = {}

    def _cargar_corpus(self) -> list[Document]:
        if self._corpus is None:
            self._corpus = construir_corpus()
        return self._corpus

    def _retriever_bm25(self) -> BM25Retriever:
        """BM25 local sobre el corpus de data/ (rank_bm25, sin red)."""
        if self._bm25 is None:
            self._bm25 = BM25Retriever.from_documents(
                self._cargar_corpus(), k=self.top_k
            )
        return self._bm25

    def _vectorstore(self, namespace: str) -> PineconeVectorStore:
        """Vectorstore de un namespace, cacheado por namespace (lazy)."""
        if namespace not in self._vectorstores:
            cliente = Pinecone(api_key=PINECONE_API_KEY)
            self._vectorstores[namespace] = PineconeVectorStore(
                index=cliente.Index(INDEX_NAME),
                embedding=get_embeddings(),
                text_key="texto",
                namespace=namespace,
            )
        return self._vectorstores[namespace]

    def _ensemble(self, namespace: str) -> EnsembleRetriever:
        """Ensemble RRF (c=60, pesos 0.5/0.5, D8) entre BM25 y el namespace."""
        vectorial = self._vectorstore(namespace).as_retriever(
            search_kwargs={"k": self.top_k}
        )
        return EnsembleRetriever(
            retrievers=[self._retriever_bm25(), vectorial],
            weights=list(RRF_WEIGHTS),
            c=RRF_C,
        )

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        namespace: str | None = None,
    ) -> list[Document]:
        """Recupera top-k documentos híbridos con metadata para citar (RF-3).

        ``namespace`` acepta una fuente (features/tutorial), un namespace
        directo o None (todos los namespaces de la ingesta); una fuente
        desconocida cae en el fallback "docs". Dedupe a nivel documento (D9):
        cada document_id aparece una sola vez, con el chunk de mayor ranking.
        Sin coincidencias devuelve lista vacía, sin excepción.
        """
        k = k or self.top_k
        namespaces = resolver_namespaces(namespace)
        rankings: list[list[str]] = []
        documentos: dict[str, Document] = {}
        for ns in namespaces:
            hits = self._ensemble(ns).invoke(query)
            for hit in hits:
                doc_id = hit.metadata.get("document_id")
                if doc_id:
                    documentos.setdefault(doc_id, hit)
            rankings.append(
                [
                    hit.metadata["document_id"]
                    for hit in hits
                    if hit.metadata.get("document_id")
                ]
            )

        if not rankings:
            return []
        # Fusión entre namespaces (misma fórmula RRF) y top-k a nivel documento.
        puntajes = rrf_combine(rankings, c=RRF_C)
        ordenados = sorted(puntajes, key=puntajes.get, reverse=True)
        return [documentos[doc_id] for doc_id in ordenados[:k]]
