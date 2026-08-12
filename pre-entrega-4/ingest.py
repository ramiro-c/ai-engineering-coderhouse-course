"""Ingesta pura de la pre-entrega 4: chunking por tokens e ids deterministas.

Fase 3 (sin red): build_chunks() combina MarkdownHeaderTextSplitter (cabeceras
h1-h3 como contexto, D6) con RecursiveCharacterTextSplitter medido en tokens
(tiktoken cl100k_base, D3) y empaqueta las piezas para que cada chunk quede en
el rango de 500-800 tokens de RF-2. Los helpers chunk_id/build_namespace/
validate_metadata_size sostienen la idempotencia (D5) y el mapeo multi-tenant
(D7). El wiring de PineconeVectorStore (upsert, texto en metadata) se agrega
en la fase 4.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FUENTE_NAMESPACES,
    NAMESPACE_DEFAULT,
)

# Limite de metadata por vector de Pinecone (~40KB, RF-2 edge).
METADATA_SIZE_LIMIT = 40_000

# cl100k_base es el codificador de los modelos text-embedding-3 (OpenAI).
_ENC = tiktoken.get_encoding("cl100k_base")


def len_tokens(texto: str) -> int:
    """Cuenta tokens de un texto con tiktoken cl100k_base (D3)."""
    return len(_ENC.encode(texto))


def build_chunks(markdown_text: str, source: str) -> list[Document]:
    """Chunkifica markdown en trozos de 500-800 tokens con contexto de seccion.

    ``source`` es la ruta relativa del .md (ej. "features/routing.md"); el
    document_id es el nombre del archivo con extension (decision U2) y se usa
    como metadata ``source`` y ``document_id`` a la vez (contrato D9).

    Combina MarkdownHeaderTextSplitter (cabeceras h1-h3 a metadata, D6) con
    RecursiveCharacterTextSplitter medido en tokens (CHUNK_SIZE/CHUNK_OVERLAP,
    D3). Cada pieza lleva la ruta de cabeceras antepuesta para que una query
    que nombre el concepto matchee aunque el cuerpo no lo repita.
    """
    document_id = Path(source).name
    etiquetas = [Path(source).parent.name] if Path(source).parent.name else []

    seccion_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
        strip_headers=True,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len_tokens,
        separators=["\n\n", "\n", " ", ""],
    )

    piezas: list[Document] = []
    for seccion in seccion_splitter.split_text(markdown_text):
        titulo = " > ".join(seccion.metadata.values())
        texto_seccion = f"{titulo}\n{seccion.page_content}" if titulo else seccion.page_content
        for trozo in splitter.split_text(texto_seccion):
            piezas.append(Document(page_content=trozo, metadata={"seccion": titulo}))

    return _empaquetar(piezas, document_id, etiquetas)


def _empaquetar(
    piezas: list[Document], document_id: str, etiquetas: list[str]
) -> list[Document]:
    """Agrupa piezas cortas en chunks de hasta CHUNK_SIZE+CHUNK_OVERLAP tokens.

    El splitter por tokens garantiza el tope (CHUNK_SIZE) pero no un piso: las
    secciones del corpus real pesan ~50-150 tokens. Agrupar piezas consecutivas
    (cada una con su ruta de cabeceras antepuesta) deja cada chunk dentro del
    rango de 500-800 tokens de RF-2 cuando el documento lo permite. La
    metadata ``seccion`` del chunk es la ruta de la primera pieza del grupo.
    """
    tope = CHUNK_SIZE + CHUNK_OVERLAP
    chunks: list[Document] = []
    texto_actual = ""
    seccion_actual = ""
    for pieza in piezas:
        if texto_actual:
            candidato = f"{texto_actual}\n\n{pieza.page_content}"
            if len_tokens(candidato) > tope:
                chunks.append(_documento(texto_actual, seccion_actual, document_id, etiquetas))
                texto_actual = pieza.page_content
                seccion_actual = pieza.metadata.get("seccion", "")
                continue
            texto_actual = candidato
        else:
            texto_actual = pieza.page_content
            seccion_actual = pieza.metadata.get("seccion", "")
    if texto_actual:
        chunks.append(_documento(texto_actual, seccion_actual, document_id, etiquetas))
    return chunks


def _documento(
    texto: str, seccion: str, document_id: str, etiquetas: list[str]
) -> Document:
    return Document(
        page_content=texto,
        metadata={
            "source": document_id,
            "document_id": document_id,
            "seccion": seccion,
            "etiquetas": etiquetas,
        },
    )


def chunk_id(namespace: str, content: str) -> str:
    """Id determinista del chunk (D5): sha1(namespace:contenido)[:16].

    La re-ejecucion de la ingesta produce los mismos ids y el upsert reemplaza
    los vectores existentes en lugar de duplicarlos (idempotencia RF-2).
    """
    return hashlib.sha1(f"{namespace}:{content}".encode("utf-8")).hexdigest()[:16]


def build_namespace(fuente: str) -> str:
    """Mapea la subcarpeta de origen al namespace de Pinecone (D7).

    Las fuentes registradas en config.FUENTE_NAMESPACES usan su namespace;
    cualquier otra cae en NAMESPACE_DEFAULT ("docs") como fallback explicito.
    """
    return FUENTE_NAMESPACES.get(fuente, NAMESPACE_DEFAULT)


def validate_metadata_size(metadata: dict) -> bool:
    """True si el JSON de metadata entra en el limite de Pinecone (~40KB, RF-2).

    Validacion previa al upsert: un vector no puede guardar el documento
    completo (~50KB+); los chunks reales (~2-4KB) siempre pasan.
    """
    return len(json.dumps(metadata)) < METADATA_SIZE_LIMIT
