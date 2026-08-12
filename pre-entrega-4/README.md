# Pre-entrega 4 — RAG híbrido escalable con Pinecone

Sistema de **retrieval-augmented generation** sobre documentación de FastAPI con
recuperación **híbrida**: búsqueda léxica local (BM25) + búsqueda semántica
(embeddings de OpenAI) fusionadas con **Reciprocal Rank Fusion (RRF)**, sobre un
índice **Pinecone Serverless**. La ingesta chunkifica el corpus Markdown por
tokens (500–800) y la evaluación mide la calidad de la recuperación contra un
golden set (Recall@5 ≥ 0.8).

## Requisitos

- Python 3.13+ y `pip`
- Cuenta de [Pinecone](https://www.pinecone.io/) (free tier alcanza)
- API key de OpenAI (embeddings `text-embedding-3-small`, 1536 dimensiones)

## Inicio rápido (replicar el índice)

```bash
cd pre-entrega-4
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # completá PINECONE_API_KEY y OPENAI_API_KEY
python init_index.py        # crea/verifica el índice Serverless (1536d, cosine)
python ingest.py            # indexa data/ por namespace (idempotente)
python evaluate.py          # golden set → tabla + promedios + PASS/FAIL
```

> **El venv no es opcional.** Si corrés los scripts sin activarlo vas a ver
> `ModuleNotFoundError`. Los 12 documentos del corpus (8 en `data/features/`,
> 4 en `data/tutorial/`) son la base de la recuperación y de las preguntas de
> evaluación.

### Paso a paso

1. **`pip install -r requirements.txt`** — instala las versiones fijadas:
   `pinecone==7.3.0`, `langchain-pinecone==0.2.13`,
   `langchain-community==0.4.2` (BM25Retriever), `langchain-classic`
   (EnsembleRetriever), `langchain-openai`, `tiktoken` y `rank-bm25`.

2. **`.env`** — copiá `.env.example` y completá las claves:
   - `PINECONE_API_KEY`: de api.pinecone.io.
   - `OPENAI_API_KEY`: para los embeddings.
   - `INDEX_NAME=pre-entrega-4-rag` (por defecto; se crea solo).

3. **`python init_index.py`** — verifica si el índice Serverless existe y lo
   crea si no (`DIMENSION=1536`, métrica `cosine`, región `aws/us-east-1`),
   esperando hasta el estado `READY` (poll cada 10s, timeout 5 min). Es
   **idempotente**: si ya existe, verifica dimensión/métrica y continúa. Sin
   `PINECONE_API_KEY` sale con error claro antes de tocar la red.

4. **`python ingest.py`** — chunkifica todo `data/` (500–800 tokens por chunk,
   overlap 100, cabeceras h1–h3 antepuestas como contexto de sección) y hace
   upsert por **namespace de fuente**: `features/` → `fastapi-core`,
   `tutorial/` → `fastapi-tutorial`. Los ids son deterministas (sha1 del
   chunk), así que re-ejecutarlo **no duplica** vectores: reemplaza los mismos
   ids (idempotencia).

5. **`python evaluate.py`** — lee `golden_set.json` (5 pares
   `{pregunta, documento_id_esperado}`), consulta el recuperador híbrido por
   pregunta e imprime una tabla con P@5/R@5/MRR por caso, promedios y el
   veredicto `PASS`/`FAIL` con criterio **Recall@5 ≥ 0.8** (4 de 5). Sale con
   código 0 (PASS) o 1 (FAIL).

## Arquitectura

```
data/*.md ──> ingest.py (tiktoken 500-800, ids sha1) ──> PineconeVectorStore
                                                          (namespaces por fuente)
golden_set.json ──> evaluate.py ──> RAGSystem.retrieve() ──> EnsembleRetriever (RRF)
                                     BM25Retriever (corpus local) + PineconeVectorStore
                                     └─> top-5 con metadata (document_id, texto) ──> P@5/R@5/MRR
```

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Variables de entorno con defaults tipados y constantes del pipeline |
| `schemas.py` | Modelos Pydantic: GoldenSet, RetrievalHit, EvalResult |
| `init_index.py` | Crea/verifica el índice Serverless idempotente (poll a READY) |
| `ingest.py` | Chunking por tokens, ids deterministas, upsert por namespace |
| `embeddings.py` | Cliente de embeddings cacheado (text-embedding-3-small) |
| `rag_system.py` | RAGSystem: BM25 + vectores con EnsembleRetriever (RRF c=60, 0.5/0.5) |
| `evaluate.py` | Métricas puras + CLI de evaluación contra el golden set |

## Recuperación híbrida

`RAGSystem.retrieve(pregunta, k=5, namespace=None)` combina dos rankings:

- **BM25** (lexical, `langchain_community.retrievers`): rankea el corpus local
  de `data/` con `rank_bm25` — exacto para términos como `Depends` o `TestClient`.
- **Vectorial** (semántico, `langchain_pinecone`): busca por similitud coseno
  sobre los embeddings del índice.

Ambos se fusionan con `EnsembleRetriever` de `langchain_classic.retrievers`
usando **RRF con c=60 y pesos 0.5/0.5**: cada documento suma `peso / (c + rango)`
por lista donde aparece. Los top-k se deduplican **a nivel documento** (varios
chunks del mismo `.md` cuentan como uno) y cada hit expone en metadata su
`document_id` y su texto original para poder citar. Sin coincidencias devuelve
lista vacía, sin excepción. El namespace se mapea por fuente: `None` consulta
los namespaces de la ingesta (`fastapi-core` + `fastapi-tutorial`), una fuente
conocida usa el suyo y una desconocida cae en el fallback `docs`.

## Métricas de evaluación

| Métrica | Definición | Criterio |
|---|---|---|
| **Precision@5** | 1/5 si el documento esperado está en el top-5 | — |
| **Recall@5** | 1 si el documento esperado está en el top-5, 0 si no | **≥ 0.8 promedio** |
| **MRR** | 1/rango del documento esperado (complemento) | — |

El Recall@5 es a nivel **documento**, no chunk: si `dependencies.md` aporta 3
chunks al top-5, cuenta como un solo documento. Con 5 preguntas, PASS = 4 o 5
aciertos.

## Tests

```bash
python -m pytest tests -m "not slow" -q     # suite unit sin red (~73 tests)
python -m pytest tests -m slow -q           # integración real (necesita .env completo)
```

Los tests de integración (`test_integration.py`) corren contra el índice real y
se saltean con mensaje claro si faltan `PINECONE_API_KEY` u `OPENAI_API_KEY`
(patrón de pre-entrega-3). La suite unit usa stubs y funciones puras: no toca
la red.

## Evolución B — generación de respuestas con LLM (próxima fase)

La capa de **generación** (responder la pregunta con un LLM citando los
`document_id` recuperados) se implementa en la **próxima fase** del cambio
(U6): una factory multi-proveedor (`clients/factory.py`, patrón de
pre-entrega-3) con `LLM_PROVIDER` configurable (default `gemini`) y un
`responder()` en `rag_system.py` que devuelve respuestas estructuradas solo a
partir del contexto recuperado. No participa en las métricas de esta entrega:
la evaluación mide recuperación, no generación.
