# Pre-entrega 4 — RAG híbrido escalable con Pinecone

Sistema de **retrieval-augmented generation** sobre documentación de FastAPI con
recuperación **híbrida**: búsqueda léxica local (BM25) + búsqueda semántica
(embeddings **locales** HuggingFace) fusionadas con **Reciprocal Rank Fusion
(RRF)**, sobre un índice **Pinecone Serverless**. La ingesta chunkifica el
corpus Markdown por tokens (500–800) y la evaluación mide la calidad de la
recuperación contra un golden set (Recall@5 ≥ 0.8). La generación de respuestas
(evolución B) usa un LLM vía `clients/factory.py` (default: Gemini en Vertex AI).

## Requisitos

- Python 3.13+ y `pip`
- Cuenta de [Pinecone](https://www.pinecone.io/) (free tier alcanza)
- Modelo de embeddings **local** `sentence-transformers/all-MiniLM-L6-v2`
  (384d, HuggingFace): se descarga una sola vez a disco, sin API key
- (Opcional) Proyecto GCP con Vertex AI + service account con rol
  *Vertex AI User*, solo para la generación con LLM (provider `gemini`)

## Inicio rápido (replicar el índice)

```bash
cd pre-entrega-4
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # completá PINECONE_API_KEY (y GCP si usás generación)
python init_index.py        # crea/verifica el índice Serverless (384d, cosine)
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
   (EnsembleRetriever), `tiktoken`, `rank-bm25`, y las dependencias de la
   enmienda: `sentence-transformers` + `torch` + `langchain-huggingface`
   (embeddings locales) y `langchain-google-vertexai` + `google-cloud-aiplatform`
   (generación vía Vertex).

2. **`.env`** — copiá `.env.example` y completá las claves:
   - `PINECONE_API_KEY`: de api.pinecone.io.
   - `GOOGLE_APPLICATION_CREDENTIALS` (ruta al JSON de la service account),
     `GOOGLE_CLOUD_PROJECT` y `GOOGLE_CLOUD_LOCATION`: solo si vas a usar la
     generación con LLM (provider `gemini`).
   - `INDEX_NAME=pre-entrega-4-rag` (por defecto; se crea solo).
   - Los embeddings NO requieren ninguna API key: son locales.

3. **`python init_index.py`** — verifica si el índice Serverless existe y lo
   crea si no (`DIMENSION=384`, métrica `cosine`, región `aws/us-east-1`),
   esperando hasta el estado `READY` (poll cada 10s, timeout 5 min). Es
   **idempotente**: si ya existe, verifica dimensión/métrica y continúa. Sin
   `PINECONE_API_KEY` sale con error claro antes de tocar la red.

4. **`python ingest.py`** — chunkifica todo `data/` (500–800 tokens por chunk,
   overlap 100, cabeceras h1–h3 antepuestas como contexto de sección) y hace
   upsert por **namespace de fuente**: `features/` → `fastapi-core`,
   `tutorial/` → `fastapi-tutorial`. Los ids son deterministas (sha1 del
   chunk), así que re-ejecutarlo **no duplica** vectores: reemplaza los mismos
   ids (idempotencia). Los embeddings los genera el modelo local de
   HuggingFace (primera corrida más lenta: descarga/carga del modelo).

5. **`python evaluate.py`** — lee `golden_set.json` (5 pares
   `{pregunta, documento_id_esperado}`), consulta el recuperador híbrido por
   pregunta e imprime una tabla con P@5/R@5/MRR por caso, promedios y el
   veredicto `PASS`/`FAIL` con criterio **Recall@5 ≥ 0.8** (4 de 5). Sale con
   código 0 (PASS) o 1 (FAIL).

## Elección del modelo de embeddings

Los embeddings son **locales**: `sentence-transformers/all-MiniLM-L6-v2`
(384 dimensiones, el mismo modelo de pre-entrega-3) vía `HuggingFaceEmbeddings`
de `langchain-huggingface`. La consigna menciona `text-embedding-3-small` de
OpenAI como **ejemplo**, no como requisito; acá se usa un modelo local por dos
razones:

- **No requiere API key.** `OPENAI_API_KEY` deja de ser un prerrequisito para
  indexar: el modelo se descarga una sola vez a disco y luego se cachea
  (`get_embeddings()` con `lru_cache`, misma instancia para indexar y
  consultar).
- **Local ≠ "en la nube".** La inferencia corre en tu máquina; nada del texto
  viaja a un servicio de embeddings. Solo el índice de vectores (Pinecone) y la
  generación opcional con LLM (Vertex) son servicios remotos.

El error a evitar de la consigna es el **mismatch de dimensiones** entre
embeddings e índice: acá ambos son **384d** (índice recreado a 384d + modelo de
384d), así que no hay mismatch. No cambies `EMBEDDING_MODEL` sin recrear el
índice con la dimensión correspondiente.

> **Modo offline (lección #793).** Si no querés que HuggingFace descargue el
> modelo automáticamente, exportá `HF_HUB_OFFLINE=1` en el **entorno del
> proceso** (shell o `.venv/bin/activate`), NO en `.env`: `huggingface_hub`
> lee esa variable en *import time* y `load_dotenv()` llega tarde.

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
| `schemas.py` | Modelos Pydantic: GoldenSet, RetrievalHit, EvalResult, LlmAnswer |
| `init_index.py` | Crea/verifica el índice Serverless idempotente (poll a READY) |
| `ingest.py` | Chunking por tokens, ids deterministas, upsert por namespace |
| `embeddings.py` | Cliente de embeddings cacheado (HuggingFace local, 384d, sin API key) |
| `rag_system.py` | RAGSystem: BM25 + vectores con EnsembleRetriever (RRF c=60, 0.5/0.5) y `responder()` |
| `clients/factory.py` | Factory multi-proveedor LLM (`build_chat_model`, default `LLM_PROVIDER`) |
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
python -m pytest tests -m "not slow" -q     # suite unit sin red (~96 tests)
python -m pytest tests -m slow -q           # integración real (índice 384d)
```

Los tests de integración (`test_integration.py`) corren contra el índice real y
se saltean con mensaje claro si el índice Pinecone no está recreado a **384d**
(embeddings HF locales; el orquestador lo recrea en el harness). La suite unit
usa stubs y funciones puras: no toca la red. El factory LLM se testea con
módulos falsos en `sys.modules` (incluida la excepción real de Vertex sin ADC,
`DefaultCredentialsError`) y `responder()` con un modelo fake
(`RunnableLambda`): nunca se llama a una API real.

## Generación de respuestas (evolución B)

La capa de generación responde una pregunta con un LLM **citando los
`document_id` recuperados**. `RAGSystem.responder(pregunta, k=5, namespace=None)`
recupera el top-k híbrido, arma el contexto con los chunks y su metadata, y
genera una respuesta estructurada `LlmAnswer` (`pregunta`, `respuesta`,
`answered`, `fuentes`) con un prompt estricto en español neutro: usar SOLO el
contexto provisto y citar `document_id` reales de la metadata (nunca
inventados).

```python
from rag_system import RAGSystem

rag = RAGSystem()
respuesta = rag.responder("¿Cómo se definen las rutas en FastAPI?")
print(respuesta.respuesta)   # "Según routing.md, las rutas se definen con..."
print(respuesta.fuentes)     # ["routing.md", ...]
print(respuesta.answered)    # True si el contexto alcanzó para responder
```

Si no hay contexto suficiente, o si la API del proveedor falla, `responder()`
devuelve `answered=false` con un mensaje claro y `fuentes=[]`, sin romper el
flujo (máx 2 reintentos internos ante errores de parseo/API).

### Proveedor del LLM (`LLM_PROVIDER`)

El modelo lo elige la factory `clients/factory.py` (patrón de pre-entrega-3)
según la variable `LLM_PROVIDER` (default `gemini`):

| `LLM_PROVIDER` | Credencial | Modelo por defecto |
|---|---|---|
| `gemini` (default) | Vertex AI / ADC (`GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) | `gemini-2.5-flash` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest` |
| `openrouter` | clave de OpenRouter | `cohere/north-mini-code:free` |

> **`gemini` usa Vertex AI, no Google AI Studio.** No se usa `GEMINI_API_KEY`:
> `ChatVertexAI` autentica con ADC/service account. Si falta la credencial GCP
> (o el ADC no resuelve), la construcción del modelo falla y `responder()`
> degrada a `answered=false` (edge controlado, sin crash). Solo hace falta la
> credencial del provider activo (las deps de los demás providers son imports
> lazy y se instalan en `requirements.txt`).

> **NOTA — la generación NO participa en la evaluación.** Las métricas
> (Precision@5/Recall@5/MRR) y el criterio `Recall@5 ≥ 0.8` miden SOLO la
> recuperación híbrida; la capa de generación es una demostración opcional.
> Un buen score de evaluación no garantiza buenas respuestas generadas (y
> viceversa).

### Demo interactiva

Un script de demostración end-to-end de la generación de respuestas: recupera
el top-5 híbrido (con `document_id`/score RRF, el rank 1 destacado) y genera
la respuesta con el LLM configurado (`LLM_PROVIDER`), imprimiendo ambos en
consola.

```bash
cd pre-entrega-4 && source .venv/bin/activate
HF_HUB_OFFLINE=1 python3 pre-entrega-4/demo.py "¿Cómo defino un decorador POST en FastAPI?"
python3 pre-entrega-4/demo.py        # sin argumento: pregunta por prompt interactivo
```

Sin argumento, el script pide la pregunta por prompt (Enter usa la pregunta
de ejemplo del corpus). Si el LLM no puede responder (`answered=false`:
contexto insuficiente o error del proveedor), imprime un mensaje claro de
degradación sin romper el flujo. La demo es un script en archivo
(deliberadamente, lección #866: `load_dotenv` no resuelve `.env` desde
`python3 -c`), así que ve las claves del `.env` como el resto de los CLIs.
