# Pre-entrega 3 — RAG semántico local sobre apuntes

Sistema de **retrieval-augmented generation (RAG) 100% local** para responder
preguntas sobre un corpus de apuntes propios (modelos mentales de *Super
Thinking* y notas sobre *knowledge bases* con LLMs), sin requerir API key para
los embeddings.

## Arquitectura

```
data/*.md ──ingest.py──> RecursiveCharacterTextSplitter (800/50)
              │  HuggingFaceEmbeddings(paraphrase-multilingual-MiniLM-L12-v2)
              ▼
        Chroma ./vectorstore (colección "apuntes", hnsw:space=cosine)
              ▲
get_rag_response(query) ──retriever (top_k=4)──> gate (>= 0.30) ──> cadena LCEL async ──> RagResponse
        0 relevantes ───────────────────────────────────────────> RagResponse("No lo sé", [])
```

- **Ingesta idempotente**: si la colección ya existe en `./vectorstore`, se
  saltea el reindexado (no duplica chunks).
- **Gate de relevancia**: solo los fragmentos con score >= `SIMILARITY_THRESHOLD`
  llegan al LLM y aparecen en `references`. Con 0 fragmentos relevantes, se
  responde "No lo sé" **sin llamar al LLM** (rápido y sin costo).
- **Generación**: cadena A con `PydanticOutputParser` (reintentos) y cadena B de
  fallback con `with_structured_output`; si ambas fallan, se devuelve
  `RagGenerationError`.

## Requisitos

- Python 3.13
- Descarga inicial del modelo de embeddings (~470MB la primera vez que se usa)
  y dependencias de PyTorch (llegan vía `sentence-transformers`).

## Instalación

```bash
cd pre-entrega-3
python -m venv .venv && source .venv/bin/activate   # opcional pero recomendado
pip install -r requirements.txt
cp .env.example .env   # y completá las claves LLM si las tenés
```

Los embeddings son locales y no necesitan clave. Para la **generación** de
respuestas se usa el mismo patrón LLM que la pre-entrega 2 (`LLM_PROVIDER`,
default `gemini`); el `.env.example` está espejado sin claves reales.

## Uso

```bash
# 1. Ingesta del corpus (idempotente: re-ejecutar no duplica)
python -m ingest

# 2. Chat interactivo: hacé cualquier pregunta sobre el corpus
python -m main
```

En el chat escribís preguntas variadas y el sistema responde grounded con sus
referencias; `salir`, `q` o `exit` terminan la sesión. También hay un modo demo
con las dos preguntas fijas (una respondible y una trampa):

```bash
python -m main --demo
```

Ejemplo de salida del chat:

```json
{
  "text": "La Matriz de Eisenhower funciona categorizando las tareas en dos ejes: urgente vs. no urgente e importante vs. no importante...",
  "references": [
    {
      "source": "Super Thinking - Decisiones y Priorización.md",
      "snippet": "## Matriz de Decisiones de Eisenhower\n\n> Categorizá las tareas en dos ejes: ..."
    }
  ]
}
```

En el chat, en lugar del JSON crudo, la respuesta se muestra como texto legible
seguido de las referencias usadas.

Si la pregunta no tiene relación con el corpus, `references` queda vacía y el
texto expresa que no sabe ("No lo sé").

## Tests

```bash
# Tests unitarios rápidos (gate de relevancia + schemas, sin modelo)
pytest -m "not slow"

# Suite completa (descarga/usa el modelo local, ~470MB la primera vez)
pytest
```

Los tests de integración usan un fixture **session-scoped** que reusa el índice
de `./vectorstore` (idempotente). Con `HF_HUB_OFFLINE=1` y sin caché ni índice
persistido, se saltean con un mensaje claro. `pytest` está incluido en
`requirements.txt`.

## Configuración (variables de entorno)

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Proveedor del LLM de generación (patrón pre-entrega 2) |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo local de embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `50` | Tamaño de chunk y solape del splitter |
| `TOP_K` | `4` | Fragmentos recuperados por consulta |
| `SIMILARITY_THRESHOLD` | `0.30` | Umbral del gate de relevancia (score coseno) |
| `VECTORSTORE_DIR` | `./vectorstore` | Carpeta del índice Chroma persistente |
| `COLLECTION_NAME` | `apuntes` | Nombre de la colección Chroma |

## Limitaciones conocidas

- **Truncado del embedder**: `MiniLM-L12-v2` trunca los embeddings alrededor de
  128 word pieces; el vector pierde el final de chunks largos, aunque el LLM
  recibe el chunk completo como contexto.
- **Calibración del umbral**: `SIMILARITY_THRESHOLD=0.30` fue calibrado
  empíricamente contra este corpus (consultas afines puntúan 0.33-0.63; ajenas,
  0.04-0.26). Si cambiás el corpus o el modelo, recalibrá con
  `retriever.similarity_search_with_relevance_scores`.
- **Sin claves LLM**: la ingesta y la recuperación funcionan sin credenciales,
  pero la generación necesita el proveedor configurado en `.env`.
