# Pre-entrega 3 — RAG semántico local sobre apuntes

Sistema de **retrieval-augmented generation (RAG)** para responder preguntas
sobre un corpus de apuntes propios (modelos mentales de *Super Thinking* y notas
sobre *knowledge bases* con LLMs). Los embeddings y la búsqueda corren 100%
locales y **no necesitan API key**; solo la generación de la respuesta final usa
un LLM.

## Inicio rápido

```bash
cd pre-entrega-3
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # completá la clave de tu proveedor LLM
python -m main
```

La primera corrida descarga el modelo de embeddings (~470MB) e indexa el corpus
sola. Las siguientes reusan todo.

> **El venv no es opcional.** Las dependencias se instalan ahí; si corrés
> `python -m main` sin activarlo vas a ver
> `ModuleNotFoundError: No module named 'langchain_chroma'`.

## Preguntas de ejemplo

Copiá y pegá cualquiera de estas en el chat. Todas están verificadas contra el
corpus actual.

### Sesgos y objetividad

```
¿Qué son los sesgos cognitivos?
¿Qué es el sesgo de disponibilidad?
¿Qué es el sesgo de confirmación?
¿Qué es un error no forzado?
¿Qué es ser antifrágil?
¿Cómo evito caer en trampas mentales?
¿Por qué importa pensar en probabilidades y no en certezas?
```

### Decisiones y priorización

```
¿Cómo funciona la matriz de Eisenhower?
¿En qué cuadrante de Eisenhower conviene enfocarse?
¿Qué dice la ley de Hofstadter?
```

### Sistemas, mercados e incentivos

```
¿Qué es una externalidad?
¿Qué es la tragedia de los comunes?
¿Qué es el problema del free rider?
¿Qué dice la ley de Goodhart?
¿Qué es el problema principal-agente?
¿Qué es la información asimétrica?
¿Cuál es la diferencia entre decisiones reversibles e irreversibles?
¿Qué dice la ley de Murphy?
```

### Knowledge bases con LLMs

```
¿Cómo armo una base de conocimiento con LLMs?
¿Para qué sirve Obsidian Web Clipper?
```

### Sobre los apuntes mismos

```
¿Quién escribió Super Thinking?
¿De qué libro salen estos apuntes?
```

### Preguntas trampa (tienen que devolver "No lo sé")

Sirven para comprobar que el sistema no alucina cuando la respuesta no está en
los apuntes:

```
¿Cuál es la capital de Australia?
¿Quién ganó el mundial del 86?
¿Cómo hago una milanesa napolitana?
¿Cómo configuro nginx?
```

Ejemplo de una sesión real:

```
Tú > ¿Qué es el sesgo de confirmación?

  El sesgo de confirmación es la tendencia a buscar o interpretar información
  que confirma nuestras creencias preexistentes, y a desacreditar o ignorar
  las pruebas que las contradicen. En la práctica, si creés que tu idea de
  proyecto es brillante, vas a buscar colegas que estén de acuerdo y vas a
  tildar de "pesimista" a quien señala un fallo fatal.

  Referencias:
    - Super Thinking - Sesgos y Objetividad.md — Sesgos Cognitivos y Pensamiento Objetivo > Sesgo de Confirmación
    - Super Thinking - Sesgos y Objetividad.md — Sesgos Cognitivos y Pensamiento Objetivo > Sesgo de Probabilidad Optimista

Tú > ¿Cuál es la capital de Australia?

  No lo sé. No encontré información relacionada con tu pregunta en mis apuntes,
  así que prefiero no inventar una respuesta.

Tú > salir
Chau 👋
```

> Las referencias muestran **archivo y sección**, no solo el archivo. Con el
> nombre del `.md` solo, un fragmento irrelevante del archivo correcto parece
> una cita válida y esconde que el retrieval trajo cualquier cosa.

## Uso

```bash
python -m main              # chat interactivo (indexa solo si hace falta)
python -m main --demo       # dos preguntas fijas, salida JSON cruda
python -m main --reindex    # reindexar y entrar al chat

python -m ingest            # solo indexar (idempotente)
python -m ingest --reindex  # forzar reindexado
```

En el chat, `salir`, `salí`, `q`, `quit`, `exit` o `chau` terminan la sesión.

El modo `--demo` imprime el `RagResponse` completo:

```json
{
  "text": "La matriz de Eisenhower funciona categorizando las tareas en dos ejes: Urgente vs. No Urgente e Importante vs. No Importante...",
  "references": [
    {
      "source": "Super Thinking - Decisiones y Priorización.md",
      "section": "Decisiones y Priorización > Matriz de Decisiones de Eisenhower",
      "snippet": "Decisiones y Priorización > Matriz de Decisiones de Eisenhower\nCategorizá las tareas en dos ejes: ..."
    }
  ]
}
```

## Arquitectura

```
data/*.md ──ingest.py──> MarkdownHeaderTextSplitter (h1/h2/h3)
              │  RecursiveCharacterTextSplitter (400/50) dentro de cada sección
              │  HuggingFaceEmbeddings(paraphrase-multilingual-MiniLM-L12-v2)
              ▼
        Chroma ./vectorstore (colección "apuntes", hnsw:space=cosine)
              ▲   store.py: única fuente de verdad sobre el estado del índice
              │
get_rag_response(query) ─retriever (top_k=4)─> gate (>= 0.30) ─> cadena LCEL async ─> RagResponse
        0 relevantes ──────────────────────────────────────────> RagResponse("No lo sé", [])
```

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Carga el `.env` y expone la configuración con defaults |
| `store.py` | Estado del índice Chroma: existencia, contenido y huella del corpus |
| `ingest.py` | Trocea el corpus y lo persiste (idempotente) |
| `embeddings.py` | Factory única del embedder local, cacheada |
| `retriever.py` | Búsqueda por similitud y gate de relevancia |
| `chain.py` | Generación con cadena A + fallback a cadena B |
| `rag.py` | Orquesta recuperación → gate → generación → citas |
| `main.py` | CLI: chat interactivo y demo |

### Decisiones de diseño

**Chunking por secciones.** Se trocea primero por encabezados de markdown y
recién ahí se subdivide si una sección es larga. Además **el título de la sección
se escribe dentro del texto que se embebe**: sin eso, la sección "Sesgo de
Disponibilidad" no matchea la consulta "¿qué es el sesgo de disponibilidad?",
porque el cuerpo explica el concepto sin repetir nunca su nombre. Medido sobre
esa consulta, el troceado plano de 800 caracteres devolvía un fragmento del
archivo equivocado; el troceado por secciones devuelve la sección correcta
primera.

**`CHUNK_SIZE=400` y no 800.** MiniLM embebe como mucho 128 word pieces (~400
caracteres de español) y **descarta el resto en silencio**. Con 800, el 67% de
los chunks quedaba truncado y su segunda mitad era invisible para la búsqueda,
aunque figurara en el índice. La ingesta ahora avisa si más del 10% de los
chunks se pasa del límite del embedder.

**No se descarta ninguna sección.** Incluso "Notas relacionadas" aporta: es la
que lleva la atribución del libro, y es la que responde "¿quién escribió Super
Thinking?" (0.652). Esto depende de que los apuntes estén curados — con
wikilinks crudos de Obsidian esas secciones son ruido y conviene filtrarlas.

**Doble guarda contra la alucinación.** Primero el gate de similitud, barato y
local; después el LLM, al que se le prohíbe salir del contexto. Con 0 fragmentos
sobre el umbral se responde "No lo sé" **sin llamar al LLM**.

**El umbral prioriza recall a propósito.** Los scores de afines y ajenas se
superponen (peor afín 0.310, peor ajeno 0.329): ningún umbral acierta de los dos
lados. Se elige el permisivo porque los errores no cuestan lo mismo — un ajeno
que pasa el gate lo ataja el LLM diciendo "No lo sé", pero un afín bloqueado en
el gate nunca llega al LLM y no tiene segunda chance.

**Las citas no las escribe el LLM.** El modelo devuelve únicamente `text`
(`LlmAnswer`); las `references` las arma `rag.py` desde los fragmentos que
pasaron el gate, deduplicadas por sección. Así una cita no puede ser inventada.
Si el modelo contesta que no sabe, la respuesta va sin referencias: no
fundamentaron nada.

**El índice se valida, no se supone.** Que exista `chroma.sqlite3` no alcanza. Se
chequea que la colección exista, tenga documentos y corresponda al corpus actual
(huella de contenido + modelo + chunking); si no, se lanza `IndexNotReadyError`.
El retriever usa `create_collection_if_not_exists=False` porque el default de
`langchain_chroma` crea una colección vacía en silencio, y eso hace que **todas**
las consultas respondan "No lo sé" como si el gate estuviera funcionando bien.

**Ingesta idempotente con huella.** Si cambiás un apunte, `EMBEDDING_MODEL`,
`CHUNK_SIZE`, `CHUNK_OVERLAP` o la estrategia de troceado, el índice se
reconstruye solo. No hace falta borrar `./vectorstore` a mano.

## Tests

```bash
# Rápidos: gate, guardas del índice, fallback de cadenas, detección de rechazo,
# schemas. Sin modelo, sin red, sin credenciales.
pytest -m "not slow"

# Suite completa (usa el modelo local de embeddings)
pytest
```

Los tests de integración comparten un fixture session-scoped que reusa el índice
de `./vectorstore`. Los que necesitan **generar** con el LLM real se saltean si
`LLM_PROVIDER` no tiene credenciales, para que un `.env` vacío o un 429 no rompan
la suite. Con `HF_HUB_OFFLINE=1` y sin caché ni índice, también se saltean con un
mensaje claro.

## Configuración

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Proveedor de generación: `gemini`, `openai`, `anthropic`, `openrouter` |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo local de embeddings |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `400` / `50` | Troceado dentro de cada sección |
| `TOP_K` | `4` | Fragmentos recuperados por consulta |
| `SIMILARITY_THRESHOLD` | `0.30` | Umbral del gate de relevancia (score coseno) — ver abajo |
| `VECTORSTORE_DIR` | `./vectorstore` | Carpeta del índice Chroma |
| `COLLECTION_NAME` | `apuntes` | Nombre de la colección |
| `DATA_DIR` | `./data` | Carpeta de apuntes `.md` |

### Elegir el umbral: recall o cuota

Medido sobre este corpus (16 consultas afines, 12 ajenas), los dos grupos se
superponen: el peor afín puntúa **0.310** y el peor ajeno **0.329**. No hay
umbral que acierte de los dos lados, así que hay que elegir qué error preferís.

| Umbral | Qué gana | Qué pierde |
|---|---|---|
| **`0.30`** (default del código) | Responde consultas al límite como "¿qué significa ser antifrágil?" (0.310) | Deja pasar ~2 de 12 ajenas al LLM, que las rechaza. Cuesta una llamada por cada una |
| **`0.35`** | Corta las ajenas en el gate, sin gastar llamadas | Bloquea afines al límite y devuelve "No lo sé" a preguntas válidas |

El default es `0.30` porque los errores no son simétricos: un ajeno que pasa el
gate lo ataja el LLM (verificado con "¿cómo configuro nginx?" y "receta de
ñoquis"), pero un afín bloqueado nunca llega al LLM y no tiene segunda chance.
Si estás con una cuota ajustada de un free tier, `0.35` en el `.env` cambia
recall por llamadas ahorradas.

### Agregar tus propios apuntes

Poné cualquier `.md` en `data/` y corré `python -m main`: el corpus se descubre
por glob y el índice se reconstruye solo. Para que el retrieval funcione bien,
conviene que los apuntes usen encabezados (`##`) descriptivos, porque el título
de la sección es lo que más pesa a la hora de encontrarla.

### Arranque más rápido

El modelo de embeddings se cachea en `~/.cache/huggingface` la primera vez y **no
se vuelve a descargar**. La barra `Loading weights:` de cada arranque es la carga
de los pesos desde disco, no una descarga — confunde, pero es local.

Lo que sí ocurre en cada arranque es una revalidación contra el Hub. Se saltea
con `HF_HUB_OFFLINE=1`: medido en este proyecto, la corrida baja de **14.5s a
8.2s**.

```bash
HF_HUB_OFFLINE=1 python -m main

# o, para no repetirlo, en tu shell o en .venv/bin/activate:
export HF_HUB_OFFLINE=1
```

> Esta variable **no** funciona desde el `.env`, a diferencia del resto de la
> configuración. `huggingface_hub` la lee en import time, y la importan
> transitivamente `langchain_core` y `chromadb` antes de que `config.py` llegue a
> ejecutar `load_dotenv()`. Tiene que estar en el entorno del proceso.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_chroma'` | El venv no está activado. `source .venv/bin/activate` |
| `Rate limit exceeded: free-models-per-day` | Se agotó la cuota diaria del proveedor. Cambiá `LLM_PROVIDER` o esperá al reset. La ingesta y la búsqueda siguen funcionando sin LLM |
| `WARNING: ... superan los 128 word pieces` | `CHUNK_SIZE` es más grande de lo que el embedder puede leer. Bajalo al valor que sugiere el mensaje y reindexá |
| `IndexNotReadyError` | El índice no existe o no corresponde al corpus. `python -m ingest --reindex` |
| Responde "No lo sé" a algo que sí está en los apuntes | Probá reformular (ver limitaciones). Si pasa seguido, bajá `SIMILARITY_THRESHOLD` y recalibrá |
| Arranque lento y `unauthenticated requests to the HF Hub` | Falta `HF_HUB_OFFLINE=1` en el entorno |

## Limitaciones conocidas

- **El embedder trunca en 128 word pieces.** `MiniLM-L12-v2` no lee más que eso
  por chunk. Es la razón de `CHUNK_SIZE=400`: con chunks más largos, el final
  queda indexado pero invisible para la búsqueda.
- **Los scores absolutos no están bien calibrados.** El ranking suele ser
  correcto, pero el score no: se midió un fragmento equivocado en 0.456 y uno
  correcto en 0.310. Por eso el umbral es permisivo y la decisión final la toma
  el LLM.
- **El retrieval es sensible a cómo formulás la pregunta.** "¿Qué es ser
  antifrágil?" puntúa 0.377 y responde; "¿Qué significa ser antifrágil?" puntúa
  0.310 y queda al borde. Si no te contesta, reformulá con las palabras que usa
  el apunte.
- **Los sub-temas dentro de una lista no se encuentran por nombre.** "Navaja de
  Ockham" es un ítem dentro de "Tácticas Prácticas": el embedding del chunk está
  dominado por el resto de la lista y la consulta directa no lo alcanza.
- **La calibración es de este corpus.** `SIMILARITY_THRESHOLD=0.30` se midió con
  16 consultas afines y 12 ajenas sobre estos apuntes. Si cambiás el corpus, el
  modelo o el chunking, recalibrá con
  `retriever.similarity_search_with_relevance_scores`.
- **La generación necesita credenciales.** La ingesta y la recuperación funcionan
  sin ninguna clave; solo la respuesta final requiere el proveedor configurado.
