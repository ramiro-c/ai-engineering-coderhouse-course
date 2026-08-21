# Pre-entrega 5 — Agente ReAct con LangGraph

Agente **ReAct** (Reason + Act) sobre un catálogo in-memory de clientes y
pedidos. El grafo LangGraph alterna entre un nodo **agent** (LLM con
`bind_tools`) y un nodo **tools** (`ToolNode`), con arista condicional
`tools_condition` que cierra el ciclo hasta una respuesta final. La memoria de
conversación persiste en **SqliteSaver** local identificada por **`thread_id`**
(checkpoint SQLite gitignored). El LLM se elige vía `clients/factory.py`
(default: `gemini` con **Vertex/ADC**; también `openai`, `anthropic`,
`openrouter`).

## Requisitos

- Python 3.12+ y `pip`
- Credencial LLM según `LLM_PROVIDER` (default `gemini` + Vertex/ADC; ver abajo)
- (Opcional) Credenciales alternativas si cambiás el provider

## Inicio rápido

```bash
cd pre-entrega-5
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # completá credenciales Vertex (o la del provider activo)
python demo.py                # REPL interactivo
python demo.py --trace        # demo scriptada cliente 102 + dump de trazas
```

> **El venv no es opcional.** Si corrés los scripts sin activarlo vas a ver
> `ModuleNotFoundError`. El catálogo de clientes/pedidos vive en
> `data/pedidos.py` (in-memory, sin base de datos externa).

### Paso a paso

1. **`pip install -r requirements.txt`** — instala las versiones fijadas:
   `langchain==1.3.14`, `langgraph==1.2.11`, `langgraph-checkpoint-sqlite==3.1.1`,
   `langgraph-prebuilt==1.1.0`, `langchain-google-genai==4.3.3` (Gemini vía
   `ChatGoogleGenerativeAI`), y las dependencias de los otros providers
   (`langchain-openai`, `langchain-anthropic`, `langchain-openrouter`).

2. **`.env`** — copiá `.env.example` y completá las claves:
   - `LLM_PROVIDER=gemini` (default).
   - **Vertex (default en `.env.example`):** `GOOGLE_GENAI_USE_VERTEXAI=true`
     + `GOOGLE_APPLICATION_CREDENTIALS` (ruta al JSON de service account),
     `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.
   - **Developer API (debug rápido):** `GEMINI_API_KEY` con
     `GOOGLE_GENAI_USE_VERTEXAI` ausente o `false`.
   - `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY`: solo si
     cambiás el provider.
   - `CHECKPOINT_PATH` (opcional): ruta del SQLite de checkpoints (default
     `checkpoints.sqlite` en este directorio).

3. **`python demo.py`** — REPL async: escribí preguntas sobre pedidos de
   clientes (ej. cliente 102), salí con `salir`, `exit`, `q` o Ctrl+C/Ctrl+D.
   Usá `--thread-id mi-sesion` para fijar la sesión persistente (default:
   `demo`).

4. **`python demo.py --trace`** — corre una demo scriptada de dos turnos sobre
   el cliente 102 y escribe las trazas ReAct en `traces/react-trace.json` y
   `traces/react-trace.log`. Usa un `thread_id` efímero (`trace-<uuid>`) para
   que regenerar trazas no acumule historial de corridas anteriores; pasá
   `--thread-id` solo si querés fijar la sesión a propósito. Requiere
   credenciales Vertex configuradas (validación explícita antes de invocar el LLM).

> **Nunca commitees claves.** `.env` y archivos `*.sqlite` (p. ej.
> `checkpoints.sqlite`) están en `.gitignore`. El JSON de ADC/service account
> **no** se ignora por ruta arbitraria: mantenelo fuera del repo o bajo
> `.secrets/` (directorio sí ignorado). Solo versioná `.env.example`.

## Arquitectura del grafo

```
START → agent ──tools_condition──> tools ──> agent ──> END
              └─ (respuesta final) ────────────────> END
```

| Componente | Responsabilidad |
|---|---|
| `AgentState` | Subclase de `MessagesState`; el reducer `add_messages` acumula historial |
| `agent` | LLM async con `bind_tools`; system prompt obliga `buscar_cliente` → `buscar_pedidos` |
| `tools` | `ToolNode` con `buscar_cliente` y `buscar_pedidos` |
| `tools_condition` | Prebuilt de LangGraph: si hay `tool_calls` → `tools`, si no → fin |
| `recursion_limit` | **10** (via `invoke_config()`); tope de ciclos agent↔tools por turno |

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Variables de entorno, `CHECKPOINT_PATH`, `RECURSION_LIMIT=10` |
| `graph.py` | `StateGraph`, `SqliteSaver` local (wrappers async), `build_graph`, `invoke_config` |
| `tools.py` | Herramientas `@tool` sobre el catálogo in-memory |
| `data/pedidos.py` | Clientes 101/102 y pedidos de ejemplo |
| `clients/factory.py` | Factory multi-proveedor (`build_chat_model`) |
| `demo.py` | CLI async: REPL, `--thread-id`, `--trace` y dump de trazas |

## Persistencia (SqliteSaver + thread_id)

Cada conversación se identifica con un **`thread_id`** en la config de
invocación:

```python
config = {
    "configurable": {"thread_id": "mi-sesion"},
    "recursion_limit": 10,
}
```

`open_checkpointer()` abre (o crea) `checkpoints.sqlite` en
`CHECKPOINT_PATH`. El archivo **no se commitea** (`**/*.sqlite` en
`.gitignore`). Un segundo turno con el mismo `thread_id` recupera el historial
del primero vía checkpoint — probado en `tests/test_persistence.py`.

## Proveedor del LLM (`LLM_PROVIDER`)

El modelo lo elige `clients/factory.py` según `LLM_PROVIDER` (default
`gemini`):

| `LLM_PROVIDER` | Credencial | Modelo por defecto |
|---|---|---|
| `gemini` (default) | `GEMINI_API_KEY` **o** Vertex (`GOOGLE_GENAI_USE_VERTEXAI=true` + vars GCP) | `gemini-2.5-flash` |
| `openai` | `OPENAI_API_KEY` | `gpt-4o-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-haiku-latest` |
| `openrouter` | `OPENROUTER_API_KEY` | `cohere/north-mini-code:free` |

> **Gemini: dos modos (patrón de pre-entrega-3/4).** Por defecto en
> `.env.example`, `GOOGLE_GENAI_USE_VERTEXAI=true`: `ChatGoogleGenerativeAI`
> autentica con ADC/service account (`GOOGLE_APPLICATION_CREDENTIALS`,
> `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`) e ignora la API key. Para
> debug rápido o cuando Vertex no está disponible, desactivá Vertex y usá
> `GEMINI_API_KEY` (Developer API). Para evitar 429 del free tier, Vertex. Si
> falta la credencial del modo activo, el error sale al invocar el LLM. Solo
> hace falta la credencial del provider activo.

```bash
# Cambiar proveedor sin tocar código:
LLM_PROVIDER=openrouter python demo.py
```

## Trazas ReAct (`--trace`)

Tras `python demo.py --trace`, encontrás:

| Archivo | Contenido |
|---|---|
| `traces/react-trace.json` | Lista JSON de mensajes (`tipo`, `content`, `tool_calls`, `name`) |
| `traces/react-trace.log` | Mismo recorrido en formato legible (Usuario / Pensamiento / Acción / Observación) |

Los tests `tests/test_trace_format.py` validan el JSON generado (tipos
`human`/`ai`/`tool` y llamadas a `buscar_cliente` o `buscar_pedidos`). Corré
`--trace` al menos una vez antes de esos tests, o regenerá el archivo si cambiás
el grafo.

## Tests

```bash
cd pre-entrega-5 && source .venv/bin/activate
python -m pytest tests -q                    # unit por default (excluye slow)
python -m pytest tests -m "not slow" -q      # explícito: ~35 tests sin red
python -m pytest tests -m slow -q            # smoke real contra Vertex (1 test)
```

`pytest.ini` excluye `slow` por default (`addopts = -m "not slow"`). Los tests
marcados `slow` (integración con Vertex en `test_integration.py`) se saltean
con mensaje claro si no hay ADC configurado — no falla CI sin keys. La suite
unit usa fakes/stubs (`FakeChatModel`, `:memory:` checkpointer): no toca la red.

## Demo interactiva

```bash
cd pre-entrega-5 && source .venv/bin/activate
python demo.py                              # REPL: varias preguntas
python demo.py --thread-id sesion-carlos    # misma sesión persistente
python demo.py --trace                      # demo scriptada + trazas en traces/

# Preguntas de ejemplo (cliente 102 tiene 3 pedidos, total 14500):
# ¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?
# ¿y el último?
```

Sin flags, el script entra en modo interactivo (REPL) pensado para uso natural:
escribí una pregunta, Enter consulta, repetí hasta `salir`/`exit`/`q` (o
Ctrl+C/Ctrl+D). `--trace` corre la secuencia fija de dos turnos del cliente 102
y guarda el historial completo en `traces/`. Debe ser un script en archivo
(lección #866: `load_dotenv` no resuelve `.env` desde `python3 -c`), así que ve
las claves del `.env` como el resto de los CLIs.
