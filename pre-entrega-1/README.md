# Pre-entrega 1: Unified Async LLM Client

Cliente unificado asíncrono para Gemini, OpenAI y Anthropic.

## Qué incluye

- `schemas.py` con validación Pydantic para mensajes y configuración del modelo.
- `clients/` con `BaseLLMClient`, `GeminiClient`, `OpenAIClient` y `AnthropicClient`.
- `manager.py` con `AsyncLLMManager` para elegir proveedor por `LLM_PROVIDER`.
- `main.py` para probar una respuesta normal y otra en streaming.
- `data/` con capturas de salida de ejemplo (no las usa el código; son referencia de una corrida exitosa).

## Estructura

```text
pre-entrega-1/
├── clients/                 # proveedores LLM (base + gemini/openai/anthropic)
├── data/
│   └── response-example.md  # ejemplo de salida de `python3 main.py`
├── config.py
├── manager.py
├── schemas.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

### Carpeta `data/`

Guarda ejemplos de respuestas reales del script de prueba. Sirve para documentar cómo se ve una corrida correcta (modo normal + streaming) sin tener que ejecutar el cliente.

- [`data/response-example.md`](data/response-example.md): salida de ejemplo al preguntar *"¿Qué es la entropía?"* con Gemini.

## Requisitos

- Python 3.12
- Variables de entorno en `.env`

## Variables de entorno

Copia `.env.example` a `.env`.

### Gemini con Vertex AI (recomendado si la API key da 429)

```bash
LLM_PROVIDER=gemini
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=tu-proyecto-gcp
GOOGLE_CLOUD_LOCATION=us-central1
```

Autenticación con Application Default Credentials:

```bash
gcloud auth application-default login
gcloud config set project tu-proyecto-gcp
```

En GCP necesitás Vertex AI API habilitada y permisos (p. ej. `roles/aiplatform.user`).

### Gemini Developer API (API key)

```bash
LLM_PROVIDER=gemini
GOOGLE_GENAI_USE_VERTEXAI=false
GEMINI_API_KEY=tu-api-key
```

### Otros proveedores

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecución

```bash
python3 main.py
```

Por defecto usa Gemini. Con `GOOGLE_GENAI_USE_VERTEXAI=true` va por Vertex; si no, usa `GEMINI_API_KEY`.
