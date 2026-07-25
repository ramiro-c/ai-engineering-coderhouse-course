# Pre-entrega 2: Pipeline de procesamiento validado

Pipeline de extracción de entidades técnicas. Recibe un párrafo de texto sin
procesar (descripción de arquitectura, log de error, etc.) y devuelve un
objeto validado con Pydantic, usando LangChain (`ChatPromptTemplate` +
`with_structured_output` + `with_retry`).

## Qué incluye

- `schemas.py`: modelo Pydantic `TechnicalExtraction` (`tecnologias`,
  `nivel_de_criticidad`, `resumen_tecnico`) y `ExtractionError`, el modelo
  que se devuelve si el pipeline agota los reintentos sin lograr una
  extracción válida.
- `clients/factory.py`: factory que arma el chat model de LangChain
  (`ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI` o `ChatOpenRouter`)
  según `LLM_PROVIDER`, reutilizando el patrón de selección de proveedor de
  `pre-entrega-1/manager.py`.
- `chain.py`: `ChatPromptTemplate` + `model.with_structured_output(TechnicalExtraction)`
  compuestos con LCEL, con `.with_retry()` para reintentar ante JSON mal
  formado o incompleto. Expone `process_text(text: str)` asíncrono, que
  devuelve `TechnicalExtraction` o, si falla, `ExtractionError`.
- `main.py`: mini-script de prueba que corre tres casos (descripción de
  arquitectura, texto ambiguo y texto sin información técnica) y muestra
  la validación en logs.

## Estructura

```text
pre-entrega-2/
├── clients/
│   ├── __init__.py
│   └── factory.py        # factory de chat models LangChain por proveedor
├── config.py
├── schemas.py             # TechnicalExtraction (Pydantic)
├── chain.py               # prompt + LCEL + with_structured_output + with_retry
├── main.py                # mini-script de prueba asíncrono
├── requirements.txt
├── .env.example
└── README.md
```

## Requisitos

- Python 3.12
- Variables de entorno en `.env`

## Variables de entorno

Copia `.env.example` a `.env`. Mismo esquema que `pre-entrega-1`:

```bash
LLM_PROVIDER=gemini            # gemini | openai | anthropic | openrouter

# Gemini vía Vertex AI (recomendado si la API key da 429)
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=tu-proyecto-gcp
GOOGLE_CLOUD_LOCATION=us-central1

# o Gemini Developer API
GEMINI_API_KEY=tu-api-key

OPENAI_API_KEY=
OPENROUTER_API_KEY=
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

## Ejemplo de salida esperada

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones concurrentes."
}
```

Salida real de una corrida contra Gemini (`LLM_PROVIDER=gemini`, Vertex AI):

```json
{
  "tecnologias": [
    "FastAPI",
    "Redis",
    "PostgreSQL",
    "API",
    "Database Connection Pooling"
  ],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El servicio de pagos, implementado con FastAPI, utiliza Redis para caché y PostgreSQL para persistencia. Bajo carga concurrente, el pool de conexiones a PostgreSQL se agota, resultando en timeouts en las solicitudes."
}
```

### Prueba de estrés (texto ambiguo)

`main.py` incluye dos casos sin tecnologías explícitas ("¿algo se rompió?"
y "hola"). El modelo no lanza excepción: infiere tecnologías genéricas a
partir del contexto en vez de devolver una lista vacía, gracias a la
instrucción explícita del prompt y a la validación `min_length=1` en
`tecnologias`.

## Resiliencia

- `chain.with_retry(stop_after_attempt=3)` reintenta toda la cadena
  (prompt → modelo → parseo estructurado) si el LLM devuelve un JSON
  incompleto o que no valida contra `TechnicalExtraction`.
- La validación Pydantic (`min_length=1` en `tecnologias` y
  `resumen_tecnico`) actúa como guardia contra respuestas truncadas por
  límite de tokens: si el JSON queda incompleto, la validación falla y
  dispara un reintento en vez de propagar un objeto a medio construir.
- `with_structured_output` puede devolver `None` si el modelo no llega a
  invocar la función estructurada (por ejemplo, si se niega o corta la
  respuesta). `chain.py` agrega un paso `_ensure_complete` que convierte
  ese `None` en una excepción real, para que `.with_retry()` —que solo
  reacciona ante excepciones— también reintente en ese caso.
- Si se agotan los 3 intentos, `process_text()` no propaga la excepción:
  devuelve un `ExtractionError` con el tipo de error y una explicación,
  para que el caller siempre reciba un objeto Pydantic validado.
