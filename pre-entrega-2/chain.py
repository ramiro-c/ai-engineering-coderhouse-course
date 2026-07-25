from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from clients import build_chat_model
from schemas import ExtractionError, TechnicalExtraction

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "Sos un analista técnico senior. A partir del texto de entrada (puede ser una "
    "descripción de arquitectura de software o un log de error) extraé la "
    "información técnica solicitada: tecnologias (lista no vacía), "
    "nivel_de_criticidad (baja, media o alta) y resumen_tecnico (texto breve).\n\n"
    "Si el texto no menciona tecnologías de forma explícita, inferí las más "
    "probables a partir del contexto (frameworks, bases de datos, protocolos, "
    "servicios). Nunca dejes la lista de tecnologías vacía."
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{texto}"),
    ]
)

def _ensure_complete(result: TechnicalExtraction | None) -> TechnicalExtraction:
    if result is None:
        raise ValueError(
            "El LLM no devolvió una extracción estructurada válida "
            "(JSON incompleto, sin llamada de función o corte por longitud)"
        )
    return result


model = build_chat_model(provider=None, temperature=0)
structured_model = model.with_structured_output(TechnicalExtraction)

chain = (prompt | structured_model | RunnableLambda(_ensure_complete)).with_retry(
    stop_after_attempt=MAX_ATTEMPTS
)


async def process_text(text: str) -> TechnicalExtraction | ExtractionError:
    logger.info("Procesando texto de entrada (%d caracteres)", len(text))
    try:
        result = await chain.ainvoke({"texto": text})
    except Exception as exc:
        logger.exception("La cadena agotó los reintentos sin devolver un JSON válido")
        return ExtractionError(
            error=type(exc).__name__,
            detalle=f"No se pudo obtener una extracción válida tras los reintentos: {exc}",
        )
    logger.info(
        "Extracción validada: %d tecnologías, criticidad=%s",
        len(result.tecnologias),
        result.nivel_de_criticidad.value,
    )
    return result
