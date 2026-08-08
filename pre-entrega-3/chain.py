from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from clients import build_chat_model
from schemas import RagGenerationError, RagResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas sobre un corpus de apuntes propios "
    "(modelos mentales de Super Thinking y notas sobre knowledge bases con LLMs). "
    "Respondé SIEMPRE en español, con tono directo y voseo.\n\n"
    "Reglas estrictas:\n"
    "1. Basate EXCLUSIVAMENTE en el contexto provisto. No uses conocimiento externo.\n"
    "2. Si la respuesta no está en el contexto, respondé exactamente 'No lo sé' "
    "y nada más, sin inventar contenido.\n"
    "3. Si el contexto cubre solo parte de la pregunta, respondé esa parte y aclaralo "
    "qué parte no sabés.\n"
    "4. No inventes datos, cifras ni citas que no aparezcan en el contexto."
)


@lru_cache(maxsize=1)
def _build_chains() -> tuple[Runnable, Runnable]:
    """Construye (cadena A, cadena B) de forma perezosa.

    El modelo se crea recién acá para que recuperación y gate (RAG-GEN-02)
    funcionen sin credenciales del LLM: solo la generación las necesita.
    """
    model = build_chat_model(provider=None, temperature=0.2)

    # Cadena A: parser Pydantic didáctico con reintentos (RAG-GEN-01)
    parser = PydanticOutputParser(pydantic_object=RagResponse)

    prompt_parser = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Contexto:\n{contexto}\n\nPregunta:\n{pregunta}\n\n{formato}",
            ),
        ]
    ).partial(formato=parser.get_format_instructions())

    prompt_structured = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Contexto:\n{contexto}\n\nPregunta:\n{pregunta}"),
        ]
    )

    chain_a = (prompt_parser | model | parser).with_retry(stop_after_attempt=MAX_ATTEMPTS)
    # Cadena B (fallback): salida estructurada nativa del proveedor
    chain_b = prompt_structured | model.with_structured_output(RagResponse)
    return chain_a, chain_b


async def generate_response(pregunta: str, contexto: str) -> RagResponse | RagGenerationError:
    """Genera la respuesta con la cadena A; si falla, con la B; si B falla, error."""
    chain_a, chain_b = _build_chains()

    try:
        return await chain_a.ainvoke({"contexto": contexto, "pregunta": pregunta})
    except Exception as exc:
        logger.warning("Cadena A (parser Pydantic) falló: %s", exc)

    try:
        return await chain_b.ainvoke({"contexto": contexto, "pregunta": pregunta})
    except Exception as exc:
        logger.exception("Cadena B (with_structured_output) falló")
        return RagGenerationError(
            error=type(exc).__name__,
            detalle=f"No se pudo generar una respuesta estructurada: {exc}",
        )
