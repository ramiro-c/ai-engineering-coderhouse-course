from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from clients import build_chat_model
from schemas import LlmAnswer, RagGenerationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas sobre un corpus de apuntes propios "
    "(modelos mentales del libro Super Thinking y notas sobre knowledge bases con LLMs). "
    "Respondé SIEMPRE en español, con tono directo y voseo.\n\n"
    "Reglas estrictas:\n"
    "1. Basate EXCLUSIVAMENTE en el contexto provisto. No uses conocimiento externo.\n"
    "2. 'No lo sé' es SOLO para cuando el contexto no habla del tema de la pregunta. "
    "Si el contexto habla del tema, tenés que responder con lo que hay.\n"
    "3. Si el contexto no define el concepto pero sí da ejemplos, tipos, casos o "
    "consecuencias, respondé con eso y aclará qué no está en los apuntes. Resumir "
    "lo que sí está NO es inventar: es responder.\n"
    "4. Si el contexto cubre solo parte de la pregunta, respondé esa parte y aclará "
    "qué parte falta.\n"
    "5. No inventes datos, cifras ni citas que no aparezcan en el contexto.\n"
    "6. Poné answered=false SOLO si tuviste que decir que no sabés; si pudiste "
    "responder aunque sea en parte, va en true."
)


@lru_cache(maxsize=1)
def _build_chains() -> tuple[Runnable, Runnable]:
    """Construye (cadena A, cadena B) de forma perezosa.

    El modelo se crea recién acá para que recuperación y gate
    funcionen sin credenciales del LLM: solo la generación las necesita.
    """
    model = build_chat_model(provider=None, temperature=0.2)

    # Cadena A: parser Pydantic didáctico con reintentos
    parser = PydanticOutputParser(pydantic_object=LlmAnswer)

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

    chain_a = (prompt_parser | model | parser).with_retry(
        stop_after_attempt=MAX_ATTEMPTS
    )
    # Cadena B (fallback): salida estructurada nativa del proveedor
    chain_b = prompt_structured | model.with_structured_output(LlmAnswer)
    return chain_a, chain_b


async def generate_response(
    pregunta: str,
    contexto: str,
    chains: tuple[Runnable, Runnable] | None = None,
) -> LlmAnswer | RagGenerationError:
    """Genera la respuesta con la cadena A; si falla, con la B; si B falla, error.

    `chains` existe para poder testear el fallback A -> B sin red ni credenciales:
    en producción se deja en None y se usan las cadenas reales.
    """
    chain_a, chain_b = chains if chains is not None else _build_chains()

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
