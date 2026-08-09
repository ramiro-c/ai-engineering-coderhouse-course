"""Tests del fallback de generación (cadena A -> cadena B).

Sin red ni credenciales: `generate_response` acepta las cadenas por parámetro,
así que se le inyectan dobles que fallan o responden a voluntad.
"""

from __future__ import annotations

import asyncio

from chain import generate_response
from schemas import LlmAnswer, RagGenerationError


class _CadenaFalsa:
    """Doble de un Runnable: responde algo fijo o revienta con una excepción."""

    def __init__(
        self, resultado: LlmAnswer | None = None, error: Exception | None = None
    ) -> None:
        self._resultado = resultado
        self._error = error
        self.llamadas = 0

    async def ainvoke(self, _payload: dict) -> LlmAnswer:
        self.llamadas += 1
        if self._error is not None:
            raise self._error
        assert self._resultado is not None
        return self._resultado


def test_cadena_a_ok_no_toca_la_b():
    a = _CadenaFalsa(resultado=LlmAnswer(text="respuesta de A"))
    b = _CadenaFalsa(resultado=LlmAnswer(text="respuesta de B"))

    resultado = asyncio.run(generate_response("p", "c", chains=(a, b)))

    assert isinstance(resultado, LlmAnswer)
    assert resultado.text == "respuesta de A"
    assert b.llamadas == 0, "si A responde, B no debe ejecutarse"


def test_si_falla_la_a_responde_la_b():
    a = _CadenaFalsa(error=ValueError("el parser Pydantic no pudo con la salida"))
    b = _CadenaFalsa(resultado=LlmAnswer(text="respuesta de B"))

    resultado = asyncio.run(generate_response("p", "c", chains=(a, b)))

    assert isinstance(resultado, LlmAnswer)
    assert resultado.text == "respuesta de B"
    assert a.llamadas == 1
    assert b.llamadas == 1


def test_si_fallan_las_dos_devuelve_error_estructurado():
    a = _CadenaFalsa(error=ValueError("A rota"))
    b = _CadenaFalsa(error=TimeoutError("B rota"))

    resultado = asyncio.run(generate_response("p", "c", chains=(a, b)))

    assert isinstance(resultado, RagGenerationError)
    assert resultado.error == "TimeoutError"
    assert "B rota" in resultado.detalle


def test_el_schema_del_llm_no_pide_referencias():
    """El modelo solo devuelve texto: las citas las arma el pipeline."""
    assert set(LlmAnswer.model_fields) == {"text", "answered"}


def test_el_prompt_no_arrastra_comentarios_de_diseno():
    """El JSON schema que viaja en el prompt no debe traer notas internas.

    Pydantic copia el docstring de la clase a `description` del schema, y
    PydanticOutputParser lo inyecta en cada prompt: por eso LlmAnswer usa
    comentarios `#` para el razonamiento de diseño y no un docstring.
    """
    schema = LlmAnswer.model_json_schema()
    assert "description" not in schema, (
        "un docstring en LlmAnswer se le manda al modelo en cada consulta"
    )
