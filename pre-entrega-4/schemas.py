"""Schemas Pydantic de la pre-entrega 4: datos de evaluación y recuperación.

Contratos D9 del diseño: GoldenSetItem, GoldenSet, RetrievalHit y EvalResult.
La evaluación opera a nivel documento (no chunk): el recall es 0 o 1 por
pregunta porque los hits se deduplican por document_id antes de medir.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["gemini", "openai", "anthropic", "openrouter"]


class GoldenSetItem(BaseModel):
    pregunta: str = Field(description="Pregunta en español de evaluación")
    documento_id_esperado: str = Field(
        description="Nombre del .md esperado (== metadata.document_id)"
    )


class GoldenSet(BaseModel):
    casos: list[GoldenSetItem] = Field(
        description="Pares pregunta -> documento esperado del golden set"
    )


class RetrievalHit(BaseModel):
    document_id: str = Field(description="Nombre del .md de origen")
    source: str = Field(description="Ruta del archivo de origen en data/")
    seccion: str = Field(
        default="",
        description="Ruta de encabezados de la sección, p. ej. 'Guía > Inicio'",
    )
    snippet: str = Field(description="Fragmento del chunk (~200 caracteres)")


class EvalResult(BaseModel):
    pregunta: str = Field(description="Pregunta evaluada")
    recuperados: list[str] = Field(
        description="Document ids recuperados, deduplicados a nivel documento"
    )
    esperado: str = Field(description="Document id esperado del golden set")
    precision_at_5: float = Field(description="Precision@5 sobre hits deduplicados")
    recall_at_5: float = Field(
        description="Recall@5 a nivel documento: 0 o 1 (1 si el esperado aparece)"
    )
    mrr: float = Field(description="Reciprocal rank del esperado en el ranking")


# Lo único que se le pide al LLM: las citas las sanea responder() contra la
# metadata del contexto recuperado, así no puede inventarlas (RF-6).
# Comentario y no docstring a propósito: Pydantic copia el docstring al JSON
# schema y PydanticOutputParser lo inyecta en el prompt de cada consulta.
class LlmAnswer(BaseModel):
    pregunta: str = Field(description="Pregunta original que se intentó responder")
    respuesta: str = Field(
        description="Respuesta en español neutro, fundamentada SOLO en el contexto recuperado"
    )
    answered: bool = Field(
        default=True,
        description=(
            "true si el contexto alcanzó para responder; false si el contexto "
            "no alcanza o hubo un error al generar"
        ),
    )
    fuentes: list[str] = Field(
        default_factory=list,
        description=(
            "Document ids citados; salen SOLO de la metadata del contexto "
            "recuperado, nunca inventados"
        ),
    )
