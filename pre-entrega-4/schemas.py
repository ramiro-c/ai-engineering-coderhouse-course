"""Schemas Pydantic de la pre-entrega 4: datos de evaluación y recuperación.

Contratos D9 del diseño: GoldenSetItem, GoldenSet, RetrievalHit y EvalResult.
La evaluación opera a nivel documento (no chunk): el recall es 0 o 1 por
pregunta porque los hits se deduplican por document_id antes de medir.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
