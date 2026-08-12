"""Tests unitarios de los schemas Pydantic del RAG híbrido (sin red ni modelo).

Cubren los contratos D9: GoldenSetItem, GoldenSet, RetrievalHit y EvalResult.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas import EvalResult, GoldenSet, GoldenSetItem, RetrievalHit


def test_golden_set_item_minimo():
    item = GoldenSetItem(
        pregunta="¿Qué es FastAPI?",
        documento_id_esperado="features/01-intro.md",
    )
    assert item.pregunta.startswith("¿Qué")
    assert item.documento_id_esperado.endswith(".md")


def test_golden_set_item_requiere_documento_esperado():
    with pytest.raises(ValidationError):
        GoldenSetItem(pregunta="¿Qué es FastAPI?")


def test_golden_set_con_casos():
    golden = GoldenSet(
        casos=[
            GoldenSetItem(
                pregunta="¿Qué es FastAPI?",
                documento_id_esperado="features/01-intro.md",
            ),
            GoldenSetItem(
                pregunta="¿Cómo definir una ruta?",
                documento_id_esperado="features/02-rutas.md",
            ),
        ]
    )
    assert len(golden.casos) == 2
    assert golden.casos[1].documento_id_esperado == "features/02-rutas.md"


def test_golden_set_rechaza_caso_incompleto():
    with pytest.raises(ValidationError):
        GoldenSet(casos=[{"pregunta": "¿Qué es FastAPI?"}])


def test_retrieval_hit_seccion_por_defecto_vacia():
    hit = RetrievalHit(
        document_id="features/01-intro.md",
        source="features/01-intro.md",
        snippet="FastAPI es un framework moderno...",
    )
    assert hit.seccion == ""
    assert hit.document_id == "features/01-intro.md"


def test_retrieval_hit_requiere_snippet():
    with pytest.raises(ValidationError):
        RetrievalHit(document_id="features/01-intro.md", source="features/01-intro.md")


def test_eval_result_completo():
    resultado = EvalResult(
        pregunta="¿Qué es FastAPI?",
        recuperados=["features/01-intro.md", "features/02-rutas.md"],
        esperado="features/01-intro.md",
        precision_at_5=0.5,
        recall_at_5=1.0,
        mrr=1.0,
    )
    assert resultado.precision_at_5 == 0.5
    assert resultado.recall_at_5 == 1.0
    assert resultado.mrr == 1.0


def test_eval_result_requiere_metricas():
    with pytest.raises(ValidationError):
        EvalResult(pregunta="¿Qué es FastAPI?", recuperados=["a.md"], esperado="a.md")
