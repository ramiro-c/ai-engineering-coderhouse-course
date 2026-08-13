"""Métricas de evaluación del RAG híbrido (Fase 5, RF-4/D9).

Cubren las funciones puras de evaluate.py: precision_at_k/recall_at_k/mrr con
casos conocidos, el dedupe a nivel documento (la evaluación mide Recall@5 por
documento, NO por chunk) y las piezas del CLI (carga del golden set,
evaluación por caso, promedios y el criterio PASS/FAIL Recall@5 >= 0.8).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluate import (
    cargar_golden,
    decidir_pase,
    dedupe_documentos,
    evaluar_casos,
    mrr,
    precision_at_k,
    promedios,
    recall_at_k,
)


def _hit(document_id: str) -> SimpleNamespace:
    """Hit mínimo para las métricas puras (solo importa metadata.document_id)."""
    return SimpleNamespace(metadata={"document_id": document_id})


def test_metricas_caso_conocido_esperado_en_top1():
    recuperados = [
        "routing.md",
        "dependencies.md",
        "testing.md",
        "middleware.md",
        "security.md",
    ]
    assert precision_at_k(recuperados, "routing.md") == pytest.approx(0.2)
    assert recall_at_k(recuperados, "routing.md") == pytest.approx(1.0)
    assert mrr(recuperados, "routing.md") == pytest.approx(1.0)


def test_metricas_esperado_en_rango_3():
    recuperados = ["a.md", "b.md", "dependencies.md", "c.md", "d.md"]
    assert precision_at_k(recuperados, "dependencies.md") == pytest.approx(0.2)
    assert recall_at_k(recuperados, "dependencies.md") == pytest.approx(1.0)
    assert mrr(recuperados, "dependencies.md") == pytest.approx(1 / 3)


def test_metricas_esperado_ausente():
    recuperados = ["a.md", "b.md", "c.md", "d.md", "e.md"]
    assert precision_at_k(recuperados, "testing.md") == pytest.approx(0.0)
    assert recall_at_k(recuperados, "testing.md") == pytest.approx(0.0)
    assert mrr(recuperados, "testing.md") == pytest.approx(0.0)


def test_metricas_sin_recuperados():
    """Sin hits (query sin coincidencias): todas las métricas en 0."""
    assert precision_at_k([], "testing.md") == pytest.approx(0.0)
    assert recall_at_k([], "testing.md") == pytest.approx(0.0)
    assert mrr([], "testing.md") == pytest.approx(0.0)


def test_k_trunca_precision_y_recall_pero_no_mrr():
    """Precision@5/Recall@5 miran el top-k; MRR usa el ranking completo."""
    recuperados = ["a.md", "b.md", "c.md", "d.md", "e.md", "f.md", "testing.md"]
    assert precision_at_k(recuperados, "testing.md", k=5) == pytest.approx(0.0)
    assert recall_at_k(recuperados, "testing.md", k=5) == pytest.approx(0.0)
    assert mrr(recuperados, "testing.md") == pytest.approx(1 / 7)


def test_dedupe_documentos_quita_duplicados_preserva_orden():
    """Chunks del mismo archivo se deduplican a nivel documento (D9)."""
    hits = [
        _hit("routing.md"),
        _hit("testing.md"),
        _hit("routing.md"),
        _hit("middleware.md"),
    ]
    assert dedupe_documentos(hits) == ["routing.md", "testing.md", "middleware.md"]


def test_evaluar_casos_construye_evalresult_por_pregunta():
    class RAGStub:
        def __init__(self, hits_por_pregunta):
            self.hits_por_pregunta = hits_por_pregunta

        def retrieve(self, pregunta, k=5):
            return self.hits_por_pregunta[pregunta]

    pregunta = "¿Qué decorador uso para una ruta POST?"
    rag = RAGStub(
        {pregunta: [_hit("routing.md"), _hit("routing.md"), _hit("testing.md")]}
    )
    golden = SimpleNamespace(
        casos=[SimpleNamespace(pregunta=pregunta, documento_id_esperado="routing.md")]
    )

    resultados = evaluar_casos(rag, golden, k=5)

    assert len(resultados) == 1
    resultado = resultados[0]
    assert resultado.pregunta == pregunta
    assert resultado.recuperados == ["routing.md", "testing.md"]  # dedupe D9
    assert resultado.esperado == "routing.md"
    assert resultado.precision_at_5 == pytest.approx(0.2)
    assert resultado.recall_at_5 == pytest.approx(1.0)
    assert resultado.mrr == pytest.approx(1.0)


def test_promedios_y_criterio_de_pase_recall_08():
    """Criterio de aceptación: Recall@5 promedio >= 0.8 (>= 4 de 5 preguntas)."""
    from schemas import EvalResult

    def _resultado(pregunta, esperado, acierto: bool) -> EvalResult:
        recuperados = [esperado] if acierto else ["otro.md"]
        return EvalResult(
            pregunta=pregunta,
            recuperados=recuperados,
            esperado=esperado,
            precision_at_5=0.2 if acierto else 0.0,
            recall_at_5=1.0 if acierto else 0.0,
            mrr=1.0 if acierto else 0.0,
        )

    cuatro_aciertos = [
        _resultado(f"p{i}", f"doc{i}.md", True) for i in range(4)
    ]
    un_fallo = [_resultado("p4", "doc5.md", False)]
    assert decidir_pase(cuatro_aciertos + un_fallo) is True  # recall 0.8

    tres_aciertos = cuatro_aciertos[:3] + [_resultado("p3", "doc3.md", False)]
    assert decidir_pase(tres_aciertos + un_fallo) is False  # recall 0.6

    promedios_resultado = promedios(cuatro_aciertos + un_fallo)
    assert promedios_resultado["precision"] == pytest.approx(0.16)
    assert promedios_resultado["recall"] == pytest.approx(0.8)
    assert promedios_resultado["mrr"] == pytest.approx(0.8)


def test_cargar_golden_desde_archivo_real():
    """El CLI lee el golden_set.json real y lo valida contra el schema."""
    ruta = Path(__file__).resolve().parent.parent / "golden_set.json"
    golden = cargar_golden(ruta)
    assert len(golden.casos) == 5
    assert golden.casos[0].documento_id_esperado == "routing.md"
    assert golden.casos[0].pregunta  # sin campos vacíos
