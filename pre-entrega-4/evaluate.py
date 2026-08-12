"""Evaluación del RAG híbrido (Fase 5, RF-4).

Métricas puras a nivel documento (D9): precision_at_k/recall_at_k/mrr sobre
ids deduplicados, con el criterio de aceptación Recall@5 >= 0.8 (>= 4 de 5
preguntas del golden set). El CLI lee golden_set.json, consulta RAGSystem por
pregunta, imprime la tabla por pregunta + promedios + MRR y sale 0 (PASS) o
1 (FAIL).
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import TOP_K
from schemas import EvalResult, GoldenSet

# Ruta por defecto del golden set (junto al módulo).
RUTA_GOLDEN = Path(__file__).resolve().parent / "golden_set.json"

# Umbral de aceptación: Recall@5 promedio (RF-4).
UMBRAL_RECALL = 0.8


def dedupe_documentos(hits) -> list[str]:
    """Document ids únicos en orden de aparición (dedupe a nivel documento, D9).

    Varios chunks del mismo archivo cuentan como UN documento en la
    evaluación: el recall es 0 o 1 por pregunta porque se mide si el
    documento esperado aparece entre los recuperados.
    """
    recuperados: list[str] = []
    for hit in hits:
        doc_id = hit.metadata["document_id"]
        if doc_id not in recuperados:
            recuperados.append(doc_id)
    return recuperados


def precision_at_k(recuperados: list[str], esperado: str, k: int = 5) -> float:
    """Precision@k con un documento relevante: 1/k si el esperado está en el top-k."""
    if esperado in recuperados[:k]:
        return 1.0 / k
    return 0.0


def recall_at_k(recuperados: list[str], esperado: str, k: int = 5) -> float:
    """Recall@k a nivel documento: 1.0 si el esperado está en el top-k, si no 0."""
    return 1.0 if esperado in recuperados[:k] else 0.0


def mrr(recuperados: list[str], esperado: str) -> float:
    """Reciprocal rank del documento esperado sobre el ranking completo."""
    try:
        return 1.0 / (recuperados.index(esperado) + 1)
    except ValueError:
        return 0.0


def cargar_golden(ruta: Path | str = RUTA_GOLDEN) -> GoldenSet:
    """Carga y valida el golden set contra el schema (5 pares pregunta->documento)."""
    return GoldenSet.model_validate_json(Path(ruta).read_text(encoding="utf-8"))


def evaluar_casos(rag, golden: GoldenSet, k: int = TOP_K) -> list[EvalResult]:
    """Evalúa cada pregunta del golden set contra el recuperador (RF-4).

    Por caso: retrieve(k) -> ids deduplicados -> métricas puras. El
    recuperador se inyecta para poder testear sin red (stub con retrieve).
    """
    resultados: list[EvalResult] = []
    for caso in golden.casos:
        hits = rag.retrieve(caso.pregunta, k=k)
        recuperados = dedupe_documentos(hits)
        resultados.append(
            EvalResult(
                pregunta=caso.pregunta,
                recuperados=recuperados,
                esperado=caso.documento_id_esperado,
                precision_at_5=precision_at_k(recuperados, caso.documento_id_esperado, k=k),
                recall_at_5=recall_at_k(recuperados, caso.documento_id_esperado, k=k),
                mrr=mrr(recuperados, caso.documento_id_esperado),
            )
        )
    return resultados


def promedios(resultados: list[EvalResult]) -> dict[str, float]:
    """Promedios de P@5, R@5 y MRR sobre todos los casos."""
    total = len(resultados) or 1
    return {
        "precision": sum(r.precision_at_5 for r in resultados) / total,
        "recall": sum(r.recall_at_5 for r in resultados) / total,
        "mrr": sum(r.mrr for r in resultados) / total,
    }


def decidir_pase(resultados: list[EvalResult], umbral: float = UMBRAL_RECALL) -> bool:
    """Criterio de aceptación: Recall@5 promedio >= 0.8 (>= 4 de 5 preguntas)."""
    return promedios(resultados)["recall"] >= umbral


def formatear_tabla(resultados: list[EvalResult]) -> str:
    """Tabla por pregunta: esperado, recuperados (dedup), P@5, R@5 y MRR."""
    lineas = [
        f"{'Pregunta':<58} | {'Esperado':<18} | {'Recuperados':<38} | "
        f"{'P@5':>4} | {'R@5':>4} | {'MRR':>4}"
    ]
    lineas.append("-" * len(lineas[0]))
    for r in resultados:
        pregunta = r.pregunta if len(r.pregunta) <= 57 else r.pregunta[:54] + "..."
        recuperados = ", ".join(r.recuperados) if r.recuperados else "(ninguno)"
        lineas.append(
            f"{pregunta:<58} | {r.esperado:<18} | {recuperados:<38} | "
            f"{r.precision_at_5:>4.2f} | {r.recall_at_5:>4.2f} | {r.mrr:>4.2f}"
        )
    return "\n".join(lineas)


def main() -> int:
    """CLI de evaluación: python evaluate.py — golden set -> tabla + PASS/FAIL."""
    from rag_system import RAGSystem

    golden = cargar_golden()
    resultados = evaluar_casos(RAGSystem(), golden)
    print(formatear_tabla(resultados))

    p = promedios(resultados)
    print(
        f"\nPromedios: Precision@5={p['precision']:.2f} "
        f"Recall@5={p['recall']:.2f} MRR={p['mrr']:.2f}"
    )
    if decidir_pase(resultados):
        print(f"RESULTADO: PASS (Recall@5 >= {UMBRAL_RECALL})")
        return 0
    print(f"RESULTADO: FAIL (Recall@5 < {UMBRAL_RECALL})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
