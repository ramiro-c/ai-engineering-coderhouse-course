"""Tests de los datos de la pre-entrega 4: corpus Markdown y golden set.

El corpus en data/features/ y data/tutorial/ es la fuente de verdad de la
recuperación: cada archivo .md es un documento cuyo document_id es el nombre
del archivo (p. ej. "routing.md"). El golden set debe apuntar exactamente a
esos ids reales y validar contra los schemas GoldenSetItem/GoldenSet.
"""

from __future__ import annotations

import json
from pathlib import Path

import schemas
from config import DATA_DIR

SUBCARPETAS = ("features", "tutorial")
GOLDEN_SET_PATH = DATA_DIR.parent / "golden_set.json"


def _documentos_corpus() -> list[Path]:
    return sorted(
        doc
        for subcarpeta in SUBCARPETAS
        for doc in (DATA_DIR / subcarpeta).glob("*.md")
    )


def _cargar_golden_set() -> schemas.GoldenSet:
    datos = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    return schemas.GoldenSet.model_validate(datos)


def test_corpus_completo_con_ambas_subcarpetas():
    docs = _documentos_corpus()
    assert len(docs) >= 8, "El corpus debe tener al menos 8 documentos"
    for subcarpeta in SUBCARPETAS:
        assert list((DATA_DIR / subcarpeta).glob("*.md")), (
            f"Falta la subcarpeta {subcarpeta} con documentos"
        )


def test_cada_documento_es_legible_no_vacio_con_heading():
    for doc in _documentos_corpus():
        texto = doc.read_text(encoding="utf-8")
        assert texto.strip(), f"{doc.name} está vacío"
        assert any(
            linea.strip().startswith("#") for linea in texto.splitlines()
        ), f"{doc.name} no tiene encabezados markdown"


def test_golden_set_valida_contra_schema_y_no_vacio():
    golden = _cargar_golden_set()
    assert len(golden.casos) == 5, "El golden set debe tener exactamente 5 pares"
    for caso in golden.casos:
        assert caso.pregunta.strip(), "La pregunta no puede estar vacía"
        assert caso.documento_id_esperado.strip(), (
            "El documento esperado no puede estar vacío"
        )


def test_documentos_esperados_existen_en_el_corpus():
    golden = _cargar_golden_set()
    ids_reales = {doc.name for doc in _documentos_corpus()}
    for caso in golden.casos:
        assert caso.documento_id_esperado in ids_reales, (
            f"{caso.documento_id_esperado} no existe en el corpus de data/"
        )
