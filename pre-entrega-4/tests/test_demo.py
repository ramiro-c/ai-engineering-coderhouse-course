"""Tests de la demo interactiva del pipeline RAG (demo.py).

Solicitud del orquestador (m0144: "sí, agregalo"): script de demostración
interactiva de la generación de respuestas. Contratos verificados SIN red:

- importar demo.py NO ejecuta red ni instancia LLM/embeddings al import
  (los imports pesados de rag_system/embeddings/factory ya son lazy; el
  módulo demo solo importa config + stdlib a nivel de módulo).
- main() con RAGSystem stubeado (monkeypatch sobre rag_system.RAGSystem,
  semántica real: retrieve/responder devuelven lo fijado por el test):
  happy path (answered=True -> imprime respuesta y fuentes), answered=False
  (imprime degradación, exit code 0 sin crash) y salida con document_ids.
- _obtener_pregunta(): argumento CLI > prompt interactivo > pregunta default
  del corpus; _score_desde_rango(): fórmula RRF (peso/(c+rango), lección
  rrf_combine de rag_system).

El import a nivel de módulo es parte del contrato: si demo.py instanciara
algo pesado al importar, la recolección de este módulo fallaría (sin ADC,
ChatVertexAI lanzaría DefaultCredentialsError; sin credenciales, el modelo
de embeddings no se toca).
"""

from __future__ import annotations

import importlib

import pytest

import demo
import rag_system
from config import RRF_C
from schemas import LlmAnswer

PREGUNTA_EJEMPLO = "¿Cómo defino un decorador POST en FastAPI?"


class _DocumentoStub:
    """Reemplazo mínimo de langchain Document para los hits de retrieve()."""

    def __init__(self, document_id: str, seccion: str = "", texto: str = "contenido"):
        self.page_content = texto
        self.metadata = {"document_id": document_id, "seccion": seccion}


class _RAGSystemStub:
    """Stub de RAGSystem con semántica real (sin red): devuelve lo fijado.

    retrieve()/responder() tienen la misma firma que el contrato real y
    registran las llamadas; el demo los consume sin saber que son un stub.
    """

    def __init__(self, hits=None, respuesta: LlmAnswer | None = None):
        self._hits = list(hits or [])
        self._respuesta = respuesta
        self.llamadas = {"retrieve": 0, "responder": 0}

    def retrieve(self, pregunta: str, k: int = 5, namespace: str | None = None):
        self.llamadas["retrieve"] += 1
        return self._hits

    def responder(self, pregunta: str) -> LlmAnswer:
        self.llamadas["responder"] += 1
        return self._respuesta


# --- Contrato de import (lazy, sin red) ---


def test_importar_demo_no_instancia_llm_ni_embeddings(monkeypatch):
    """Importar demo.py NO ejecuta red ni instancia LLM/embeddings (lazy).

    Se recarga el módulo con spies en build_chat_model/get_embeddings: si el
    cuerpo del módulo los llamara al importar, el spy lanzaría. Además el
    módulo no debe bindear una instancia de RAGSystem (construcción lazy).
    """
    def _prohibida(*args, **kwargs):
        raise AssertionError("instanciación prohibida durante el import de demo")

    monkeypatch.setattr(rag_system, "get_embeddings", _prohibida)
    monkeypatch.setattr(rag_system, "build_chat_model", _prohibida)

    importlib.reload(demo)

    assert not hasattr(demo, "rag"), "no debe construirse RAGSystem al importar"
    assert callable(demo.main)


# --- main() con RAGSystem stubeado (sin red) ---


def _stubear_rag(monkeypatch, hits, respuesta: LlmAnswer):
    """Inyecta el RAGSystem stub en rag_system.RAGSystem (import por valor)."""
    stub = _RAGSystemStub(hits=hits, respuesta=respuesta)
    monkeypatch.setattr(rag_system, "RAGSystem", lambda **kwargs: stub)
    return stub


def test_main_happy_path_imprime_respuesta_y_fuentes(monkeypatch, capsys):
    """RF-6 happy en la demo: answered=True -> imprime respuesta y fuentes."""
    hits = [
        _DocumentoStub(
            "routing.md", seccion="Routing en FastAPI > Organizacion con APIRouter"
        ),
        _DocumentoStub("dependencies.md", seccion="Dependencias"),
    ]
    respuesta = LlmAnswer(
        pregunta=PREGUNTA_EJEMPLO,
        respuesta="Segun routing.md, las rutas POST se definen con @app.post.",
        answered=True,
        fuentes=["routing.md", "dependencies.md"],
    )
    _stubear_rag(monkeypatch, hits=hits, respuesta=respuesta)

    codigo = demo.main([PREGUNTA_EJEMPLO])

    assert codigo == 0
    salida = capsys.readouterr().out
    assert PREGUNTA_EJEMPLO in salida  # la pregunta se imprime
    assert "Segun routing.md" in salida  # la respuesta generada
    assert "Fuentes: routing.md, dependencies.md" in salida  # citas de metadata


def test_main_answered_false_imprime_degradacion_sin_crash(monkeypatch, capsys):
    """RF-6 edge en la demo: answered=False -> mensaje claro, exit code 0."""
    respuesta = LlmAnswer(
        pregunta="pregunta sin contexto",
        respuesta="No puedo responder esta pregunta con el contexto recuperado: "
        "no encontre informacion suficiente en las fuentes indexadas.",
        answered=False,
        fuentes=[],
    )
    _stubear_rag(monkeypatch, hits=[], respuesta=respuesta)

    codigo = demo.main(["pregunta sin contexto"])

    assert codigo == 0, "la degradacion es controlada: no debe crashear"
    salida = capsys.readouterr().out
    assert "answered=false" in salida
    assert "No puedo responder" in salida


def test_main_salida_muestra_document_ids_y_rank_1_destacado(monkeypatch, capsys):
    """La demo imprime el top-k con document_id/score y resalta el rank 1."""
    hits = [
        _DocumentoStub("routing.md", seccion="Routing"),
        _DocumentoStub("dependencies.md", seccion="Dependencias"),
    ]
    respuesta = LlmAnswer(
        pregunta="pregunta", respuesta="r", answered=True, fuentes=["routing.md"]
    )
    _stubear_rag(monkeypatch, hits=hits, respuesta=respuesta)

    demo.main(["pregunta"])

    salida = capsys.readouterr().out
    assert "1. routing.md" in salida  # rank 1 con su document_id
    assert "2. dependencies.md" in salida  # ranks siguientes
    assert "score" in salida  # columna de score RRF de posicion
    assert "mejor coincidencia" in salida  # el rank 1 se destaca


# --- Obtencion de la pregunta (CLI > prompt > default) ---


def test_obtener_pregunta_usa_argumento_cli():
    pregunta = demo._obtener_pregunta(["¿Cómo", "defino", "un", "decorador?"])
    assert pregunta == "¿Cómo defino un decorador?"


def test_obtener_pregunta_sin_argumento_pide_por_prompt(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args: "¿pregunta interactiva?")
    assert demo._obtener_pregunta([]) == "¿pregunta interactiva?"


def test_obtener_pregunta_input_vacio_usa_default_del_corpus(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *args: "")
    assert demo._obtener_pregunta([]) == demo.PREGUNTA_DEFAULT
    assert demo.PREGUNTA_DEFAULT == PREGUNTA_EJEMPLO


# --- Score RRF de posicion (misma formula que rrf_combine) ---


def test_score_desde_rango_aplica_formula_rrf():
    assert demo._score_desde_rango(1) == pytest.approx(1.0 / (RRF_C + 1))
    assert demo._score_desde_rango(5) == pytest.approx(1.0 / (RRF_C + 5))
    assert demo._score_desde_rango(1) > demo._score_desde_rango(5)
