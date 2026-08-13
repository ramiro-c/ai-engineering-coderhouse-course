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
el SDK de Vertex lanzaría DefaultCredentialsError al invocar; sin
credenciales, el modelo de embeddings no se toca).
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


# --- Obtencion de la pregunta (CLI > default; el REPL maneja su propio input) ---


def test_obtener_pregunta_usa_argumento_cli():
    pregunta = demo._obtener_pregunta(["¿Cómo", "defino", "un", "decorador?"])
    assert pregunta == "¿Cómo defino un decorador?"


def test_obtener_pregunta_sin_argumento_usa_default_del_corpus():
    """Sin argumento CLI, la demo en modo una-pregunta usa el ejemplo del corpus.

    El input interactivo vive en el REPL (_modo_interactivo), no aqui.
    """
    assert demo._obtener_pregunta([]) == demo.PREGUNTA_DEFAULT
    assert demo.PREGUNTA_DEFAULT == PREGUNTA_EJEMPLO


# --- Comandos de salida del modo interactivo (uso natural por humano) ---


@pytest.mark.parametrize(
    "texto",
    ["salir", "exit", "quit", "q", "s", "SALIR", " Salir ", "EXIT", "q "],
)
def test_es_comando_salida_reconoce_variantes(texto):
    """El humano puede salir del REPL con varias palabras, mayus/minus y espacios."""
    assert demo._es_comando_salida(texto)


@pytest.mark.parametrize(
    "texto", ["", "¿Cómo defino un POST?", "hola", "salir ahora", "salida"]
)
def test_es_comando_salida_rechaza_no_comandos(texto):
    """Frases que no son comandos de salida (incluso 'salida' y prefijos)."""
    assert not demo._es_comando_salida(texto)


# --- Modo interactivo (REPL): varias preguntas hasta salir ---


def test_modo_interactivo_hace_loop_hasta_salir(monkeypatch, capsys):
    """El REPL pregunta -> retrieve/responder -> vuelve a preguntar hasta 'salir'."""
    respuesta = LlmAnswer(
        pregunta="¿pregunta uno?", respuesta="r1", answered=True, fuentes=["routing.md"]
    )
    stub = _stubear_rag(monkeypatch, hits=[_DocumentoStub("routing.md")], respuesta=respuesta)
    entradas = iter(["¿pregunta uno?", "salir"])
    monkeypatch.setattr("builtins.input", lambda *a: next(entradas))

    codigo = demo._modo_interactivo(stub)

    assert codigo == 0
    assert stub.llamadas["retrieve"] == 1
    assert stub.llamadas["responder"] == 1
    salida = capsys.readouterr().out
    # BUG REAL DEL USUARIO (m0225): la pregunta se repetía. input() ya muestra
    # el prompt "Pregunta: " y el terminal refleja lo tipeado; el REPL NO debe
    # volver a imprimir la pregunta (eso duplicaba "como funcionan..." 2 veces).
    assert "Pregunta: ¿pregunta uno?" not in salida
    assert "r1" in salida
    assert "Hasta luego" in salida  # despedida al salir


def test_modo_interactivo_acepta_varias_preguntas(monkeypatch, capsys):
    """Dos preguntas antes de salir -> retrieve/responder corre dos veces."""
    respuesta = LlmAnswer(pregunta="p", respuesta="r", answered=True, fuentes=[])
    stub = _stubear_rag(monkeypatch, hits=[], respuesta=respuesta)
    entradas = iter(["¿primera?", "¿segunda?", "exit"])
    monkeypatch.setattr("builtins.input", lambda *a: next(entradas))

    demo._modo_interactivo(stub)

    assert stub.llamadas["retrieve"] == 2
    assert stub.llamadas["responder"] == 2


def test_modo_interactivo_input_vacio_sigue_sin_llamar(monkeypatch, capsys):
    """Enter vacio no llama al RAG y vuelve a preguntar (no crashea)."""
    respuesta = LlmAnswer(pregunta="p", respuesta="r", answered=True, fuentes=[])
    stub = _stubear_rag(monkeypatch, hits=[], respuesta=respuesta)
    entradas = iter(["", "salir"])
    monkeypatch.setattr("builtins.input", lambda *a: next(entradas))

    codigo = demo._modo_interactivo(stub)

    assert codigo == 0
    assert stub.llamadas["retrieve"] == 0
    assert "Pregunta vacía" in capsys.readouterr().out


def test_modo_interactivo_eof_termina_sin_crash(monkeypatch, capsys):
    """Ctrl+D (EOF) termina la demo limpiamente, sin crash."""
    stub = _RAGSystemStub(hits=[], respuesta=None)

    def _eof(*_a):
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)

    assert demo._modo_interactivo(stub) == 0
    assert "Hasta luego" in capsys.readouterr().out


def test_modo_interactivo_ctrl_c_termina_sin_crash(monkeypatch, capsys):
    """Ctrl+C (KeyboardInterrupt) termina la demo limpiamente, sin crash."""
    stub = _RAGSystemStub(hits=[], respuesta=None)

    def _ctrl_c(*_a):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _ctrl_c)

    assert demo._modo_interactivo(stub) == 0
    assert "Hasta luego" in capsys.readouterr().out


def test_main_sin_argumentos_entra_modo_interactivo(monkeypatch):
    """main([]) sin argumentos delega en el REPL (uso natural, no una sola pregunta)."""
    stub = _RAGSystemStub(hits=[], respuesta=None)
    monkeypatch.setattr(rag_system, "RAGSystem", lambda **kwargs: stub)
    llamado = {"n": 0}

    def _fake_interactivo(rag):
        llamado["n"] += 1
        assert rag is stub
        return 0

    monkeypatch.setattr(demo, "_modo_interactivo", _fake_interactivo)

    assert demo.main([]) == 0
    assert llamado["n"] == 1


def test_main_con_argumento_usa_una_pregunta_no_repl(monkeypatch, capsys):
    """Con argumento CLI la demo responde una pregunta y termina (sin REPL)."""
    respuesta = LlmAnswer(pregunta="p", respuesta="r", answered=True, fuentes=[])
    _stubear_rag(monkeypatch, hits=[], respuesta=respuesta)
    llamado = {"n": 0}

    def _no_debe_llamarse(*_a):
        llamado["n"] += 1
        return 0

    monkeypatch.setattr(demo, "_modo_interactivo", _no_debe_llamarse)

    assert demo.main(["¿pregunta puntual?"]) == 0
    assert llamado["n"] == 0
    assert "r" in capsys.readouterr().out


def test_correr_una_pregunta_sin_mostrar_pregunta_no_imprime_la_pregunta(
    monkeypatch, capsys
):
    """Fix del bug m0225: en REPL la pregunta NO se imprime (el prompt y el
    echo del terminal ya la muestran). Con mostrar_pregunta=False, el output
    NO contiene el renglón 'Pregunta: <texto>' (evita la duplicación que vio
    el usuario: 'como funcionan...' aparecía dos veces).
    """
    respuesta = LlmAnswer(
        pregunta="¿como funcionan?", respuesta="r", answered=True, fuentes=[]
    )
    stub = _RAGSystemStub(hits=[], respuesta=respuesta)

    demo._correr_una_pregunta(stub, "¿como funcionan?", mostrar_pregunta=False)

    salida = capsys.readouterr().out
    assert "Pregunta: ¿como funcionan?" not in salida
    assert "r" in salida  # la consulta se procesó igual


def test_correr_una_pregunta_con_mostrar_pregunta_imprime_la_pregunta(
    monkeypatch, capsys
):
    """En modo CLI (una pregunta con argumento) la pregunta SÍ se imprime:
    es la única ocasión en que el humano la ve (no hay prompt de input).
    """
    respuesta = LlmAnswer(
        pregunta="¿como funcionan?", respuesta="r", answered=True, fuentes=[]
    )
    stub = _RAGSystemStub(hits=[], respuesta=respuesta)

    demo._correr_una_pregunta(stub, "¿como funcionan?", mostrar_pregunta=True)

    salida = capsys.readouterr().out
    assert "Pregunta: ¿como funcionan?" in salida
    assert "r" in salida


# --- Score RRF de posicion (misma formula que rrf_combine) ---


def test_score_desde_rango_aplica_formula_rrf():
    assert demo._score_desde_rango(1) == pytest.approx(1.0 / (RRF_C + 1))
    assert demo._score_desde_rango(5) == pytest.approx(1.0 / (RRF_C + 5))
    assert demo._score_desde_rango(1) > demo._score_desde_rango(5)
