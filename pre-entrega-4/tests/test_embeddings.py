"""Tests de embeddings locales HuggingFace (Fase 7, ENMIENDA 2026-08-12, RF-2/D11).

get_embeddings() DEBE devolver UNA instancia cacheada de HuggingFaceEmbeddings
(sentence-transformers/all-MiniLM-L6-v2, 384d) — mismo modelo local de
pre-entrega-3 — SIN exigir ninguna API key (OPENAI_API_KEY deja de aplicar).
El constructor se testea con un fake inyectado en el módulo embeddings (la
clase se importa a nivel módulo, así que monkeypatch.setattr alcanza): nunca
se instancia el modelo real (torch pesado + descarga). La lección #793 se
documenta con un test: HF_HUB_OFFLINE se exporta en el ENTORNO DEL PROCESO,
no en .env — el módulo no debe romper cuando la variable está seteada.
"""

from __future__ import annotations

import pytest

import embeddings


@pytest.fixture(autouse=True)
def _cache_embeddings_limpio():
    """El lru_cache de get_embeddings persiste entre tests: se limpia por test."""
    embeddings.get_embeddings.cache_clear()
    yield
    embeddings.get_embeddings.cache_clear()


def _fake_hf(nombre: str = "HuggingFaceEmbeddings"):
    """Constructor falso que registra los kwargs con que lo llamaron."""
    llamadas: list[dict] = []

    class _Fake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            llamadas.append(kwargs)

    _Fake.__name__ = nombre
    return _Fake, llamadas


def test_get_embeddings_devuelve_huggingface_local_384d(monkeypatch):
    """RF-2 (enmienda): el cliente es HuggingFaceEmbeddings con all-MiniLM-L6-v2."""
    fake, llamadas = _fake_hf()
    monkeypatch.setattr(
        embeddings, "HuggingFaceEmbeddings", fake
    )
    monkeypatch.setattr(
        embeddings, "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    cliente = embeddings.get_embeddings()

    assert isinstance(cliente, fake)
    assert llamadas[0]["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_get_embeddings_es_cacheadas_misma_instancia(monkeypatch):
    """Indexar y consultar usan la MISMA instancia (patrón pre-entrega-3)."""
    fake, llamadas = _fake_hf()
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", fake)

    primera = embeddings.get_embeddings()
    segunda = embeddings.get_embeddings()

    assert primera is segunda
    assert len(llamadas) == 1, "el lru_cache debe construir una sola instancia"


def test_get_embeddings_sin_api_key(monkeypatch):
    """ENMIENDA: los embeddings locales NO exigen OPENAI_API_KEY ni ninguna key."""
    fake, llamadas = _fake_hf()
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", fake)

    cliente = embeddings.get_embeddings()

    assert isinstance(cliente, fake)
    assert "api_key" not in llamadas[0], (
        "el constructor de HuggingFaceEmbeddings no recibe api_key: la "
        "credencial de OpenAI ya no aplica a los embeddings"
    )
    # El módulo ni siquiera importa la clave de OpenAI: no hay de qué depender.
    assert not hasattr(embeddings, "OPENAI_API_KEY")


def test_importar_embeddings_no_instancia_el_modelo(monkeypatch):
    """Lazy: importar embeddings NO instancia el modelo (torch pesado)."""
    fake, llamadas = _fake_hf()
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", fake)

    assert llamadas == [], "el import del módulo no debe construir el modelo"

    embeddings.get_embeddings()

    assert len(llamadas) == 1, "la instancia se crea recién al llamar get_embeddings()"


def test_hf_hub_offline_en_el_proceso_no_rompe(monkeypatch):
    """Lección #793: HF_HUB_OFFLINE se exporta en el ENTORNO DEL PROCESO.

    huggingface_hub lo lee en import time, así que no se configura desde .env.
    El módulo debe funcionar igual cuando la variable está seteada antes.
    """
    fake, _ = _fake_hf()
    monkeypatch.setattr(embeddings, "HuggingFaceEmbeddings", fake)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    cliente = embeddings.get_embeddings()

    assert isinstance(cliente, fake), "HF_HUB_OFFLINE no debe romper get_embeddings()"
