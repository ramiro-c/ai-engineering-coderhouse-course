"""Factory multi-proveedor de modelos de chat (pre-entrega-5, agente ReAct).

Especifica el contrato de clients/factory.py: DEFAULT_MODELS idénticos a P4,
_normalize_provider (provider inválido -> ValueError), build_chat_model(provider,
temperature=0.2) con lazy imports (importar factory NO instancia modelos) y
default desde LLM_PROVIDER (gemini). El provider gemini se construye con
ChatGoogleGenerativeAI (langchain-google-genai), patrón idéntico de P4: Vertex
vía GOOGLE_GENAI_USE_VERTEXAI=TRUE en .env, ADC/service account, api_key
siempre pasada aunque esté vacía. Los constructores se inyectan como módulos
falsos en sys.modules: los tests no tocan la red ni API keys. El constructor
del SDK nuevo es LAZY (no lanza sin ADC al construir; el error sale al invocar).
"""

from __future__ import annotations

import sys
import types

import pytest

import clients.factory as factory
from clients.factory import DEFAULT_MODELS, _normalize_provider, build_chat_model


def _inyectar_provider(monkeypatch, nombre_modulo: str, **clases) -> None:
    """Reemplaza un módulo de provider por uno falso en sys.modules.

    El factory importa las clases con `from langchain_x import Clase` DENTRO
    de build_chat_model (lazy): Python resuelve ese import contra sys.modules,
    así que inyectar un módulo falso evita las librerías reales (y sus API
    keys).
    """
    modulo = types.ModuleType(nombre_modulo)
    for attr, clase in clases.items():
        setattr(modulo, attr, clase)
    monkeypatch.setitem(sys.modules, nombre_modulo, modulo)


def _fake_chat(nombre: str):
    """Constructor falso que registra los kwargs con que lo llamaron."""
    llamadas: list[dict] = []

    class _ChatFake:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            llamadas.append(kwargs)

    _ChatFake.__name__ = nombre
    return _ChatFake, llamadas


def test_build_chat_model_gemini_usa_modelo_default(monkeypatch):
    """Gemini se construye con ChatGoogleGenerativeAI (langchain-google-genai)."""
    fake, llamadas = _fake_chat("ChatGoogleGenerativeAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_genai", ChatGoogleGenerativeAI=fake
    )

    modelo = build_chat_model(provider="gemini")

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == DEFAULT_MODELS["gemini"] == "gemini-2.5-flash"
    assert llamadas[0]["temperature"] == 0.2
    assert llamadas[0]["api_key"] == factory.GEMINI_API_KEY


def test_build_chat_model_openai_usa_modelo_default(monkeypatch):
    fake, llamadas = _fake_chat("ChatOpenAI")
    _inyectar_provider(monkeypatch, "langchain_openai", ChatOpenAI=fake)

    modelo = build_chat_model(provider="openai")

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == DEFAULT_MODELS["openai"] == "gpt-4o-mini"
    assert llamadas[0]["temperature"] == 0.2


def test_build_chat_model_anthropic_usa_modelo_default(monkeypatch):
    fake, llamadas = _fake_chat("ChatAnthropic")
    _inyectar_provider(monkeypatch, "langchain_anthropic", ChatAnthropic=fake)

    modelo = build_chat_model(provider="anthropic")

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == DEFAULT_MODELS["anthropic"] == "claude-3-5-haiku-latest"


def test_build_chat_model_openrouter_usa_modelo_default(monkeypatch):
    fake, llamadas = _fake_chat("ChatOpenRouter")
    _inyectar_provider(monkeypatch, "langchain_openrouter", ChatOpenRouter=fake)

    modelo = build_chat_model(provider="openrouter")

    assert isinstance(modelo, fake)
    assert (
        llamadas[0]["model"] == DEFAULT_MODELS["openrouter"] == "cohere/north-mini-code:free"
    )


def test_provider_invalido_lanza_value_error():
    with pytest.raises(ValueError):
        build_chat_model(provider="groq")


def test_normalize_provider_normaliza_espacios_y_mayusculas():
    assert _normalize_provider("  OpenAI ") == "openai"
    assert _normalize_provider("GEMINI") == "gemini"


def test_default_usa_llm_provider_gemini(monkeypatch):
    fake, llamadas = _fake_chat("ChatGoogleGenerativeAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_genai", ChatGoogleGenerativeAI=fake
    )
    monkeypatch.setattr(factory, "LLM_PROVIDER", "gemini")

    modelo = build_chat_model()

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == "gemini-2.5-flash"


def test_default_respeta_llm_provider_cambiado(monkeypatch):
    fake, llamadas = _fake_chat("ChatOpenAI")
    _inyectar_provider(monkeypatch, "langchain_openai", ChatOpenAI=fake)
    monkeypatch.setattr(factory, "LLM_PROVIDER", "openai")

    modelo = build_chat_model()

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == "gpt-4o-mini"


def test_importar_factory_no_instancia_modelos():
    """Lazy imports: importar factory no expone ni instancia los providers."""
    for attr in ("ChatOpenAI", "ChatGoogleGenerativeAI", "ChatAnthropic", "ChatOpenRouter"):
        assert not hasattr(factory, attr), (
            f"el import de factory no debe exponer {attr}: los imports de "
            "provider deben ser lazy (dentro de build_chat_model)"
        )


def test_gemini_sin_adc_construye_lazy_y_error_sale_al_invocar(monkeypatch):
    """El SDK nuevo (langchain-google-genai) construye sin validar credenciales."""
    fake, llamadas = _fake_chat("ChatGoogleGenerativeAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_genai", ChatGoogleGenerativeAI=fake
    )

    modelo = build_chat_model(provider="gemini")

    assert isinstance(modelo, fake)
    assert "api_key" in llamadas[0]
