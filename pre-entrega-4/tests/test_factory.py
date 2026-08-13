"""Factory multi-proveedor de modelos de chat (Fase 6, RF-6/D10).

Especifica el contrato de clients/factory.py, copia adaptada del patrón de
pre-entrega-3: DEFAULT_MODELS idénticos, _normalize_provider (provider
inválido -> ValueError), build_chat_model(provider, temperature=0.2) con lazy
imports (importar factory NO instancia modelos) y default desde LLM_PROVIDER
(gemini). ENMIENDA 2026-08-12 (U7): el provider gemini se construye con
ChatVertexAI (langchain_google_vertexai, auth vía ADC/service account, sin
api_key) en lugar de ChatGoogleGenerativeAI. Los constructores de las
librerías se inyectan como módulos falsos en sys.modules: los tests no tocan
la red ni API keys. Lección #870: los stubs reproducen la semántica de
errores real (excepciones), no solo valores de retorno — el stub de Vertex
lanza DefaultCredentialsError cuando falta ADC, igual que la librería real.
"""

from __future__ import annotations

import sys
import types

import pytest
from google.auth.exceptions import DefaultCredentialsError

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
    """ENMIENDA U7: gemini se construye con ChatVertexAI (Vertex AI, sin api_key)."""
    fake, llamadas = _fake_chat("ChatVertexAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_vertexai", ChatVertexAI=fake
    )

    modelo = build_chat_model(provider="gemini")

    assert isinstance(modelo, fake)
    assert llamadas[0]["model_name"] == DEFAULT_MODELS["gemini"] == "gemini-2.5-flash"
    assert llamadas[0]["temperature"] == 0.2
    # Vertex AI autentica con ADC/service account (GOOGLE_APPLICATION_CREDENTIALS):
    # el constructor NO recibe api_key (no hay GEMINI_API_KEY en la enmienda U7).
    assert "api_key" not in llamadas[0]


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
    fake, llamadas = _fake_chat("ChatVertexAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_vertexai", ChatVertexAI=fake
    )
    monkeypatch.setattr(factory, "LLM_PROVIDER", "gemini")

    modelo = build_chat_model()  # sin provider: default LLM_PROVIDER

    assert isinstance(modelo, fake)
    assert llamadas[0]["model_name"] == "gemini-2.5-flash"


def test_default_respeta_llm_provider_cambiado(monkeypatch):
    fake, llamadas = _fake_chat("ChatOpenAI")
    _inyectar_provider(monkeypatch, "langchain_openai", ChatOpenAI=fake)
    monkeypatch.setattr(factory, "LLM_PROVIDER", "openai")

    modelo = build_chat_model()

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == "gpt-4o-mini"


def test_importar_factory_no_instancia_modelos():
    """Lazy imports: importar factory no expone ni instancia los providers."""
    for attr in ("ChatOpenAI", "ChatVertexAI", "ChatAnthropic", "ChatOpenRouter"):
        assert not hasattr(factory, attr), (
            f"el import de factory no debe exponer {attr}: los imports de "
            "provider deben ser lazy (dentro de build_chat_model)"
        )


def test_gemini_sin_adc_propaga_error_de_autenticacion(monkeypatch):
    """Lección #870: el stub reproduce la excepción REAL de Vertex sin ADC.

    Si falta GOOGLE_APPLICATION_CREDENTIALS (o el ADC no resuelve), la librería
    lanza google.auth.exceptions.DefaultCredentialsError al construir
    ChatVertexAI. build_chat_model NO debe tragarla: la propaga para que
    responder() la convierta en answered=False (RF-6 edge controlado).
    """

    class _ChatVertexAISinADC:
        def __init__(self, **kwargs):
            raise DefaultCredentialsError(
                "Could not automatically determine credentials"
            )

    _inyectar_provider(
        monkeypatch, "langchain_google_vertexai", ChatVertexAI=_ChatVertexAISinADC
    )

    with pytest.raises(DefaultCredentialsError, match="credentials"):
        build_chat_model(provider="gemini")
