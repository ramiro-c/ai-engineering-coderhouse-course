"""Factory multi-proveedor de modelos de chat (Fase 6, RF-6/D10).

Especifica el contrato de clients/factory.py, copia adaptada del patrón de
pre-entrega-3: DEFAULT_MODELS idénticos, _normalize_provider (provider
inválido -> ValueError), build_chat_model(provider, temperature=0.2) con lazy
imports (importar factory NO instancia modelos) y default desde LLM_PROVIDER
(gemini). ENMIENDA 2026-08-13: el provider gemini se construye con
ChatGoogleGenerativeAI (langchain-google-genai), patrón idéntico de
pre-entrega-3. Sigue usando Vertex AI porque GOOGLE_GENAI_USE_VERTEXAI=TRUE
está en .env: el SDK nuevo autentica con ADC/service account y NO llama a
Cloud Resource Manager (el SDK viejo ChatVertexAI generaba el 403
"Failed to convert project number to project ID"). api_key se pasa igual
que en P3 aunque esté vacía: con Vertex activo el SDK la ignora. Los
constructores de las librerías se inyectan como módulos falsos en sys.modules:
los tests no tocan la red ni API keys. Lección #870: el constructor del SDK
nuevo es LAZY (no lanza sin ADC al construir; el DefaultCredentialsError
sale al invocar y responder() lo degrada a answered=False, RF-6).
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
    """ENMIENDA 2026-08-13: gemini se construye con ChatGoogleGenerativeAI
    (patrón P3, langchain-google-genai). Sigue Vertex vía GOOGLE_GENAI_USE_VERTEXAI."""
    fake, llamadas = _fake_chat("ChatGoogleGenerativeAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_genai", ChatGoogleGenerativeAI=fake
    )

    modelo = build_chat_model(provider="gemini")

    assert isinstance(modelo, fake)
    assert llamadas[0]["model"] == DEFAULT_MODELS["gemini"] == "gemini-2.5-flash"
    assert llamadas[0]["temperature"] == 0.2
    # Patrón P3: api_key se pasa siempre (GEMINI_API_KEY de config). Con
    # GOOGLE_GENAI_USE_VERTEXAI=TRUE el SDK nuevo la ignora y usa ADC/Vertex.
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

    modelo = build_chat_model()  # sin provider: default LLM_PROVIDER

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
    """Lección #870: el SDK nuevo (langchain-google-genai) es LAZY sin ADC.

    A diferencia de ChatVertexAI (que lanzaba DefaultCredentialsError al
    construir), ChatGoogleGenerativeAI construye sin validar credenciales;
    el error sale al INVOCAR y responder() lo degrada a answered=False
    (RF-6 edge controlado, ya cubierto en test_rag_system). La factory
    construye y devuelve el modelo; si el constructor lanzara igual, la
    factory lo propaga (no lo traga).
    """
    fake, llamadas = _fake_chat("ChatGoogleGenerativeAI")
    _inyectar_provider(
        monkeypatch, "langchain_google_genai", ChatGoogleGenerativeAI=fake
    )

    modelo = build_chat_model(provider="gemini")

    assert isinstance(modelo, fake)
    assert "api_key" in llamadas[0]  # patrón P3: la factory siempre la pasa
