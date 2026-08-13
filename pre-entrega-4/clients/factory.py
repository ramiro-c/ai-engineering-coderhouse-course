"""Factory multi-proveedor de modelos de chat (evolución B, RF-6/D10).

Copia adaptada del patrón de pre-entrega-3/clients/factory.py: los mismos
DEFAULT_MODELS, `_normalize_provider` (provider inválido -> ValueError) y
`build_chat_model` con imports lazy por provider. El proveedor por defecto
sale de LLM_PROVIDER (config, default "gemini").

ENMIENDA 2026-08-13: el provider gemini se construye con ChatGoogleGenerativeAI
(langchain-google-genai), patrón idéntico de pre-entrega-3. Sigue usando
Vertex AI porque GOOGLE_GENAI_USE_VERTEXAI=TRUE está en .env: el SDK nuevo
autentica con ADC/service account (GOOGLE_APPLICATION_CREDENTIALS,
GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION) y NO llama a Cloud Resource
Manager — el SDK viejo ChatVertexAI (langchain-google-vertexai) generaba el
403 "Failed to convert project number to project ID" y quedó deprecado en
LangChain 3.2.0. La api_key se pasa igual que en P3 aunque esté vacía: con
Vertex activo el SDK la ignora. Sin ADC configurado, el constructor del SDK
nuevo es LAZY (no lanza al construir); el error de autenticación sale al
invocar y la factory lo propaga para que responder() lo degrade a
answered=False (RF-6 edge controlado). OpenAI/Anthropic/OpenRouter siguen
disponibles vía LLM_PROVIDER.
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel

from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)
from schemas import ProviderName

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "openrouter": "cohere/north-mini-code:free",
}


def _normalize_provider(provider: str) -> ProviderName:
    value = provider.strip().lower()
    if value not in {"gemini", "openai", "anthropic", "openrouter"}:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
    return cast(ProviderName, value)


def build_chat_model(
    provider: str | None = None, temperature: float = 0.2
) -> BaseChatModel:
    resolved = _normalize_provider(provider or LLM_PROVIDER)

    if resolved == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODELS["openai"],
            temperature=temperature,
        )

    if resolved == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model=DEFAULT_MODELS["anthropic"],
            temperature=temperature,
        )

    if resolved == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Patrón P3 exacto: se pasa GEMINI_API_KEY (config) aunque esté vacía;
        # con GOOGLE_GENAI_USE_VERTEXAI=TRUE en .env el SDK nuevo usa
        # ADC/service account e ignora la key. El constructor es lazy: sin
        # ADC no lanza acá, el error sale al invocar y responder() lo
        # convierte en answered=False.
        return ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=DEFAULT_MODELS["gemini"],
            temperature=temperature,
        )

    if resolved == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model=DEFAULT_MODELS["openrouter"], temperature=temperature
        )

    raise ValueError(f"Unsupported provider: {resolved}")
