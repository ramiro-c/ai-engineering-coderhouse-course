"""Factory multi-proveedor de modelos de chat (evolución B, RF-6/D10).

Copia adaptada del patrón de pre-entrega-3/clients/factory.py: los mismos
DEFAULT_MODELS, `_normalize_provider` (provider inválido -> ValueError) y
`build_chat_model` con imports lazy por provider. El proveedor por defecto
sale de LLM_PROVIDER (config, default "gemini"); sin API key configurada el
constructor del modelo se crea igual (el error aparece recién al invocar,
y responder() lo convierte en answered=False).
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel

from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
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
