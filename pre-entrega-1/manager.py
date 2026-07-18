from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

from clients.anthropic_client import AnthropicClient
from clients.base import BaseLLMClient
from clients.gemini_client import GeminiClient
from clients.openai_client import OpenAIClient
from config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_GENAI_USE_VERTEXAI,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)
from schemas import ChatMessage, ModelConfig, ModelResponse, ProviderName


class AsyncLLMManager:
    default_models: dict[str, str] = {
        "gemini": GeminiClient.default_model,
        "openai": OpenAIClient.default_model,
        "anthropic": AnthropicClient.default_model,
    }

    def __init__(self, provider: ProviderName | None = None) -> None:
        self.provider: ProviderName = provider or self._normalize_provider(LLM_PROVIDER)
        self.client = self._build_client(self.provider)

    @staticmethod
    def _normalize_provider(provider: str) -> ProviderName:
        value = provider.strip().lower()
        if value not in {"gemini", "openai", "anthropic"}:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
        return cast(ProviderName, value)

    def _build_client(self, provider: ProviderName) -> BaseLLMClient:
        if provider == "gemini":
            return GeminiClient(
                api_key=GEMINI_API_KEY,
                model=self.default_models[provider],
                use_vertexai=GOOGLE_GENAI_USE_VERTEXAI,
                project=GOOGLE_CLOUD_PROJECT,
                location=GOOGLE_CLOUD_LOCATION or "us-central1",
            )
        if provider == "openai":
            return OpenAIClient(api_key=OPENAI_API_KEY, model=self.default_models[provider])
        if provider == "anthropic":
            return AnthropicClient(api_key=ANTHROPIC_API_KEY, model=self.default_models[provider])
        raise ValueError(f"Unsupported provider: {provider}")

    @property
    def default_model(self) -> str:
        return self.default_models[self.provider]

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        model_config = config or ModelConfig(model=self.default_model)
        return await self.client.generate(messages=messages, config=model_config)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        model_config = config or ModelConfig(model=self.default_model)
        async for token in self.client.stream(messages=messages, config=model_config):
            yield token

