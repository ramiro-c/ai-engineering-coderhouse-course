from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from clients.base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class OpenAIClient(BaseLLMClient):
    provider_name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIClient")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model or self.default_model

    @staticmethod
    def _to_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": message.role, "content": message.content} for message in messages]

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        model_config = config or ModelConfig(model=self.model)
        try:
            response = await self.client.chat.completions.create(
                model=model_config.model,
                messages=self._to_messages(messages),
                temperature=model_config.temperature,
                max_completion_tokens=model_config.max_tokens,
            )
            choice = response.choices[0] if response.choices else None
            return ModelResponse(
                provider=self.provider_name,
                model=model_config.model,
                text=(choice.message.content if choice and choice.message.content else ""),
                finish_reason=choice.finish_reason if choice else None,
            )
        except Exception as exc:  # pragma: no cover - defensive API boundary
            return ModelResponse(
                provider=self.provider_name,
                model=model_config.model,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        model_config = config or ModelConfig(model=self.model)
        try:
            stream = await self.client.chat.completions.create(
                model=model_config.model,
                messages=self._to_messages(messages),
                temperature=model_config.temperature,
                max_completion_tokens=model_config.max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:  # pragma: no cover - defensive API boundary
            yield f"Error: {type(exc).__name__}: {exc}"

