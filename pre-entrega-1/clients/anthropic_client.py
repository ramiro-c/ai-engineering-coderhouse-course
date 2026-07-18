from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from clients.base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class AnthropicClient(BaseLLMClient):
    provider_name = "anthropic"
    default_model = "claude-3-5-haiku-latest"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for AnthropicClient")
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model or self.default_model

    @staticmethod
    def _split_messages(
        messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict[str, str]]]:
        system_messages: list[str] = []
        payload: list[dict[str, str]] = []

        for message in messages:
            if message.role == "system":
                system_messages.append(message.content)
                continue
            payload.append({"role": message.role, "content": message.content})

        system_prompt = "\n".join(system_messages).strip() or None
        return system_prompt, payload

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        model_config = config or ModelConfig(model=self.model)
        try:
            system_prompt, payload = self._split_messages(messages)
            request: dict[str, object] = {
                "model": model_config.model,
                "max_tokens": model_config.max_tokens,
                "messages": payload,
                "temperature": model_config.temperature,
            }
            if system_prompt:
                request["system"] = system_prompt

            response = await self.client.messages.create(**request)
            text = getattr(response, "text", "") or ""
            return ModelResponse(
                provider=self.provider_name,
                model=model_config.model,
                text=text,
                finish_reason=getattr(response, "stop_reason", None),
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
            system_prompt, payload = self._split_messages(messages)
            request: dict[str, object] = {
                "model": model_config.model,
                "max_tokens": model_config.max_tokens,
                "messages": payload,
                "temperature": model_config.temperature,
            }
            if system_prompt:
                request["system"] = system_prompt

            async with self.client.messages.stream(**request) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:  # pragma: no cover - defensive API boundary
            yield f"Error: {type(exc).__name__}: {exc}"

