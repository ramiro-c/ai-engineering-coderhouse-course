from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from clients.base import BaseLLMClient
from schemas import ChatMessage, ModelConfig, ModelResponse


class GeminiClient(BaseLLMClient):
    provider_name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        use_vertexai: bool = False,
        project: str | None = None,
        location: str = "us-central1",
    ) -> None:
        if use_vertexai:
            if not project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT is required when GOOGLE_GENAI_USE_VERTEXAI=true"
                )
            # Auth via Application Default Credentials (gcloud ADC / service account).
            self.client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for GeminiClient")
            self.client = genai.Client(api_key=api_key)

        self.model = model or self.default_model

    @staticmethod
    def _split_messages(messages: list[ChatMessage]) -> tuple[str | None, str]:
        system_messages: list[str] = []
        transcript_lines: list[str] = []

        for message in messages:
            if message.role == "system":
                system_messages.append(message.content)
                continue
            transcript_lines.append(f"{message.role}: {message.content}")

        system_instruction = "\n".join(system_messages).strip() or None
        transcript = "\n".join(transcript_lines).strip()
        return system_instruction, transcript

    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        model_config = config or ModelConfig(model=self.model)
        try:
            system_instruction, transcript = self._split_messages(messages)
            response = await self.client.aio.models.generate_content(
                model=model_config.model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=model_config.temperature,
                    max_output_tokens=model_config.max_tokens,
                ),
            )
            return ModelResponse(
                provider=self.provider_name,
                model=model_config.model,
                text=response.text or "",
                finish_reason=getattr(response, "finish_reason", None),
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
            system_instruction, transcript = self._split_messages(messages)
            stream = await self.client.aio.models.generate_content_stream(
                model=model_config.model,
                contents=transcript,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=model_config.temperature,
                    max_output_tokens=model_config.max_tokens,
                ),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # pragma: no cover - defensive API boundary
            yield f"Error: {type(exc).__name__}: {exc}"
