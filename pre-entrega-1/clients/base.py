from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from schemas import ChatMessage, ModelConfig, ModelResponse


class BaseLLMClient(ABC):
    provider_name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> ModelResponse:
        """Return a full response from the model."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[str]:
        """Yield text fragments from the model stream."""
        pass
