from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ChatRole = Literal["system", "user", "assistant"]
ProviderName = Literal["gemini", "openai", "anthropic"]


class ChatMessage(BaseModel):
    role: ChatRole
    content: str = Field(min_length=1)


class ModelConfig(BaseModel):
    model: str = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1)

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model cannot be empty")
        return value


class ModelResponse(BaseModel):
    provider: str
    model: str
    text: str = ""
    finish_reason: str | None = None
    error: str | None = None

