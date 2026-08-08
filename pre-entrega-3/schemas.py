from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["gemini", "openai", "anthropic", "openrouter"]


class RagReference(BaseModel):
    source: str = Field(description="Nombre del archivo .md de origen")
    snippet: str = Field(
        description="Fragmento del apunte usado como contexto (~200 caracteres)"
    )


class RagResponse(BaseModel):
    text: str = Field(
        description="Respuesta en español, fundamentada solo en el contexto recuperado"
    )
    references: list[RagReference] = Field(
        default_factory=list,
        description="Fragmentos del corpus que pasaron el gate de relevancia",
    )


class RagGenerationError(BaseModel):
    error: str = Field(
        description="Tipo de error, p. ej. la clase de la excepción original"
    )
    detalle: str = Field(
        description="Explicación legible de qué falló en el pipeline RAG"
    )
