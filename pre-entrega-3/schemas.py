from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["gemini", "openai", "anthropic", "openrouter"]


class RagReference(BaseModel):
    source: str = Field(description="Nombre del archivo .md de origen")
    section: str = Field(
        default="",
        description="Ruta de encabezados de la sección, p. ej. 'Sesgos > Antifrágil'",
    )
    snippet: str = Field(
        description="Fragmento del apunte usado como contexto (~200 caracteres)"
    )


# Lo único que se le pide al LLM: las referencias las arma rag.py desde los
# fragmentos recuperados, así no puede inventarlas.
# Comentario y no docstring a propósito: Pydantic copia el docstring al JSON
# schema y PydanticOutputParser lo inyecta en el prompt de cada consulta. Las
# Field(description=...) sí van dirigidas al modelo.
class LlmAnswer(BaseModel):
    text: str = Field(
        description="Respuesta en español, fundamentada solo en el contexto recuperado"
    )
    answered: bool = Field(
        default=True,
        description=(
            "true si el contexto alcanzó para responder; false si tuviste que "
            "decir que no sabés porque el contexto no trata el tema"
        ),
    )


class RagResponse(BaseModel):
    """Respuesta que devuelve el pipeline: texto del LLM + citas verificables."""

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
