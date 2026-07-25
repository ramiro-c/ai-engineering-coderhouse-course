from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["gemini", "openai", "anthropic", "openrouter"]


class NivelCriticidad(str, Enum):
    baja = "baja"
    media = "media"
    alta = "alta"


class TechnicalExtraction(BaseModel):
    tecnologias: list[str] = Field(min_length=1, description="Tecnologías mencionadas o inferidas del texto")
    nivel_de_criticidad: NivelCriticidad
    resumen_tecnico: str = Field(min_length=1, description="Resumen técnico conciso del texto de entrada")


class ExtractionError(BaseModel):
    error: str = Field(description="Tipo de error, p. ej. la clase de la excepción original")
    detalle: str = Field(description="Explicación legible de qué falló en el pipeline")
