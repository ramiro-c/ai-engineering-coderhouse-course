"""Fixtures compartidos para tests de pre-entrega-5."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_CLOUD_LOCATION,
    GOOGLE_CLOUD_PROJECT,
    LLM_PROVIDER,
)

_vertex_credenciales_ok: bool | None = None


def _vertex_listo() -> bool:
    """¿Hay ADC/Vertex configurado para tests slow? (cacheado por sesión)."""
    global _vertex_credenciales_ok
    if _vertex_credenciales_ok is None:
        if LLM_PROVIDER != "gemini":
            _vertex_credenciales_ok = False
        elif (
            not GOOGLE_APPLICATION_CREDENTIALS
            or not GOOGLE_CLOUD_PROJECT
            or not GOOGLE_CLOUD_LOCATION
        ):
            _vertex_credenciales_ok = False
        elif not Path(GOOGLE_APPLICATION_CREDENTIALS).is_file():
            _vertex_credenciales_ok = False
        else:
            _vertex_credenciales_ok = True
    return _vertex_credenciales_ok


@pytest.fixture(autouse=True)
def _skip_slow_sin_vertex(request: pytest.FixtureRequest) -> None:
    """Saltea tests `slow` si no hay credenciales Vertex (no falla CI sin keys)."""
    if request.node.get_closest_marker("slow") and not _vertex_listo():
        pytest.skip(
            "Test marcado `slow`: requiere credenciales Vertex (ADC). "
            "Corré tests unit con -m 'not slow'."
        )
