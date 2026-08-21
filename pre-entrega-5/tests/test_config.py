"""Tests unitarios de config.py: helpers de entorno y constantes del agente ReAct.

Los helpers _env_int/_env_str se prueban con monkeypatch para no depender del
.env real; las constantes se prueban contra sus valores por defecto.
"""

from __future__ import annotations

import importlib

import pytest

import config


def test_env_int_valido(monkeypatch):
    monkeypatch.setenv("TEST_INT", "42")
    assert config._env_int("TEST_INT", 7) == 42


def test_env_int_vacio_usa_default(monkeypatch):
    monkeypatch.setenv("TEST_INT", "")
    assert config._env_int("TEST_INT", 7) == 7


def test_env_int_ausente_usa_default(monkeypatch):
    monkeypatch.delenv("TEST_INT", raising=False)
    assert config._env_int("TEST_INT", 7) == 7


def test_env_int_invalido_lanza_value_error(monkeypatch):
    monkeypatch.setenv("TEST_INT", "no-es-numero")
    with pytest.raises(ValueError):
        config._env_int("TEST_INT", 7)


def test_env_str_valido(monkeypatch):
    monkeypatch.setenv("TEST_STR", "  valor  ")
    assert config._env_str("TEST_STR", "default") == "valor"


def test_env_str_ausente_usa_default(monkeypatch):
    monkeypatch.delenv("TEST_STR", raising=False)
    assert config._env_str("TEST_STR", "default") == "default"


def test_env_str_vacio_usa_default(monkeypatch):
    monkeypatch.setenv("TEST_STR", "")
    assert config._env_str("TEST_STR", "default") == "default"


def test_gcp_vars_de_vertex_expuestas():
    """Config expone las variables de Vertex AI para ADC/service account."""
    assert hasattr(config, "GOOGLE_APPLICATION_CREDENTIALS")
    assert hasattr(config, "GOOGLE_CLOUD_PROJECT")
    assert hasattr(config, "GOOGLE_CLOUD_LOCATION")


def test_llm_provider_default_gemini(monkeypatch):
    """Default gemini cuando LLM_PROVIDER no está en el entorno (sin .env)."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    importlib.reload(config)
    assert config.LLM_PROVIDER == "gemini"


def test_recursion_limit_es_10():
    assert config.RECURSION_LIMIT == 10


def test_checkpoint_path_default_es_sqlite_en_basedir():
    assert config.CHECKPOINT_PATH == config.BASE_DIR / "checkpoints.sqlite"
