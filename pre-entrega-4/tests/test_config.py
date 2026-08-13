"""Tests unitarios de config.py: helpers de entorno y constantes del RAG.

Los helpers _env_int/_env_float/_env_str se prueban con monkeypatch para no
depender del .env real; las constantes se prueban contra sus valores por
defecto (patrón pre-entrega-3).
"""

from __future__ import annotations

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


def test_env_float_valido(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "0.85")
    assert config._env_float("TEST_FLOAT", 0.5) == 0.85


def test_env_float_ausente_usa_default(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    assert config._env_float("TEST_FLOAT", 0.5) == 0.5


def test_env_str_valido(monkeypatch):
    monkeypatch.setenv("TEST_STR", "  valor  ")
    assert config._env_str("TEST_STR", "default") == "valor"


def test_env_str_ausente_usa_default(monkeypatch):
    monkeypatch.delenv("TEST_STR", raising=False)
    assert config._env_str("TEST_STR", "default") == "default"


def test_constantes_rng_por_defecto():
    assert config.CHUNK_SIZE == 700
    assert config.CHUNK_OVERLAP == 100
    assert config.TOP_K == 5
    assert config.RRF_C == 60
    assert config.DIMENSION == 1536
    assert config.NAMESPACE_DEFAULT == "docs"


def test_fuente_namespaces_mapea_carpetas():
    assert config.FUENTE_NAMESPACES["features"] == "fastapi-core"
    assert config.FUENTE_NAMESPACES["tutorial"] == "fastapi-tutorial"
