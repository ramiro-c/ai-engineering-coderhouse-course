"""Tests unit de init_index.py (Fase 4, sin red): verificación/creación del índice.

Cubren los edges de RF-1 con un stub del cliente de Pinecone: creación con la
spec Serverless correcta (cloud/región de config, 1536d, cosine) y espera a
READY vía poll, la idempotencia cuando el índice ya existe (no se recrea),
la advertencia si la dimensión/métrica del índice existente difieren, y el
fallo claro sin PINECONE_API_KEY antes de cualquier llamada de red (el cliente
Pinecone ni siquiera se instancia).
"""

from __future__ import annotations

import pytest

import init_index
from config import DIMENSION, INDEX_NAME, METRIC, PINECONE_CLOUD, PINECONE_REGION


class _Status:
    def __init__(self, ready: bool):
        self.ready = ready
        self.state = "Ready" if ready else "Initializing"


class _IndiceFalso:
    """Descripción de índice: empieza no-ready y pasa a ready tras N describes."""

    def __init__(self, describes_hasta_ready: int = 1, dimension: int = DIMENSION):
        self.describes_hasta_ready = describes_hasta_ready
        self._describes = 0
        self.dimension = dimension
        self.metric = METRIC
        self.status = _Status(ready=False)

    def avanzar(self) -> None:
        self._describes += 1
        if self._describes >= self.describes_hasta_ready:
            self.status = _Status(ready=True)


class _PineconeStub:
    """Stub del cliente: describe_index (None si no existe) y create_index."""

    def __init__(self, indice_existente: _IndiceFalso | None = None):
        self.indices: dict[str, _IndiceFalso] = {}
        self.create_calls: list[dict] = []
        self.describes = 0
        if indice_existente is not None:
            self.indices[INDEX_NAME] = indice_existente

    def describe_index(self, nombre: str):
        self.describes += 1
        indice = self.indices.get(nombre)
        if indice is not None:
            indice.avanzar()
        return indice

    def create_index(self, name, dimension, metric, spec):
        self.create_calls.append(
            {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
        )
        # El índice creado tarda 2 describes en quedar ready: fuerza un ciclo
        # de poll para validar la espera sin dormir en el test.
        self.indices[name] = _IndiceFalso(describes_hasta_ready=2)


class _ServerlessSpecFalso:
    def __init__(self, cloud, region):
        self.cloud = cloud
        self.region = region


class _ModuloPinecone:
    Pinecone = _PineconeStub
    ServerlessSpec = _ServerlessSpecFalso


def test_crea_indice_y_espera_ready(monkeypatch):
    """Índice inexistente: lo crea con la spec correcta y espera a READY (RF-1 happy)."""
    stub = _PineconeStub()
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone)

    resumen = init_index.init_index(poll_intervalo=0)

    assert len(stub.create_calls) == 1
    llamada = stub.create_calls[0]
    assert llamada["name"] == INDEX_NAME
    assert llamada["dimension"] == DIMENSION == 1536
    assert llamada["metric"] == METRIC == "cosine"
    assert llamada["spec"].cloud == PINECONE_CLOUD == "aws"
    assert llamada["spec"].region == PINECONE_REGION == "us-east-1"
    assert stub.describes >= 3  # existencia + al menos un ciclo de poll
    assert resumen["creado"] is True
    assert resumen["estado"] == "ready"


def test_indice_existente_es_idempotente(monkeypatch):
    """Índice ya existente: verifica y continúa sin recrear (RF-1 edge)."""
    stub = _PineconeStub(indice_existente=_IndiceFalso(describes_hasta_ready=1))
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone)

    resumen = init_index.init_index(poll_intervalo=0)

    assert stub.create_calls == []
    assert resumen["creado"] is False
    assert resumen["estado"] == "ready"


def test_indice_existente_con_dim_distinta_advierte(monkeypatch, capsys):
    """Índice existente con dimensión distinta: advertencia clara, sin recrear."""
    stub = _PineconeStub(
        indice_existente=_IndiceFalso(describes_hasta_ready=1, dimension=384)
    )
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone)

    resumen = init_index.init_index(poll_intervalo=0)

    assert stub.create_calls == []
    assert "ATENCIÓN" in capsys.readouterr().err
    assert resumen["creado"] is False


def test_sin_api_key_sale_sin_red(monkeypatch, capsys):
    """Sin PINECONE_API_KEY: SystemExit con mensaje claro y sin tocar Pinecone."""

    class _PineconeProhibido:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Pinecone no debe instanciarse sin API key (sin red)")

    class _ModuloPineconeProhibido:
        Pinecone = _PineconeProhibido
        ServerlessSpec = _ServerlessSpecFalso

    monkeypatch.setattr(init_index, "pinecone", _ModuloPineconeProhibido)
    monkeypatch.setattr(init_index, "PINECONE_API_KEY", None)

    with pytest.raises(SystemExit) as exc:
        init_index.main()

    assert exc.value.code == 1
    assert "PINECONE_API_KEY" in capsys.readouterr().err
