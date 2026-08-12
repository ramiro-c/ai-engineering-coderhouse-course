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
from pinecone.exceptions import NotFoundException


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
    """Stub del cliente: describe_index (NotFoundException si no existe) y create_index."""

    def __init__(self, indice_existente: _IndiceFalso | None = None):
        self.indices: dict[str, _IndiceFalso] = {}
        self.create_calls: list[dict] = []
        self.describes = 0
        if indice_existente is not None:
            self.indices[INDEX_NAME] = indice_existente

    def describe_index(self, nombre: str):
        self.describes += 1
        indice = self.indices.get(nombre)
        if indice is None:
            # Semántica real de pinecone 7.3.0 (SDK v6.x): un índice inexistente
            # lanza pinecone.exceptions.NotFoundException (404), NO devuelve None.
            raise NotFoundException(
                status=404, reason=f"Resource {nombre} not found"
            )
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


class _ExceptionsFalso:
    """Estructura pinecone.exceptions del paquete real (verificado en 7.3.0):
    expone NotFoundException para que init_index la capte en el except."""

    NotFoundException = NotFoundException


class _ModuloPinecone:
    """Módulo pinecone stub: Pinecone() devuelve SIEMPRE la instancia del test."""

    def __init__(self, cliente: _PineconeStub):
        self._cliente = cliente
        self.ServerlessSpec = _ServerlessSpecFalso
        self.exceptions = _ExceptionsFalso()

    def Pinecone(self, api_key=None):
        return self._cliente


def test_crea_indice_y_espera_ready(monkeypatch):
    """Índice inexistente: lo crea con la spec correcta y espera a READY (RF-1 happy)."""
    stub = _PineconeStub()
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone(stub))
    monkeypatch.setattr(init_index, "PINECONE_API_KEY", "clave-dummy-de-test")

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
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone(stub))
    monkeypatch.setattr(init_index, "PINECONE_API_KEY", "clave-dummy-de-test")

    resumen = init_index.init_index(poll_intervalo=0)

    assert stub.create_calls == []
    assert resumen["creado"] is False
    assert resumen["estado"] == "ready"


def test_indice_existente_con_dim_distinta_advierte(monkeypatch, capsys):
    """Índice existente con dimensión distinta: advertencia clara, sin recrear."""
    stub = _PineconeStub(
        indice_existente=_IndiceFalso(describes_hasta_ready=1, dimension=384)
    )
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone(stub))
    monkeypatch.setattr(init_index, "PINECONE_API_KEY", "clave-dummy-de-test")

    resumen = init_index.init_index(poll_intervalo=0)

    assert stub.create_calls == []
    assert "ATENCIÓN" in capsys.readouterr().err
    assert resumen["creado"] is False


def test_poll_tolera_not_found_hasta_timeout(monkeypatch):
    """Defensivo: si describe_index sigue lanzando NotFound tras crear (el índice
    tarda en propagarse), el poll continúa esperando y el timeout devuelve
    RuntimeError — no la excepción cruda de la SDK."""

    class _StubSiempreNotFound(_PineconeStub):
        def describe_index(self, nombre: str):
            self.describes += 1
            # El índice nunca llega a ser visible: describe siempre lanza 404.
            raise NotFoundException(
                status=404, reason=f"Resource {nombre} not found"
            )

    stub = _StubSiempreNotFound()
    monkeypatch.setattr(init_index, "pinecone", _ModuloPinecone(stub))
    monkeypatch.setattr(init_index, "PINECONE_API_KEY", "clave-dummy-de-test")

    with pytest.raises(RuntimeError, match="no quedó READY"):
        init_index.init_index(timeout_segundos=0.01, poll_intervalo=0)


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
