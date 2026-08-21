"""Tests de las herramientas LangChain buscar_cliente y buscar_pedidos.

Catálogo in-memory en data/pedidos.py. Cliente 102: 3 pedidos (501/502/503),
total 14500, último pedido 503. IDs desconocidos devuelven dict con clave error.
"""

from __future__ import annotations

from tools import TOOLS, buscar_cliente, buscar_pedidos


def test_buscar_cliente_102_existe():
    ficha = buscar_cliente.invoke({"cliente_id": 102})

    assert ficha["id"] == 102
    assert "nombre" in ficha
    assert "error" not in ficha


def test_buscar_pedidos_102_tres_pedidos_total_14500():
    resultado = buscar_pedidos.invoke({"cliente_id": 102})

    assert "error" not in resultado
    assert resultado["cantidad"] == 3
    assert resultado["total"] == 14500
    assert len(resultado["pedidos"]) == 3
    assert [p["id"] for p in resultado["pedidos"]] == [501, 502, 503]
    assert resultado["pedidos"][-1]["id"] == 503


def test_buscar_cliente_id_desconocido_retorna_error():
    resultado = buscar_cliente.invoke({"cliente_id": 999})

    assert "error" in resultado
    assert isinstance(resultado["error"], str)
    assert resultado["error"]


def test_buscar_pedidos_id_desconocido_retorna_error():
    resultado = buscar_pedidos.invoke({"cliente_id": 999})

    assert "error" in resultado
    assert isinstance(resultado["error"], str)
    assert "buscar_cliente" in resultado["error"].lower()


def test_tools_son_langchain_tools_con_nombres_correctos():
    assert buscar_cliente.name == "buscar_cliente"
    assert buscar_pedidos.name == "buscar_pedidos"
    assert TOOLS == [buscar_cliente, buscar_pedidos]
