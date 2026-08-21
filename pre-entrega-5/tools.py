"""Herramientas LangChain para consultar clientes y pedidos del catálogo in-memory."""

from __future__ import annotations

from langchain.tools import tool

from data.pedidos import CLIENTES, PEDIDOS


@tool
def buscar_cliente(cliente_id: int) -> dict:
    """Busca la ficha de un cliente por su identificador numérico.

    Usá esta herramienta cuando el usuario pregunte quién es un cliente, pida
    datos de contacto (nombre, email, teléfono) o necesites confirmar que un
    cliente_id existe antes de responder sobre pedidos.

    Args:
        cliente_id: Identificador entero del cliente en el catálogo (ej. 102).

    Returns:
        Dict con la ficha del cliente (id, nombre, email, telefono, etc.) si
        existe. Si no existe, devuelve {"error": "<mensaje>"} indicando que el
        id no está en el catálogo; en ese caso no inventes datos: informá el
        error al usuario o pedí un id válido.
    """
    cliente = CLIENTES.get(cliente_id)
    if cliente is None:
        return {
            "error": f"Cliente {cliente_id} no encontrado en el catálogo in-memory."
        }
    return dict(cliente)


@tool
def buscar_pedidos(cliente_id: int) -> dict:
    """Lista los pedidos de un cliente y calcula total y cantidad.

    Usá esta herramienta cuando el usuario pregunte cuántos pedidos tuvo un
    cliente, el monto total, el último pedido o detalle de compras. Requiere un
    cliente_id válido; si no conocés el id, primero usá buscar_cliente o pedí
    aclaración al usuario.

    Args:
        cliente_id: Identificador entero del cliente cuyos pedidos se consultan.

    Returns:
        Si el cliente existe: {"pedidos": [...], "total": int, "cantidad": int}
        donde pedidos es la lista ordenada por id (el último tiene el id más
        alto) y total es la suma de los campos total de cada pedido. Si el
        cliente no existe: {"error": "<mensaje>"} sugiriendo llamar a
        buscar_cliente o usar un id válido del catálogo (ciclo de retorno).
    """
    if cliente_id not in CLIENTES:
        return {
            "error": (
                f"Cliente {cliente_id} no encontrado. "
                "Usá buscar_cliente con un id válido o confirmá el id con el usuario."
            )
        }

    pedidos = sorted(PEDIDOS.get(cliente_id, []), key=lambda p: p["id"])
    total = sum(p["total"] for p in pedidos)
    return {
        "pedidos": pedidos,
        "total": total,
        "cantidad": len(pedidos),
    }


TOOLS = [buscar_cliente, buscar_pedidos]
