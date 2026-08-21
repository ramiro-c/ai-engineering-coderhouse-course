"""E2E del ciclo de retorno: error en buscar_pedidos y recuperación con id válido."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
from graph import build_graph, invoke_config


class _CicloRetornoFakeChatModel:
    """Primero buscar_pedidos(999) falla; luego buscar_cliente(102) + buscar_pedidos(102)."""

    def __init__(self) -> None:
        self.saw_pedidos_error = False
        self.saw_recovery = False

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

        if len(tool_msgs) == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_pedidos",
                        "args": {"cliente_id": 999},
                        "id": "call_bad",
                        "type": "tool_call",
                    }
                ],
            )

        if len(tool_msgs) == 1 and tool_msgs[0].name == "buscar_pedidos":
            payload = tool_msgs[0].content
            if isinstance(payload, str):
                payload = json.loads(payload)
            assert "error" in payload
            self.saw_pedidos_error = True
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_cliente",
                        "args": {"cliente_id": 102},
                        "id": "call_cliente",
                        "type": "tool_call",
                    }
                ],
            )

        if len(tool_msgs) == 2 and tool_msgs[-1].name == "buscar_cliente":
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_pedidos",
                        "args": {"cliente_id": 102},
                        "id": "call_pedidos",
                        "type": "tool_call",
                    }
                ],
            )

        if len(tool_msgs) == 3 and tool_msgs[-1].name == "buscar_pedidos":
            payload = tool_msgs[-1].content
            if isinstance(payload, str):
                payload = json.loads(payload)
            assert "error" not in payload
            assert payload.get("cantidad") == 3
            self.saw_recovery = True
            return AIMessage(content="El cliente 102 tiene 3 pedidos con total 14500.")

        raise AssertionError(
            f"Estado inesperado del fake: {len(tool_msgs)} tool messages"
        )


@pytest.mark.anyio
async def test_ciclo_retorno_tras_error_en_buscar_pedidos(monkeypatch, memory_checkpointer):
    fake = _CicloRetornoFakeChatModel()
    monkeypatch.setattr(graph, "build_chat_model", lambda: fake)

    compiled = build_graph(memory_checkpointer)
    result = await compiled.ainvoke(
        {"messages": [HumanMessage(content="¿Cuántos pedidos tiene el cliente 999?")]},
        config=invoke_config("ciclo-retorno-thread"),
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 3
    assert tool_messages[0].name == "buscar_pedidos"
    assert "error" in str(tool_messages[0].content).lower()
    assert {m.name for m in tool_messages[1:]} == {"buscar_cliente", "buscar_pedidos"}

    assert fake.saw_pedidos_error
    assert fake.saw_recovery

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "14500" in last.content
