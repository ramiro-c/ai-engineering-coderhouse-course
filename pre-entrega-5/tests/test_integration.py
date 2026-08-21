"""Smoke de integración real contra Vertex (slow, skip sin credenciales)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import build_graph, create_checkpointer, invoke_config

pytestmark = pytest.mark.slow


@pytest.mark.anyio
async def test_vertex_smoke_cliente_102_con_tools():
    compiled = build_graph(create_checkpointer(":memory:"))
    result = await compiled.ainvoke(
        {"messages": [HumanMessage(content="¿Cuántos pedidos tiene el cliente 102?")]},
        config=invoke_config("vertex-smoke-thread"),
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages
    assert {m.name for m in tool_messages} >= {"buscar_cliente"}

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content
