"""Tests del StateGraph ReAct: agent async, ToolNode, tools_condition, SqliteSaver."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import tools_condition

import graph
from config import RECURSION_LIMIT
from graph import AgentState, MessagesState, build_graph, create_checkpointer, invoke_config, open_checkpointer


@pytest.fixture
def memory_checkpointer():
    return create_checkpointer(":memory:")


def test_graph_tiene_nodos_agent_y_tools(memory_checkpointer):
    compiled = build_graph(memory_checkpointer)

    assert "agent" in compiled.nodes
    assert "tools" in compiled.nodes


def test_agent_tiene_arista_condicional_via_tools_condition(memory_checkpointer):
    compiled = build_graph(memory_checkpointer)
    branch = compiled.builder.branches["agent"]["tools_condition"]

    assert branch.path.func is tools_condition

    drawn = compiled.get_graph()
    agent_edges = [edge for edge in drawn.edges if edge.source == "agent" and edge.conditional]
    targets = {edge.target for edge in agent_edges}

    assert "tools" in targets
    assert "__end__" in targets


def test_invoke_config_recursion_limit_es_10():
    config = invoke_config("thread-test")

    assert config["recursion_limit"] == 10
    assert config["recursion_limit"] == RECURSION_LIMIT
    assert config["configurable"]["thread_id"] == "thread-test"


def test_open_checkpointer_usa_sqlite_saver():
    checkpointer = open_checkpointer()

    assert isinstance(checkpointer, graph.SqliteSaver)


class _FakeChatModel:
    """Stub con bind_tools + ainvoke para simular dos tool calls y respuesta final."""

    def __init__(self) -> None:
        self._calls = 0

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, _messages):
        self._calls += 1
        if self._calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_cliente",
                        "args": {"cliente_id": 102},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        if self._calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "buscar_pedidos",
                        "args": {"cliente_id": 102},
                        "id": "call_2",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="Cliente 102 tiene 3 pedidos con total 14500.")


@pytest.mark.anyio
async def test_ainvoke_con_fake_llm_dos_tools_y_respuesta_final(
    monkeypatch, memory_checkpointer
):
    fake = _FakeChatModel()
    monkeypatch.setattr(graph, "build_chat_model", lambda: fake)

    compiled = build_graph(memory_checkpointer)
    result = await compiled.ainvoke(
        {"messages": [HumanMessage(content="¿Cuántos pedidos tiene el cliente 102?")]},
        config=invoke_config("test-thread"),
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    assert {m.name for m in tool_messages} == {"buscar_cliente", "buscar_pedidos"}

    last = result["messages"][-1]
    assert isinstance(last, AIMessage)
    assert last.content
    assert "14500" in last.content or "3 pedidos" in last.content.lower()


def test_agent_state_es_subclase_messages_state():
    assert issubclass(AgentState, MessagesState)
    assert "messages" in AgentState.__annotations__
    assert AgentState.__doc__ is not None
    assert "add_messages" in AgentState.__doc__
