"""Persistencia de conversación con SqliteSaver (sin red).

El fake LLM emite tool_calls en el primer turno y, en el segundo turno con el
mismo thread_id, valida que el historial del primer turno llega al agente vía
checkpoint (state resilience).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import graph
from graph import build_graph, invoke_config


class _PersistenceFakeChatModel:
    """Primera vuelta ReAct con tools; segunda vuelta exige historial previo."""

    def __init__(self) -> None:
        self.saw_turn1_history_on_turn2 = False

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]

        if len(human_msgs) == 1:
            if len(tool_msgs) == 0:
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
            if len(tool_msgs) == 1:
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
            return AIMessage(content="El cliente 102 tiene 3 pedidos con total 14500.")

        if len(human_msgs) >= 2:
            assert any(
                isinstance(m, ToolMessage) and m.name == "buscar_pedidos" for m in messages
            )
            blob = " ".join(
                str(getattr(m, "content", ""))
                for m in messages
                if isinstance(m, (HumanMessage, ToolMessage, AIMessage))
            )
            assert "102" in blob or "14500" in blob or "Carlos" in blob
            self.saw_turn1_history_on_turn2 = True
            return AIMessage(content="El último pedido es el 503.")

        raise AssertionError(
            f"Estado inesperado del fake: {len(human_msgs)} human, {len(tool_msgs)} tool"
        )


@pytest.mark.anyio
async def test_segundo_turno_recupera_mensajes_del_primero(monkeypatch, memory_checkpointer):
    fake = _PersistenceFakeChatModel()
    monkeypatch.setattr(graph, "build_chat_model", lambda: fake)

    compiled = build_graph(memory_checkpointer)
    config = invoke_config("persist-test-thread")

    turn1 = await compiled.ainvoke(
        {"messages": [HumanMessage(content="¿Cuántos pedidos tiene el cliente 102?")]},
        config=config,
    )

    tool_messages = [m for m in turn1["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    assert turn1["messages"][-1].content

    turn2 = await compiled.ainvoke(
        {"messages": [HumanMessage(content="¿Y el último?")]},
        config=config,
    )

    assert fake.saw_turn1_history_on_turn2
    assert len(turn2["messages"]) > len(turn1["messages"])

    human_contents = [m.content for m in turn2["messages"] if isinstance(m, HumanMessage)]
    assert any("102" in c for c in human_contents)
    assert any("último" in c.lower() for c in human_contents)

    last = turn2["messages"][-1]
    assert isinstance(last, AIMessage)
    assert "503" in last.content
