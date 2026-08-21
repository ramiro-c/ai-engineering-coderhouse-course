"""StateGraph ReAct con agent async, ToolNode, tools_condition y SqliteSaver."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.sqlite import SqliteSaver as _BaseSqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from clients.factory import build_chat_model
from config import CHECKPOINT_PATH, RECURSION_LIMIT
from tools import TOOLS

SYSTEM_PROMPT = (
    "Sos un asistente de pedidos. Para preguntas sobre pedidos de un cliente, "
    "primero verificá al cliente con la herramienta buscar_cliente y después "
    "consultá sus pedidos con buscar_pedidos. No inventes datos: usá solo lo "
    "que devuelven las herramientas."
)


class SqliteSaver(_BaseSqliteSaver):
    """SqliteSaver sync con delegación async para graph.ainvoke (LangGraph 1.x)."""

    async def aget_tuple(self, config: RunnableConfig):
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)


class AgentState(MessagesState):
    """Historial de mensajes; el reducer add_messages acumula, no reemplaza."""


def _messages_with_system(messages: list) -> list:
    if messages and isinstance(messages[0], SystemMessage):
        return messages
    return [SystemMessage(content=SYSTEM_PROMPT), *messages]


def build_graph(checkpointer: SqliteSaver) -> CompiledStateGraph:
    llm = build_chat_model()
    tools = TOOLS

    async def agent(state: AgentState) -> dict:
        messages = _messages_with_system(state["messages"])
        response = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)


def create_checkpointer(path: str | sqlite3.Connection = ":memory:") -> SqliteSaver:
    if isinstance(path, sqlite3.Connection):
        conn = path
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def open_checkpointer() -> SqliteSaver:
    return create_checkpointer(str(CHECKPOINT_PATH))


def invoke_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
