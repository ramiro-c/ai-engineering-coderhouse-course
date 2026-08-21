"""Valida el formato del trace JSON generado por demo.py --trace."""

from __future__ import annotations

import json
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent.parent / "traces" / "react-trace.json"


def test_trace_tiene_tipos_de_mensaje():
    data = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    tipos = {entry["tipo"] for entry in data}

    assert "human" in tipos
    assert "ai" in tipos
    assert "tool" in tipos


def test_trace_tiene_tool_call_buscar_cliente_o_pedidos():
    data = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    tool_names: set[str] = set()

    for entry in data:
        for tool_call in entry.get("tool_calls") or []:
            tool_names.add(tool_call["name"])
        if entry.get("tipo") == "tool" and entry.get("name"):
            tool_names.add(entry["name"])

    assert tool_names & {"buscar_cliente", "buscar_pedidos"}
