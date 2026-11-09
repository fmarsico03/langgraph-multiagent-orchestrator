"""Utilidades compartidas por los nodos especialistas."""

from __future__ import annotations

import warnings
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from llm import message_text

# create_react_agent está marcado como deprecado en LangGraph 1.x a favor de
# langchain.agents.create_agent, pero sigue siendo funcional y es lo que pide la consigna.
warnings.filterwarnings("ignore", message=".*create_react_agent.*")


def summarize_agent_run(messages: list[BaseMessage]) -> tuple[str, list[str]]:
    """Devuelve (texto final del agente, lista de herramientas usadas)."""
    tools_used = [m.name for m in messages if isinstance(m, ToolMessage) and m.name]
    final = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not m.tool_calls:
            final = message_text(m).strip()
            if final:
                break
    return final or "(el agente no produjo una respuesta final)", tools_used


def contribution(agent: str, step: int, instruction: str, output: str, tools: list[str]) -> dict[str, Any]:
    return {"agent": agent, "iteration": step, "instruction": instruction, "output": output, "tool_calls": tools}
