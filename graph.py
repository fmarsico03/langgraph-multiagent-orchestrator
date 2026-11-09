"""Construcción del grafo jerárquico: Supervisor -> {Researcher, Analyst} -> Supervisor ... -> Synthesizer."""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import analyst_node, research_node, supervisor_node, synthesizer_node
from state import OrchestratorState
from tools import credit_guard


def route_from_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "synthesizer"]:
    """Mapea la decisión del supervisor a nombres de nodos del grafo."""
    nxt = state.get("next_agent", "FINISH")
    if nxt == "researcher":
        return "researcher"
    if nxt == "analyst":
        return "analyst"
    return "synthesizer"


def build_graph():
    builder = StateGraph(OrchestratorState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "analyst": "analyst", "synthesizer": "synthesizer"},
    )
    # Los especialistas siempre reportan de vuelta al supervisor (topología jerárquica)
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("analyst", "supervisor")
    builder.add_edge("synthesizer", END)
    return builder.compile()


graph = build_graph()


def initial_state(query: str, max_steps: int = 8) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "next_agent": "researcher",
        "current_instruction": "",
        "validation_feedback": "",
        "step_count": 0,
        "max_steps": max_steps,
        "revision_counts": {},
        "research_findings": "",
        "analysis_results": "",
        "contributions": [],
        "decisions": [],
        "task_completed": False,
        "final_answer": "",
    }


def run_stream(query: str, max_steps: int = 8):
    """Ejecuta el grafo emitiendo (nodo, update) por cada paso. Resetea el límite de búsquedas por ejecución."""
    credit_guard.reset_run()
    yield from (
        (node, update)
        for chunk in graph.stream(initial_state(query, max_steps), {"recursion_limit": 40}, stream_mode="updates")
        for node, update in chunk.items()
    )
