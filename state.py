"""Esquema de estado compartido del orquestador.

Hereda de MessagesState (campo `messages` con reducer add_messages) y agrega
campos estructurados para rastrear qué agente aportó qué información.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState

AgentName = Literal["researcher", "analyst"]
RouteName = Literal["researcher", "analyst", "FINISH"]


class Contribution(TypedDict):
    """Registro de un aporte de un especialista."""

    agent: str  # "researcher" | "analyst"
    iteration: int  # step del supervisor en el que se produjo
    instruction: str  # tarea puntual que recibió el especialista
    output: str  # resultado entregado
    tool_calls: list[str]  # herramientas que usó


class SupervisorDecisionLog(TypedDict):
    """Registro de cada decisión del supervisor (trazabilidad del ruteo)."""

    step: int
    next: str
    reasoning: str
    validation_passed: bool
    forced: bool  # True si la decisión la forzó una regla de parada en código


class OrchestratorState(MessagesState):
    # Consulta original del usuario (fuente de verdad, no se pierde en el historial)
    user_query: str

    # Control de flujo
    next_agent: RouteName
    current_instruction: str  # instrucción acotada para el especialista de turno
    validation_feedback: str  # feedback del supervisor cuando pide un refinamiento
    step_count: int
    max_steps: int
    revision_counts: dict[str, int]  # cantidad de refinamientos por agente

    # Resultados de cada dominio (última versión)
    research_findings: str
    analysis_results: str

    # Trazabilidad (append-only gracias al reducer operator.add)
    contributions: Annotated[list[Contribution], operator.add]
    decisions: Annotated[list[SupervisorDecisionLog], operator.add]

    # Cierre
    task_completed: bool
    final_answer: str
