"""Agente de Búsqueda / Investigación.

Herramientas acotadas: solo `web_search` (Tavily con guardián de créditos) y `mock_search`.
Aislamiento de contexto: recibe únicamente la instrucción del supervisor (+ feedback si es
un refinamiento), no el historial completo de la conversación.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agents._common import contribution, summarize_agent_run
from llm import call_with_fallback, get_llm, today
from state import OrchestratorState
from tools import mock_search, web_search

RESEARCH_PROMPT = """Sos el AGENTE DE INVESTIGACIÓN de un sistema multi-agente.
Tu único trabajo es conseguir información factual de fuentes externas para la tarea asignada.

Reglas:
- Usá `web_search` con consultas precisas. Tenés un presupuesto de créditos: hacé como MÁXIMO 3 búsquedas
  y no repitas consultas equivalentes. Usá `mock_search` solo si web_search falla.
- NO hagas cálculos ni interpretaciones: eso lo hace el Agente de Análisis.
- Si encontrás datos numéricos, transcribilos EXACTOS con su período/unidad.
- Si la información es insuficiente o contradictoria entre fuentes, decilo explícitamente.

Formato de salida (obligatorio):
## Hallazgos
- <dato concreto> (fuente: <URL>)
## Datos numéricos estructurados
<lista "etiqueta: valor unidad" en orden cronológico, o "N/A">
## Textos para análisis de sentimiento
<titulares/extractos textuales relevantes, uno por línea, o "N/A">
## Limitaciones
<qué faltó o qué es dudoso>
"""


@lru_cache(maxsize=8)
def build_research_agent(model: str):
    return create_react_agent(get_llm(model), tools=[web_search, mock_search], prompt=RESEARCH_PROMPT, name="researcher")


def research_node(state: OrchestratorState) -> dict:
    instruction = state.get("current_instruction") or state["user_query"]
    parts = [
        f"Fecha de hoy: {today()}. Priorizá los datos MÁS RECIENTES publicados a esta fecha.",
        f"Tarea asignada por el supervisor:\n{instruction}",
    ]
    if state.get("validation_feedback") and state.get("research_findings"):
        parts.append(f"Esto es un REFINAMIENTO. Feedback del supervisor:\n{state['validation_feedback']}")
        parts.append(f"Tu entrega anterior (mejorala, no la repitas):\n{state['research_findings']}")

    task = [HumanMessage(content="\n\n".join(parts))]
    result = call_with_fallback(
        lambda model: build_research_agent(model).invoke({"messages": task}, {"recursion_limit": 12})
    )
    output, tools_used = summarize_agent_run(result["messages"])

    return {
        "research_findings": output,
        "contributions": [contribution("researcher", state.get("step_count", 0), instruction, output, tools_used)],
        "messages": [AIMessage(content=output, name="researcher")],
    }
