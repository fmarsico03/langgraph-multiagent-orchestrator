"""Agente de Análisis / Cómputo.

Herramientas acotadas: `calculator`, `compute_statistics`, `sentiment_analysis`. No tiene acceso
a búsqueda: trabaja solo con los hallazgos que le pasa el supervisor (evita inventar datos).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

from agents._common import contribution, summarize_agent_run
from llm import call_with_fallback, get_llm
from state import OrchestratorState
from tools import calculator, compute_statistics, sentiment_analysis

ANALYST_PROMPT = """Sos el AGENTE DE ANÁLISIS de un sistema multi-agente.
Procesás datos YA OBTENIDOS por el Agente de Investigación. No tenés acceso a internet.

Reglas:
- Todo número que reportes debe salir de una herramienta (`calculator`, `compute_statistics`)
  aplicada a los datos recibidos. No calcules de memoria.
- Para series de porcentajes (ej. inflación mensual) usá compute_statistics con are_percentages=True
  para obtener el acumulado compuesto.
- Para sentimiento usá `sentiment_analysis` sobre los textos provistos y luego matizá con tu criterio.
- Si los datos recibidos son insuficientes o inconsistentes, NO los completes: escribí
  "DATOS_INSUFICIENTES: <qué falta>" para que el supervisor pida más investigación.

Formato de salida (obligatorio):
## Cálculos
- <métrica>: <valor> (herramienta usada)
## Sentimiento
<resultado agregado e interpretación, o "N/A">
## Interpretación
<2-4 frases conectando los números con la pregunta>
## Advertencias
<supuestos, datos faltantes o "Ninguna">
"""


@lru_cache(maxsize=8)
def build_analyst_agent(model: str):
    return create_react_agent(
        get_llm(model), tools=[calculator, compute_statistics, sentiment_analysis], prompt=ANALYST_PROMPT, name="analyst"
    )


def analyst_node(state: OrchestratorState) -> dict:
    instruction = state.get("current_instruction") or state["user_query"]
    parts = [
        f"Tarea asignada por el supervisor:\n{instruction}",
        f"Datos del Agente de Investigación:\n{state.get('research_findings') or 'N/A (no hay datos de investigación)'}",
    ]
    if state.get("validation_feedback") and state.get("analysis_results"):
        parts.append(f"Esto es un REFINAMIENTO. Feedback del supervisor:\n{state['validation_feedback']}")
        parts.append(f"Tu análisis anterior (corregilo):\n{state['analysis_results']}")

    task = [HumanMessage(content="\n\n".join(parts))]
    result = call_with_fallback(
        lambda model: build_analyst_agent(model).invoke({"messages": task}, {"recursion_limit": 16})
    )
    output, tools_used = summarize_agent_run(result["messages"])

    return {
        "analysis_results": output,
        "contributions": [contribution("analyst", state.get("step_count", 0), instruction, output, tools_used)],
        "messages": [AIMessage(content=output, name="analyst")],
    }
