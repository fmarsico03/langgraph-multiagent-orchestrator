"""Supervisor: router inteligente + validador con rúbrica, y nodo de síntesis final.

Condiciones de parada (en código, no dependen del LLM):
  1. step_count >= max_steps           -> FINISH forzado.
  2. revision_counts[agente] > MAX_REVISIONS -> no se vuelve a delegar a ese agente (FINISH forzado).
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from llm import call_with_fallback, get_llm, message_text, today
from state import OrchestratorState

MAX_REVISIONS = 2  # refinamientos permitidos por agente (además de la primera entrega)

SUPERVISOR_PROMPT = """Sos el SUPERVISOR de un equipo multi-agente. Dada la consulta del usuario y lo que
aportó cada especialista hasta ahora, decidí quién debe intervenir ahora o si es momento de finalizar.

Especialistas disponibles:
- "researcher": busca información en fuentes externas (web). NO calcula.
- "analyst": hace cálculos, estadísticas y análisis de sentimiento SOBRE los datos ya investigados. NO busca.
- "FINISH": la información reunida es suficiente; se pasa a la síntesis final.

Flujo típico: researcher -> analyst -> FINISH. El analyst necesita datos del researcher primero.

RÚBRICA DE VALIDACIÓN (aplicala antes de elegir FINISH; validation_passed=True solo si R1-R5 se cumplen):
  R1. Hay datos factuales de fuentes externas con su fuente (URL) citada.
  R2. El análisis usa esos datos y no inventa números (los cálculos provienen de herramientas).
  R3. Los cálculos son coherentes con los datos (ej. cantidad de períodos correcta).
  R4. Todas las partes de la consulta del usuario están cubiertas.
  R5. Recencia: si la consulta pide datos "recientes"/"últimos", los períodos reportados deben ser los más
      recientes disponibles respecto de la FECHA DE HOY (no datos de hace un año si existen más nuevos).
Si un criterio falla, delegá al especialista responsable con una `instruction` que incluya el feedback
concreto de qué corregir. Si el analyst reportó DATOS_INSUFICIENTES, delegá al researcher.
COHERENCIA: si el researcher entregó hallazgos NUEVOS después del último análisis, el análisis quedó
desactualizado: delegá al analyst para que lo rehaga sobre los datos nuevos antes de elegir FINISH.
Criterio de SUFICIENCIA: no busques perfección. Si los datos disponibles responden razonablemente la
consulta y las limitaciones están declaradas, elegí FINISH.

La `instruction` debe ser autocontenida: el especialista NO ve la conversación, solo tu instrucción."""


class SupervisorDecision(BaseModel):
    next: Literal["researcher", "analyst", "FINISH"] = Field(description="Siguiente nodo a ejecutar")
    instruction: str = Field(description="Instrucción autocontenida para el especialista (vacía si FINISH)")
    reasoning: str = Field(description="Justificación breve de la decisión")
    validation_passed: bool = Field(description="True si la rúbrica R1-R5 se cumple completa")
    feedback: str = Field(default="", description="Qué debe corregir el especialista si es un refinamiento")


def _build_context(state: OrchestratorState) -> str:
    contribs = state.get("contributions", [])
    history = "\n".join(
        f"- step {c['iteration']}: {c['agent']} (tools: {', '.join(c['tool_calls']) or 'ninguna'})" for c in contribs
    ) or "- (ningún especialista intervino todavía)"
    return f"""FECHA DE HOY: {today()}

CONSULTA DEL USUARIO:
{state['user_query']}

HISTORIAL DE DELEGACIONES:
{history}

ÚLTIMOS HALLAZGOS DEL RESEARCHER:
{state.get('research_findings') or 'N/A'}

ÚLTIMO ANÁLISIS DEL ANALYST:
{state.get('analysis_results') or 'N/A'}

Paso actual: {state.get('step_count', 0) + 1} de {state.get('max_steps', 8)}.
Refinamientos usados: {state.get('revision_counts', {})} (máximo {MAX_REVISIONS} por agente)."""


def _analysis_is_stale(state: OrchestratorState) -> bool:
    """True si hay un análisis previo pero el researcher aportó algo después de él."""
    agents = [c["agent"] for c in state.get("contributions", [])]
    if "analyst" not in agents:
        return False
    last_analyst = len(agents) - 1 - agents[::-1].index("analyst")
    return "researcher" in agents[last_analyst + 1 :]


def supervisor_node(state: OrchestratorState) -> dict:
    step = state.get("step_count", 0) + 1
    max_steps = state.get("max_steps", 8)
    revisions = dict(state.get("revision_counts", {}))

    def finish(reason: str, forced: bool, passed: bool = False) -> dict:
        return {
            "next_agent": "FINISH",
            "step_count": step,
            "current_instruction": "",
            "decisions": [
                {"step": step, "next": "FINISH", "reasoning": reason, "validation_passed": passed, "forced": forced}
            ],
        }

    # Regla de parada 1: contador de pasos
    if step >= max_steps:
        return finish(f"Límite de {max_steps} pasos alcanzado: se cierra con la información disponible.", True)

    prompt = [SystemMessage(content=SUPERVISOR_PROMPT), HumanMessage(content=_build_context(state))]
    decision: SupervisorDecision = call_with_fallback(
        lambda model: get_llm(model).with_structured_output(SupervisorDecision).invoke(prompt)
    )

    if decision.next == "FINISH":
        # Guardia de coherencia (en código): si el researcher refinó después del último análisis,
        # el análisis está desactualizado -> se rehace antes de cerrar (respetando MAX_REVISIONS).
        if _analysis_is_stale(state) and revisions.get("analyst", 0) < MAX_REVISIONS:
            revisions["analyst"] = revisions.get("analyst", 0) + 1
            instruction = (
                "El investigador actualizó sus hallazgos después de tu último análisis. Rehacé el análisis "
                f"completo sobre los datos nuevos para responder la consulta: {state['user_query']}"
            )
            return {
                "next_agent": "analyst",
                "current_instruction": instruction,
                "validation_feedback": "Tu análisis anterior se basó en datos que ya fueron reemplazados.",
                "step_count": step,
                "revision_counts": revisions,
                "decisions": [
                    {
                        "step": step,
                        "next": "analyst",
                        "reasoning": "Análisis desactualizado respecto de los últimos hallazgos: se rehace antes "
                        f"de FINISH. (El supervisor proponía cerrar: {decision.reasoning})",
                        "validation_passed": False,
                        "forced": True,
                    }
                ],
                "messages": [AIMessage(content=f"[supervisor -> analyst] {instruction}", name="supervisor")],
            }
        return finish(decision.reasoning, False, decision.validation_passed)

    # ¿Es un refinamiento? (el agente ya había entregado algo)
    agent = decision.next
    already_delivered = bool(state.get("research_findings" if agent == "researcher" else "analysis_results"))
    if already_delivered:
        revisions[agent] = revisions.get(agent, 0) + 1
        # Regla de parada 2: tope de refinamientos por agente
        if revisions[agent] > MAX_REVISIONS:
            return finish(
                f"{agent} ya fue refinado {MAX_REVISIONS} veces; se evita el bucle y se cierra. "
                f"(El supervisor pedía: {decision.reasoning})",
                True,
            )

    return {
        "next_agent": agent,
        "current_instruction": decision.instruction,
        "validation_feedback": decision.feedback if already_delivered else "",
        "step_count": step,
        "revision_counts": revisions,
        "decisions": [
            {
                "step": step,
                "next": agent,
                "reasoning": decision.reasoning,
                "validation_passed": decision.validation_passed,
                "forced": False,
            }
        ],
        "messages": [AIMessage(content=f"[supervisor -> {agent}] {decision.instruction}", name="supervisor")],
    }


SYNTH_PROMPT = """Sos el SINTETIZADOR final. Redactá la respuesta al usuario en español, clara y estructurada,
usando EXCLUSIVAMENTE la información de los especialistas. No agregues datos nuevos.
Estructura:
1. Respuesta directa (2-3 frases).
2. Datos clave (con fuentes) — aporte del Agente de Investigación.
3. Análisis (cálculos y sentimiento) — aporte del Agente de Análisis.
4. Limitaciones / advertencias (incluí si el cierre fue forzado por límite de pasos)."""


def synthesizer_node(state: OrchestratorState) -> dict:
    last = state.get("decisions", [])[-1] if state.get("decisions") else {}
    context = f"""CONSULTA: {state['user_query']}

INVESTIGACIÓN:
{state.get('research_findings') or 'N/A'}

ANÁLISIS:
{state.get('analysis_results') or 'N/A'}

CIERRE: {'forzado por regla de parada' if last.get('forced') else 'decidido por el supervisor'} —
validación de rúbrica: {'aprobada' if last.get('validation_passed') else 'no aprobada / parcial'}.
Motivo: {last.get('reasoning', '')}"""

    prompt = [SystemMessage(content=SYNTH_PROMPT), HumanMessage(content=context)]
    response = call_with_fallback(lambda model: get_llm(model).invoke(prompt))
    answer = message_text(response).strip()
    return {
        "final_answer": answer,
        "task_completed": True,
        "messages": [AIMessage(content=answer, name="synthesizer")],
    }
