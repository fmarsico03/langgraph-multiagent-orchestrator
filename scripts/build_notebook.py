"""Genera demo.ipynb (luego se ejecuta con: jupyter nbconvert --to notebook --execute --inplace demo.ipynb)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

nb.cells = [
    md(
        "# Demo: Orquestador Multi-Agente (LangGraph + Gemini)\n\n"
        "Este notebook demuestra el **flujo de delegación jerárquico**:\n\n"
        "`supervisor → researcher → supervisor → analyst → supervisor (rúbrica) → synthesizer → END`\n\n"
        "- **Caso 1:** consulta que requiere investigación externa (Tavily) + análisis (estadísticas y sentimiento).\n"
        "- **Caso 2:** demostración de la **condición de parada** (`max_steps`) que evita el *supervisor infinito*.\n\n"
        "> Cada búsqueda Tavily consume 1 crédito; el guardián limita a 4 búsquedas por ejecución."
    ),
    code(
        "from IPython.display import Image, Markdown, display\n"
        "from graph import graph\n"
        "from main import run\n"
        "from tools import credit_guard\n"
        "from llm import candidate_models\n\n"
        "print('Modelos Gemini (principal + fallbacks):', candidate_models())\n"
        "print(f'Créditos Tavily usados antes de la demo: {credit_guard.credits_used}/{credit_guard.budget}')"
    ),
    md("## Topología del grafo"),
    code("display(Image(graph.get_graph().draw_mermaid_png()))"),
    md(
        "## Caso 1 — Investigación + Análisis + Síntesis\n"
        "El supervisor debe delegar primero al **Investigador** (datos con fuentes), luego al **Analista** "
        "(cálculos con herramientas) y validar con la rúbrica R1–R5 antes de cerrar."
    ),
    code(
        "query = (\"Buscá la inflación mensual de Argentina (INDEC) de los últimos 6 meses publicados, calculá el \"\n"
        "         \"promedio, la inflación acumulada del período y la tendencia, y analizá el sentimiento de los \"\n"
        "         \"titulares de noticias recientes sobre la inflación.\")\n"
        "trace = run(query)"
    ),
    md("### Trazabilidad del estado compartido\nQué decidió el supervisor en cada paso y qué aportó cada agente:"),
    code(
        "print('DECISIONES DEL SUPERVISOR')\n"
        "for d in trace['decisions']:\n"
        "    print(f\"  step {d['step']}: -> {d['next']:<10} rúbrica_ok={d['validation_passed']!s:<5} forzado={d['forced']}\")\n\n"
        "print('\\nCONTRIBUCIONES (state[\"contributions\"])')\n"
        "for c in trace['contributions']:\n"
        "    print(f\"  [{c['agent']}] step {c['iteration']} | tools={c['tool_calls']} | {len(c['output'])} chars\")"
    ),
    code("display(Markdown(trace['final_answer']))"),
    md(
        "## Caso 2 — Condición de parada (anti *supervisor infinito*)\n"
        "Con `max_steps=2`, el supervisor delega al investigador en el paso 1 y en el paso 2 la regla en código "
        "**fuerza `FINISH`** (sin consultar al LLM), aunque falte el análisis. La síntesis declara la limitación."
    ),
    code(
        "trace2 = run(\"Buscá el precio actual del Bitcoin en USD y calculá su variación porcentual de los últimos 7 días.\", "
        "max_steps=2)\n"
        "assert trace2['decisions'][-1]['forced'], 'El cierre debió ser forzado'"
    ),
    code("print(f'Créditos Tavily usados después de la demo: {credit_guard.credits_used}/{credit_guard.budget}')"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "demo.ipynb")
print("demo.ipynb generado")
