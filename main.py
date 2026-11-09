"""CLI del orquestador.

Uso:
    python main.py "tu consulta"            # ejecuta el flujo mostrando cada delegación
    python main.py --draw                   # exporta el diagrama del grafo a docs/
    python main.py --credits                # muestra créditos de Tavily usados
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

from graph import graph, run_stream
from tools import credit_guard

DEFAULT_QUERY = (
    "Buscá la inflación mensual de Argentina (INDEC) de los últimos 6 meses publicados, calculá el promedio, "
    "la inflación acumulada del período y la tendencia, y analizá el sentimiento de los titulares de noticias "
    "recientes sobre la inflación."
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, format="   ⚠️  %(message)s")


def draw(out_dir: Path = Path("docs")) -> None:
    out_dir.mkdir(exist_ok=True)
    g = graph.get_graph()
    mermaid = g.draw_mermaid()
    (out_dir / "graph.mmd").write_text(mermaid, encoding="utf-8")
    print(mermaid)
    try:
        (out_dir / "graph.png").write_bytes(g.draw_mermaid_png())
        print(f"PNG guardado en {out_dir / 'graph.png'}")
    except Exception as exc:  # noqa: BLE001  (draw_mermaid_png usa la API de mermaid.ink)
        print(f"No se pudo generar el PNG ({exc}); quedó el .mmd")


def _short(text: str, width: int = 600) -> str:
    return textwrap.shorten(" ".join(text.split()), width=width, placeholder=" …")


def run(query: str, max_steps: int = 8) -> dict:
    """Ejecuta el grafo imprimiendo cada delegación. Devuelve la traza (decisiones, aportes, respuesta)."""
    print(f"\n🧑 USUARIO: {query}\n")
    trace: dict = {"decisions": [], "contributions": [], "final_answer": ""}
    for node, update in run_stream(query, max_steps):
        trace["decisions"] += update.get("decisions", [])
        trace["contributions"] += update.get("contributions", [])
        if node == "supervisor":
            d = update["decisions"][-1]
            tag = " (FORZADO)" if d["forced"] else ""
            print(f"🧭 SUPERVISOR [step {d['step']}] -> {d['next']}{tag} | rúbrica OK={d['validation_passed']}")
            print(f"   motivo: {_short(d['reasoning'], 300)}")
            if update.get("current_instruction"):
                print(f"   instrucción: {_short(update['current_instruction'], 300)}")
        elif node in ("researcher", "analyst"):
            c = update["contributions"][-1]
            icon = "🔎" if node == "researcher" else "📊"
            print(f"\n{icon} {node.upper()} (tools: {', '.join(c['tool_calls']) or 'ninguna'})")
            print(textwrap.indent(_short(c["output"], 900), "   "), "\n")
        elif node == "synthesizer":
            trace["final_answer"] = update["final_answer"]
            print("\n" + "=" * 80 + "\n✅ RESPUESTA FINAL\n" + "=" * 80)
            print(update["final_answer"])
    print(f"\n💳 Créditos Tavily usados (total histórico): {credit_guard.credits_used}/{credit_guard.budget}")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente (LangGraph + Gemini)")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--draw", action="store_true", help="Exporta el diagrama del grafo a docs/")
    parser.add_argument("--credits", action="store_true", help="Muestra créditos Tavily usados")
    args = parser.parse_args()

    if args.draw:
        draw()
    elif args.credits:
        print(f"Créditos usados: {credit_guard.credits_used} / {credit_guard.budget} (restan {credit_guard.remaining})")
    else:
        run(args.query, args.max_steps)


if __name__ == "__main__":
    main()
