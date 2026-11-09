"""Herramientas del Agente de Investigación.

- `web_search`: búsqueda real con Tavily, protegida por un guardián de créditos.
- `mock_search`: búsqueda simulada sobre un corpus local (fallback sin key o sin créditos).

Control de créditos de Tavily:
  * search_depth="basic"  -> 1 crédito por búsqueda.
  * Límite global persistente (TAVILY_CREDIT_BUDGET, default 1500) contado en .tavily_usage.json.
  * Límite por ejecución del grafo (TAVILY_MAX_SEARCHES_PER_RUN, default 4).
  * Caché en memoria por consulta normalizada (repetir una búsqueda cuesta 0 créditos).
  Si se alcanza cualquiera de los dos, se usa mock_search y se informa al agente.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
USAGE_FILE = ROOT / ".tavily_usage.json"
MOCK_CORPUS = ROOT / "data" / "mock_corpus.json"
CREDITS_PER_SEARCH = 1  # search_depth="basic"


class TavilyCreditGuard:
    """Cuenta créditos usados (persistente) y limita búsquedas por ejecución."""

    def __init__(self) -> None:
        self.budget = int(os.getenv("TAVILY_CREDIT_BUDGET", "1500"))
        self.max_per_run = int(os.getenv("TAVILY_MAX_SEARCHES_PER_RUN", "4"))
        self.run_searches = 0
        self._lock = threading.Lock()

    # --- persistencia ---
    def _load(self) -> dict:
        if USAGE_FILE.exists():
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        return {"credits_used": 0, "history": []}

    def _save(self, data: dict) -> None:
        USAGE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def credits_used(self) -> int:
        return self._load()["credits_used"]

    @property
    def remaining(self) -> int:
        return self.budget - self.credits_used

    def reset_run(self) -> None:
        self.run_searches = 0

    def try_consume(self, query: str) -> tuple[bool, str]:
        """Reserva créditos antes de llamar a Tavily. Devuelve (ok, motivo)."""
        with self._lock:
            data = self._load()
            if data["credits_used"] + CREDITS_PER_SEARCH > self.budget:
                return False, f"presupuesto global agotado ({data['credits_used']}/{self.budget})"
            if self.run_searches >= self.max_per_run:
                return False, f"límite de {self.max_per_run} búsquedas por ejecución alcanzado"
            data["credits_used"] += CREDITS_PER_SEARCH
            data["history"].append(
                {"ts": datetime.now().isoformat(timespec="seconds"), "query": query, "credits": CREDITS_PER_SEARCH}
            )
            data["history"] = data["history"][-200:]
            self._save(data)
            self.run_searches += 1
            return True, "ok"


credit_guard = TavilyCreditGuard()
_SEARCH_CACHE: dict[str, str] = {}


def _mock_lookup(query: str, k: int = 3) -> list[dict]:
    corpus = json.loads(MOCK_CORPUS.read_text(encoding="utf-8"))
    terms = {t for t in query.lower().split() if len(t) > 3}
    scored = []
    for doc in corpus:
        haystack = f"{doc['title']} {doc['content']} {' '.join(doc.get('tags', []))}".lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]


def _format_results(results: list[dict], source: str) -> str:
    if not results:
        return f"[{source}] Sin resultados."
    lines = [f"[{source}] {len(results)} resultados:"]
    for i, r in enumerate(results, 1):
        content = (r.get("content") or "").strip().replace("\n", " ")
        lines.append(f"{i}. {r.get('title', 'sin título')}\n   URL: {r.get('url', 'n/a')}\n   {content[:700]}")
    return "\n".join(lines)


@tool
def mock_search(query: str) -> str:
    """Búsqueda simulada sobre un corpus local de documentos. Usar solo si web_search no está disponible."""
    return _format_results(_mock_lookup(query), "mock_search")


@tool
def web_search(query: str) -> str:
    """Busca información actualizada en la web (Tavily). Devuelve títulos, URLs y extractos.

    Cada llamada consume créditos: formulá consultas precisas y evitá repetirlas.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return "TAVILY_API_KEY no configurada -> " + mock_search.invoke({"query": query})

    # Caché en memoria: si el agente (o un reintento por fallback de modelo) repite la consulta, no gasta créditos.
    key = " ".join(query.lower().split())
    if key in _SEARCH_CACHE:
        return _SEARCH_CACHE[key] + "\n(resultado en caché: 0 créditos)"

    ok, reason = credit_guard.try_consume(query)
    if not ok:
        return f"Búsqueda web bloqueada ({reason}) -> " + mock_search.invoke({"query": query})

    from langchain_tavily import TavilySearch

    tavily = TavilySearch(max_results=5, search_depth="basic", include_answer=False)
    try:
        response = tavily.invoke({"query": query})
    except Exception as exc:  # noqa: BLE001
        return f"Error en Tavily ({exc}) -> " + mock_search.invoke({"query": query})
    results = response.get("results", []) if isinstance(response, dict) else []
    _SEARCH_CACHE[key] = _format_results(results, "tavily")
    return _SEARCH_CACHE[key]
