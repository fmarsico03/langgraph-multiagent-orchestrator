"""Factory del LLM (Gemini) compartido por todos los agentes, con fallback entre modelos.

Los modelos Gemini devuelven 503 (alta demanda) / 429 / 504 de forma intermitente. Para que el grafo
no se caiga, `call_with_fallback` reintenta la misma operación con el siguiente modelo de la lista
GEMINI_MODEL + GEMINI_FALLBACK_MODELS.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from datetime import date
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Evita el warning "Both GOOGLE_API_KEY and GEMINI_API_KEY are set": el proyecto usa GOOGLE_API_KEY del .env.
if os.getenv("GOOGLE_API_KEY"):
    os.environ.pop("GEMINI_API_KEY", None)

log = logging.getLogger("orchestrator.llm")

# Gemini 3.x ignora `temperature` (sampling fijo) y lo avisa en cada llamada.
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")


def today() -> str:
    """Fecha actual (ISO). Se inyecta en los prompts para que los agentes evalúen recencia."""
    return date.today().isoformat()

DEFAULT_MODEL = "gemini-3.5-flash"
DEFAULT_FALLBACKS = "gemini-3.6-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "504", "DEADLINE_EXCEEDED", "overloaded")

T = TypeVar("T")


def candidate_models() -> list[str]:
    primary = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    fallbacks = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",") if m.strip()]
    return list(dict.fromkeys([primary, *fallbacks]))


@lru_cache(maxsize=16)
def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model or candidate_models()[0],
        temperature=temperature,
        max_retries=1,  # fallar rápido y pasar al siguiente modelo
        timeout=90,
    )


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}"
    return any(marker in text for marker in TRANSIENT_MARKERS)


def call_with_fallback(fn: Callable[[str], T], rounds: int = 2) -> T:
    """Ejecuta fn(model_name) probando cada modelo candidato ante errores transitorios."""
    last_exc: Exception | None = None
    for attempt in range(rounds):
        for model in candidate_models():
            try:
                return fn(model)
            except Exception as exc:  # noqa: BLE001
                if not _is_transient(exc):
                    raise
                last_exc = exc
                log.warning("Modelo %s no disponible (%s). Probando el siguiente...", model, str(exc)[:80])
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Ningún modelo Gemini disponible: {last_exc}") from last_exc


def message_text(message: BaseMessage) -> str:
    """Extrae texto plano de un mensaje (Gemini puede devolver content como lista de partes)."""
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)
