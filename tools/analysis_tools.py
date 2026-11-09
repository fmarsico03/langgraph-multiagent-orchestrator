"""Herramientas del Agente de Análisis: cálculo seguro, estadísticas y sentimiento."""

from __future__ import annotations

import ast
import math
import operator as op
import re
import statistics

from langchain_core.tools import tool

# ---------------------------------------------------------------- calculadora
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]  # type: ignore[return-value]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError(f"Expresión no permitida: {ast.dump(node)[:80]}")


@tool
def calculator(expression: str) -> str:
    """Evalúa una expresión matemática de forma segura.

    Soporta + - * / ** % //, paréntesis y funciones sqrt, log, log10, exp, abs, round, min, max, sum.
    Ejemplo: "((1+0.027)*(1+0.022)-1)*100"
    """
    try:
        result = _eval(ast.parse(expression, mode="eval"))
        return f"{expression} = {result:.6g}" if isinstance(result, float) else f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001
        return f"Error al evaluar '{expression}': {exc}"


# ---------------------------------------------------------------- estadísticas
@tool
def compute_statistics(values: list[float], are_percentages: bool = False) -> str:
    """Calcula estadísticas descriptivas de una serie ordenada cronológicamente.

    Devuelve n, media, mediana, desvío estándar, mínimo, máximo, variación entre primer y último
    valor y tendencia (pendiente lineal). Si `are_percentages` es True (ej. inflación mensual en %),
    también calcula la variación ACUMULADA compuesta: prod(1 + v/100) - 1.
    """
    if not values:
        return "Error: la lista de valores está vacía."
    n = len(values)
    mean = statistics.fmean(values)
    out = {
        "n": n,
        "media": round(mean, 4),
        "mediana": round(statistics.median(values), 4),
        "desvio_std": round(statistics.stdev(values), 4) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "primero": values[0],
        "ultimo": values[-1],
    }
    if values[0] != 0:
        out["variacion_primero_ultimo_pct"] = round((values[-1] - values[0]) / abs(values[0]) * 100, 2)
    if n > 1:
        xs = range(n)
        x_mean = (n - 1) / 2
        slope = sum((x - x_mean) * (y - mean) for x, y in zip(xs, values)) / sum((x - x_mean) ** 2 for x in xs)
        out["pendiente_por_periodo"] = round(slope, 4)
        out["tendencia"] = "alcista" if slope > 0.05 * abs(mean) / max(n, 1) else (
            "bajista" if slope < -0.05 * abs(mean) / max(n, 1) else "estable"
        )
    if are_percentages:
        acc = math.prod(1 + v / 100 for v in values) - 1
        out["acumulado_compuesto_pct"] = round(acc * 100, 2)
    return "\n".join(f"{k}: {v}" for k, v in out.items())


# ---------------------------------------------------------------- sentimiento
_POS = {
    "baja", "bajó", "desacelera", "desaceleración", "mejora", "mejoró", "crecimiento", "crece", "récord",
    "positivo", "positiva", "estabilidad", "estable", "recuperación", "optimismo", "superávit", "logro",
    "éxito", "favorable", "alivio", "avance", "sube el consumo", "improve", "growth", "positive", "gain",
    "strong", "recovery", "optimism", "beat", "success", "record high", "surplus",
}
_NEG = {
    "suba", "aumento", "acelera", "aceleración", "crisis", "caída", "cae", "cayó", "negativo", "negativa",
    "pobreza", "déficit", "recesión", "incertidumbre", "preocupación", "pérdida", "riesgo", "devaluación",
    "desempleo", "conflicto", "alarma", "decline", "loss", "risk", "crisis", "recession", "negative",
    "weak", "fear", "uncertainty", "inflation surge", "layoffs", "deficit",
}
_NEGATORS = {"no", "sin", "nunca", "not", "never"}


def _score_text(text: str) -> tuple[float, list[str], list[str]]:
    lowered = text.lower()
    tokens = re.findall(r"[\wáéíóúñü]+", lowered)
    pos_hits, neg_hits = [], []
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in _NEGATORS
        if tok in _POS:
            (neg_hits if negated else pos_hits).append(tok)
        elif tok in _NEG:
            (pos_hits if negated else neg_hits).append(tok)
    for phrase in (p for p in _POS if " " in p):
        if phrase in lowered:
            pos_hits.append(phrase)
    for phrase in (p for p in _NEG if " " in p):
        if phrase in lowered:
            neg_hits.append(phrase)
    total = len(pos_hits) + len(neg_hits)
    score = 0.0 if total == 0 else (len(pos_hits) - len(neg_hits)) / total
    return score, pos_hits, neg_hits


@tool
def sentiment_analysis(texts: list[str]) -> str:
    """Analiza el sentimiento (léxico ES/EN) de una lista de textos (titulares, extractos).

    Devuelve un score por texto en [-1, 1], su etiqueta y el agregado.
    Nota: es un analizador léxico simple; interpretá los resultados con criterio.
    """
    if not texts:
        return "Error: no se recibieron textos."
    lines, scores = [], []
    for i, t in enumerate(texts, 1):
        s, pos, neg = _score_text(t)
        scores.append(s)
        label = "positivo" if s > 0.15 else "negativo" if s < -0.15 else "neutral"
        lines.append(f"{i}. score={s:+.2f} ({label}) +{pos} -{neg} | {t[:90]}")
    avg = statistics.fmean(scores)
    label = "positivo" if avg > 0.15 else "negativo" if avg < -0.15 else "neutral/mixto"
    dist = {
        "positivos": sum(s > 0.15 for s in scores),
        "neutrales": sum(-0.15 <= s <= 0.15 for s in scores),
        "negativos": sum(s < -0.15 for s in scores),
    }
    lines.append(f"AGREGADO: score_medio={avg:+.2f} -> {label} | distribución={dist}")
    return "\n".join(lines)
