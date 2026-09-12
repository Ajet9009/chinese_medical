"""Corrective RAG v1：用检索分分级，incorrect 则改写重检索。

不接云端 rerank；调用方把稠密余弦 / 融合后还原的相关分当作 top1。
"""

from __future__ import annotations

import os

GRADE_CORRECT = "correct"
GRADE_AMBIGUOUS = "ambiguous"
GRADE_INCORRECT = "incorrect"


def _thresholds(high: float | None, low: float | None) -> tuple[float, float]:
    _high = float(os.getenv("CRAG_HIGH", "0.72")) if high is None else float(high)
    _low = float(os.getenv("CRAG_LOW", "0.3")) if low is None else float(low)
    return _high, _low


def grade(
    top1_score: float,
    n_contexts: int,
    rerank_ok: bool = True,
    es: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> tuple[str, str]:
    """检索结果分级。返回 (grade, reason)。"""
    if not rerank_ok:
        return GRADE_AMBIGUOUS, "rerank 未启用，无法可靠分级（保守降级）"
    if n_contexts == 0:
        return GRADE_INCORRECT, "检索无结果"
    _high, _low = _thresholds(high, low)
    score = float(es) if es is not None else float(top1_score)
    label = "es" if es is not None else "top1 相关性"
    if score >= _high:
        return GRADE_CORRECT, f"{label} {score:.2f} ≥ {_high}"
    if score < _low:
        return GRADE_INCORRECT, f"{label} {score:.2f} < {_low}"
    return GRADE_AMBIGUOUS, f"{label} {score:.2f} 介于阈值间"


def confidence_of(grade: str, rewritten: bool) -> str:
    """分级 + 是否改写重检索 → high/medium/refused。"""
    if grade == GRADE_CORRECT and not rewritten:
        return "high"
    if grade == GRADE_INCORRECT and rewritten:
        return "refused"
    return "medium"
