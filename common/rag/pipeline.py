"""文献检索 + CRAG v1 纠错闭环（一次 query 改写重检索）。"""

from __future__ import annotations

from typing import Any, Callable

from common.rag import crag

SearchFn = Callable[[str], list[dict[str, Any]]]
RewriteFn = Callable[[str], str]


def retrieve_with_crag(
    question: str,
    search_fn: SearchFn,
    rewrite_fn: RewriteFn | None = None,
) -> dict[str, Any]:
    q = (question or "").strip()
    hits = search_fn(q) if q else []
    top1 = float(hits[0].get("score") or 0.0) if hits else 0.0
    grade, reason = crag.grade(top1, len(hits), rerank_ok=True)
    action = "normal"
    if grade == crag.GRADE_INCORRECT and rewrite_fn:
        new_q = (rewrite_fn(q) or "").strip()
        if new_q and new_q != q:
            hits2 = search_fn(new_q) or []
            if hits2:
                hits = hits2
            top1 = float(hits[0].get("score") or 0.0) if hits else 0.0
            grade, reason = crag.grade(top1, len(hits), rerank_ok=True)
            action = "rewritten"
            if grade == crag.GRADE_INCORRECT:
                action = "refused"
    rewritten = action.startswith("rewritten") or action == "refused"
    confidence = crag.confidence_of(grade, rewritten)
    if grade == crag.GRADE_INCORRECT:
        hits = []
    return {
        "hits": hits,
        "crag_grade": grade,
        "crag_action": action,
        "crag_confidence": confidence,
        "crag_reason": reason,
    }
