"""引用校验：答案与文献块的字面重叠，可选余弦；不做 NLI。"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import numpy as np


def _bigrams(text: str) -> set[str]:
    t = "".join((text or "").split())
    if len(t) < 2:
        return {t} if t else set()
    return {t[i : i + 2] for i in range(len(t) - 1)}


def verify_citations(
    answer: str,
    chunks: Sequence[dict[str, Any]] | None,
    threshold: float | None = None,
    encode_fn: Callable[[list[str]], np.ndarray] | None = None,
    cosine_threshold: float | None = None,
) -> dict[str, Any]:
    """返回 ok / rewrite_needed / overlap。核心事实与文献对不上则 rewrite_needed。"""
    cut = float(os.getenv("CITATION_OVERLAP", "0.08")) if threshold is None else float(threshold)
    ans = (answer or "").strip()
    blob = "".join(str(c.get("text") or "") for c in (chunks or []))
    if not ans or not blob:
        return {"ok": False, "rewrite_needed": True, "overlap": 0.0}
    a = _bigrams(ans)
    b = _bigrams(blob)
    overlap = len(a & b) / max(1, len(a))
    ok = overlap >= cut
    if not ok and encode_fn is not None:
        vecs = np.asarray(encode_fn([ans, blob]), dtype="float32")
        va, vb = vecs[0], vecs[1]
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
        cos = float(np.dot(va, vb) / denom)
        cos_cut = (
            float(os.getenv("CITATION_VERIFY_SIM_THRESHOLD", "0.35"))
            if cosine_threshold is None
            else float(cosine_threshold)
        )
        ok = cos >= cos_cut
    return {"ok": bool(ok), "rewrite_needed": (not ok), "overlap": round(float(overlap), 4)}
