"""MMR（Maximal Marginal Relevance）：jieba token Jaccard 去冗余。"""

from functools import lru_cache

import jieba


@lru_cache(maxsize=4096)
def _tokens(text: str) -> frozenset:
    t = text or ""
    return frozenset(w for w in jieba.cut(t) if len(w.strip()) >= 2)


def mmr(candidates: list[dict], topk: int, lambda_: float = 0.5) -> list[dict]:
    """candidates: [{text, score, ...}]（建议已按相关性降序）。返回 diverse topk。"""
    if len(candidates) <= topk:
        return candidates
    cg = [_tokens(c.get("text", "")) for c in candidates]
    selected = [0]
    while len(selected) < topk:
        best_j, best_score = -1, -1e9
        for j in range(len(candidates)):
            if j in selected:
                continue
            rel = candidates[j].get("score", 0.0)
            div = max(
                len(cg[j] & cg[s]) / max(1, len(cg[j] | cg[s])) for s in selected
            )
            score = lambda_ * rel - (1 - lambda_) * div
            if score > best_score:
                best_score, best_j = score, j
        if best_j < 0:
            break
        selected.append(best_j)
    return [candidates[i] for i in selected]
