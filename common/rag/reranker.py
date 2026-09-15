"""可选 CrossEncoder/BGE reranker。

默认不强制加载模型；配置 DOC_RERANK_MODEL_PATH 后再启用。
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from typing import Any

logger = logging.getLogger("rag.reranker")


@lru_cache(maxsize=2)
def load_bge_reranker_step_01(model_path: str | None = None) -> Any | None:
    """步骤 01：按需加载 BGE/CrossEncoder reranker；无路径则返回 None。"""
    path = (model_path or os.getenv("DOC_RERANK_MODEL_PATH") or "").strip()
    if not path:
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        logger.warning("未安装 sentence-transformers，跳过 reranker: %s", exc)
        return None
    try:
        return CrossEncoder(path)
    except Exception as exc:
        logger.warning("加载 reranker 失败: %s", exc)
        return None


def rerank_hits_step_02(
    question: str,
    hits: list[dict[str, Any]],
    model: Any | None = None,
    weight: float | None = None,
) -> list[dict[str, Any]]:
    """步骤 02：用 CrossEncoder 分数对候选文档块做二次排序。"""
    q = (question or "").strip()
    if not q or len(hits) <= 1:
        return list(hits)
    reranker = model if model is not None else load_bge_reranker_step_01()
    if reranker is None:
        return list(hits)

    pairs = [(q, _hit_text_step_03(hit)) for hit in hits]
    raw_scores = list(reranker.predict(pairs))
    norm_scores = _normalize_scores_step_04([float(x) for x in raw_scores])
    rerank_weight = float(weight if weight is not None else os.getenv("DOC_RERANK_WEIGHT", "0.25"))

    ranked: list[dict[str, Any]] = []
    for rank, (hit, raw, norm) in enumerate(zip(hits, raw_scores, norm_scores)):
        item = dict(hit)
        base_score = float(item.get("score") or item.get("rerank_score") or 0.0)
        item["cross_score"] = float(raw)
        item["rerank_score"] = round(base_score + rerank_weight * norm, 6)
        item["score"] = min(1.0, float(item["rerank_score"]))
        item["_rerank_order"] = rank
        ranked.append(item)
    ranked.sort(
        key=lambda x: (
            -float(x.get("rerank_score") or 0.0),
            -float(x.get("cross_score") or 0.0),
            int(x.get("_rerank_order") or 0),
        )
    )
    for item in ranked:
        item.pop("_rerank_order", None)
    return ranked


def _hit_text_step_03(hit: dict[str, Any]) -> str:
    """步骤 03：拼出 reranker 输入文本，保留标题和父块线索。"""
    parts = [
        str(hit.get("parent_title") or hit.get("title") or ""),
        str(hit.get("doc_name") or ""),
        str(hit.get("text") or ""),
    ]
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _normalize_scores_step_04(scores: list[float]) -> list[float]:
    """步骤 04：把 CrossEncoder 原始分数压到 0-1，避免不同模型尺度影响主分。"""
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [0.5 for _ in scores]
    span = high - low
    return [(score - low) / span for score in scores]
