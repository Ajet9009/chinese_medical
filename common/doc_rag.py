"""W3 helpers: chunk text, mix KGQA with document hits, refuse with CRAG.

Document FAISS is a separate index from entity matching. Redis LTRIM / ContextCompressor
are short-term memory and are not used here.

CRAG guards the document path. Knowledge-graph facts still answer even if docs are weak.
"""

from __future__ import annotations

from typing import Any, Sequence

REFUSE_ANSWER = (
    "图谱与文献中均未查到与该问题相关的依据，无法作答。"
    "请改问具体的方剂、药材或证型。"
)

UNCERTAIN_PREFIX = "依据有限，仅供参考。"

_EMPTY_KG_MARKERS = ("", "(无图谱数据)")


def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    parts: list[str] = []
    start = 0
    n = len(raw)
    while start < n:
        end = min(n, start + size)
        parts.append(raw[start:end])
        if end >= n:
            break
        start = end - overlap
    return parts


def kg_context_empty(neo4j_answer: str | None) -> bool:
    text = (neo4j_answer or "").strip()
    return text in _EMPTY_KG_MARKERS


def should_refuse(
    neo4j_answer: str | None,
    doc_chunks: Sequence[dict[str, Any]] | None,
    crag_grade: str | None = None,
    crag_action: str | None = None,
    crag_confidence: str | None = None,
) -> bool:
    """图谱有事实则不拒答。文献 incorrect/refused 且图谱空才拒答。"""
    if not kg_context_empty(neo4j_answer):
        return False
    if crag_confidence == "refused" or crag_action in ("refused", "rewritten_failed"):
        return True
    chunks = list(doc_chunks or [])
    grade = (crag_grade or "").strip()
    if not grade:
        if not chunks:
            return True
        from common.rag.crag import GRADE_INCORRECT, grade as crag_grade_fn

        top1 = float(chunks[0].get("score") or 0.0)
        grade, _ = crag_grade_fn(top1, len(chunks), rerank_ok=True)
        return grade == GRADE_INCORRECT
    if grade == "incorrect":
        return True
    return not bool(chunks)


def format_doc_context(doc_chunks: Sequence[dict[str, Any]] | None) -> str:
    if not doc_chunks:
        return ""
    lines: list[str] = []
    for item in doc_chunks:
        name = str(item.get("doc_name") or "文献")
        idx = item.get("chunk_idx", 0)
        body = str(item.get("text") or "").strip()
        lines.append(f"[文献: {name}#{idx}] {body}")
    return "\n".join(lines)
