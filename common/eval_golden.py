"""Golden-set load/validate and offline keyword scoring.

CI uses this without a live LLM. `--live` in scripts/eval_golden.py is local only.
source=feedback rows may have empty expect (待标注); they are skipped in scoring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common.env_loader import load_app_env

CATEGORIES = frozenset({"方剂", "本草", "证候", "文献", "拒答", "用户反馈"})
_REPO_ROOT = Path(__file__).resolve().parent.parent


def golden_path() -> Path:
    load_app_env()
    raw = (os.getenv("GOLDEN_QA_PATH") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / raw
        return path
    return _REPO_ROOT / "eval" / "golden_qa.json"


def load_golden(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or golden_path()
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_qa.json 必须是数组")
    return [item for item in data if isinstance(item, dict)]


def save_golden(items: list[dict[str, Any]], path: Path | None = None) -> Path:
    p = path or golden_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(items, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def validate_golden(items: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for i, item in enumerate(items):
        query = str(item.get("query") or "").strip()
        if not query:
            errors.append(f"[{i}] query 为空")
        elif query in seen:
            errors.append(f"[{i}] 重复问句: {query}")
        seen.add(query)
        expect = item.get("expect")
        if not isinstance(expect, list):
            errors.append(f"[{i}] expect 必须是字符串数组")
            expect = []
        elif any(not isinstance(x, str) or not str(x).strip() for x in expect):
            errors.append(f"[{i}] expect 含空项")
        category = str(item.get("category") or "").strip()
        if not category:
            errors.append(f"[{i}] category 为空")
        elif category not in CATEGORIES:
            errors.append(f"[{i}] category 非法: {category}")
        source = str(item.get("source") or "seed").strip() or "seed"
        if source != "feedback" and not expect:
            errors.append(f"[{i}] 非反馈条目 expect 不能为空")
        docs = item.get("relevant_docs")
        if docs is not None and not (
            isinstance(docs, list) and all(isinstance(x, str) for x in docs)
        ):
            errors.append(f"[{i}] relevant_docs 必须是字符串数组")
    return errors


def keyword_hit_ratio(expect: list[str], text: str) -> float:
    needles = [str(x).strip() for x in expect if str(x).strip()]
    if not needles:
        return 0.0
    hay = text or ""
    hits = sum(1 for n in needles if n in hay)
    return hits / len(needles)


def evaluate_offline(
    items: list[dict[str, Any]],
    answers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score labeled rows. Empty-expect feedback rows are skipped, not failed."""
    answers = answers or {}
    scored: list[dict[str, Any]] = []
    for item in items:
        query = str(item.get("query") or "").strip()
        expect = [str(x) for x in (item.get("expect") or []) if str(x).strip()]
        if not expect:
            scored.append({
                "query": query,
                "skipped": True,
                "reason": "待标注",
                "hit_ratio": None,
                "passed": None,
            })
            continue
        text = answers.get(query, "")
        hits = [n for n in expect if n in (text or "")]
        misses = [n for n in expect if n not in (text or "")]
        ratio = keyword_hit_ratio(expect, text)
        scored.append({
            "query": query,
            "category": item.get("category") or "",
            "skipped": False,
            "hit_ratio": ratio,
            "passed": ratio == 1.0,
            "hits": hits,
            "misses": misses,
        })
    labeled = [s for s in scored if not s.get("skipped")]
    passed = sum(1 for s in labeled if s.get("passed"))
    return {
        "total": len(items),
        "labeled": len(labeled),
        "passed": passed,
        "failed": len(labeled) - passed,
        "skipped": len(scored) - len(labeled),
        "pass_rate": (passed / len(labeled)) if labeled else 0.0,
        "items": scored,
    }


def add_from_feedback(query: str, category: str = "用户反馈") -> dict[str, Any]:
    q = (query or "").strip()
    items = load_golden()
    if not q:
        return {"added": False, "total": len(items), "query": "", "reason": "问句为空"}
    if any(str(it.get("query") or "").strip() == q for it in items):
        return {"added": False, "total": len(items), "query": q, "reason": "该问题已在黄金集"}
    items.append({
        "query": q,
        "expect": [],
        "category": category or "用户反馈",
        "source": "feedback",
    })
    save_golden(items)
    return {"added": True, "total": len(items), "query": q, "reason": ""}


def golden_summary() -> dict[str, Any]:
    items = load_golden()
    labeled = sum(1 for it in items if it.get("expect"))
    return {
        "path": str(golden_path()),
        "total": len(items),
        "labeled": labeled,
        "pending": len(items) - labeled,
    }
