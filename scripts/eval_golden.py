#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate / score the TCM golden set. CI: no --live (no LLM)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_golden import (  # noqa: E402
    evaluate_offline,
    golden_path,
    load_golden,
    validate_golden,
)


def _load_answers(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return {str(k).strip(): str(v or "") for k, v in data.items()}
    if isinstance(data, list):
        out: dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            q = str(item.get("query") or "").strip()
            if q:
                out[q] = str(item.get("answer") or "")
        return out
    raise ValueError("score file 必须是 {query: answer} 或 [{query, answer}]")


def _live_answers(items: list[dict], base: str, token: str) -> dict[str, str]:
    import urllib.error
    import urllib.request

    out: dict[str, str] = {}
    for item in items:
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        req = urllib.request.Request(
            base.rstrip("/") + "/ask",
            data=json.dumps({"question": query}, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            out[query] = str(body.get("answer") or "")
        except urllib.error.URLError as exc:
            print(f"  [skip] {query}: {exc}", file=sys.stderr)
            out[query] = ""
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="黄金集校验与离线关键词命中")
    parser.add_argument("--path", default="", help="覆盖 GOLDEN_QA_PATH")
    parser.add_argument("--score-file", default="", help="离线答案 JSON，不调 LLM")
    parser.add_argument("--live", action="store_true", help="打本机 /ask（不进 CI）")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="", help="登录 JWT；--live 时需要")
    args = parser.parse_args()

    path = Path(args.path) if args.path else golden_path()
    items = load_golden(path)
    errors = validate_golden(items)
    print(f"golden: {path}  ({len(items)} 条)")
    if errors:
        for err in errors:
            print(f"  [err] {err}")
        return 1
    print("  validate: ok")

    answers: dict[str, str] | None = None
    if args.score_file:
        answers = _load_answers(Path(args.score_file))
    elif args.live:
        if not args.token:
            print("--live 需要 --token", file=sys.stderr)
            return 2
        answers = _live_answers(items, args.base, args.token)

    if answers is not None:
        report = evaluate_offline(items, answers)
        print(
            f"  labeled={report['labeled']} passed={report['passed']} "
            f"failed={report['failed']} skipped={report['skipped']} "
            f"pass_rate={report['pass_rate']:.2%}"
        )
        for row in report["items"]:
            if row.get("skipped"):
                print(f"  skip  {row['query']}")
                continue
            flag = "ok" if row.get("passed") else "fail"
            print(f"  {flag:4} {row['query']}  hits={row.get('hits')} misses={row.get('misses')}")
        return 0 if report["failed"] == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
