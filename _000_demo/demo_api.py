#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调用中医问答 API 的验证脚本。

用法:
    python _000_demo/demo_api.py                          # 交互模式
    python _000_demo/demo_api.py "四君子汤有什么功效？"     # 单次
    python _000_demo/demo_api.py --batch                   # 批量预置用例
    python _000_demo/demo_api.py --url http://x.x.x.x:8000  # 自定义地址
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from typing import Any

DEFAULT_URL = "http://localhost:8000"

PRESET_CASES = [

    "肾虚用什么方剂调理？",
]


def color(text: str, c: str) -> str:
    codes = {"green": "32", "red": "31", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
    return f"\033[{codes.get(c, '0')}m{text}\033[0m"


def call_api(base_url: str, question: str) -> dict[str, Any]:
    """调用 /ask 接口，返回完整响应 JSON。"""
    data = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/ask",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_result(r: dict[str, Any]):
    """结构化打印 API 返回结果。"""
    intent_tag = color("✓ 中医", "green") if r["is_zhongyi_intent"] else color("✗ 普通", "red")
    print(f"  意图: {intent_tag}  ({r['intent_reason']})")
    print(f"  耗时: {r['elapsed_ms']}ms")

    # 用户实体
    ue = r.get("user_entities", {})
    non_empty_ue = {k: v for k, v in ue.items() if v}
    if non_empty_ue:
        print(f"  用户实体:")
        for k, v in non_empty_ue.items():
            print(f"    {k}: {v}")

    # 匹配实体
    me = r.get("matched_entities", {})
    non_empty_me = {k: v for k, v in me.items() if v}
    if non_empty_me:
        print(f"  匹配实体:")
        for k, v in non_empty_me.items():
            names = [f"{e['name']}({e['score']:.2f})" for e in v]
            print(f"    {k}: {', '.join(names)}")

    # Cypher
    cyphers = r.get("cypher_queries", [])
    if cyphers:
        print(f"  Cypher ({len(cyphers)}条):")
        for i, c in enumerate(cyphers):
            print(f"    [{i+1}] {c[:120]}")

    # KG 上下文
    kg = r.get("kg_context", "")
    if kg:
        print(f"  KG上下文: {len(kg)} 字符")

    # 最终回答
    answer = r.get("answer", "")
    print(f"\n  {color('回答:', 'bold')}")
    print(f"  {answer}")


def interactive(base_url: str):
    """交互模式。"""
    print(color("=" * 64, "bold"))
    print(color("  中医问答 API · 交互验证", "bold"))
    print(color(f"  服务地址: {base_url}", "cyan"))
    print(color("  输入问题，空行退出", "dim"))
    print(color("=" * 64, "bold"))
    print()

    # 先检查服务
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            h = json.loads(resp.read())
            print(color(f"  服务状态: {h.get('status', '?')}", "green"))
    except Exception as exc:
        print(color(f"  无法连接 {base_url}: {exc}", "red"))
        return

    print()
    while True:
        try:
            question = input(color("问题 → ", "yellow")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            print("  退出。")
            break
        print()
        try:
            t0 = time.time()
            result = call_api(base_url, question)
            print_result(result)
        except Exception as exc:
            print(color(f"  错误: {exc}", "red"))
        print()


def batch_test(base_url: str):
    """批量预置用例。"""
    total = len(PRESET_CASES)
    print(color(f"  中医问答 API · 批量验证（{total} 条）", "bold"))
    print(color(f"  服务地址: {base_url}", "cyan"))
    print()

    ok_count = 0
    fail_count = 0
    total_ms = 0.0

    for i, question in enumerate(PRESET_CASES, 1):
        print(f"[{i}/{total}] {question[:50]}")
        try:
            result = call_api(base_url, question)
            ms = result.get("elapsed_ms", 0)
            total_ms += ms
            answer_preview = result.get("answer", "")[:60]
            print(f"        意图={result['intent']:<8s}  耗时={ms:.0f}ms")
            print(f"        回答={answer_preview}")
            ok_count += 1
        except Exception as exc:
            print(color(f"        ERR: {exc}", "red"))
            fail_count += 1
        print()

    print(color("═" * 64, "bold"))
    print(f"  成功: {color(str(ok_count), 'green')}  /  失败: {color(str(fail_count), 'red')}  /  共 {total}")
    avg = total_ms / ok_count if ok_count else 0
    print(f"  平均耗时: {avg:.0f}ms")
    print(color("═" * 64, "bold"))


def parse_args(argv: list[str]) -> tuple[str, str | None, bool]:
    """解析命令行参数。返回 (base_url, single_question | None, batch)。"""
    base_url = DEFAULT_URL
    question = None
    batch = False
    skip = False
    for i, arg in enumerate(argv):
        if skip:
            skip = False
            continue
        if arg == "--url" and i + 1 < len(argv):
            base_url = argv[i + 1]
            skip = True
        elif arg == "--batch":
            batch = True
        else:
            question = " ".join(argv[i:])
            break
    return base_url, question, batch


def main():
    base_url, question, batch = parse_args(sys.argv[1:])

    if question:
        result = call_api(base_url, question)
        print_result(result)
    elif batch:
        batch_test(base_url)
    else:
        interactive(base_url)


if __name__ == "__main__":
    main()
