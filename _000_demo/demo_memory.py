#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三层记忆效果演示：短期 / 长期 / 画像 + 冲突遗忘 + 用户隔离。

用法:
    python _000_demo/demo_memory.py
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / "common" / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def main() -> None:
    from _008_memory.factory import create_memory_manager, reset_memory_manager

    reset_memory_manager()
    _banner("初始化 MemoryManager")
    mm = create_memory_manager()
    # 演示要立刻看到长期结果 → 同步抽取
    mm._cfg.async_extract = False
    print(f"  短期后端: {type(mm._short).__name__}")
    print(f"  长期后端: {type(mm._long).__name__}")
    print(f"  画像后端: {type(mm._profile).__name__}")

    user_a = f"demo-user-a-{uuid.uuid4().hex[:8]}"
    user_b = f"demo-user-b-{uuid.uuid4().hex[:8]}"
    session_1 = f"sess-{uuid.uuid4().hex[:8]}"
    session_2 = f"sess-{uuid.uuid4().hex[:8]}"

    _banner("① 短期记忆：同会话滑动窗口")
    turns = [
        ("user", "你好，我最近总是失眠多梦"),
        ("assistant", "失眠多梦常见于心脾两虚或阴虚火旺，建议先辨证。"),
        ("user", "我偏向中药调理，不太想吃安眠药"),
        ("assistant", "好的，我们可以从安神类方剂和药材入手。"),
    ]
    for role, content in turns:
        mm.add_message(user_a, session_1, role, content, wait_extract=True)
        print(f"  [{role}] {content}")

    short = mm._short.recent(user_a, session_1, 20)
    print(f"\n  短期窗口条数: {len(short)}")
    for m in short:
        print(f"    - {m.role}: {m.content[:40]}")

    ctx1 = mm.get_context(user_a, session_1, "给我推荐改善睡眠的思路")
    print("\n  get_context（同会话）片段:")
    for line in ctx1.splitlines()[:18]:
        print(f"    {line}")

    _banner("② 长期记忆 + 画像：提取可复用事实")
    for content in [
        "我有气虚体质，平时怕冷",
        "我偏好喝温热的汤药，不喜欢苦味太重",
        "我计划下周开始调理脾胃",
    ]:
        print(f"  写入: {content}")
        mm.add_message(user_a, session_1, "user", content, wait_extract=True)

    items = mm._long._memory.get(user_a, []) if hasattr(mm._long, "_memory") else []
    print(f"\n  长期记忆条数(user_a): {len(items)}")
    for it in items:
        print(f"    - {it['subject']} | {it['predicate']} | {it['object']}")

    profile = mm._profile.get_all(user_a)
    print(f"\n  用户画像(user_a): {profile}")

    _banner("③ 跨会话：新 session 仍有长期，无旧会话短期")
    ctx2 = mm.get_context(user_a, session_2, "我怕冷，调理时要注意什么")
    print("  get_context（新会话 session_2）:")
    for line in ctx2.splitlines()[:20]:
        print(f"    {line}")
    has_long = "【长期记忆】" in ctx2
    # 旧会话短期原文不应出现在「近期对话」；长期里可有「失眠多梦」事实
    short_block = ""
    if "【近期对话】" in ctx2:
        short_block = ctx2.split("【近期对话】", 1)[1].split("【当前输入】")[0]
    leaked_short = "偏向中药调理" in short_block or "安眠药" in short_block
    print(f"\n  含长期记忆块: {has_long}（期望 True）")
    print(f"  新会话误带旧短期原文: {leaked_short}（期望 False）")

    _banner("④ 信息冲突与遗忘：偏好变更")
    print("  写入: 我不再偏好苦味重的药了，改喜欢清淡安神方")
    before = len(mm._long._memory.get(user_a, [])) if hasattr(mm._long, "_memory") else -1
    mm.add_message(
        user_a, session_2, "user",
        "我不再偏好苦味重的药了，改喜欢清淡安神方",
        wait_extract=True,
    )
    after_items = mm._long._memory.get(user_a, []) if hasattr(mm._long, "_memory") else []
    print(f"  长期条数: {before} → {len(after_items)}")
    prefs = [it for it in after_items if it.get("predicate") == "偏好"]
    print(f"  偏好三元组: {prefs}")
    print(f"  画像: {mm._profile.get_all(user_a)}")

    _banner("⑤ 用户隔离：同 session_id 不同 user 不串短期")
    mm.add_message(user_b, session_2, "user", "我只想了解感冒风寒怎么处理", wait_extract=True)
    ctx_b = mm.get_context(user_b, session_2, "有什么方剂推荐")
    print("  user_b 上下文:")
    for line in ctx_b.splitlines()[:14]:
        print(f"    {line}")
    leak = any(k in ctx_b for k in ("气虚", "怕冷", "清淡安神方", "苦味", "脾胃"))
    print(f"\n  是否泄漏 user_a 事实: {leak}（期望 False）")

    _banner("演示结束")
    print("  落盘: data/memory/long_term.json , data/memory/profile.json")
    ok = has_long and not leaked_short and not leak
    print(f"  隔离/跨会话检查: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
