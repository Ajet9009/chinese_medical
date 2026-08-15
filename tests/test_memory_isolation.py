#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆系统单元测试：用户隔离 / 冲突覆盖 / 画像落盘（不依赖真实 LLM/BGE）。"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from _008_memory.factory import InMemoryShortTermStore, JsonLongTermStore
from _008_memory.memory_manager import MemoryConfig, MemoryManager
from _008_memory.stores import JsonProfileStore, Message, short_term_key


class FakeLLM:
    def __init__(self, payloads: list[str] | None = None) -> None:
        self._payloads = list(payloads or [])
        self.calls = 0

    def invoke(self, prompt: str) -> str:
        self.calls += 1
        if self._payloads:
            return self._payloads.pop(0)
        return "[]"


class FakeEmb:
    def embed_query(self, text: str) -> list[float]:
        # 简单 hash → 8 维伪向量，同文同向量
        vals = [((ord(c) % 13) / 13.0) for c in (text or "x")[:8]]
        while len(vals) < 8:
            vals.append(0.1)
        # L2 normalize
        import math
        n = math.sqrt(sum(v * v for v in vals)) or 1.0
        return [v / n for v in vals]


class FakeTok:
    def count(self, text: str) -> int:
        return len(text)


class TestShortTermIsolation(unittest.TestCase):
    def test_same_session_different_users_isolated(self) -> None:
        store = InMemoryShortTermStore(max_rounds=10)
        sid = "shared-session"
        store.append("user-a", sid, Message("user", "我有气虚"))
        store.append("user-b", sid, Message("user", "我只问感冒"))

        a = store.recent("user-a", sid, 10)
        b = store.recent("user-b", sid, 10)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertIn("气虚", a[0].content)
        self.assertIn("感冒", b[0].content)
        self.assertNotEqual(short_term_key("user-a", sid), short_term_key("user-b", sid))


class TestMemoryManagerIsolation(unittest.TestCase):
    def _mgr(self, llm: FakeLLM, tmp: Path) -> MemoryManager:
        return MemoryManager(
            short_term=InMemoryShortTermStore(),
            long_term=JsonLongTermStore(tmp, FakeEmb()),
            profile=JsonProfileStore(tmp),
            llm=llm,
            embeddings=FakeEmb(),
            tokenizer=FakeTok(),
            config=MemoryConfig(async_extract=False, sanitize_pii=False),
        )

    def test_user_b_does_not_see_user_a_short_term(self) -> None:
        llm = FakeLLM([
            '[{"subject":"用户","predicate":"偏好","object":"清淡安神方","action":"set"}]',
            "[]",
        ])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mm = self._mgr(llm, tmp)
            sid = "sess-shared"
            mm.add_message("ua", sid, "user", "我喜欢清淡安神方", wait_extract=True)
            mm.add_message("ub", sid, "user", "感冒风寒怎么处理", wait_extract=True)

            ctx_b = mm.get_context("ub", sid, "有什么方剂")
            self.assertNotIn("清淡安神方", ctx_b)
            self.assertIn("感冒风寒", ctx_b)

            ctx_a = mm.get_context("ua", sid, "安神")
            self.assertIn("清淡安神方", ctx_a)
            self.assertNotIn("感冒风寒", ctx_a)

    def test_profile_persists_and_conflict_overwrite(self) -> None:
        llm = FakeLLM([
            '[{"subject":"用户","predicate":"偏好","object":"苦味不重","action":"set"}]',
            '[{"subject":"用户","predicate":"偏好","object":"清淡安神方","action":"set"}]',
        ])
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mm = self._mgr(llm, tmp)
            mm.add_message("u1", "s1", "user", "偏好苦味不重", wait_extract=True)
            self.assertEqual(mm._profile.get("u1", "偏好"), "苦味不重")

            mm.add_message("u1", "s1", "user", "改喜欢清淡安神方", wait_extract=True)
            self.assertEqual(mm._profile.get("u1", "偏好"), "清淡安神方")

            # 落盘可读
            profile2 = JsonProfileStore(tmp)
            self.assertEqual(profile2.get("u1", "偏好"), "清淡安神方")

            items = mm._long._memory.get("u1", [])
            prefs = [it for it in items if it["predicate"] == "偏好"]
            self.assertEqual(len(prefs), 1)
            self.assertEqual(prefs[0]["object"], "清淡安神方")


class FakeAudit:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def record(self, user_id, action, *, subject="", predicate="", object_value="", detail="") -> None:
        self.events.append((user_id, action, subject, predicate, object_value, detail))


class TestAuditAndQueue(unittest.TestCase):
    def test_audit_records_triple_upsert(self) -> None:
        llm = FakeLLM([
            '[{"subject":"用户","predicate":"偏好","object":"清淡","action":"set"}]',
        ])
        audit = FakeAudit()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mm = MemoryManager(
                short_term=InMemoryShortTermStore(),
                long_term=JsonLongTermStore(tmp, FakeEmb()),
                profile=JsonProfileStore(tmp),
                llm=llm,
                embeddings=FakeEmb(),
                tokenizer=FakeTok(),
                config=MemoryConfig(async_extract=False, sanitize_pii=False),
                audit=audit,
            )
            mm.add_message("u1", "s1", "user", "我喜欢清淡", wait_extract=True)
            actions = [e[1] for e in audit.events]
            self.assertIn("triple_upsert", actions)
            self.assertIn("profile_upsert", actions)

    def test_extract_queue_drops_when_full(self) -> None:
        llm = FakeLLM(["[]"] * 10)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mm = MemoryManager(
                short_term=InMemoryShortTermStore(),
                long_term=JsonLongTermStore(tmp, FakeEmb()),
                profile=JsonProfileStore(tmp),
                llm=llm,
                embeddings=FakeEmb(),
                tokenizer=FakeTok(),
                config=MemoryConfig(
                    async_extract=True,
                    sanitize_pii=False,
                    extract_workers=1,
                    extract_queue_size=1,
                ),
            )
            # 占满唯一 slot
            self.assertTrue(mm._extract_slots.acquire(blocking=False))
            mm.add_message("u1", "s1", "user", "第一条会被排队丢弃", wait_extract=False)
            self.assertGreaterEqual(mm._dropped_extracts, 1)
            mm._extract_slots.release()


if __name__ == "__main__":
    unittest.main()
