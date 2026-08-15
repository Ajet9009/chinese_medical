#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MemoryManager：企业级 Agent 记忆系统核心。

职责：
  - 写入（add_message）：短期记忆 → 价值判断 → 三元组提取 → 长期记忆 + 遗忘
  - 读取（get_context）：长期记忆检索 + 短期上下文 + Token 预算压缩 → Prompt

解决的核心问题：
  1. 【幻觉】每条长期记忆带 source（来源）+ timestamp，检索可追溯，不凭空生成。
  2. 【信息冲突】三元组写入前检测 subject+predicate 是否已存在，
     存在则覆盖（更新），否定意图则删除（遗忘），避免重复存储矛盾信息。
  3. 【Token 爆仓】超阈值不直接截断，而是触发 LLM 摘要压缩，保留语义。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from .stores import (
    LongTermStore,
    Message,
    MemoryTriple,
    ProfileStore,
    ShortTermStore,
)

logger = logging.getLogger("memory.manager")


class AuditStore(Protocol):
    """记忆变更审计（医疗合规）。"""

    def record(
        self,
        user_id: str,
        action: str,
        *,
        subject: str = "",
        predicate: str = "",
        object_value: str = "",
        detail: str = "",
    ) -> None: ...


# ============================================================
# LLM / Embedding / Tokenizer 接口（可注入）
# ============================================================

class LLM(Protocol):
    """LLM 抽象：用于三元组提取与摘要压缩。"""
    def invoke(self, prompt: str) -> str: ...


class Embeddings(Protocol):
    """Embedding 抽象：文本 → 向量。"""
    def embed_query(self, text: str) -> list[float]: ...


class Tokenizer(Protocol):
    """Tokenizer 抽象：计算 token 数（tiktoken 实现）。"""
    def count(self, text: str) -> int: ...


# ============================================================
# 三元组提取器（含遗忘 / 信息冲突解决）
# ============================================================

_TRIPLE_EXTRACT_PROMPT = """从用户消息中提取值得长期记忆的事实三元组。

规则：
1. 只提取稳定、可复用的信息（偏好、身份、计划、重要事实），忽略寒暄。
2. 每个三元组格式：subject|predicate|object|action
   - action = "set"（新增/更新）或 "delete"（用户明确否定/不再适用）
3. 识别"信息冲突"与"遗忘"：
   - "我不喝咖啡了" → action=delete, subject=用户, predicate=偏好
   - "我改用尾号5678的卡" → 先 delete 旧卡，再 set 新卡
4. 无价值则返回空列表。

用户消息：
{content}

输出 JSON 数组：
[{{"subject":"...","predicate":"...","object":"...","action":"set|delete"}}]"""


class TripleExtractor:
    """从消息提取三元组，并处理信息冲突（遗忘）。"""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def extract(self, content: str) -> list[dict]:
        """调用 LLM 提取三元组。失败返回空（不阻塞主流程）。"""
        import json
        import re
        try:
            raw = self._llm.invoke(_TRIPLE_EXTRACT_PROMPT.format(content=content))
            # 容错：提取 JSON 数组
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                return json.loads(m.group())
        except Exception:
            pass
        return []


# ============================================================
# Token 预算 + 摘要压缩
# ============================================================

class ContextCompressor:
    """上下文 Token 预算管理。

    超阈值时不直接截断（会丢失语义），而是用 LLM 摘要压缩历史。
    """

    def __init__(self, llm: LLM, tokenizer: Tokenizer, budget: int = 4000) -> None:
        self._llm = llm
        self._tokenizer = tokenizer
        self._budget = budget

    def count(self, text: str) -> int:
        return self._tokenizer.count(text)

    def _summarize(self, messages: list[Message]) -> str:
        """LLM 摘要压缩历史对话，保留关键信息。"""
        history = "\n".join(f"{m.role}: {m.content}" for m in messages)
        prompt = f"""以下是历史对话，请压缩为一段摘要，保留关键事实、用户偏好、未完成任务：

{history}

摘要："""
        try:
            return self._llm.invoke(prompt).strip()
        except Exception:
            # 摘要失败降级为截断（保底策略）
            return history[-self._budget:]

    def fit_budget(self, messages: list[Message], prefix: str, suffix: str) -> str:
        """把 messages（短期对话）压到预算内，返回拼接后的上下文文本。

        策略：能放下就全放；放不下则把最老的消息摘要压缩，保留最近消息。
        """
        if not messages:
            return prefix + suffix

        reserved = self.count(prefix) + self.count(suffix)
        budget_left = max(0, self._budget - reserved)

        # 1) 全放得下
        full = "【近期对话】\n" + "\n".join(f"{m.role}: {m.content}" for m in messages)
        if self.count(full) <= budget_left:
            return prefix + full + suffix

        # 2) 放不下 → 摘要压缩：从中间切分，压缩前半段，保留最近几条
        #    切分点：找使"摘要 + 最近消息"恰好放下的位置
        recent: list[Message] = []
        older: list[Message] = list(messages)
        # 从后往前保留最近消息，直到剩余放得下
        while older and self.count("\n".join(f"{m.role}: {m.content}" for m in older)) > budget_left:
            recent.insert(0, older.pop())

        # 摘要压缩前半段
        summary = self._summarize(older) if older else ""
        recent_text = "\n".join(f"{m.role}: {m.content}" for m in recent)
        compressed = f"[历史摘要]\n{summary}\n\n[最近对话]\n{recent_text}"
        return prefix + compressed + suffix


# ============================================================
# MemoryManager 主类
# ============================================================

@dataclass
class MemoryConfig:
    """记忆系统配置。"""
    short_term_rounds: int = 20        # 短期记忆窗口大小
    long_term_top_k: int = 5           # 长期记忆检索条数
    token_budget: int = 4000           # 上下文 token 预算阈值
    min_confidence: float = 0.5        # 三元组最低置信度（预留）
    async_extract: bool = True         # 长期抽取异步，避免阻塞问答主路径
    sanitize_pii: bool = True          # 写入前脱敏手机号/身份证/邮箱
    extract_workers: int = 2           # 抽取线程池大小
    extract_queue_size: int = 64       # 抽取排队上限；满则丢弃并打日志


class MemoryManager:
    """企业级 Agent 记忆系统。"""

    def __init__(
        self,
        short_term: ShortTermStore,
        long_term: LongTermStore,
        profile: ProfileStore,
        llm: LLM,
        embeddings: Embeddings,
        tokenizer: Tokenizer,
        config: MemoryConfig | None = None,
        audit: AuditStore | None = None,
    ) -> None:
        self._short = short_term
        self._long = long_term
        self._profile = profile
        self._llm = llm
        self._emb = embeddings
        self._tokenizer = tokenizer
        self._cfg = config or MemoryConfig()
        self._audit = audit
        self._extractor = TripleExtractor(llm)
        self._compressor = ContextCompressor(llm, tokenizer, self._cfg.token_budget)
        self._extract_pool = ThreadPoolExecutor(
            max_workers=max(1, self._cfg.extract_workers),
            thread_name_prefix="mem-extract",
        )
        self._extract_slots = threading.Semaphore(max(1, self._cfg.extract_queue_size))
        self._dropped_extracts = 0

    def _sanitize(self, content: str) -> str:
        if not self._cfg.sanitize_pii:
            return content
        try:
            from common.sanitizer import sanitize
            return sanitize(content)
        except Exception:
            return content

    def _audit_record(
        self,
        user_id: str,
        action: str,
        *,
        subject: str = "",
        predicate: str = "",
        object_value: str = "",
        detail: str = "",
    ) -> None:
        if self._audit is None:
            return
        try:
            self._audit.record(
                user_id,
                action,
                subject=subject,
                predicate=predicate,
                object_value=object_value,
                detail=detail,
            )
        except Exception as exc:
            logger.warning("记忆审计写入失败: %s", exc)

    def health(self) -> dict[str, Any]:
        """后端健康快照（供 /health 使用）。"""
        info: dict[str, Any] = {
            "short": type(self._short).__name__,
            "long": type(self._long).__name__,
            "profile": type(self._profile).__name__,
            "audit": type(self._audit).__name__ if self._audit else None,
            "async_extract": self._cfg.async_extract,
            "dropped_extracts": self._dropped_extracts,
            "ok": True,
        }
        for key, store in (
            ("short_ok", self._short),
            ("long_ok", self._long),
            ("profile_ok", self._profile),
            ("audit_ok", self._audit),
        ):
            if store is None:
                info[key] = None
                continue
            ping = getattr(store, "ping", None)
            if callable(ping):
                try:
                    info[key] = bool(ping())
                except Exception:
                    info[key] = False
            else:
                info[key] = True
            if info[key] is False:
                info["ok"] = False
        return info

    # ============================================================
    # 写入
    # ============================================================

    def add_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        *,
        wait_extract: bool | None = None,
    ) -> None:
        """写入一条消息：短期同步；长期抽取默认同步或异步。

        Args:
            user_id: 用户 ID（跨会话稳定）。
            session_id: 会话 ID（单次对话）。
            role: user / assistant。
            content: 消息内容。
            wait_extract: 覆盖配置；True 同步抽长期（测试用），False 强制异步。
        """
        content = self._sanitize(content)
        msg = Message(role=role, content=content, timestamp=time.time())

        # 1) 短期记忆：按 user_id + session_id 隔离写入
        self._short.append(user_id, session_id, msg)

        # 2) 只处理用户消息（assistant 消息不提取长期记忆）
        if role != "user":
            return

        # 3) 价值判断 + 三元组提取（默认有界线程池，不阻塞主请求）
        sync = self._cfg.async_extract is False if wait_extract is None else wait_extract
        if sync:
            self._extract_and_apply(user_id, content)
        else:
            self._submit_extract(user_id, content)

    def _submit_extract(self, user_id: str, content: str) -> None:
        if not self._extract_slots.acquire(blocking=False):
            self._dropped_extracts += 1
            logger.warning(
                "抽取队列已满，丢弃任务 user=%s dropped=%s",
                user_id,
                self._dropped_extracts,
            )
            self._audit_record(user_id, "extract_dropped", detail="queue_full")
            return

        def _job() -> None:
            try:
                self._extract_and_apply(user_id, content)
            finally:
                self._extract_slots.release()

        self._extract_pool.submit(_job)

    def _extract_and_apply(self, user_id: str, content: str) -> None:
        try:
            triples = self._extractor.extract(content)
            for t in triples:
                self._apply_triple(user_id, t, source=content)
        except Exception as exc:
            logger.warning("长期记忆抽取失败 user=%s: %s", user_id, exc)
            self._audit_record(user_id, "extract_error", detail=str(exc)[:200])

    def _apply_triple(self, user_id: str, triple: dict, source: str) -> None:
        """应用单个三元组：set 更新 / delete 遗忘。

        信息冲突解决的核心：subject+predicate 唯一，新值覆盖旧值。
        """
        subject = str(triple.get("subject", "")).strip()
        predicate = str(triple.get("predicate", "")).strip()
        obj = str(triple.get("object", "")).strip()
        action = str(triple.get("action", "set")).strip() or "set"

        if not subject or not predicate:
            return

        if action == "delete":
            self._long.delete(user_id, subject, predicate)
            # 画像字段同步清除
            if subject == "用户" and predicate in ("偏好", "身份", "计划"):
                self._profile.upsert(user_id, predicate, None)
            self._audit_record(
                user_id, "forget", subject=subject, predicate=predicate, object_value=obj
            )
            return

        # 【信息冲突】先删除旧的同 subject+predicate，再写入新的
        self._long.delete(user_id, subject, predicate)

        # 【画像同步】结构化偏好落画像库
        if subject == "用户" and predicate in ("偏好", "身份", "计划") and obj:
            self._profile.upsert(user_id, predicate, obj)
            self._audit_record(
                user_id, "profile_upsert", subject=subject, predicate=predicate, object_value=obj
            )

        mem = MemoryTriple(
            subject=subject,
            predicate=predicate,
            object=obj,
            source=source,
            timestamp=time.time(),
        )
        embedding = self._emb.embed_query(mem.to_text())
        self._long.add(user_id, mem, embedding)
        self._audit_record(
            user_id, "triple_upsert", subject=subject, predicate=predicate, object_value=obj
        )

    # ============================================================
    # 读取
    # ============================================================

    def get_context(self, user_id: str, session_id: str, current_input: str) -> str:
        """读取记忆上下文，拼接成 Prompt 前缀。

        流程：长期检索 + 画像（固定 prefix）+ 短期窗口（预算压缩）→ Prompt。
        """
        parts: list[str] = []

        query_vec = self._emb.embed_query(current_input)
        long_mems = self._long.search(user_id, query_vec, self._cfg.long_term_top_k)
        if long_mems:
            lines = "\n".join(
                f"- {m.subject} {m.predicate} {m.object}" for m in long_mems
            )
            parts.append(f"【长期记忆】\n{lines}")

        profile = {
            k: v for k, v in (self._profile.get_all(user_id) or {}).items()
            if v is not None and v != ""
        }
        if profile:
            plines = "\n".join(f"- {k}: {v}" for k, v in profile.items())
            parts.append(f"【用户画像】\n{plines}")

        short_msgs = self._short.recent(user_id, session_id, self._cfg.short_term_rounds)

        prefix = "\n\n".join(parts) + "\n\n" if parts else ""
        suffix = f"\n\n【当前输入】\n{current_input}"
        return self._compressor.fit_budget(short_msgs, prefix, suffix)
