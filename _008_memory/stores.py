#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分层存储抽象：短期（Redis）、长期（向量库）、画像（结构化 DB）。

用 Protocol 定义接口，实现可注入替换，便于测试与切换后端。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Message:
    """单条对话消息。"""
    role: str                       # user / assistant
    content: str
    timestamp: float = field(default_factory=lambda: 0.0)


@dataclass
class MemoryTriple:
    """长期记忆三元组：主语-谓语-宾语，附来源与时间。

    解决"幻觉"的关键：每条记忆都带 source（来源消息）与 timestamp，
    检索时可追溯，避免凭空生成。
    """
    subject: str                    # 实体（如 "用户"、"银行卡"）
    predicate: str                  # 关系（如 "偏好"、"拥有"、"不使用"）
    object: str                     # 值（如 "喝咖啡"、"尾号1234"）
    source: str = ""                # 来源消息，供追溯，防幻觉
    timestamp: float = 0.0          # 写入时间，冲突时取最新
    confidence: float = 1.0         # 置信度，低置信可被覆盖

    def to_text(self) -> str:
        """三元组序列化为可向量化文本。"""
        return f"{self.subject} {self.predicate} {self.object}"


# ============================================================
# 存储接口（Protocol）
# ============================================================

def short_term_key(user_id: str, session_id: str) -> str:
    """短期记忆键：必须 user_id + session_id 联合隔离，防止同 session 串用户。"""
    return f"mem:short:{user_id}:{session_id}"


class ShortTermStore(Protocol):
    """短期记忆：Redis 滑动窗口（按 user_id + session_id 隔离）。"""

    def append(self, user_id: str, session_id: str, message: Message) -> None:
        """追加一条消息到会话窗口末尾。"""
        ...

    def recent(self, user_id: str, session_id: str, n: int) -> list[Message]:
        """返回最近 n 条消息（时间序）。"""
        ...

    def trim(self, user_id: str, session_id: str, keep: int) -> None:
        """裁剪窗口，只保留最近 keep 条。"""
        ...


class LongTermStore(Protocol):
    """长期记忆：向量库三元组。"""

    def add(self, user_id: str, triple: MemoryTriple, embedding: list[float]) -> None:
        """写入三元组及其向量。"""
        ...

    def search(self, user_id: str, query_embedding: list[float], top_k: int) -> list[MemoryTriple]:
        """按语义相似度检索 top_k 条三元组。"""
        ...

    def delete(self, user_id: str, subject: str, predicate: str) -> int:
        """删除匹配 subject+predicate 的旧三元组，返回删除数（遗忘）。"""
        ...


class ProfileStore(Protocol):
    """用户画像：结构化偏好。"""

    def upsert(self, user_id: str, key: str, value: Any) -> None:
        """写入/更新一个偏好字段。"""
        ...

    def get(self, user_id: str, key: str) -> Any | None:
        """读取一个偏好字段。"""
        ...

    def get_all(self, user_id: str) -> dict[str, Any]:
        """读取全部偏好。"""
        ...


# ============================================================
# Redis 短期记忆实现
# ============================================================

class RedisShortTermStore:
    """Redis 滑动窗口实现。

    用 Redis List（LPUSH + LTRIM）维护最近 N 条，天然滑动窗口。
    Key = user_id:session_id，并设置 TTL，避免会话键无限堆积。
    """

    def __init__(
        self,
        redis_client: Any,
        max_rounds: int = 20,
        ttl_seconds: int = 7 * 24 * 3600,
    ) -> None:
        self._r = redis_client
        self._max_rounds = max_rounds
        self._ttl = ttl_seconds

    def append(self, user_id: str, session_id: str, message: Message) -> None:
        import json
        key = short_term_key(user_id, session_id)
        payload = json.dumps({
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
        }, ensure_ascii=False)
        pipe = self._r.pipeline()
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, self._max_rounds - 1)
        if self._ttl > 0:
            pipe.expire(key, self._ttl)
        pipe.execute()

    def recent(self, user_id: str, session_id: str, n: int) -> list[Message]:
        import json
        key = short_term_key(user_id, session_id)
        items = self._r.lrange(key, 0, n - 1)
        messages = []
        for raw in reversed(items):
            d = json.loads(raw)
            messages.append(Message(d["role"], d["content"], d.get("timestamp", 0.0)))
        return messages

    def trim(self, user_id: str, session_id: str, keep: int) -> None:
        key = short_term_key(user_id, session_id)
        self._r.ltrim(key, 0, keep - 1)

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


# ============================================================
# LangChain 向量库长期记忆实现
# ============================================================

class LangChainLongTermStore:
    """基于 LangChain VectorStore 的长期记忆。

    三元组序列化为文本后向量化，存 vectorstore。
    delete 通过 metadata 过滤实现遗忘。
    """

    def __init__(self, vectorstore: Any, embeddings: Any) -> None:
        self._vs = vectorstore
        self._emb = embeddings

    def add(self, user_id: str, triple: MemoryTriple, embedding: list[float]) -> None:
        self._vs.add_texts(
            texts=[triple.to_text()],
            metadatas=[{
                "user_id": user_id,
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "source": triple.source,
                "timestamp": triple.timestamp,
            }],
        )

    def search(self, user_id: str, query_embedding: list[float], top_k: int) -> list[MemoryTriple]:
        docs = self._vs.similarity_search_by_vector(
            query_embedding,
            k=top_k,
            filter={"user_id": user_id},  # 按用户隔离，避免串记忆
        )
        triples = []
        for d in docs:
            m = d.metadata
            triples.append(MemoryTriple(
                subject=m.get("subject", ""),
                predicate=m.get("predicate", ""),
                object=m.get("object", ""),
                source=m.get("source", ""),
                timestamp=m.get("timestamp", 0.0),
            ))
        return triples

    def delete(self, user_id: str, subject: str, predicate: str) -> int:
        # 通过 metadata 过滤删除，实现遗忘
        ids = self._vs.get_document_ids(
            filter={"user_id": user_id, "subject": subject, "predicate": predicate}
        )
        if ids:
            self._vs.delete(ids=ids)
        return len(ids)


# ============================================================
# 结构化用户画像实现（Postgres 抽象，可换 Mongo）
# ============================================================

class DictProfileStore:
    """内存字典实现（测试 / 临时 fallback；生产用 JsonProfileStore 或 DB）。"""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}

    def upsert(self, user_id: str, key: str, value: Any) -> None:
        self._data.setdefault(user_id, {})
        if value is None:
            self._data[user_id].pop(key, None)
        else:
            self._data[user_id][key] = value

    def get(self, user_id: str, key: str) -> Any | None:
        return self._data.get(user_id, {}).get(key)

    def get_all(self, user_id: str) -> dict[str, Any]:
        return dict(self._data.get(user_id, {}))


class JsonProfileStore:
    """JSON 落盘用户画像（重启不丢；接口可替换为 Postgres/Mongo）。"""

    def __init__(self, data_dir: Any) -> None:
        from pathlib import Path
        import threading

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "profile.json"
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        import json
        if self._file.exists():
            try:
                raw = json.loads(self._file.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self) -> None:
        import json
        try:
            self._file.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError:
            pass

    def upsert(self, user_id: str, key: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault(user_id, {})
            if value is None:
                self._data[user_id].pop(key, None)
            else:
                self._data[user_id][key] = value
            self._save()

    def get(self, user_id: str, key: str) -> Any | None:
        return self._data.get(user_id, {}).get(key)

    def get_all(self, user_id: str) -> dict[str, Any]:
        return dict(self._data.get(user_id, {}))
