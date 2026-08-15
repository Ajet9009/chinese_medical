#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆系统工厂：注入真实依赖（Redis / 向量库 / LLM / BGE / tiktoken）。

企业级特性：
  - 依赖降级：Redis 不可用 → 内存；Milvus 不可用 → JSON；MySQL 不可用 → JSON
  - 单例复用：BGE 模型、Redis 客户端全局单例，避免重复加载
  - 持久化：长期记忆优先 Milvus；画像优先 MySQL；均可落盘降级

用法:
    from _008_memory.factory import create_memory_manager
    mm = create_memory_manager()
    mm.add_message("u1", "s1", "user", "我最近失眠")
    ctx = mm.get_context("u1", "s1", "吃什么能改善睡眠")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .memory_manager import LLM, Embeddings, MemoryManager, Tokenizer
from .stores import (
    DictProfileStore,
    JsonProfileStore,
    LongTermStore,
    MemoryTriple,
    Message,
    ProfileStore,
    RedisShortTermStore,
    ShortTermStore,
    short_term_key,
)

logger = logging.getLogger("memory.factory")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# 记忆数据落盘位置（企业级：重启不丢）
MEMORY_DATA_DIR = ROOT / "data" / "memory"

# ============================================================
# 依赖单例（延迟加载，避免启动阻塞）
# ============================================================

_bge_model = None
_bge_lock = threading.Lock()
_redis_client = None


def _get_redis_client():
    """Redis 客户端单例。不可用返回 None（触发内存 fallback）。"""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            password = os.getenv("REDIS_PASSWORD", "") or None
            _redis_client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=password,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            _redis_client.ping()  # 立即验证
        except Exception as exc:
            logger.warning("Redis 不可用，降级为内存短期记忆: %s", exc)
            _redis_client = None
    return _redis_client


def _create_long_term_store(embeddings: Embeddings) -> LongTermStore:
    """优先 Milvus，失败回退 JSON 向量库。"""
    prefer = os.getenv("MEMORY_LONG_BACKEND", "milvus").strip().lower()
    if prefer in ("milvus", "auto"):
        try:
            from .backends import MilvusLongTermStore
            dim = int(os.getenv("MEMORY_EMBED_DIM", "1024"))
            store = MilvusLongTermStore(
                host=os.getenv("MILVUS_HOST", "127.0.0.1"),
                port=int(os.getenv("MILVUS_PORT", "19530")),
                database=os.getenv("MILVUS_DATABASE", "itcast"),
                # 独立 collection，避免覆盖业务 edurag
                collection=os.getenv("MILVUS_MEMORY_COLLECTION", "agent_long_term_memory"),
                dim=dim,
            )
            logger.info("长期记忆后端: Milvus")
            return store
        except Exception as exc:
            logger.warning("Milvus 不可用，长期记忆降级 JSON: %s", exc)
    store = JsonLongTermStore(MEMORY_DATA_DIR, embeddings)
    logger.info("长期记忆后端: JsonLongTermStore")
    return store


def _create_profile_store() -> ProfileStore:
    """优先 MySQL，失败回退 JSON 画像。"""
    prefer = os.getenv("MEMORY_PROFILE_BACKEND", "mysql").strip().lower()
    if prefer in ("mysql", "auto"):
        try:
            from .backends import MySQLProfileStore
            store = MySQLProfileStore(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "subjects_kg"),
                table=os.getenv("MYSQL_PROFILE_TABLE", "agent_user_profile"),
                pool_size=int(os.getenv("MYSQL_POOL_SIZE", "4")),
            )
            logger.info("画像后端: MySQL")
            return store
        except Exception as exc:
            logger.warning("MySQL 不可用，画像降级 JSON: %s", exc)
    store = JsonProfileStore(MEMORY_DATA_DIR)
    logger.info("画像后端: JsonProfileStore")
    return store


def _create_audit_store():
    """优先 MySQL 审计表，失败回退 JSONL。"""
    prefer = os.getenv("MEMORY_AUDIT_BACKEND", "mysql").strip().lower()
    if prefer in ("mysql", "auto"):
        try:
            from .backends import MySQLAuditStore
            store = MySQLAuditStore(
                host=os.getenv("MYSQL_HOST", "localhost"),
                port=int(os.getenv("MYSQL_PORT", "3306")),
                user=os.getenv("MYSQL_USER", "root"),
                password=os.getenv("MYSQL_PASSWORD", ""),
                database=os.getenv("MYSQL_DATABASE", "subjects_kg"),
                table=os.getenv("MYSQL_AUDIT_TABLE", "agent_memory_audit"),
                pool_size=int(os.getenv("MYSQL_AUDIT_POOL_SIZE", "2")),
            )
            logger.info("审计后端: MySQL")
            return store
        except Exception as exc:
            logger.warning("MySQL 审计不可用，降级 JSONL: %s", exc)
    if prefer in ("off", "none", "false", "0"):
        logger.info("审计后端: 关闭")
        return None
    from .backends import JsonAuditStore
    store = JsonAuditStore(MEMORY_DATA_DIR)
    logger.info("审计后端: JsonAuditStore")
    return store


def _get_bge_model():
    """BGE embedding 模型单例（延迟加载，首次调用加载权重）。"""
    global _bge_model
    with _bge_lock:
        if _bge_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_path = os.getenv("EMBEDDING_MODEL_PATH")
                if not model_path:
                    raise ValueError("EMBEDDING_MODEL_PATH 未配置")
                _bge_model = SentenceTransformer(model_path)
            except Exception as exc:
                logger.warning("BGE 模型加载失败: %s", exc)
                _bge_model = None
        return _bge_model


# ============================================================
# LLM / Embedding / Tokenizer 适配器
# ============================================================

class ChatLLMAdapter:
    """把 ChatOpenAI 包装成 LLM 接口（invoke(str) -> str）。"""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def invoke(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        resp = self._llm.invoke([HumanMessage(content=prompt)])
        return str(resp.content)


class BgeEmbeddings:
    """BGE embedding 适配器：text -> 向量。"""

    def embed_query(self, text: str) -> list[float]:
        model = _get_bge_model()
        if model is None:
            # 降级：返回零向量（检索会退化，但不阻塞）
            return [0.0] * 1024
        vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return vec[0].tolist()


class TiktokenTokenizer:
    """tiktoken 适配器：计算 token 数。"""

    def __init__(self, model_name: str = "cl100k_base") -> None:
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(model_name)
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # 降级：粗略估算（中文 1 字 ≈ 1 token，英文 1 词 ≈ 1.3 token）
        return len(text)


# ============================================================
# 长期记忆：JSON 持久化向量库（替代 FAISS，支持 metadata 过滤）
# ============================================================

class JsonLongTermStore:
    """JSON 文件持久化的长期记忆向量库。

    按 user_id 隔离，numpy 余弦相似度检索，支持 delete（遗忘）。
    接口与 LangChainLongTermStore 一致，可随时替换为 Milvus/Chroma。
    """

    def __init__(self, data_dir: Path, embeddings: Embeddings) -> None:
        self._data_dir = data_dir
        self._emb = embeddings
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "long_term.json"
        self._memory: dict[str, list[dict]] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> dict[str, list[dict]]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("长期记忆文件损坏，重新初始化")
        return {}

    def _save(self) -> None:
        try:
            self._file.write_text(
                json.dumps(self._memory, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("长期记忆落盘失败: %s", exc)

    def add(self, user_id: str, triple: MemoryTriple, embedding: list[float]) -> None:
        with self._lock:
            self._memory.setdefault(user_id, []).append({
                "subject": triple.subject,
                "predicate": triple.predicate,
                "object": triple.object,
                "source": triple.source,
                "timestamp": triple.timestamp,
                "embedding": embedding,
            })
            self._save()

    def search(self, user_id: str, query_embedding: list[float], top_k: int) -> list[MemoryTriple]:
        items = self._memory.get(user_id, [])
        if not items:
            return []
        q = np.asarray(query_embedding, dtype="float32")
        results = []
        for it in items:
            emb = np.asarray(it.get("embedding", [0.0] * len(q)), dtype="float32")
            # 余弦相似度（embedding 已归一化，点积即余弦）
            sim = float(np.dot(q, emb)) if len(emb) == len(q) and np.any(emb) else 0.0
            results.append((sim, it))
        results.sort(key=lambda x: x[0], reverse=True)
        return [
            MemoryTriple(
                subject=it["subject"], predicate=it["predicate"], object=it["object"],
                source=it.get("source", ""), timestamp=it.get("timestamp", 0.0),
            )
            for sim, it in results[:top_k] if sim > 0.0
        ]

    def delete(self, user_id: str, subject: str, predicate: str) -> int:
        with self._lock:
            items = self._memory.get(user_id, [])
            before = len(items)
            self._memory[user_id] = [
                it for it in items
                if not (it["subject"] == subject and it["predicate"] == predicate)
            ]
            removed = before - len(self._memory[user_id])
            if removed:
                self._save()
            return removed


class InMemoryShortTermStore:
    """内存短期记忆（Redis 降级 fallback）。键 = user_id:session_id。"""

    def __init__(self, max_rounds: int = 20) -> None:
        self._max = max_rounds
        self._data: dict[str, list[Message]] = {}

    def append(self, user_id: str, session_id: str, message: Message) -> None:
        key = short_term_key(user_id, session_id)
        self._data.setdefault(key, []).append(message)
        self._data[key] = self._data[key][-self._max:]

    def recent(self, user_id: str, session_id: str, n: int) -> list[Message]:
        key = short_term_key(user_id, session_id)
        return self._data.get(key, [])[-n:]

    def trim(self, user_id: str, session_id: str, keep: int) -> None:
        key = short_term_key(user_id, session_id)
        self._data[key] = self._data.get(key, [])[-keep:]


# ============================================================
# 工厂
# ============================================================

_memory_manager: MemoryManager | None = None
_factory_lock = threading.Lock()


def create_llm():
    """复用项目 LLM（ChatOpenAI）。"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL"),
        temperature=0,
    )


def create_memory_manager() -> MemoryManager:
    """创建 MemoryManager（全局单例，惰性初始化）。"""
    global _memory_manager
    with _factory_lock:
        if _memory_manager is not None:
            return _memory_manager

        from .memory_manager import MemoryConfig

        embeddings = BgeEmbeddings()
        tokenizer = TiktokenTokenizer()
        llm = ChatLLMAdapter(create_llm())

        ttl = int(os.getenv("MEMORY_SHORT_TTL_SECONDS", str(7 * 24 * 3600)))
        max_rounds = int(os.getenv("MEMORY_SHORT_ROUNDS", "20"))

        # 短期记忆：Redis 优先，降级内存
        redis_client = _get_redis_client()
        short_term: ShortTermStore = (
            RedisShortTermStore(redis_client, max_rounds=max_rounds, ttl_seconds=ttl)
            if redis_client is not None
            else InMemoryShortTermStore(max_rounds=max_rounds)
        )

        # 长期记忆：Milvus 优先 → JSON
        long_term: LongTermStore = _create_long_term_store(embeddings)

        # 画像：MySQL 优先 → JSON
        profile: ProfileStore = _create_profile_store()

        async_extract = os.getenv("MEMORY_ASYNC_EXTRACT", "true").strip().lower() not in (
            "false", "0", "no", "off",
        )
        extract_workers = int(os.getenv("MEMORY_EXTRACT_WORKERS", "2"))
        extract_queue_size = int(os.getenv("MEMORY_EXTRACT_QUEUE_SIZE", "64"))

        audit = _create_audit_store()

        _memory_manager = MemoryManager(
            short_term=short_term,
            long_term=long_term,
            profile=profile,
            llm=llm,
            embeddings=embeddings,
            tokenizer=tokenizer,
            config=MemoryConfig(
                short_term_rounds=max_rounds,
                async_extract=async_extract,
                extract_workers=extract_workers,
                extract_queue_size=extract_queue_size,
            ),
            audit=audit,
        )
        logger.info(
            "MemoryManager 初始化完成 short=%s long=%s profile=%s audit=%s async_extract=%s",
            type(short_term).__name__,
            type(long_term).__name__,
            type(profile).__name__,
            type(audit).__name__ if audit else None,
            async_extract,
        )
        return _memory_manager


def get_memory_manager() -> MemoryManager | None:
    """获取 MemoryManager（初始化失败返回 None，不阻塞主流程）。"""
    try:
        return create_memory_manager()
    except Exception as exc:
        logger.warning("MemoryManager 初始化失败，记忆功能降级: %s", exc)
        return None


def reset_memory_manager() -> None:
    """重置单例（仅测试用）。"""
    global _memory_manager
    with _factory_lock:
        _memory_manager = None
