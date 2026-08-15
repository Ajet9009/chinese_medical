#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生产级后端：MySQL 画像/审计 + Milvus 长期记忆。

依赖降级：连接失败时由 factory 回退 JsonProfileStore / JsonLongTermStore。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .stores import MemoryTriple

logger = logging.getLogger("memory.backends")


def _milvus_escape(value: str) -> str:
    """转义 Milvus 布尔表达式中的字符串字面量，防止过滤注入。"""
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


class _MySQLPool:
    """轻量连接池（避免每次 upsert 新建 TCP）。"""

    def __init__(self, cfg: dict[str, Any], size: int = 4) -> None:
        import pymysql

        self._cfg = cfg
        self._size = max(1, size)
        self._pool: queue.Queue = queue.Queue(maxsize=self._size)
        self._created = 0
        self._lock = threading.Lock()
        self._pymysql = pymysql

    def _new(self):
        return self._pymysql.connect(**self._cfg)

    def acquire(self):
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._created < self._size:
                    self._created += 1
                    return self._new()
            return self._pool.get(timeout=5)

    def release(self, conn) -> None:
        if conn is None:
            return
        try:
            conn.ping(reconnect=True)
            self._pool.put_nowait(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created = max(0, self._created - 1)

    def close_all(self) -> None:
        while True:
            try:
                conn = self._pool.get_nowait()
            except queue.Empty:
                break
            try:
                conn.close()
            except Exception:
                pass


class MySQLProfileStore:
    """MySQL 用户画像（subjects_kg.agent_user_profile）。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "subjects_kg",
        table: str = "agent_user_profile",
        pool_size: int = 4,
    ) -> None:
        self._cfg = dict(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=5,
        )
        self._table = table
        self._pool = _MySQLPool(self._cfg, size=pool_size)
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS `{self._table}` (
                      `user_id` VARCHAR(64) NOT NULL,
                      `pref_key` VARCHAR(64) NOT NULL,
                      `pref_value` TEXT NULL,
                      `updated_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                      PRIMARY KEY (`user_id`, `pref_key`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
        finally:
            self._pool.release(conn)
        logger.info("MySQLProfileStore ready db=%s table=%s pool=%s", database, table, pool_size)

    def upsert(self, user_id: str, key: str, value: Any) -> None:
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                if value is None:
                    cur.execute(
                        f"DELETE FROM `{self._table}` WHERE user_id=%s AND pref_key=%s",
                        (user_id, key),
                    )
                else:
                    cur.execute(
                        f"""
                        INSERT INTO `{self._table}` (user_id, pref_key, pref_value)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE pref_value=VALUES(pref_value)
                        """,
                        (user_id, key, str(value)),
                    )
        finally:
            self._pool.release(conn)

    def get(self, user_id: str, key: str) -> Any | None:
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT pref_value FROM `{self._table}` WHERE user_id=%s AND pref_key=%s",
                    (user_id, key),
                )
                row = cur.fetchone()
                return row[0] if row else None
        finally:
            self._pool.release(conn)

    def get_all(self, user_id: str) -> dict[str, Any]:
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT pref_key, pref_value FROM `{self._table}` WHERE user_id=%s",
                    (user_id,),
                )
                return {k: v for k, v in cur.fetchall() if v is not None}
        finally:
            self._pool.release(conn)

    def ping(self) -> bool:
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            return False
        finally:
            self._pool.release(conn)


class MySQLAuditStore:
    """记忆变更审计（医疗合规：谁改了什么，不含完整原文 PHI）。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "subjects_kg",
        table: str = "agent_memory_audit",
        pool_size: int = 2,
    ) -> None:
        self._cfg = dict(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset="utf8mb4",
            autocommit=True,
            connect_timeout=5,
        )
        self._table = table
        self._pool = _MySQLPool(self._cfg, size=pool_size)
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS `{self._table}` (
                      `id` BIGINT NOT NULL AUTO_INCREMENT,
                      `user_id` VARCHAR(64) NOT NULL,
                      `action` VARCHAR(32) NOT NULL,
                      `subject` VARCHAR(128) NULL,
                      `predicate` VARCHAR(128) NULL,
                      `object_preview` VARCHAR(64) NULL,
                      `detail` VARCHAR(255) NULL,
                      `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      PRIMARY KEY (`id`),
                      KEY `idx_user_time` (`user_id`, `created_at`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                    """
                )
        finally:
            self._pool.release(conn)
        logger.info("MySQLAuditStore ready table=%s", table)

    def record(
        self,
        user_id: str,
        action: str,
        *,
        subject: str = "",
        predicate: str = "",
        object_value: str = "",
        detail: str = "",
    ) -> None:
        preview = (object_value or "")[:64]
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO `{self._table}`
                      (user_id, action, subject, predicate, object_preview, detail)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        action[:32],
                        (subject or "")[:128],
                        (predicate or "")[:128],
                        preview,
                        (detail or "")[:255],
                    ),
                )
        finally:
            self._pool.release(conn)

    def ping(self) -> bool:
        conn = self._pool.acquire()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            return False
        finally:
            self._pool.release(conn)


class JsonAuditStore:
    """审计 JSONL 降级（MySQL 不可用时）。"""

    def __init__(self, data_dir: Any) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "memory_audit.jsonl"
        self._lock = threading.Lock()

    def record(
        self,
        user_id: str,
        action: str,
        *,
        subject: str = "",
        predicate: str = "",
        object_value: str = "",
        detail: str = "",
    ) -> None:
        row = {
            "ts": time.time(),
            "user_id": user_id,
            "action": action,
            "subject": subject,
            "predicate": predicate,
            "object_preview": (object_value or "")[:64],
            "detail": (detail or "")[:255],
        }
        with self._lock:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def ping(self) -> bool:
        return self._dir.exists()


class MilvusLongTermStore:
    """Milvus 长期记忆三元组（独立 collection，不占用业务 edurag）。"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 19530,
        database: str = "default",
        collection: str = "agent_long_term_memory",
        dim: int = 1024,
        alias: str = "memory",
    ) -> None:
        from pymilvus import (
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            connections,
            db,
            utility,
        )

        self._alias = alias
        self._collection_name = collection
        self._dim = dim
        self._lock = threading.Lock()

        connections.connect(alias=alias, host=host, port=str(port))

        try:
            existing = db.list_database(using=alias)
            if database not in existing:
                db.create_database(database, using=alias)
            db.using_database(database, using=alias)
        except Exception as exc:
            logger.warning("Milvus database 切换跳过(%s): %s", database, exc)

        if not utility.has_collection(collection, using=alias):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="predicate", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="object", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="timestamp", dtype=DataType.DOUBLE),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="agent long-term memory triples")
            col = Collection(name=collection, schema=schema, using=alias)
            col.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 128},
                },
            )
            try:
                col.create_index(
                    field_name="user_id",
                    index_params={"index_type": "TRIE"},
                )
            except Exception as exc:
                logger.warning("Milvus user_id 标量索引跳过: %s", exc)
            logger.info("Milvus collection created: %s", collection)
        self._col = Collection(collection, using=alias)
        self._col.load()
        logger.info("MilvusLongTermStore ready collection=%s dim=%s", collection, dim)

    def add(self, user_id: str, triple: MemoryTriple, embedding: list[float]) -> None:
        with self._lock:
            vec = list(embedding)
            if len(vec) < self._dim:
                vec = vec + [0.0] * (self._dim - len(vec))
            elif len(vec) > self._dim:
                vec = vec[: self._dim]
            row_id = uuid.uuid4().hex
            self._col.insert([
                [row_id],
                [user_id],
                [triple.subject[:128]],
                [triple.predicate[:128]],
                [triple.object[:512]],
                [(triple.source or "")[:1024]],
                [float(triple.timestamp or time.time())],
                [vec],
            ])
            self._col.flush()

    def search(self, user_id: str, query_embedding: list[float], top_k: int) -> list[MemoryTriple]:
        vec = list(query_embedding)
        if len(vec) < self._dim:
            vec = vec + [0.0] * (self._dim - len(vec))
        elif len(vec) > self._dim:
            vec = vec[: self._dim]

        expr = f'user_id == "{_milvus_escape(user_id)}"'
        res = self._col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=max(top_k * 5, top_k),
            expr=expr,
            output_fields=["subject", "predicate", "object", "source", "timestamp"],
        )
        out: list[MemoryTriple] = []
        for hits in res:
            for hit in hits:
                entity = hit.entity
                out.append(
                    MemoryTriple(
                        subject=entity.get("subject") or "",
                        predicate=entity.get("predicate") or "",
                        object=entity.get("object") or "",
                        source=entity.get("source") or "",
                        timestamp=float(entity.get("timestamp") or 0.0),
                    )
                )
                if len(out) >= top_k:
                    break
        return out

    def delete(self, user_id: str, subject: str, predicate: str) -> int:
        with self._lock:
            expr = (
                f'user_id == "{_milvus_escape(user_id)}" and '
                f'subject == "{_milvus_escape(subject)}" and '
                f'predicate == "{_milvus_escape(predicate)}"'
            )
            rows = self._col.query(expr=expr, output_fields=["id"])
            if not rows:
                return 0
            self._col.delete(expr)
            self._col.flush()
            return len(rows)

    def ping(self) -> bool:
        try:
            self._col.num_entities
            return True
        except Exception:
            return False
