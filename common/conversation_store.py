"""SQLite conversations + Redis List hot path (LTRIM count cap).

Short-term memory for the *model* is W2 ContextCompressor (token_budget /
fit_budget). This store only caps Redis by message *count*.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.env_loader import load_app_env

logger = logging.getLogger("conversation_store")

_REDIS_KEY = "cm:conv:{conv_id}:msgs"


def title_from_question(question: str, max_len: int = 20) -> str:
    text = " ".join(question.strip().split())
    if len(text) <= max_len:
        return text or "新对话"
    return text[:max_len] + "…"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    deleted_at: str | None = None
    user_id: str = ""


@dataclass
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    details: dict[str, Any] | None = None
    created_at: str = ""


@dataclass
class Favorite:
    id: str
    query: str
    answer: str
    created_at: str
    user_id: str = ""


def _row_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=row["id"],
        title=row["title"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        deleted_at=row["deleted_at"],
        user_id=row["user_id"] if "user_id" in row.keys() else "",
    )


def _row_message(row: sqlite3.Row) -> Message:
    raw = row["details_json"]
    details = None
    if raw:
        try:
            details = json.loads(raw)
        except json.JSONDecodeError:
            details = None
    return Message(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        details=details,
        created_at=row["created_at"],
    )


class ConversationStore:
    """SQLite is source of truth; Redis holds recent {role, content} for the graph."""

    def __init__(
        self,
        db_path: str | Path,
        redis_client: Any | None = None,
        redis_history_max: int | None = None,
        graph_history_limit: int | None = None,
    ) -> None:
        load_app_env()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.redis = redis_client
        self.redis_history_max = redis_history_max or int(os.getenv("REDIS_HISTORY_MAX", "50"))
        self.graph_history_limit = graph_history_limit or int(os.getenv("GRAPH_HISTORY_LIMIT", "6"))
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                details_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conv_created
                ON messages(conversation_id, created_at);
            CREATE TABLE IF NOT EXISTS favorites (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        self._add_column_if_missing("conversations", "user_id", "TEXT NOT NULL DEFAULT ''")
        self._add_column_if_missing("favorites", "user_id", "TEXT NOT NULL DEFAULT ''")
        self._conn.commit()

    def attach_orphans(self, user_id: str) -> None:
        """挂到种子管理员：旧行 user_id 为空时只给该用户看见。"""
        if not user_id:
            return
        self._conn.execute(
            "UPDATE conversations SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
            (user_id,),
        )
        self._conn.execute(
            "UPDATE favorites SET user_id = ? WHERE user_id IS NULL OR user_id = ''",
            (user_id,),
        )
        self._conn.commit()

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        cols = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _visible(self, conv: Conversation, user_id: str | None, is_admin: bool) -> bool:
        if is_admin:
            return True
        if not user_id:
            return True
        return (conv.user_id or "") == user_id

    def _redis_key(self, conv_id: str) -> str:
        return _REDIS_KEY.format(conv_id=conv_id)

    def create(self, title: str = "", user_id: str = "") -> Conversation:
        now = _utcnow()
        conv = Conversation(
            id=str(uuid.uuid4()),
            title=title or "新对话",
            created_at=now,
            updated_at=now,
            user_id=user_id or "",
        )
        self._conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, deleted_at, user_id) VALUES (?, ?, ?, ?, NULL, ?)",
            (conv.id, conv.title, conv.created_at, conv.updated_at, conv.user_id),
        )
        self._conn.commit()
        return conv

    def get(
        self,
        conv_id: str,
        include_deleted: bool = False,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> Conversation | None:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if row is None:
            return None
        conv = _row_conversation(row)
        if conv.deleted_at and not include_deleted:
            return None
        if not self._visible(conv, user_id, is_admin):
            return None
        return conv

    def list_conversations(
        self,
        keyword: str = "",
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[Conversation]:
        if user_id and not is_admin:
            rows = self._conn.execute(
                "SELECT * FROM conversations WHERE deleted_at IS NULL AND user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM conversations WHERE deleted_at IS NULL ORDER BY updated_at DESC"
            ).fetchall()
        convs = [_row_conversation(r) for r in rows]
        kw = keyword.strip().lower()
        if not kw:
            return convs
        return [c for c in convs if kw in (c.title or "").lower()]

    def soft_delete_many(
        self,
        ids: list[str],
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> int:
        n = 0
        for conv_id in ids:
            try:
                self.soft_delete(conv_id, user_id=user_id, is_admin=is_admin)
                n += 1
            except KeyError:
                continue
        return n

    def get_message(
        self,
        conv_id: str,
        message_id: str,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> Message | None:
        if self.get(conv_id, user_id=user_id, is_admin=is_admin) is None:
            raise KeyError(conv_id)
        row = self._conn.execute(
            "SELECT * FROM messages WHERE id = ? AND conversation_id = ?",
            (message_id, conv_id),
        ).fetchone()
        return _row_message(row) if row else None

    def update_message_content(self, conv_id: str, message_id: str, content: str) -> Message:
        msg = self.get_message(conv_id, message_id)
        if msg is None:
            raise KeyError(message_id)
        now = _utcnow()
        self._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ? AND conversation_id = ?",
            (content, message_id, conv_id),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id),
        )
        self._conn.commit()
        self._hydrate_redis_from_sqlite(conv_id)
        msg.content = content
        return msg

    def patch_message_details(self, conv_id: str, message_id: str, patch: dict[str, Any]) -> Message:
        msg = self.get_message(conv_id, message_id)
        if msg is None:
            raise KeyError(message_id)
        details = dict(msg.details or {})
        details.update(patch)
        now = _utcnow()
        self._conn.execute(
            "UPDATE messages SET details_json = ? WHERE id = ? AND conversation_id = ?",
            (json.dumps(details, ensure_ascii=False), message_id, conv_id),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id),
        )
        self._conn.commit()
        msg.details = details
        return msg

    def delete_messages(
        self,
        conv_id: str,
        ids: list[str],
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> int:
        if self.get(conv_id, user_id=user_id, is_admin=is_admin) is None:
            raise KeyError(conv_id)
        n = 0
        for message_id in ids:
            cur = self._conn.execute(
                "DELETE FROM messages WHERE id = ? AND conversation_id = ?",
                (message_id, conv_id),
            )
            n += cur.rowcount or 0
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utcnow(), conv_id),
        )
        self._conn.commit()
        self._hydrate_redis_from_sqlite(conv_id)
        return n

    def delete_messages_after(self, conv_id: str, message_id: str) -> int:
        msgs = self.get_messages(conv_id)
        idx = next((i for i, m in enumerate(msgs) if m.id == message_id), None)
        if idx is None:
            raise KeyError(message_id)
        return self.delete_messages(conv_id, [m.id for m in msgs[idx + 1 :]])

    def add_favorite(self, query: str, answer: str, user_id: str = "") -> Favorite:
        now = _utcnow()
        fav = Favorite(id=str(uuid.uuid4()), query=query, answer=answer, created_at=now, user_id=user_id or "")
        self._conn.execute(
            "INSERT INTO favorites (id, query, answer, created_at, user_id) VALUES (?, ?, ?, ?, ?)",
            (fav.id, fav.query, fav.answer, fav.created_at, fav.user_id),
        )
        self._conn.commit()
        return fav

    def list_favorites(
        self,
        keyword: str = "",
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[Favorite]:
        if user_id and not is_admin:
            rows = self._conn.execute(
                "SELECT * FROM favorites WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM favorites ORDER BY created_at DESC"
            ).fetchall()
        out = [
            Favorite(
                id=r["id"],
                query=r["query"],
                answer=r["answer"],
                created_at=r["created_at"],
                user_id=r["user_id"] if "user_id" in r.keys() else "",
            )
            for r in rows
        ]
        kw = keyword.strip().lower()
        if not kw:
            return out
        return [f for f in out if kw in f.query.lower() or kw in f.answer.lower()]

    def delete_favorite(
        self,
        fav_id: str,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> None:
        row = self._conn.execute("SELECT * FROM favorites WHERE id = ?", (fav_id,)).fetchone()
        if row is None:
            raise KeyError(fav_id)
        owner = row["user_id"] if "user_id" in row.keys() else ""
        if user_id and not is_admin and owner != user_id:
            raise KeyError(fav_id)
        self._conn.execute("DELETE FROM favorites WHERE id = ?", (fav_id,))
        self._conn.commit()

    def list_feedbacks(self, feedback: str = "") -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT m.id AS message_id, m.content AS answer, m.details_json, m.created_at,
                   c.id AS conversation_id, c.title, c.user_id
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.role = 'assistant' AND m.details_json IS NOT NULL
            ORDER BY m.created_at DESC
            """
        ).fetchall()
        want = feedback.strip().lower()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                details = json.loads(row["details_json"] or "{}")
            except json.JSONDecodeError:
                continue
            fb = str(details.get("feedback") or "")
            if not fb:
                continue
            if want and fb != want:
                continue
            prev = self._conn.execute(
                """
                SELECT content FROM messages
                WHERE conversation_id = ? AND role = 'user' AND created_at <= ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (row["conversation_id"], row["created_at"]),
            ).fetchone()
            refused = bool(details.get("refused"))
            evidence_gap = bool(details.get("evidence_gap") or refused)
            out.append({
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "title": row["title"] or "",
                "user_id": row["user_id"] or "",
                "question": prev["content"] if prev else "",
                "answer": row["answer"] or "",
                "feedback": fb,
                "refused": refused,
                "evidence_gap": evidence_gap,
                "created_at": row["created_at"],
            })
        return out

    def get_feedback_item(self, message_id: str) -> dict[str, Any] | None:
        mid = (message_id or "").strip()
        if not mid:
            return None
        for item in self.list_feedbacks():
            if item["message_id"] == mid:
                return item
        return None

    def rename(
        self,
        conv_id: str,
        title: str,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> Conversation:
        conv = self.get(conv_id, user_id=user_id, is_admin=is_admin)
        if conv is None:
            raise KeyError(conv_id)
        now = _utcnow()
        self._conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, conv_id),
        )
        self._conn.commit()
        conv.title = title
        conv.updated_at = now
        return conv

    def soft_delete(
        self,
        conv_id: str,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> None:
        conv = self.get(conv_id, user_id=user_id, is_admin=is_admin)
        if conv is None:
            raise KeyError(conv_id)
        now = _utcnow()
        self._conn.execute(
            "UPDATE conversations SET deleted_at = ?, updated_at = ? WHERE id = ?",
            (now, now, conv_id),
        )
        self._conn.commit()
        self._redis_delete(conv_id)

    def ensure(
        self,
        conv_id: str | None,
        first_question: str,
        user_id: str = "",
        is_admin: bool = False,
    ) -> Conversation:
        if conv_id:
            conv = self.get(conv_id, user_id=user_id, is_admin=is_admin)
            if conv is None:
                raise KeyError(conv_id)
            return conv
        return self.create(title=title_from_question(first_question), user_id=user_id)

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        details: dict[str, Any] | None = None,
    ) -> Message:
        conv = self.get(conv_id)
        if conv is None:
            raise KeyError(conv_id)
        now = _utcnow()
        msg = Message(
            id=str(uuid.uuid4()),
            conversation_id=conv_id,
            role=role,
            content=content,
            details=details,
            created_at=now,
        )
        details_json = json.dumps(details, ensure_ascii=False) if details else None
        self._conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, details_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg.id, conv_id, role, content, details_json, now),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conv_id),
        )
        self._conn.commit()
        self._redis_push(conv_id, {"role": role, "content": content})
        return msg

    def get_messages(
        self,
        conv_id: str,
        user_id: str | None = None,
        is_admin: bool = False,
    ) -> list[Message]:
        if self.get(conv_id, user_id=user_id, is_admin=is_admin) is None:
            raise KeyError(conv_id)
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        ).fetchall()
        return [_row_message(r) for r in rows]

    def get_recent_for_llm(self, conv_id: str) -> list[dict[str, str]]:
        """Hot window after Redis LTRIM (count). Token packing is ContextCompressor."""
        items = self._redis_lrange(conv_id)
        if not items:
            items = self._hydrate_redis_from_sqlite(conv_id)
        return items

    def _hydrate_redis_from_sqlite(self, conv_id: str) -> list[dict[str, str]]:
        rows = self._conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        ).fetchall()
        items = [{"role": r["role"], "content": r["content"]} for r in rows]
        if self.redis is not None:
            key = self._redis_key(conv_id)
            try:
                self.redis.delete(key)
                if items:
                    self.redis.rpush(key, *[json.dumps(x, ensure_ascii=False) for x in items])
                    self.redis.ltrim(key, -self.redis_history_max, -1)
            except Exception as exc:
                from common.obs import degraded

                degraded("redis_hydrate", exc)
                logger.warning("Redis hydrate 失败: %s", exc)
        return items[-self.redis_history_max :]

    def _redis_push(self, conv_id: str, payload: dict[str, str]) -> None:
        if self.redis is None:
            return
        key = self._redis_key(conv_id)
        try:
            self.redis.rpush(key, json.dumps(payload, ensure_ascii=False))
            self.redis.ltrim(key, -self.redis_history_max, -1)
        except Exception as exc:
            from common.obs import degraded

            degraded("redis_write", exc)
            logger.warning("Redis 写入失败，已落 SQLite: %s", exc)

    def _redis_lrange(self, conv_id: str) -> list[dict[str, str]]:
        if self.redis is None:
            return []
        try:
            raw = self.redis.lrange(self._redis_key(conv_id), 0, -1) or []
        except Exception as exc:
            from common.obs import degraded

            degraded("redis_read", exc)
            logger.warning("Redis 读取失败: %s", exc)
            return []
        out: list[dict[str, str]] = []
        for item in raw:
            try:
                obj = json.loads(item)
            except (TypeError, json.JSONDecodeError):
                continue
            role = str(obj.get("role", ""))
            content = str(obj.get("content", ""))
            if role and content:
                out.append({"role": role, "content": content})
        return out

    def _redis_delete(self, conv_id: str) -> None:
        if self.redis is None:
            return
        try:
            self.redis.delete(self._redis_key(conv_id))
        except Exception as exc:
            from common.obs import degraded

            degraded("redis_delete", exc)
            logger.warning("Redis 删除失败: %s", exc)


def default_store(redis_client: Any | None = None) -> ConversationStore:
    load_app_env()
    path = os.getenv("CONVERSATION_DB_PATH") or str(
        Path(__file__).resolve().parent.parent / "data" / "conversations.sqlite"
    )
    return ConversationStore(db_path=path, redis_client=redis_client)
