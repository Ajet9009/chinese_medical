"""SQLite users in the same file as conversations."""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from common.env_loader import load_app_env
from common.security import hash_password, verify_password

logger = logging.getLogger("user_store")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def parse_log_datetime(raw: str | None, *, end: bool = False) -> datetime | None:
    """解析查询时间。仅日期或无时区时按东八区；end 的日期取当天最后一刻。"""
    text = (raw or "").strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = f"{text}T23:59:59.999999" if end else f"{text}T00:00:00"
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (ValueError, OverflowError) as exc:
        raise ValueError("时间格式无效") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_SHANGHAI)
    return dt.astimezone(timezone.utc)


def parse_log_bound(raw: str | None, *, end: bool = False) -> str | None:
    dt = parse_log_datetime(raw, end=end)
    return dt.isoformat() if dt else None


@dataclass
class User:
    id: str
    username: str
    password_hash: str
    role: str
    status: str
    created_at: str
    dept: str = ""


def _row_user(row: sqlite3.Row) -> User:
    keys = row.keys()
    return User(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        role=row["role"] or "user",
        status=row["status"] or "active",
        created_at=row["created_at"],
        dept=row["dept"] if "dept" in keys and row["dept"] is not None else "",
    )


class UserStore:
    def __init__(self, db_path: str | Path) -> None:
        load_app_env()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                operate_type TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(users)").fetchall()]
        if "dept" not in cols:
            self._conn.execute("ALTER TABLE users ADD COLUMN dept TEXT NOT NULL DEFAULT ''")
        self._conn.commit()

    def seed_admin(self) -> User:
        username = os.getenv("ADMIN_USERNAME", "admin").strip() or "admin"
        password = os.getenv("ADMIN_PASSWORD", "admin123")
        existing = self.get_by_username(username)
        if existing:
            return existing
        user = self.create(username, password, role="admin")
        logger.info("已创建种子管理员: %s", username)
        return user

    def get(self, user_id: str) -> User | None:
        row = self._conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_user(row) if row else None

    def get_by_username(self, username: str) -> User | None:
        row = self._conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return _row_user(row) if row else None

    def authenticate(self, username: str, password: str) -> User | None:
        user = self.get_by_username(username)
        if user is None or user.status != "active":
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def create(self, username: str, password: str, role: str = "user") -> User:
        if role not in ("user", "admin"):
            raise ValueError("角色只能是 user 或 admin")
        if len(password) < 6:
            raise ValueError("密码至少 6 位")
        if self.get_by_username(username):
            raise ValueError("用户名已存在")
        now = _utcnow()
        user = User(
            id=str(uuid.uuid4()),
            username=username.strip(),
            password_hash=hash_password(password),
            role=role,
            status="active",
            created_at=now,
        )
        self._conn.execute(
            "INSERT INTO users (id, username, password_hash, role, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user.id, user.username, user.password_hash, user.role, user.status, user.created_at),
        )
        self._conn.commit()
        return user

    def list_users(self) -> list[User]:
        rows = self._conn.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
        return [_row_user(r) for r in rows]

    def _count_active_admins(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE role = 'admin' AND status = 'active'"
        ).fetchone()
        return int(row[0] if row else 0)

    def set_status(self, user_id: str, status: str, actor_id: str | None = None) -> User:
        if status not in ("active", "inactive"):
            raise ValueError("状态无效")
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        if actor_id and user_id == actor_id:
            raise ValueError("不能禁用自己")
        if (
            user.role == "admin"
            and status == "inactive"
            and self._count_active_admins() <= 1
        ):
            raise ValueError("不能禁用最后一个管理员")
        self._conn.execute("UPDATE users SET status = ? WHERE id = ?", (status, user_id))
        self._conn.commit()
        user.status = status
        return user

    def set_password(self, user_id: str, password: str) -> None:
        if len(password) < 6:
            raise ValueError("密码至少 6 位")
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        self._conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (hash_password(password), user_id),
        )
        self._conn.commit()

    def set_role(self, user_id: str, role: str, actor_id: str | None = None) -> User:
        if role not in ("user", "admin"):
            raise ValueError("角色只能是 user 或 admin")
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        if (
            user.role == "admin"
            and role != "admin"
            and self._count_active_admins() <= 1
        ):
            raise ValueError("不能降级最后一个管理员")
        self._conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        self._conn.commit()
        user.role = role
        return user

    def set_dept(self, user_id: str, dept: str) -> User:
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        dept = (dept or "").strip()
        self._conn.execute("UPDATE users SET dept = ? WHERE id = ?", (dept, user_id))
        self._conn.commit()
        user.dept = dept
        return user

    def delete_user(self, user_id: str, actor_id: str | None = None) -> None:
        user = self.get(user_id)
        if user is None:
            raise KeyError(user_id)
        if actor_id and user_id == actor_id:
            raise ValueError("不能删除自己")
        if user.role == "admin" and self._count_active_admins() <= 1:
            raise ValueError("不能删除最后一个管理员")
        self._conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self._conn.commit()

    def write_log(self, username: str, operate_type: str, content: str = "") -> None:
        self._conn.execute(
            """
            INSERT INTO audit_logs (id, username, operate_type, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), username, operate_type, content or "", _utcnow()),
        )
        self._conn.commit()

    def _log_filters(
        self,
        username: str = "",
        operate_type: str = "",
        start: str = "",
        end: str = "",
    ) -> tuple[str, list[str]]:
        clauses: list[str] = []
        args: list[str] = []
        name = (username or "").strip()
        if name:
            clauses.append("username LIKE ? ESCAPE '\\'")
            args.append(f"%{_like_escape(name)}%")
        kind = (operate_type or "").strip()
        if kind:
            clauses.append("operate_type = ?")
            args.append(kind)
        start_dt = parse_log_datetime(start, end=False)
        end_dt = parse_log_datetime(end, end=True)
        if start_dt and end_dt and start_dt > end_dt:
            raise ValueError("开始时间不能晚于结束时间")
        if start_dt:
            clauses.append("created_at >= ?")
            args.append(start_dt.isoformat())
        if end_dt:
            clauses.append("created_at <= ?")
            args.append(end_dt.isoformat())
        where = " AND ".join(clauses)
        return where, args

    def list_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        username: str = "",
        operate_type: str = "",
        start: str = "",
        end: str = "",
    ) -> dict:
        where, args = self._log_filters(username, operate_type, start, end)
        sql_where = f"WHERE {where}" if where else ""
        total = int(
            self._conn.execute(
                f"SELECT COUNT(*) FROM audit_logs {sql_where}", args
            ).fetchone()[0]
        )
        rows = self._conn.execute(
            f"SELECT * FROM audit_logs {sql_where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*args, max(1, min(int(limit), 200)), max(0, int(offset))),
        ).fetchall()
        types = [
            r[0]
            for r in self._conn.execute(
                "SELECT DISTINCT operate_type FROM audit_logs ORDER BY operate_type"
            ).fetchall()
        ]
        return {
            "total": total,
            "items": [
                {
                    "id": r["id"],
                    "username": r["username"],
                    "operate_type": r["operate_type"],
                    "content": r["content"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
            "operate_types": types,
        }


def default_db_path() -> Path:
    load_app_env()
    return Path(
        os.getenv("CONVERSATION_DB_PATH")
        or Path(__file__).resolve().parent.parent / "data" / "conversations.sqlite"
    )


def default_user_store() -> UserStore:
    return UserStore(default_db_path())
