"""Knowledge base: local files + SQLite registry + FAISS rebuild."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common.doc_chunking import build_document_chunks_step_04
from common.env_loader import load_app_env
from common.knowledge_acl import acl_ok
from common.knowledge_governance import (
    effective_state,
    is_retrievable,
    lifecycle_findings,
    missing_fields,
)

load_app_env()
logger = logging.getLogger("knowledge")

DOC_TYPES = ("方剂", "本草", "典籍", "医案", "其他")
ALLOWED_EXT = {".md", ".txt", ".pdf"}
MAX_FILES = 5
MAX_BYTES = int(os.getenv("DOC_UPLOAD_MAX_BYTES", str(8 * 1024 * 1024)))
TCM_TERMS = (
    "四君子汤", "桂枝汤", "人参", "白术", "茯苓", "甘草", "黄芪", "当归",
    "熟地", "川芎", "白芍", "陈皮", "半夏", "柴胡", "黄芩", "附子",
)

EncodeFn = Callable[[list[str]], np.ndarray]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _docs_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    return Path(os.getenv("KNOWLEDGE_DOCS_DIR") or root / "data" / "docs")


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def _safe_name(name: str) -> str:
    raw = (name or "").replace("\\", "/")
    base = Path(raw).name
    if (not base) or ".." in raw or "/" in raw:
        raise ValueError("非法文件名")
    return base


def _herb_tags(text: str) -> str:
    found = [t for t in TCM_TERMS if t and t in (text or "")]
    return ",".join(found[:20])


class KnowledgeService:
    def __init__(self, db_path: str | Path, encode_fn: EncodeFn | None = None) -> None:
        load_app_env()
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.docs_dir = _docs_root()
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._encode_fn = encode_fn
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        """步骤 01：创建知识库表，并补齐旧库缺失的 chunk metadata 列。"""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                doc_name TEXT NOT NULL UNIQUE,
                doc_type TEXT NOT NULL DEFAULT '其他',
                file_path TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                chunk_count INTEGER DEFAULT 0,
                upload_user TEXT DEFAULT '',
                herb_tags TEXT DEFAULT '',
                dept TEXT DEFAULT '',
                allowed_roles TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_document_versions (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER DEFAULT 0,
                created_by TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                text TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_metadata (
                doc_id TEXT PRIMARY KEY,
                owner TEXT DEFAULT '',
                applicable_region TEXT DEFAULT '',
                effective_at TEXT,
                expires_at TEXT,
                is_permanent INTEGER DEFAULT 0,
                review_interval_days INTEGER,
                next_review_at TEXT,
                version_label TEXT DEFAULT '',
                version_status TEXT DEFAULT 'draft',
                created_by TEXT DEFAULT '',
                updated_by TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS kb_issues (
                id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL UNIQUE,
                issue_type TEXT NOT NULL,
                severity TEXT DEFAULT 'warning',
                status TEXT DEFAULT 'open',
                doc_id TEXT NOT NULL,
                title TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                evidence_json TEXT DEFAULT '{}',
                occurrence_count INTEGER DEFAULT 1,
                detected_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                reviewer TEXT DEFAULT '',
                review_note TEXT DEFAULT '',
                reviewed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS kb_reviews (
                id TEXT PRIMARY KEY,
                issue_id TEXT NOT NULL,
                from_status TEXT DEFAULT '',
                to_status TEXT NOT NULL,
                reviewer TEXT NOT NULL,
                note TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_chunk_metadata_columns_step_02()
        self._conn.commit()

    def _ensure_chunk_metadata_columns_step_02(self) -> None:
        """步骤 02：为旧版 kb_chunks 表追加父子分块 metadata 列。"""
        existing = {
            str(row["name"])
            for row in self._conn.execute("PRAGMA table_info(kb_chunks)").fetchall()
        }
        columns = {
            "parent_id": "TEXT DEFAULT ''",
            "parent_title": "TEXT DEFAULT ''",
            "section_path": "TEXT DEFAULT '[]'",
            "text_simplified": "TEXT DEFAULT ''",
            "text_traditional": "TEXT DEFAULT ''",
            "title_simplified": "TEXT DEFAULT ''",
            "title_traditional": "TEXT DEFAULT ''",
            "parent_text_preview": "TEXT DEFAULT ''",
        }
        for name, ddl in columns.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE kb_chunks ADD COLUMN {name} {ddl}")

    def _doc_dir(self, doc_id: str) -> Path:
        path = self.docs_dir / doc_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload(
        self,
        files: list[tuple[str, bytes]],
        doc_type: str = "其他",
        username: str = "",
        dept: str = "",
        allowed_roles: str = "",
        *,
        effective_at: str = "",
        expires_at: str = "",
        is_permanent: bool = False,
        version_of: str = "",
    ) -> dict[str, Any]:
        if len(files) > MAX_FILES:
            raise ValueError(f"批量上传不超过 {MAX_FILES} 份")
        if doc_type not in DOC_TYPES:
            doc_type = "其他"
        success_list: list[str] = []
        fail_list: list[str] = []
        for name, content in files:
            try:
                name = _safe_name(name)
                if _ext(name) not in ALLOWED_EXT:
                    raise ValueError(f"不支持的格式：{_ext(name)}")
                if not content:
                    raise ValueError("文件为空")
                if len(content) > MAX_BYTES:
                    raise ValueError("文件过大")
                existing = self._conn.execute(
                    "SELECT * FROM kb_documents WHERE doc_name = ?", (name,)
                ).fetchone()
                if existing:
                    self._archive_current(existing)
                    dest = Path(existing["file_path"])
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
                    self._conn.execute(
                        """
                        UPDATE kb_documents SET file_size=?, status='pending', chunk_count=0,
                          upload_user=?, dept=?, allowed_roles=?, doc_type=?
                        WHERE id=?
                        """,
                        (len(content), username, dept, allowed_roles, doc_type, existing["id"]),
                    )
                    self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (existing["id"],))
                    self._conn.commit()
                    success_list.append(f"{name}(已换版→需重新解析)")
                    continue
                doc_id = uuid.uuid4().hex
                dest = self._doc_dir(doc_id) / name
                dest.write_bytes(content)
                now = _utcnow()
                self._conn.execute(
                    """
                    INSERT INTO kb_documents
                    (id, doc_name, doc_type, file_path, file_size, status, chunk_count,
                     upload_user, herb_tags, dept, allowed_roles, created_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, '', ?, ?, ?)
                    """,
                    (doc_id, name, doc_type, str(dest), len(content), username, dept, allowed_roles, now),
                )
                if effective_at or expires_at or is_permanent or version_of:
                    label = f"sub of {version_of[:48]}" if version_of else ""
                    self._insert_meta(
                        doc_id,
                        username,
                        effective_at=effective_at,
                        expires_at="" if is_permanent else expires_at,
                        is_permanent=is_permanent,
                        version_label=label,
                        version_status="draft",
                    )
                self._conn.commit()
                success_list.append(name)
            except Exception as exc:
                fail_list.append(f"{name}({exc})")
        return {"successList": success_list, "failList": fail_list}

    def _archive_current(self, row: sqlite3.Row) -> None:
        max_ver = self._conn.execute(
            "SELECT MAX(version) FROM kb_document_versions WHERE doc_id=?",
            (row["id"],),
        ).fetchone()[0] or 0
        src = Path(row["file_path"])
        archive = self._doc_dir(row["id"]) / "versions" / f"v{max_ver + 1}{src.suffix}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            archive.write_bytes(src.read_bytes())
        self._conn.execute(
            """
            INSERT INTO kb_document_versions (id, doc_id, version, file_path, file_size, created_by, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (uuid.uuid4().hex, row["id"], max_ver + 1, str(archive), row["file_size"], row["upload_user"], _utcnow()),
        )

    def _insert_meta(
        self,
        doc_id: str,
        username: str,
        *,
        effective_at: str = "",
        expires_at: str = "",
        is_permanent: bool = False,
        version_label: str = "",
        version_status: str = "draft",
        owner: str = "",
        applicable_region: str = "",
        review_interval_days: int | None = None,
        next_review_at: str = "",
    ) -> None:
        now = _utcnow()
        self._conn.execute(
            """
            INSERT INTO kb_metadata (
                doc_id, owner, applicable_region, effective_at, expires_at, is_permanent,
                review_interval_days, next_review_at, version_label, version_status,
                created_by, updated_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id, owner, applicable_region, effective_at or None,
                expires_at or None, 1 if is_permanent else 0, review_interval_days,
                next_review_at or None, version_label, version_status,
                username, username, now, now,
            ),
        )

    def list_documents(
        self,
        keyword: str = "",
        page: int = 1,
        size: int = 20,
        user_dept: str = "",
        user_role: str = "user",
    ) -> dict[str, Any]:
        where = ["1=1"]
        args: list[Any] = []
        if user_role != "admin":
            where.append("(dept = '' OR dept = ?)")
            args.append(user_dept)
        if keyword:
            where.append("doc_name LIKE ?")
            args.append(f"%{keyword}%")
        clause = " AND ".join(where)
        rows = self._conn.execute(
            f"SELECT * FROM kb_documents WHERE {clause} ORDER BY created_at DESC",
            args,
        ).fetchall()
        items = [self._doc_out(r) for r in rows]
        if user_role != "admin":
            items = [d for d in items if acl_ok(d["dept"], d["allowedRoles"], user_dept, user_role)]
        total = len(items)
        start = max(0, (page - 1) * size)
        return {"total": total, "list": items[start : start + size]}

    def _doc_out(self, r: sqlite3.Row) -> dict[str, Any]:
        return {
            "docId": r["id"],
            "docName": r["doc_name"],
            "docType": r["doc_type"],
            "status": r["status"],
            "chunkCount": r["chunk_count"],
            "uploadUser": r["upload_user"],
            "herbTags": r["herb_tags"] or "",
            "equipmentTags": r["herb_tags"] or "",
            "dept": r["dept"] or "",
            "allowedRoles": r["allowed_roles"] or "",
            "fileSize": r["file_size"] or 0,
            "createdAt": r["created_at"],
        }

    def get_document(self, doc_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        return self._doc_out(row)

    def stats(self) -> dict[str, Any]:
        total = int(self._conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0])
        chunks = int(self._conn.execute("SELECT COALESCE(SUM(chunk_count),0) FROM kb_documents").fetchone()[0])
        by_status = {
            "pending": 0,
            "parsed": 0,
            "vectorized": 0,
        }
        for row in self._conn.execute("SELECT status, COUNT(*) c FROM kb_documents GROUP BY status"):
            by_status[row["status"]] = row["c"]
        vectorized = int(by_status.get("vectorized") or 0)
        return {
            "docTotal": total,
            "chunkTotal": chunks,
            "vectorTotal": vectorized,
            "byStatus": by_status,
        }

    def parse(self, ids: list[str]) -> int:
        """步骤 03：解析上传文档，并按章节父子分块写入 SQLite。"""
        n = 0
        size = int(os.getenv("DOC_CHUNK_SIZE", "400"))
        overlap = int(os.getenv("DOC_CHUNK_OVERLAP", "80"))
        for doc_id in ids:
            row = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                continue
            path = Path(row["file_path"])
            from common.doc_store import read_document

            body = read_document(path) if path.suffix.lower() in {".pdf"} else path.read_text(encoding="utf-8", errors="ignore")
            chunks = build_document_chunks_step_04(
                doc_id=doc_id,
                doc_name=row["doc_name"],
                body=body,
                doc_type=row["doc_type"],
                size=size,
                overlap=overlap,
            )
            self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
            for item in chunks:
                self._conn.execute(
                    """
                    INSERT INTO kb_chunks (
                        id, doc_id, chunk_idx, text, parent_id, parent_title, section_path,
                        text_simplified, text_traditional, title_simplified, title_traditional,
                        parent_text_preview
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        doc_id,
                        item["chunk_idx"],
                        item["text"],
                        item.get("parent_id") or "",
                        item.get("parent_title") or "",
                        json.dumps(item.get("section_path") or [], ensure_ascii=False),
                        item.get("text_simplified") or "",
                        item.get("text_traditional") or "",
                        item.get("title_simplified") or "",
                        item.get("title_traditional") or "",
                        item.get("parent_text_preview") or "",
                    ),
                )
            tags = _herb_tags(body)
            self._conn.execute(
                "UPDATE kb_documents SET status='parsed', chunk_count=?, herb_tags=? WHERE id=?",
                (len(chunks), tags, doc_id),
            )
            n += 1
        self._conn.commit()
        return n

    def vectorize(self, doc_id: str) -> None:
        row = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        if row["status"] == "pending":
            raise ValueError("请先解析")
        self._conn.execute("UPDATE kb_documents SET status='vectorized' WHERE id=?", (doc_id,))
        self._conn.commit()
        self.rebuild_index()

    def vectorize_many(self, ids: list[str]) -> dict[str, Any]:
        ok, fail = [], []
        for doc_id in ids:
            try:
                self.vectorize(doc_id)
                ok.append(doc_id)
            except Exception as exc:
                fail.append({"docId": doc_id, "error": str(exc)})
        return {"successList": ok, "failList": fail}

    def rebuild_index(self) -> int:
        """步骤 01：按文献向量后端配置重建索引。"""
        from common.doc_store import default_doc_paths, reset_doc_store
        from common.doc_vector_backend import create_doc_store_step_02

        rows = self._conn.execute(
            """
            SELECT c.doc_id, d.doc_name, c.chunk_idx, c.text, c.parent_id, c.parent_title,
                   c.section_path, c.text_simplified, c.text_traditional, c.title_simplified,
                   c.title_traditional, c.parent_text_preview, d.doc_type, d.dept, d.allowed_roles
            FROM kb_chunks c JOIN kb_documents d ON d.id = c.doc_id
            WHERE d.status = 'vectorized'
            ORDER BY c.doc_id, c.chunk_idx
            """
        ).fetchall()
        if not rows:
            reset_doc_store()
            return 0
        chunks = [
            {
                "doc_id": r["doc_id"],
                "doc_name": r["doc_name"],
                "chunk_idx": r["chunk_idx"],
                "text": r["text"],
                "parent_id": r["parent_id"] or "",
                "parent_title": r["parent_title"] or "",
                "section_path": json.loads(r["section_path"] or "[]"),
                "text_simplified": r["text_simplified"] or "",
                "text_traditional": r["text_traditional"] or "",
                "title_simplified": r["title_simplified"] or "",
                "title_traditional": r["title_traditional"] or "",
                "parent_text_preview": r["parent_text_preview"] or "",
                "doc_type": r["doc_type"],
                "dept": r["dept"],
                "allowed_roles": r["allowed_roles"],
            }
            for r in rows
        ]
        index, meta = default_doc_paths()
        store = create_doc_store_step_02(index_path=index, metadata_path=meta, encode_fn=self._encode_fn)
        store.build_chunks(chunks)
        reset_doc_store()
        return len(chunks)

    def delete(self, ids: list[str]) -> int:
        n = 0
        for doc_id in ids:
            row = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
            if not row:
                continue
            self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM kb_document_versions WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM kb_metadata WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM kb_issues WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM kb_documents WHERE id=?", (doc_id,))
            n += 1
        self._conn.commit()
        if n:
            try:
                self.rebuild_index()
            except Exception as exc:
                from common.obs import degraded

                degraded("kb_rebuild", exc)
                logger.warning("删除后重建索引失败", exc_info=True)
        return n

    def preview(self, doc_id: str) -> tuple[bytes, str]:
        row = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        path = Path(row["file_path"])
        data = path.read_bytes() if path.is_file() else b""
        mime = {
            ".pdf": "application/pdf",
            ".txt": "text/plain; charset=utf-8",
            ".md": "text/plain; charset=utf-8",
        }.get(_ext(row["doc_name"]), "application/octet-stream")
        return data, mime

    def list_versions(self, doc_id: str) -> dict[str, Any]:
        row = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        archived = [
            {
                "version": r["version"],
                "fileSize": r["file_size"],
                "createdBy": r["created_by"],
                "createdAt": r["created_at"],
                "isCurrent": False,
            }
            for r in self._conn.execute(
                "SELECT * FROM kb_document_versions WHERE doc_id=? ORDER BY version DESC",
                (doc_id,),
            ).fetchall()
        ]
        current_n = (archived[0]["version"] if archived else 0) + 1
        return {
            "docId": doc_id,
            "docName": row["doc_name"],
            "current": {
                "version": current_n,
                "fileSize": row["file_size"] or 0,
                "createdBy": row["upload_user"] or "",
                "createdAt": row["created_at"],
                "isCurrent": True,
            },
            "list": archived,
        }

    def rollback(self, doc_id: str, version: int) -> None:
        current = self._conn.execute("SELECT * FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        ver = self._conn.execute(
            "SELECT * FROM kb_document_versions WHERE doc_id=? AND version=?",
            (doc_id, version),
        ).fetchone()
        if not current or not ver:
            raise KeyError(doc_id)
        src = Path(ver["file_path"])
        dest = Path(current["file_path"])
        if src.is_file():
            dest.write_bytes(src.read_bytes())
        self._conn.execute(
            "UPDATE kb_documents SET status='pending', chunk_count=0, file_size=? WHERE id=?",
            (ver["file_size"], doc_id),
        )
        self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
        self._conn.commit()

    def update_perms(self, doc_id: str, dept: str, allowed_roles: str) -> dict[str, str]:
        row = self._conn.execute("SELECT id FROM kb_documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            raise KeyError(doc_id)
        self._conn.execute(
            "UPDATE kb_documents SET dept=?, allowed_roles=? WHERE id=?",
            (dept, allowed_roles, doc_id),
        )
        self._conn.commit()
        return {"docId": doc_id, "dept": dept, "allowedRoles": allowed_roles}

    def get_profile(self, doc_id: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM kb_metadata WHERE doc_id=?", (doc_id,)).fetchone()
        if not row:
            return None
        return self._meta_out(row)

    def upsert_profile(self, doc_id: str, body: dict[str, Any], username: str) -> dict[str, Any]:
        if not self._conn.execute("SELECT id FROM kb_documents WHERE id=?", (doc_id,)).fetchone():
            raise KeyError(doc_id)
        owner = str(body.get("owner") or "")
        region = str(body.get("applicableRegion") or body.get("applicable_region") or "")
        effective_at = str(body.get("effectiveAt") or body.get("effective_at") or "") or None
        expires_at = str(body.get("expiresAt") or body.get("expires_at") or "") or None
        is_permanent = bool(body.get("isPermanent") if "isPermanent" in body else body.get("is_permanent"))
        if is_permanent:
            expires_at = None
        interval = body.get("reviewIntervalDays", body.get("review_interval_days"))
        interval = int(interval) if interval not in (None, "") else None
        next_review = str(body.get("nextReviewAt") or body.get("next_review_at") or "") or None
        label = str(body.get("versionLabel") or body.get("version_label") or "")
        status = str(body.get("versionStatus") or body.get("version_status") or "draft")
        now = _utcnow()
        existing = self._conn.execute("SELECT doc_id FROM kb_metadata WHERE doc_id=?", (doc_id,)).fetchone()
        if existing:
            self._conn.execute(
                """
                UPDATE kb_metadata SET owner=?, applicable_region=?, effective_at=?, expires_at=?,
                  is_permanent=?, review_interval_days=?, next_review_at=?, version_label=?,
                  version_status=?, updated_by=?, updated_at=?
                WHERE doc_id=?
                """,
                (
                    owner, region, effective_at, expires_at, 1 if is_permanent else 0,
                    interval, next_review, label, status, username, now, doc_id,
                ),
            )
        else:
            self._insert_meta(
                doc_id, username, effective_at=effective_at or "", expires_at=expires_at or "",
                is_permanent=is_permanent, version_label=label, version_status=status,
                owner=owner, applicable_region=region, review_interval_days=interval,
                next_review_at=next_review or "",
            )
        self._conn.commit()
        return self.get_profile(doc_id) or {}

    def _meta_out(self, r: sqlite3.Row) -> dict[str, Any]:
        return {
            "doc_id": r["doc_id"],
            "owner": r["owner"] or "",
            "applicable_region": r["applicable_region"] or "",
            "applicableRegion": r["applicable_region"] or "",
            "effective_at": r["effective_at"],
            "effectiveAt": r["effective_at"],
            "expires_at": r["expires_at"],
            "expiresAt": r["expires_at"],
            "is_permanent": bool(r["is_permanent"]),
            "isPermanent": bool(r["is_permanent"]),
            "review_interval_days": r["review_interval_days"],
            "reviewIntervalDays": r["review_interval_days"],
            "next_review_at": r["next_review_at"],
            "nextReviewAt": r["next_review_at"],
            "version_label": r["version_label"] or "",
            "versionLabel": r["version_label"] or "",
            "version_status": r["version_status"] or "",
            "versionStatus": r["version_status"] or "",
        }

    def governance_documents(self, keyword: str = "", page: int = 1, size: int = 20) -> dict[str, Any]:
        data = self.list_documents(keyword=keyword, page=page, size=size, user_role="admin")
        items = []
        for doc in data["list"]:
            meta = self.get_profile(doc["docId"])
            items.append(
                {
                    **doc,
                    "documentStatus": doc["status"],
                    "metadata": None if meta is None else meta,
                    "effectiveState": effective_state(meta),
                    "missingFields": missing_fields(meta),
                }
            )
        return {"total": data["total"], "list": items}

    def scan(self, doc_ids: list[str] | None = None) -> dict[str, Any]:
        rows = self._conn.execute("SELECT * FROM kb_documents").fetchall()
        created = updated = findings_n = scanned = 0
        for row in rows:
            if doc_ids and row["id"] not in doc_ids:
                continue
            scanned += 1
            meta = self.get_profile(row["id"])
            for finding in lifecycle_findings(dict(row), meta):
                findings_n += 1
                fp = hashlib.sha1(
                    f"{finding['issue_type']}:{finding['doc_id']}:{finding['title']}".encode("utf-8")
                ).hexdigest()
                existing = self._conn.execute(
                    "SELECT id, occurrence_count FROM kb_issues WHERE fingerprint=?", (fp,)
                ).fetchone()
                now = _utcnow()
                if existing:
                    self._conn.execute(
                        "UPDATE kb_issues SET last_seen_at=?, occurrence_count=? WHERE id=?",
                        (now, existing["occurrence_count"] + 1, existing["id"]),
                    )
                    updated += 1
                else:
                    self._conn.execute(
                        """
                        INSERT INTO kb_issues (
                            id, fingerprint, issue_type, severity, status, doc_id, title, summary,
                            evidence_json, occurrence_count, detected_at, last_seen_at
                        ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            uuid.uuid4().hex, fp, finding["issue_type"], finding["severity"],
                            finding["doc_id"], finding["title"], finding["summary"],
                            json.dumps(finding.get("evidence") or {}, ensure_ascii=False),
                            now, now,
                        ),
                    )
                    created += 1
        self._conn.commit()
        return {"documentsScanned": scanned, "findings": findings_n, "created": created, "updated": updated}

    def list_issues(
        self,
        status: str = "",
        issue_type: str = "",
        severity: str = "",
        keyword: str = "",
    ) -> dict[str, Any]:
        where = ["1=1"]
        args: list[Any] = []
        if status:
            where.append("status=?")
            args.append(status)
        if issue_type:
            where.append("issue_type=?")
            args.append(issue_type)
        if severity:
            where.append("severity=?")
            args.append(severity)
        if keyword:
            where.append("(title LIKE ? OR summary LIKE ?)")
            args.extend([f"%{keyword}%", f"%{keyword}%"])
        clause = " AND ".join(where)
        rows = self._conn.execute(
            f"SELECT * FROM kb_issues WHERE {clause} ORDER BY last_seen_at DESC", args
        ).fetchall()
        items = []
        for r in rows:
            items.append(
                {
                    "id": r["id"],
                    "type": r["issue_type"],
                    "severity": r["severity"],
                    "status": r["status"],
                    "docId": r["doc_id"],
                    "title": r["title"],
                    "summary": r["summary"],
                    "evidence": json.loads(r["evidence_json"] or "{}"),
                    "occurrenceCount": r["occurrence_count"],
                    "lastSeenAt": r["last_seen_at"],
                    "reviewer": r["reviewer"],
                    "reviewedAt": r["reviewed_at"],
                    "reviewNote": r["review_note"] or "",
                }
            )
        unresolved = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM kb_issues WHERE status IN ('open','confirmed')"
            ).fetchone()[0]
        )
        return {"total": len(items), "list": items, "unresolved": unresolved}

    def issue_stats(self) -> dict[str, Any]:
        docs = int(self._conn.execute("SELECT COUNT(*) FROM kb_documents").fetchone()[0])
        governed = int(self._conn.execute("SELECT COUNT(*) FROM kb_metadata").fetchone()[0])
        unresolved = int(
            self._conn.execute(
                "SELECT COUNT(*) FROM kb_issues WHERE status IN ('open','confirmed')"
            ).fetchone()[0]
        )
        by_status: dict[str, int] = {"open": 0, "confirmed": 0, "resolved": 0, "ignored": 0}
        for row in self._conn.execute("SELECT status, COUNT(*) c FROM kb_issues GROUP BY status"):
            by_status[row["status"]] = row["c"]
        by_type: dict[str, int] = {}
        for row in self._conn.execute("SELECT issue_type, COUNT(*) c FROM kb_issues GROUP BY issue_type"):
            by_type[row["issue_type"]] = row["c"]
        coverage = (governed / docs) if docs else 0.0
        return {
            "documents": docs,
            "governedDocuments": governed,
            "metadataCoverage": coverage,
            "unresolved": unresolved,
            "conflicts": 0,
            "byStatus": by_status,
            "byType": by_type,
        }

    def review_issue(self, issue_id: str, to_status: str, reviewer: str, note: str = "") -> None:
        allowed = {"open", "confirmed", "resolved", "ignored"}
        if to_status not in allowed:
            raise ValueError("无效状态")
        row = self._conn.execute("SELECT * FROM kb_issues WHERE id=?", (issue_id,)).fetchone()
        if not row:
            raise KeyError(issue_id)
        if to_status in {"resolved", "ignored"} and not (note or "").strip():
            raise ValueError("解决或忽略需要填写说明")
        now = _utcnow()
        self._conn.execute(
            """
            UPDATE kb_issues SET status=?, reviewer=?, review_note=?, reviewed_at=? WHERE id=?
            """,
            (to_status, reviewer, note, now, issue_id),
        )
        self._conn.execute(
            """
            INSERT INTO kb_reviews (id, issue_id, from_status, to_status, reviewer, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (uuid.uuid4().hex, issue_id, row["status"], to_status, reviewer, note, now),
        )
        self._conn.commit()

    def blocked_ids(self) -> set[str]:
        blocked: set[str] = set()
        for row in self._conn.execute("SELECT * FROM kb_metadata"):
            if not is_retrievable(self._meta_out(row)):
                blocked.add(row["doc_id"])
        return blocked

    def search(self, question: str, viewer_dept: str = "", viewer_role: str = "user") -> list[dict[str, Any]]:
        from common.doc_store import FaissDocStore, default_doc_paths

        index, meta = default_doc_paths()
        if not index.is_file():
            return []
        store = FaissDocStore(index, meta, encode_fn=self._encode_fn)
        try:
            store.load()
        except Exception as exc:
            from common.obs import degraded

            degraded("doc_index", exc)
            return []
        top_k = int(os.getenv("DOC_TOP_K", "4"))
        min_score = float(os.getenv("DOC_MIN_SCORE", "0.0" if self._encode_fn else "0.35"))
        hits = store.mixed_search(question, top_k=top_k, min_score=min_score)
        blocked = self.blocked_ids()
        kept: list[dict[str, Any]] = []
        for hit in hits:
            if hit.get("doc_id") in blocked:
                continue
            if not acl_ok(hit.get("dept"), hit.get("allowed_roles"), viewer_dept, viewer_role):
                continue
            kept.append(hit)
        return kept


_svc: KnowledgeService | None = None


def default_knowledge_db() -> Path:
    load_app_env()
    return Path(
        os.getenv("CONVERSATION_DB_PATH")
        or Path(__file__).resolve().parent.parent / "data" / "conversations.sqlite"
    )


def get_knowledge_service() -> KnowledgeService:
    global _svc
    if _svc is None:
        _svc = KnowledgeService(default_knowledge_db())
    return _svc


def reset_knowledge_service() -> None:
    global _svc
    if _svc is not None:
        try:
            _svc.close()
        except Exception:
            pass
    _svc = None
