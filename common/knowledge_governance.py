"""Governance sidecar: missing fields, effective state, retrieval gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

VERSION_STATUSES = {"draft", "active", "superseded", "withdrawn"}


def _parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _meta_get(meta: Any, *names: str) -> Any:
    if meta is None:
        return None
    if isinstance(meta, dict):
        for name in names:
            if name in meta and meta[name] not in (None, ""):
                return meta[name]
        return None
    for name in names:
        if hasattr(meta, name):
            val = getattr(meta, name)
            if val not in (None, ""):
                return val
    return None


def missing_fields(meta: Any | None) -> list[str]:
    if meta is None:
        return [
            "owner",
            "applicableRegion",
            "effectiveAt",
            "expiryPolicy",
            "reviewPolicy",
            "versionLabel",
            "versionStatus",
        ]
    missing: list[str] = []
    if not str(_meta_get(meta, "owner") or "").strip():
        missing.append("owner")
    if not str(_meta_get(meta, "applicable_region", "applicableRegion") or "").strip():
        missing.append("applicableRegion")
    if _parse_dt(_meta_get(meta, "effective_at", "effectiveAt")) is None:
        missing.append("effectiveAt")
    permanent = bool(_meta_get(meta, "is_permanent", "isPermanent") or False)
    if not permanent and _parse_dt(_meta_get(meta, "expires_at", "expiresAt")) is None:
        missing.append("expiryPolicy")
    if not _meta_get(meta, "review_interval_days", "reviewIntervalDays") and not _parse_dt(
        _meta_get(meta, "next_review_at", "nextReviewAt")
    ):
        missing.append("reviewPolicy")
    if not str(_meta_get(meta, "version_label", "versionLabel") or "").strip():
        missing.append("versionLabel")
    status = str(_meta_get(meta, "version_status", "versionStatus") or "")
    if status not in VERSION_STATUSES:
        missing.append("versionStatus")
    return missing


def effective_state(meta: Any | None, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if meta is None:
        return "metadata_incomplete"
    status = str(_meta_get(meta, "version_status", "versionStatus") or "")
    if status in {"draft", "superseded", "withdrawn"}:
        return status
    effective_at = _parse_dt(_meta_get(meta, "effective_at", "effectiveAt"))
    if effective_at is None:
        return "metadata_incomplete"
    if effective_at > now:
        return "not_yet_effective"
    permanent = bool(_meta_get(meta, "is_permanent", "isPermanent") or False)
    expires_at = _parse_dt(_meta_get(meta, "expires_at", "expiresAt"))
    if not permanent and expires_at and expires_at < now:
        return "expired"
    return "active"


def is_retrievable(meta: Any | None, now: datetime | None = None) -> bool:
    return effective_state(meta, now) not in {
        "superseded",
        "withdrawn",
        "not_yet_effective",
        "expired",
    }


def lifecycle_findings(doc: dict[str, Any], meta: Any | None, now: datetime | None = None) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    findings: list[dict[str, Any]] = []
    doc_id = str(doc.get("id") or doc.get("docId") or "")
    doc_name = str(doc.get("doc_name") or doc.get("docName") or "")
    missing = missing_fields(meta)
    if missing:
        findings.append(
            {
                "issue_type": "metadata_missing",
                "severity": "warning",
                "doc_id": doc_id,
                "title": f"文档治理元数据不完整：{doc_name}",
                "summary": f"缺少 {len(missing)} 项治理元数据，需责任人补录后复核。",
                "evidence": {"docName": doc_name, "missingFields": missing},
            }
        )
    if meta is None:
        return findings
    state = effective_state(meta, now)
    if state == "not_yet_effective":
        findings.append(
            {
                "issue_type": "not_yet_effective",
                "severity": "info",
                "doc_id": doc_id,
                "title": f"文档尚未生效：{doc_name}",
                "summary": "生效时间晚于当前，检索应排除。",
                "evidence": {"docName": doc_name},
            }
        )
    if state == "expired":
        findings.append(
            {
                "issue_type": "expired",
                "severity": "critical",
                "doc_id": doc_id,
                "title": f"文档已过期：{doc_name}",
                "summary": "失效时间已过，检索应排除。",
                "evidence": {"docName": doc_name},
            }
        )
    expires_at = _parse_dt(_meta_get(meta, "expires_at", "expiresAt"))
    permanent = bool(_meta_get(meta, "is_permanent", "isPermanent") or False)
    if state == "active" and not permanent and expires_at and expires_at - now <= timedelta(days=30):
        findings.append(
            {
                "issue_type": "expiring",
                "severity": "warning",
                "doc_id": doc_id,
                "title": f"文档即将到期：{doc_name}",
                "summary": "30 天内到期，请复核有效期。",
                "evidence": {"docName": doc_name},
            }
        )
    next_review = _parse_dt(_meta_get(meta, "next_review_at", "nextReviewAt"))
    if next_review and next_review <= now:
        findings.append(
            {
                "issue_type": "review_due",
                "severity": "warning",
                "doc_id": doc_id,
                "title": f"文档到达复审日：{doc_name}",
                "summary": "请复核后更新复审时间。",
                "evidence": {"docName": doc_name},
            }
        )
    return findings
