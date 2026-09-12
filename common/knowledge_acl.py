"""Document-level ACL: empty dept = public; admin sees all."""

from __future__ import annotations


def acl_ok(
    dept: str | None,
    allowed_roles: str | None,
    user_dept: str | None,
    user_role: str | None,
) -> bool:
    if (user_role or "") == "admin":
        return True
    doc_dept = (dept or "").strip()
    if doc_dept and (user_dept or "").strip() != doc_dept:
        return False
    raw = (allowed_roles or "").strip()
    if not raw:
        return True
    allowed = [r.strip() for r in raw.split(",") if r.strip()]
    return (not allowed) or ((user_role or "") in allowed)
