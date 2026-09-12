"""Knowledge base and governance HTTP API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from _005_fastapi.deps import get_current_user, get_users, require_admin
from common.knowledge_service import DOC_TYPES, get_knowledge_service
from common.user_store import User

router = APIRouter()


class IdsBody(BaseModel):
    ids: list[str] = Field(default_factory=list)


class ParseBody(BaseModel):
    ids: list[str] = Field(default_factory=list)


class VectorBody(BaseModel):
    docId: str = ""
    ids: list[str] = Field(default_factory=list)


class RollbackBody(BaseModel):
    docId: str
    version: int


class PermsBody(BaseModel):
    dept: str = ""
    allowedRoles: str = ""


class ProfileBody(BaseModel):
    owner: str | None = None
    applicableRegion: str | None = None
    effectiveAt: str | None = None
    expiresAt: str | None = None
    isPermanent: bool | None = None
    reviewIntervalDays: int | None = None
    nextReviewAt: str | None = None
    versionLabel: str | None = None
    versionStatus: str | None = None


class ReviewBody(BaseModel):
    status: str
    note: str = ""


class ScanBody(BaseModel):
    docIds: list[str] = Field(default_factory=list)


def _svc():
    return get_knowledge_service()


def _require_upload(user: User) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员")
    return user


@router.post("/document/upload")
async def document_upload(
    files: list[UploadFile] = File(...),
    docType: str = Form("其他"),
    dept: str = Form(""),
    allowedRoles: str = Form(""),
    effectiveAt: str = Form(""),
    expiresAt: str = Form(""),
    isPermanent: bool = Form(False),
    versionOf: str = Form(""),
    user: User = Depends(get_current_user),
):
    _require_upload(user)
    packed: list[tuple[str, bytes]] = []
    for f in files:
        packed.append((f.filename or "unnamed", await f.read()))
    try:
        data = _svc().upload(
            packed,
            doc_type=docType,
            username=user.username,
            dept=dept,
            allowed_roles=allowedRoles,
            effective_at=effectiveAt,
            expires_at=expiresAt,
            is_permanent=isPermanent,
            version_of=versionOf,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    get_users().write_log(user.username, "文档上传", f"成功 {len(data['successList'])} 份")
    return data


@router.get("/document/list")
def document_list(
    keyword: str = "",
    page: int = 1,
    size: int = 20,
    user: User = Depends(get_current_user),
):
    dept = getattr(user, "dept", "") or ""
    return _svc().list_documents(keyword=keyword, page=page, size=size, user_dept=dept, user_role=user.role)


@router.get("/document/stats")
def document_stats(user: User = Depends(get_current_user)):
    return _svc().stats()


@router.get("/document/types")
def document_types(user: User = Depends(get_current_user)):
    return {"list": list(DOC_TYPES)}


@router.get("/document/preview/{doc_id}")
def document_preview(doc_id: str, user: User = Depends(get_current_user)):
    from common.knowledge_acl import acl_ok

    try:
        doc = _svc().get_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档不存在")
    if not acl_ok(doc.get("dept"), doc.get("allowedRoles"), getattr(user, "dept", ""), user.role):
        raise HTTPException(status_code=403, detail="无权限预览")
    data, mime = _svc().preview(doc_id)
    return Response(content=data, media_type=mime)


@router.get("/document/{doc_id}/versions")
def document_versions(doc_id: str, user: User = Depends(get_current_user)):
    try:
        return _svc().list_versions(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档不存在")


@router.post("/document/rollback")
def document_rollback(payload: RollbackBody, user: User = Depends(require_admin)):
    try:
        _svc().rollback(payload.docId, payload.version)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档或版本不存在")
    get_users().write_log(user.username, "文档回滚", f"{payload.docId} → v{payload.version}")
    return {"ok": True}


@router.post("/document/parse")
def document_parse(payload: ParseBody, user: User = Depends(require_admin)):
    n = _svc().parse(payload.ids)
    get_users().write_log(user.username, "文档解析", f"{n} 份")
    return {"parsed": n}


@router.post("/document/vector/generate")
def document_vector(payload: VectorBody, user: User = Depends(require_admin)):
    doc_id = payload.docId
    try:
        _svc().vectorize(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}


@router.post("/document/vector/batch")
def document_vector_batch(payload: VectorBody, user: User = Depends(require_admin)):
    return _svc().vectorize_many(payload.ids or ([payload.docId] if payload.docId else []))


@router.delete("/document/delete")
def document_delete(payload: IdsBody, user: User = Depends(require_admin)):
    n = _svc().delete(payload.ids)
    get_users().write_log(user.username, "文档删除", f"{n} 份")
    return {"deleted": n}


@router.post("/document/batch-delete")
def document_batch_delete(payload: IdsBody, user: User = Depends(require_admin)):
    return document_delete(payload, user)


@router.get("/document/{doc_id}/perms")
def get_perms(doc_id: str, user: User = Depends(require_admin)):
    try:
        doc = _svc().get_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"docId": doc_id, "dept": doc.get("dept") or "", "allowedRoles": doc.get("allowedRoles") or ""}


@router.put("/document/{doc_id}/perms")
def put_perms(doc_id: str, payload: PermsBody, user: User = Depends(require_admin)):
    try:
        data = _svc().update_perms(doc_id, payload.dept, payload.allowedRoles)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档不存在")
    get_users().write_log(user.username, "文档授权", f"{doc_id}")
    return data


@router.get("/knowledge-governance/documents")
def gov_documents(
    keyword: str = "",
    page: int = 1,
    size: int = 20,
    user: User = Depends(require_admin),
):
    return _svc().governance_documents(keyword=keyword, page=page, size=size)


@router.get("/knowledge-governance/documents/{doc_id}/profile")
def gov_get_profile(doc_id: str, user: User = Depends(require_admin)):
    return _svc().get_profile(doc_id) or {}


@router.put("/knowledge-governance/documents/{doc_id}/profile")
def gov_put_profile(doc_id: str, payload: ProfileBody, user: User = Depends(require_admin)):
    try:
        return _svc().upsert_profile(doc_id, payload.model_dump(), user.username)
    except KeyError:
        raise HTTPException(status_code=404, detail="文档不存在")


@router.post("/knowledge-governance/scan")
def gov_scan(payload: ScanBody | None = None, user: User = Depends(require_admin)):
    ids = list((payload.docIds if payload else []) or [])
    data = _svc().scan(doc_ids=ids or None)
    get_users().write_log(user.username, "知识治理", "扫描")
    return data


@router.get("/knowledge-governance/issues")
def gov_issues(
    status: str = "",
    type: str = "",
    severity: str = "",
    keyword: str = "",
    user: User = Depends(require_admin),
):
    return _svc().list_issues(status=status, issue_type=type, severity=severity, keyword=keyword)


@router.get("/knowledge-governance/issues/stats")
def gov_issue_stats(user: User = Depends(require_admin)):
    return _svc().issue_stats()


@router.post("/knowledge-governance/issues/{issue_id}/review")
def gov_review(issue_id: str, payload: ReviewBody, user: User = Depends(require_admin)):
    try:
        _svc().review_issue(issue_id, payload.status, user.username, payload.note)
    except KeyError:
        raise HTTPException(status_code=404, detail="问题不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}
