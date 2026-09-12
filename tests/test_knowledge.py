"""Knowledge base registry: ACL, versions, governance, parse/vectorize."""

from __future__ import annotations

import numpy as np


def _encode(texts: list[str]) -> np.ndarray:
    dim = 64
    mat = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for ch in text:
            mat[i, ord(ch) % dim] += 1.0
        n = np.linalg.norm(mat[i])
        if n:
            mat[i] /= n
    return mat


def test_acl_public_and_dept_and_roles():
    from common.knowledge_acl import acl_ok

    assert acl_ok(dept="", allowed_roles="", user_dept="脾胃", user_role="user") is True
    assert acl_ok(dept="脾胃", allowed_roles="", user_dept="脾胃", user_role="user") is True
    assert acl_ok(dept="脾胃", allowed_roles="", user_dept="外科", user_role="user") is False
    assert acl_ok(dept="脾胃", allowed_roles="", user_dept="外科", user_role="admin") is True
    assert acl_ok(dept="脾胃", allowed_roles="admin", user_dept="脾胃", user_role="user") is False
    assert acl_ok(dept="脾胃", allowed_roles="admin", user_dept="脾胃", user_role="admin") is True


def test_upload_list_parse_vectorize(tmp_path, monkeypatch):
    from common.knowledge_service import KnowledgeService

    monkeypatch.setenv("KNOWLEDGE_DOCS_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("DOC_FAISS_INDEX_PATH", str(tmp_path / "docs.index"))
    monkeypatch.setenv("DOC_FAISS_METADATA_PATH", str(tmp_path / "docs.json"))
    svc = KnowledgeService(tmp_path / "k.sqlite", encode_fn=_encode)
    out = svc.upload(
        [("四君子汤.md", "# 四君子汤\n人参白术茯苓甘草。".encode("utf-8"))],
        doc_type="方剂",
        username="admin",
        dept="脾胃",
        allowed_roles="user,admin",
    )
    assert out["successList"]
    listed = svc.list_documents(user_role="admin")
    assert listed["total"] == 1
    doc_id = listed["list"][0]["docId"]
    assert listed["list"][0]["status"] == "pending"
    first_ver = svc.list_versions(doc_id)
    assert first_ver["current"]["version"] == 1
    assert first_ver["list"] == []
    svc.parse([doc_id])
    parsed = svc.get_document(doc_id)
    assert parsed["status"] == "parsed"
    assert parsed["chunkCount"] >= 1
    svc.vectorize(doc_id)
    vec = svc.get_document(doc_id)
    assert vec["status"] == "vectorized"
    hits = svc.search("四君子汤组成", viewer_dept="脾胃", viewer_role="user")
    assert hits
    assert hits[0]["doc_name"] == "四君子汤.md"
    blocked = svc.search("四君子汤组成", viewer_dept="外科", viewer_role="user")
    assert blocked == []


def test_same_name_archives_version_and_rollback(tmp_path, monkeypatch):
    from common.knowledge_service import KnowledgeService

    monkeypatch.setenv("KNOWLEDGE_DOCS_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("DOC_FAISS_INDEX_PATH", str(tmp_path / "docs.index"))
    monkeypatch.setenv("DOC_FAISS_METADATA_PATH", str(tmp_path / "docs.json"))
    svc = KnowledgeService(tmp_path / "k.sqlite", encode_fn=_encode)
    svc.upload([("方.md", b"v1-old")], doc_type="典籍", username="admin")
    svc.upload([("方.md", b"v2-new")], doc_type="典籍", username="admin")
    listed = svc.list_documents(user_role="admin")
    assert listed["total"] == 1
    doc_id = listed["list"][0]["docId"]
    versions = svc.list_versions(doc_id)
    assert versions["current"]["version"] == 2
    assert versions["current"]["isCurrent"] is True
    assert len(versions["list"]) == 1
    preview = svc.preview(doc_id)
    assert b"v2-new" in preview[0]
    svc.rollback(doc_id, versions["list"][0]["version"])
    preview2 = svc.preview(doc_id)
    assert b"v1-old" in preview2[0]
    assert svc.get_document(doc_id)["status"] == "pending"


def test_governance_blocks_expired_and_scan_missing(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    from common.knowledge_governance import effective_state, is_retrievable, missing_fields
    from common.knowledge_service import KnowledgeService

    monkeypatch.setenv("KNOWLEDGE_DOCS_DIR", str(tmp_path / "files"))
    svc = KnowledgeService(tmp_path / "k.sqlite", encode_fn=_encode)
    svc.upload([("空档.md", b"x")], doc_type="其他", username="admin")
    doc_id = svc.list_documents(user_role="admin")["list"][0]["docId"]
    assert missing_fields(None) != []
    assert effective_state(None) == "metadata_incomplete"
    assert is_retrievable(None) is True

    past = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    svc.upsert_profile(
        doc_id,
        {
            "owner": "李医师",
            "applicableRegion": "脾胃气虚",
            "effectiveAt": past,
            "expiresAt": past,
            "isPermanent": False,
            "reviewIntervalDays": 365,
            "versionLabel": "v1",
            "versionStatus": "active",
        },
        "admin",
    )
    meta = svc.get_profile(doc_id)
    assert is_retrievable(meta) is False
    assert effective_state(meta) == "expired"

    scan = svc.scan()
    assert scan["findings"] >= 1
    issues = svc.list_issues()
    assert issues["total"] >= 1
    issue_id = issues["list"][0]["id"]
    svc.review_issue(issue_id, "confirmed", "admin", "确认")
    assert svc.list_issues(status="confirmed")["total"] >= 1


def test_version_of_sets_label(tmp_path, monkeypatch):
    from common.knowledge_service import KnowledgeService

    monkeypatch.setenv("KNOWLEDGE_DOCS_DIR", str(tmp_path / "files"))
    svc = KnowledgeService(tmp_path / "k.sqlite", encode_fn=_encode)
    first = svc.upload([("旧.md", b"old")], doc_type="方剂", username="admin")
    old_id = svc.list_documents(user_role="admin")["list"][0]["docId"]
    svc.upload(
        [("新.md", b"new")],
        doc_type="方剂",
        username="admin",
        version_of=old_id,
        effective_at="2020-01-01T00:00:00",
    )
    names = {d["docName"]: d for d in svc.list_documents(user_role="admin")["list"]}
    new_id = names["新.md"]["docId"]
    profile = svc.get_profile(new_id)
    assert profile is not None
    assert "sub of" in (profile.get("version_label") or profile.get("versionLabel") or "")
