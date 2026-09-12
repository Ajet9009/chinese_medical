"""JWT login, ownership, admin guards."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeRedis


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("JWT_SECRET", "test-secret-w4-please-use-32bytes!!")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin123")

    import common.redis_client as redis_mod
    redis_mod.reset_redis_client()

    import _005_fastapi.deps as deps
    import _005_fastapi.main as main

    orig_get_store = main.get_store
    orig_get_store.cache_clear()
    deps.get_users.cache_clear()

    from common.conversation_store import ConversationStore
    from common.user_store import UserStore

    db = tmp_path / "c.sqlite"
    store = ConversationStore(db, redis_client=FakeRedis())
    users = UserStore(db)
    users.seed_admin()
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(deps, "get_users", lambda: users)
    monkeypatch.setattr(main, "get_users", lambda: users)

    from common.knowledge_service import reset_knowledge_service

    reset_knowledge_service()

    with TestClient(main.app) as c:
        yield c, store

    orig_get_store.cache_clear()
    cached = getattr(deps.get_users, "cache_clear", None)
    if cached:
        cached()


def _login(c, username="admin", password="admin123"):
    resp = c.post("/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['token']}"}


def test_login_fail(auth_client):
    c, _ = auth_client
    assert c.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401


def test_conversations_require_auth(auth_client):
    c, _ = auth_client
    assert c.get("/conversations").status_code == 401


def test_user_cannot_see_other_conversations(auth_client):
    c, _store = auth_client
    admin_h = _login(c)
    created = c.post("/conversations", json={"title": "管理员的方"}, headers=admin_h).json()
    c.post(
        "/admin/users",
        json={"username": "herb", "password": "herb1234", "role": "user"},
        headers=admin_h,
    )
    herb_h = _login(c, "herb", "herb1234")
    mine = c.get("/conversations", headers=herb_h).json()
    assert created["id"] not in [x["id"] for x in mine]
    assert c.get(f"/conversations/{created['id']}/messages", headers=herb_h).status_code == 404
    herb_conv = c.post("/conversations", json={"title": "用户的方"}, headers=herb_h).json()
    assert c.get("/conversations", headers=herb_h).json()[0]["id"] == herb_conv["id"]


def test_user_forbidden_admin(auth_client):
    c, _ = auth_client
    admin_h = _login(c)
    c.post(
        "/admin/users",
        json={"username": "herb", "password": "herb1234", "role": "user"},
        headers=admin_h,
    )
    herb_h = _login(c, "herb", "herb1234")
    assert c.get("/admin/feedbacks", headers=herb_h).status_code == 403
    assert c.get("/admin/prompts", headers=admin_h).status_code == 200
    users = c.get("/admin/users", headers=admin_h).json()
    assert any(u["username"] == "admin" for u in users)


def test_health_public(auth_client):
    c, _ = auth_client
    assert c.get("/health").status_code == 200


def test_change_password(auth_client):
    c, _ = auth_client
    h = _login(c)
    bad = c.post(
        "/auth/password",
        json={"old_password": "wrong", "new_password": "newpass1"},
        headers=h,
    )
    assert bad.status_code == 400
    ok = c.post(
        "/auth/password",
        json={"old_password": "admin123", "new_password": "newpass1"},
        headers=h,
    )
    assert ok.status_code == 200
    assert c.post("/auth/login", json={"username": "admin", "password": "admin123"}).status_code == 401
    assert c.post("/auth/login", json={"username": "admin", "password": "newpass1"}).status_code == 200


def test_cannot_disable_or_delete_self(auth_client):
    c, _ = auth_client
    h = _login(c)
    me = c.get("/auth/me", headers=h).json()
    assert c.patch(f"/admin/users/{me['id']}", json={"status": "inactive"}, headers=h).status_code == 400
    assert c.delete(f"/admin/users/{me['id']}", headers=h).status_code == 400


def test_cannot_disable_or_delete_last_admin(auth_client):
    c, _ = auth_client
    h = _login(c)
    other = c.post(
        "/admin/users",
        json={"username": "herb", "password": "herb1234", "role": "user"},
        headers=h,
    ).json()
    admin_id = c.get("/auth/me", headers=h).json()["id"]
    assert c.patch(f"/admin/users/{admin_id}", json={"status": "inactive"}, headers=h).status_code == 400
    assert c.delete(f"/admin/users/{admin_id}", headers=h).status_code == 400
    assert c.patch(f"/admin/users/{admin_id}", json={"role": "user"}, headers=h).status_code == 400
    assert c.delete(f"/admin/users/{other['id']}", headers=h).status_code == 204
    assert c.post("/auth/login", json={"username": "herb", "password": "herb1234"}).status_code == 401


def test_reset_password_and_logs(auth_client):
    c, _ = auth_client
    h = _login(c)
    created = c.post(
        "/admin/users",
        json={"username": "herb", "password": "herb1234", "role": "user"},
        headers=h,
    ).json()
    reset = c.post(
        f"/admin/users/{created['id']}/reset-password",
        json={"password": "reset5678"},
        headers=h,
    )
    assert reset.status_code == 200
    assert c.post("/auth/login", json={"username": "herb", "password": "herb1234"}).status_code == 401
    assert c.post("/auth/login", json={"username": "herb", "password": "reset5678"}).status_code == 200
    logs = c.get("/admin/logs", headers=h).json()
    types = [x["operate_type"] for x in logs["items"]]
    assert "登录" in types
    assert "用户管理" in types
    assert logs["total"] >= 1


def test_admin_upload_doc_rebuilds_index(auth_client, tmp_path, monkeypatch):
    c, _ = auth_client
    monkeypatch.setenv("KNOWLEDGE_DOCS_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("DOC_FAISS_INDEX_PATH", str(tmp_path / "docs.index"))
    monkeypatch.setenv("DOC_FAISS_METADATA_PATH", str(tmp_path / "docs.json"))
    from common.knowledge_service import reset_knowledge_service

    reset_knowledge_service()
    h = _login(c)
    files = {"files": ("四君子汤.md", "# 四君子汤\n益气健脾".encode("utf-8"), "text/markdown")}
    resp = c.post("/document/upload", files=files, data={"docType": "方剂"}, headers=h)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["successList"]
    listed = c.get("/document/list", headers=h)
    assert listed.status_code == 200
    assert listed.json()["total"] >= 1
    bad = c.post(
        "/document/upload",
        files={"files": ("../evil.md", b"x", "text/markdown")},
        data={"docType": "其他"},
        headers=h,
    )
    assert bad.status_code == 200
    assert bad.json()["failList"]


def test_login_rate_limit(auth_client, monkeypatch):
    c, _ = auth_client
    monkeypatch.setenv("LOGIN_RATE_MAX", "3")
    from common.login_limit import reset_login_limiter

    reset_login_limiter()
    for _ in range(3):
        assert c.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 401
    assert c.post("/auth/login", json={"username": "admin", "password": "nope"}).status_code == 429
