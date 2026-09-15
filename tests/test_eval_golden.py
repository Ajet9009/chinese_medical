"""Golden-set format, scoring, and admin mark-golden flywheel."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from common.doc_rag import REFUSE_ANSWER
from common.eval_golden import (
    add_from_feedback,
    evaluate_offline,
    keyword_hit_ratio,
    load_golden,
    save_golden,
    validate_golden,
)
from tests.fakes import FakeRedis
from tests.test_auth import _login


def test_repo_golden_validates():
    items = load_golden()
    assert items, "eval/golden_qa.json 应纳入版本库"
    assert validate_golden(items) == []
    queries = [str(it["query"]).strip() for it in items]
    assert len(queries) == len(set(queries))
    assert any(it.get("category") == "拒答" for it in items)
    cats = {str(it.get("category") or "") for it in items}
    assert {"方剂", "本草", "证候", "典籍", "医案", "其他", "拒答"} <= cats
    assert all(it.get("category") != "变电" for it in items)


def test_keyword_hit_ratio_and_offline_score():
    expect = ["益气", "健脾"]
    assert keyword_hit_ratio(expect, "四君子汤益气健脾") == 1.0
    assert keyword_hit_ratio(expect, "只提到益气") == 0.5
    assert keyword_hit_ratio([], "anything") == 0.0
    assert keyword_hit_ratio(["未查到"], REFUSE_ANSWER) == 1.0

    items = [
        {"query": "四君子汤有什么功效？", "expect": ["益气", "健脾"], "category": "方剂"},
        {"query": "待标注问句", "expect": [], "category": "用户反馈", "source": "feedback"},
        {"query": "今天上证指数多少？", "expect": ["未查到"], "category": "拒答"},
    ]
    report = evaluate_offline(
        items,
        {
            "四君子汤有什么功效？": "本方益气健脾",
            "今天上证指数多少？": REFUSE_ANSWER,
        },
    )
    assert report["labeled"] == 2
    assert report["skipped"] == 1
    assert report["passed"] == 2
    assert report["failed"] == 0
    assert report["pass_rate"] == 1.0

    miss = evaluate_offline(items[:1], {"四君子汤有什么功效？": "未提及关键词"})
    assert miss["passed"] == 0
    assert miss["items"][0]["misses"] == ["益气", "健脾"]


def test_validate_rejects_dup_and_empty_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_QA_PATH", str(tmp_path / "g.json"))
    bad = [
        {"query": "同一问", "expect": ["a"], "category": "方剂"},
        {"query": "同一问", "expect": ["b"], "category": "方剂"},
        {"query": "无期望", "expect": [], "category": "方剂", "source": "seed"},
        {"query": "反馈待标", "expect": [], "category": "用户反馈", "source": "feedback"},
    ]
    errs = validate_golden(bad)
    assert any("重复" in e for e in errs)
    assert any("expect 不能为空" in e for e in errs)
    assert not any("反馈待标" in e for e in errs)


def test_add_from_feedback_dedup(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_QA_PATH", str(tmp_path / "g.json"))
    save_golden([])
    first = add_from_feedback("  四君子汤有什么功效？  ")
    assert first["added"] is True
    assert first["total"] == 1
    again = add_from_feedback("四君子汤有什么功效？")
    assert again["added"] is False
    assert "已在" in again["reason"]
    items = load_golden()
    assert items[0]["source"] == "feedback"
    assert items[0]["expect"] == []
    empty = add_from_feedback("   ")
    assert empty["added"] is False


@pytest.fixture
def eval_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.setenv("GOLDEN_QA_PATH", str(tmp_path / "golden.json"))
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("JWT_SECRET", "test-secret-w4-please-use-32bytes!!")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin123")

    import common.redis_client as redis_mod

    redis_mod.reset_redis_client()
    from common.obs import reset_obs

    reset_obs()

    import _005_fastapi.deps as deps
    import _005_fastapi.main as main

    orig_get_store = main.get_store
    orig_get_store.cache_clear()
    deps.get_users.cache_clear()

    from common.conversation_store import ConversationStore
    from common.knowledge_service import reset_knowledge_service
    from common.user_store import UserStore

    db = tmp_path / "c.sqlite"
    store = ConversationStore(db, redis_client=FakeRedis())
    users = UserStore(db)
    users.seed_admin()
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(deps, "get_users", lambda: users)
    monkeypatch.setattr(main, "get_users", lambda: users)
    reset_knowledge_service()
    save_golden([])

    with TestClient(main.app) as c:
        yield c, store

    orig_get_store.cache_clear()
    cached = getattr(deps.get_users, "cache_clear", None)
    if cached:
        cached()


def test_health_includes_degraded(eval_client):
    c, _ = eval_client
    body = c.get("/health").json()
    assert body["status"] == "ok"
    assert "degraded" in body
    assert "byTag" in body["degraded"]
    assert "total" in body["degraded"]


def test_mark_golden_from_dislike(eval_client):
    c, store = eval_client
    h = _login(c)
    me = c.get("/auth/me", headers=h).json()
    conv = store.create("点踩", user_id=me["id"])
    store.add_message(conv.id, "user", "小柴胡汤治什么？")
    msg = store.add_message(
        conv.id,
        "assistant",
        "说不清",
        details={"feedback": "dislike", "refused": True, "evidence_gap": True},
    )
    fb = c.get("/admin/feedbacks", headers=h).json()
    assert fb["total"] == 1
    row = fb["items"][0]
    assert row["evidence_gap"] is True
    assert row["in_golden"] is False

    first = c.post(f"/admin/feedbacks/{msg.id}/golden", headers=h)
    assert first.status_code == 200, first.text
    assert first.json()["added"] is True
    again = c.post(f"/admin/feedbacks/{msg.id}/golden", headers=h)
    assert again.json()["added"] is False
    assert "已在" in again.json()["reason"]

    golden = c.get("/admin/golden", headers=h).json()
    assert golden["total"] == 1
    assert golden["pending"] == 1
    assert golden["items"][0]["query"] == "小柴胡汤治什么？"

    fb2 = c.get("/admin/feedbacks", headers=h).json()["items"][0]
    assert fb2["in_golden"] is True

    status = c.get("/admin/import-status", headers=h).json()
    assert status["golden"]["total"] == 1
    assert "degraded" in status

    missing = c.post("/admin/feedbacks/no-such/golden", headers=h)
    assert missing.status_code == 404

    c.post(
        "/admin/users",
        json={"username": "herb", "password": "herb1234", "role": "user"},
        headers=h,
    )
    herb = _login(c, "herb", "herb1234")
    assert c.post(f"/admin/feedbacks/{msg.id}/golden", headers=herb).status_code == 403
    assert c.get("/admin/golden", headers=herb).status_code == 403
