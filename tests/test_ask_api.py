"""Ask API + SSE token filter tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from _005_fastapi.sse import is_answer_token_event, token_text_from_event
from tests.fakes import FakeRedis


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGraph:
    def invoke(self, initial, config=None):
        return {
            **initial,
            "search_question": initial.get("user_question"),
            "intent": "tcm",
            "intent_reason": "方剂",
            "is_zhongyi_intent": True,
            "final_answer": "补气健脾",
            "cypher_queries": ["MATCH (n) RETURN n"],
            "neo4j_answer": "四君子汤 HAS_EFFECT 补气健脾",
        }

    async def astream_events(self, initial, version="v2", config=None):
        yield {
            "event": "on_chain_start",
            "name": "intent_recognition",
            "data": {"input": {}},
            "metadata": {},
        }
        yield {
            "event": "on_chain_end",
            "name": "intent_recognition",
            "data": {"output": {"intent": "tcm", "is_zhongyi_intent": True, "intent_reason": "方剂"}},
            "metadata": {"langgraph_node": "intent_recognition"},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _Chunk("MATCH")},
            "metadata": {"langgraph_node": "cypher_generation"},
        }
        yield {
            "event": "on_chat_model_stream",
            "name": "ChatOpenAI",
            "data": {"chunk": _Chunk("补气")},
            "metadata": {"langgraph_node": "answer_generation"},
        }
        yield {
            "event": "on_chain_end",
            "name": "answer_generation",
            "data": {"output": {"final_answer": "补气健脾"}},
            "metadata": {"langgraph_node": "answer_generation"},
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
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

    get_store = main.get_store
    get_store.cache_clear()
    cached = getattr(deps.get_users, "cache_clear", None)
    if cached:
        cached()

    from common.conversation_store import ConversationStore
    from common.user_store import UserStore

    db = tmp_path / "c.sqlite"
    store = ConversationStore(db, redis_client=FakeRedis())
    users = UserStore(db)
    users.seed_admin()
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(main, "get_users", lambda: users)
    monkeypatch.setattr(deps, "get_users", lambda: users)
    monkeypatch.setattr(main, "get_graph", lambda: FakeGraph())
    monkeypatch.setattr(main, "redis_status", lambda: "ok")

    from common.knowledge_service import reset_knowledge_service

    reset_knowledge_service()

    with TestClient(main.app) as c:
        resp = c.post(
            "/auth/login",
            json={"username": "admin", "password": "admin123"},
        )
        assert resp.status_code == 200, resp.text
        headers = {"Authorization": f"Bearer {resp.json()['token']}"}
        yield c, store, headers

    get_store.cache_clear()
    cached = getattr(deps.get_users, "cache_clear", None)
    if cached:
        cached()


def test_token_filter_only_answer_nodes():
    cypher_ev = {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": "cypher_generation"},
        "data": {"chunk": _Chunk("MATCH")},
    }
    answer_ev = {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": "answer_generation"},
        "data": {"chunk": _Chunk("补气")},
    }
    assert is_answer_token_event(cypher_ev) is False
    assert is_answer_token_event(answer_ev) is True
    assert token_text_from_event(answer_ev) == "补气"


def test_ask_creates_conversation(client):
    c, store, headers = client
    resp = c.post("/ask", json={"question": "四君子汤有什么功效？"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"]
    assert body["answer"] == "补气健脾"
    convs = c.get("/conversations", headers=headers).json()
    assert len(convs) == 1
    msgs = c.get(f"/conversations/{body['conversation_id']}/messages", headers=headers).json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]


def test_stream_emits_session_token_done(client):
    c, _store, headers = client
    with c.stream("POST", "/ask/stream", json={"question": "四君子汤有什么功效？"}, headers=headers) as resp:
        text = "".join(resp.iter_text())
    assert "event: session" in text
    assert "event: token" in text
    assert "补气" in text
    assert "MATCH" not in text.split("event: token")[1].split("event:")[0] if "event: token" in text else True
    assert "event: done" in text


def test_rename_and_soft_delete(client):
    c, _store, headers = client
    created = c.post("/conversations", json={"title": "旧名"}, headers=headers).json()
    cid = created["id"]
    renamed = c.patch(f"/conversations/{cid}", json={"title": "新名"}, headers=headers).json()
    assert renamed["title"] == "新名"
    assert c.delete(f"/conversations/{cid}", headers=headers).status_code == 204
    assert c.get("/conversations", headers=headers).json() == []
    assert c.get(f"/conversations/{cid}/messages", headers=headers).status_code == 404


def test_search_batch_delete_favorites_feedback(client):
    c, _store, headers = client
    a = c.post("/conversations", json={"title": "四君子汤"}, headers=headers).json()
    b = c.post("/conversations", json={"title": "人参"}, headers=headers).json()
    found = c.get("/conversations", params={"keyword": "君子"}, headers=headers).json()
    assert [x["title"] for x in found] == ["四君子汤"]
    deleted = c.post("/conversations/batch-delete", json={"ids": [a["id"], b["id"]]}, headers=headers).json()
    assert deleted["deleted"] == 2
    assert c.get("/conversations", headers=headers).json() == []

    conv = c.post("/conversations", json={"title": "收藏"}, headers=headers).json()
    c.post("/ask", json={"question": "四君子汤有什么功效？", "conversation_id": conv["id"]}, headers=headers)
    msgs = c.get(f"/conversations/{conv['id']}/messages", headers=headers).json()
    aid = next(m["id"] for m in msgs if m["role"] == "assistant")
    patched = c.patch(
        f"/conversations/{conv['id']}/messages/{aid}",
        json={"feedback": "like"},
        headers=headers,
    ).json()
    assert patched["details"]["feedback"] == "like"
    fav = c.post("/favorites", json={"query": "四君子汤功效", "answer": "益气健脾"}, headers=headers).json()
    assert c.get("/favorites", headers=headers).json()[0]["id"] == fav["id"]
    assert c.delete(f"/favorites/{fav['id']}", headers=headers).status_code == 204
    uid = next(m["id"] for m in msgs if m["role"] == "user")
    c.post(
        f"/conversations/{conv['id']}/messages/batch-delete",
        json={"ids": [aid]},
        headers=headers,
    )
    left = c.get(f"/conversations/{conv['id']}/messages", headers=headers).json()
    assert [m["id"] for m in left] == [uid]

