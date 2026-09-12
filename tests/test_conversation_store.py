"""Conversation store tests."""

from __future__ import annotations

import json

import pytest

from common.conversation_store import ConversationStore, title_from_question
from tests.fakes import FakeRedis


@pytest.fixture
def store(tmp_path):
    redis = FakeRedis()
    s = ConversationStore(
        db_path=tmp_path / "c.sqlite",
        redis_client=redis,
        redis_history_max=3,
        graph_history_limit=2,
    )
    yield s, redis
    s.close()


def test_create_and_list(store):
    s, _ = store
    a = s.create("甲")
    b = s.create("乙")
    ids = [c.id for c in s.list_conversations()]
    assert a.id in ids and b.id in ids


def test_soft_delete_hides(store):
    s, redis = store
    conv = s.create("将删")
    s.add_message(conv.id, "user", "问")
    s.soft_delete(conv.id)
    assert s.get(conv.id) is None
    assert conv.id not in [c.id for c in s.list_conversations()]
    assert redis.lists.get(f"cm:conv:{conv.id}:msgs") is None


def test_ltrim_keeps_last_n(store):
    s, redis = store
    conv = s.create("裁剪")
    for i in range(5):
        s.add_message(conv.id, "user", f"q{i}")
    key = f"cm:conv:{conv.id}:msgs"
    assert len(redis.lists[key]) == 3
    recent = s.get_recent_for_llm(conv.id)
    assert len(recent) == 3
    assert recent[-1]["content"] == "q4"


def test_redis_down_falls_back_to_sqlite(tmp_path):
    redis = FakeRedis()
    s = ConversationStore(tmp_path / "c.sqlite", redis_client=redis, redis_history_max=50, graph_history_limit=6)
    conv = s.create("降级")
    s.add_message(conv.id, "user", "第一问")
    s.add_message(conv.id, "assistant", "答")
    redis.fail = True
    recent = s.get_recent_for_llm(conv.id)
    assert recent[-1]["content"] == "答"
    s.add_message(conv.id, "user", "第二问")
    msgs = s.get_messages(conv.id)
    assert [m.content for m in msgs] == ["第一问", "答", "第二问"]
    s.close()


def test_title_from_question():
    assert title_from_question("abcdefghij", 4) == "abcd…"
    assert title_from_question("短") == "短"


def test_search_and_batch_delete(store):
    s, _ = store
    a = s.create("四君子汤")
    b = s.create("人参")
    s.create("桂枝汤")
    assert [c.title for c in s.list_conversations(keyword="君子")] == ["四君子汤"]
    n = s.soft_delete_many([a.id, b.id, "missing"])
    assert n == 2
    titles = [c.title for c in s.list_conversations()]
    assert titles == ["桂枝汤"]


def test_delete_messages_rebuilds_redis(store):
    s, redis = store
    conv = s.create("消息")
    u = s.add_message(conv.id, "user", "问")
    a = s.add_message(conv.id, "assistant", "答")
    extra = s.add_message(conv.id, "user", "追问")
    s.delete_messages_after(conv.id, u.id)
    contents = [m.content for m in s.get_messages(conv.id)]
    assert contents == ["问"]
    key = f"cm:conv:{conv.id}:msgs"
    assert json.loads(redis.lists[key][-1])["content"] == "问"
    s.delete_messages(conv.id, [u.id])
    assert s.get_messages(conv.id) == []
    assert redis.lists.get(key) in (None, [])
    _ = extra, a


def test_favorites_and_feedback(store):
    s, _ = store
    conv = s.create("点赞")
    s.add_message(conv.id, "user", "四君子汤功效")
    msg = s.add_message(
        conv.id,
        "assistant",
        "益气健脾",
        details={"elapsed_ms": 1, "refused": True},
    )
    patched = s.patch_message_details(conv.id, msg.id, {"feedback": "dislike"})
    assert patched.details["feedback"] == "dislike"
    items = s.list_feedbacks()
    assert items[0]["question"] == "四君子汤功效"
    assert items[0]["evidence_gap"] is True
    assert s.get_feedback_item(msg.id)["message_id"] == msg.id
    fav = s.add_favorite("四君子汤功效", "益气健脾")
    assert s.list_favorites(keyword="君子")[0].id == fav.id
    s.delete_favorite(fav.id)
    assert s.list_favorites() == []


def test_user_scoped_conversations(store):
    s, _ = store
    a = s.create("甲的会话", user_id="u-a")
    s.create("乙的会话", user_id="u-b")
    titles = [c.title for c in s.list_conversations(user_id="u-a")]
    assert titles == ["甲的会话"]
    assert s.get(a.id, user_id="u-b") is None
    assert s.get(a.id, user_id="u-a") is not None
    assert s.get(a.id, user_id="u-b", is_admin=True) is not None


