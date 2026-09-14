"""Intent node: plain chat JSON, no response_format."""

from __future__ import annotations

import pytest

from _004_langgraph_more_nodes.intent_recognition import (
    kb_title_zhongyi_intent,
    keyword_zhongyi_intent,
    literature_zhongyi_intent,
    make_intent_recognition_node,
    parse_intent_flag,
    resolve_zhongyi_intent,
)
from common.obs import reset_obs, snapshot
from tests.fakes import BoomLLM, FakeLLM


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setattr(
        "_004_langgraph_more_nodes.intent_recognition._get_intent_prompt",
        lambda: "intent-router",
    )
    reset_obs()


def test_tcm_question_updates_only_tcm_flag():
    node = make_intent_recognition_node(FakeLLM('{"is_zhongyi_intent": true}'))
    before = {"user_question": "黄芪有什么中医功效？"}
    update = node(before)
    assert update["is_zhongyi_intent"] is True
    assert {**before, **update}["is_zhongyi_intent"] is True


def test_general_question_sets_flag_to_false():
    node = make_intent_recognition_node(FakeLLM('{"is_zhongyi_intent": false}'))
    out = node({"user_question": "今天天气怎么样？"})
    assert out["is_zhongyi_intent"] is False


def test_parses_json_wrapped_in_prose():
    node = make_intent_recognition_node(
        FakeLLM('判断如下\n{"is_zhongyi_intent": true}\n完毕')
    )
    assert node({"user_question": "四君子汤有什么功效？"})["is_zhongyi_intent"] is True


def test_parse_failure_uses_keywords_and_degraded():
    node = make_intent_recognition_node(FakeLLM("不是 json"))
    assert node({"user_question": "四君子汤有什么功效？"})["is_zhongyi_intent"] is True
    assert node({"user_question": "今天天气怎么样？"})["is_zhongyi_intent"] is False
    assert snapshot()["byTag"].get("intent_parse", 0) >= 2


def test_invoke_error_falls_back_to_keywords():
    node = make_intent_recognition_node(BoomLLM())
    assert node({"user_question": "四君子汤有什么功效？"})["is_zhongyi_intent"] is True
    assert node({"user_question": "今天天气怎么样？"})["is_zhongyi_intent"] is False
    assert snapshot()["byTag"].get("intent_invoke", 0) >= 2
    assert "intent_parse" not in snapshot()["byTag"]


def test_node_rejects_empty_question():
    node = make_intent_recognition_node(FakeLLM("{}"))
    try:
        node({"user_question": "  "})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_parse_intent_flag():
    assert parse_intent_flag('{"is_zhongyi_intent": true}') is True
    assert parse_intent_flag('```json\n{"is_zhongyi_intent": false}\n```') is False
    assert parse_intent_flag("不是 json") is None
    assert keyword_zhongyi_intent("四君子汤功效") is True
    assert keyword_zhongyi_intent("今天天气怎么样") is False
    assert keyword_zhongyi_intent("鱼汤怎么做") is False
    assert keyword_zhongyi_intent("十四经发挥是什么") is False
    assert literature_zhongyi_intent("十四经发挥是什么") is True


def test_llm_false_still_tcm_for_classic_title():
    node = make_intent_recognition_node(FakeLLM('{"is_zhongyi_intent": false}'))
    out = node({"user_question": "十四经发挥是什么"})
    assert out["is_zhongyi_intent"] is True
    assert out["intent_reason"] == "中医典籍"


def test_llm_false_not_rescued_by_ambiguous_qianjin():
    node = make_intent_recognition_node(FakeLLM('{"is_zhongyi_intent": false}'))
    assert node({"user_question": "我家千金感冒了"})["is_zhongyi_intent"] is False


def test_kb_title_overrides_llm_false():
    assert kb_title_zhongyi_intent("十四经发挥是什么", titles=("十四经发挥",))
    assert not kb_title_zhongyi_intent("今天天气怎么样", titles=("十四经发挥",))
    zhongyi, reason = resolve_zhongyi_intent(
        "十四经发挥是什么",
        False,
        titles=("十四经发挥",),
    )
    assert zhongyi is True
    assert reason == "知识库文献"


def test_general_intent_routes_to_doc_retrieval_first():
    from _004_langgraph_more_nodes.graph import _route_after_docs, _route_by_intent

    assert _route_by_intent({"is_zhongyi_intent": False}) == "doc_retrieval"
    assert _route_by_intent({"is_zhongyi_intent": True}) == "entity_extraction"
    assert _route_after_docs({"is_zhongyi_intent": False, "doc_chunks": [], "doc_context": ""}) == "general_response"
    assert _route_after_docs({"is_zhongyi_intent": True, "doc_chunks": [], "doc_context": ""}) == "answer_generation"
    assert (
        _route_after_docs(
            {
                "is_zhongyi_intent": False,
                "doc_chunks": [{"doc_name": "十四经发挥.md", "text": "滑寿"}],
                "doc_context": "滑寿",
            }
        )
        == "answer_generation"
    )
