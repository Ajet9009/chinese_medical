"""Intent recognition node tests (mock LLM, no network)."""

from __future__ import annotations

import pytest

from _004_langgraph_more_nodes.intent_recognition import make_intent_recognition_node
from tests.fakes import FakeLLM


def test_tcm_intent_from_json():
    node = make_intent_recognition_node(FakeLLM('{"intent": "tcm", "reason": "方剂功效"}'))
    out = node({"user_question": "四君子汤有什么功效？"})
    assert out["intent"] == "tcm"
    assert out["is_zhongyi_intent"] is True
    assert "方剂" in out["intent_reason"]


def test_general_intent_from_json():
    node = make_intent_recognition_node(FakeLLM('{"intent": "general", "reason": "闲聊"}'))
    out = node({"user_question": "今天天气怎么样？"})
    assert out["intent"] == "general"
    assert out["is_zhongyi_intent"] is False


def test_empty_question_raises():
    node = make_intent_recognition_node(FakeLLM('{"intent": "tcm", "reason": "x"}'))
    with pytest.raises(ValueError):
        node({"user_question": "  "})
