"""General response node tests (mock LLM, no network)."""

from __future__ import annotations

import pytest

from _004_langgraph_more_nodes.general_response import make_general_response_node
from tests.fakes import FakeLLM


def test_returns_llm_content():
    node = make_general_response_node(FakeLLM("  这是普通回答。  "))
    out = node.invoke({"user_question": "今天天气怎么样？"})
    assert out["final_answer"] == "这是普通回答。"


def test_empty_question_raises():
    node = make_general_response_node(FakeLLM("ok"))
    with pytest.raises(ValueError):
        node.invoke({"user_question": ""})


def test_includes_history_in_prompt():
    captured: list = []

    class CaptureLLM:
        def invoke(self, messages):
            captured.append(messages)
            return type("R", (), {"content": "好的"})()

    node = make_general_response_node(CaptureLLM())
    node.invoke({
        "user_question": "那明天呢？",
        "messages": [
            {"role": "user", "content": "今天天气怎么样？"},
            {"role": "assistant", "content": "晴"},
        ],
    })
    assert captured
    human = str(captured[0][-1].content)
    assert "今天天气怎么样？" in human
    assert "那明天呢？" in human


def test_streams_chunks():
    class ChunkLLM:
        def stream(self, messages):
            yield type("R", (), {"content": "这是"})()
            yield type("R", (), {"content": "普通回答。"})()

    node = make_general_response_node(ChunkLLM())
    out = node.invoke({"user_question": "今天天气怎么样？"})
    assert out["final_answer"] == "这是普通回答。"


def test_astream_chunks():
    import asyncio

    class ChunkLLM:
        async def astream(self, messages):
            yield type("R", (), {"content": "这是"})()
            yield type("R", (), {"content": "异步回答。"})()

    node = make_general_response_node(ChunkLLM())
    out = asyncio.run(node.ainvoke({"user_question": "今天天气怎么样？"}))
    assert out["final_answer"] == "这是异步回答。"
