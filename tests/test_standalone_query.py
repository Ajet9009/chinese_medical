"""Standalone query (coreference) node tests."""

from __future__ import annotations

from _004_langgraph_more_nodes.standalone_query import make_standalone_query_node, standalone_query_node
from tests.fakes import BoomLLM, FakeLLM


def test_no_history_passthrough_without_llm():
    out = standalone_query_node({"user_question": "四君子汤有什么功效？", "messages": []})
    assert out["search_question"] == "四君子汤有什么功效？"


def test_history_rewrites_with_mock_llm():
    llm = FakeLLM("四君子汤由哪些药组成？")
    node = make_standalone_query_node(llm)
    out = node({
        "user_question": "它由哪些药组成？",
        "messages": [
            {"role": "user", "content": "四君子汤有什么功效？"},
            {"role": "assistant", "content": "补气健脾"},
        ],
    })
    assert out["search_question"] == "四君子汤由哪些药组成？"
    assert llm.calls == 1


def test_llm_failure_falls_back_to_original():
    node = make_standalone_query_node(BoomLLM())
    out = node({
        "user_question": "它呢？",
        "messages": [{"role": "user", "content": "四君子汤"}],
    })
    assert out["search_question"] == "它呢？"
