"""LLM provider catalog / resolve."""

from __future__ import annotations

import pytest

from common.llm import collect_chat_text_step_01, list_llm_providers, normalize_provider, resolve_llm


def test_normalize_blank_is_default():
    assert normalize_provider("") == ""
    assert normalize_provider("default") == ""
    assert normalize_provider("AUTO") == ""


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError, match="未知模型"):
        normalize_provider("claude")


def test_resolve_default_uses_model_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-default")
    monkeypatch.setenv("MODEL_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-v4-flash")
    spec = resolve_llm("")
    assert spec.id == ""
    assert spec.api_key == "sk-default"
    assert spec.model == "deepseek-v4-flash"


def test_resolve_deepseek_falls_back_to_model_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-default")
    monkeypatch.setenv("MODEL_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("MODEL_NAME", "deepseek-chat")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    spec = resolve_llm("deepseek")
    assert spec.id == "deepseek"
    assert spec.api_key == "sk-default"
    assert spec.label == "DeepSeek"


def test_resolve_qwen_uses_dashscope_alias(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-qwen")
    monkeypatch.setenv("QWEN_LLM_MODEL", "qwen-plus")
    spec = resolve_llm("qwen")
    assert spec.id == "qwen"
    assert spec.api_key == "sk-qwen"
    assert "dashscope" in spec.base_url
    assert spec.model == "qwen-plus"


def test_resolve_missing_default_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MODEL_API_KEY"):
        resolve_llm("")


def test_resolve_qwen_missing_key(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API Key"):
        resolve_llm("qwen")


def test_resolve_doubao_needs_endpoint(monkeypatch):
    monkeypatch.setenv("DOUBAO_API_KEY", "sk-doubao")
    monkeypatch.delenv("DOUBAO_LLM_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("DOUBAO_MODEL", raising=False)
    with pytest.raises(ValueError, match="DOUBAO_LLM_ENDPOINT_ID"):
        resolve_llm("doubao")


def test_list_marks_unconfigured(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "sk-default")
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("DOUBAO_API_KEY", raising=False)
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    items = {x["id"]: x for x in list_llm_providers()}
    assert items[""]["configured"] is True
    assert items["qwen"]["configured"] is False
    assert items["doubao"]["configured"] is False


def test_collect_chat_text_prefers_stream():
    class StreamLLM:
        def __init__(self) -> None:
            self.invoke_calls = 0
            self.stream_calls = 0

        def invoke(self, messages):
            self.invoke_calls += 1
            return type("R", (), {"content": "invoke"})()

        def stream(self, messages):
            self.stream_calls += 1
            yield type("R", (), {"content": "补"})()
            yield type("R", (), {"content": "气"})()

    llm = StreamLLM()
    assert collect_chat_text_step_01(llm, []) == "补气"
    assert llm.stream_calls == 1
    assert llm.invoke_calls == 0


def test_collect_chat_text_falls_back_to_invoke():
    class OnlyInvoke:
        def invoke(self, messages):
            return type("R", (), {"content": "  好  "})()

    assert collect_chat_text_step_01(OnlyInvoke(), []) == "好"


def test_collect_chat_text_astream():
    import asyncio

    class StreamLLM:
        def invoke(self, messages):
            return type("R", (), {"content": "invoke"})()

        async def astream(self, messages):
            yield type("R", (), {"content": "补"})()
            yield type("R", (), {"content": "气"})()

    from common.llm import collect_chat_text_astream_step_02

    assert asyncio.run(collect_chat_text_astream_step_02(StreamLLM(), [])) == "补气"


def test_collect_chat_text_stream_ignores_unknown_config_kw():
    class StreamLLM:
        def stream(self, messages):
            yield type("R", (), {"content": "好"})()

    assert collect_chat_text_step_01(StreamLLM(), [], config={"callbacks": []}) == "好"
