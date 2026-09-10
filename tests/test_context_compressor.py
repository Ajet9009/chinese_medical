"""W2 ContextCompressor: token_budget / fit_budget vs Redis LTRIM count trim."""

from __future__ import annotations

from tests.fakes import FakeLLM


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def _turns(*texts: str) -> list[dict[str, str]]:
    out = []
    for i, text in enumerate(texts):
        out.append(_msg("user" if i % 2 == 0 else "assistant", text))
    return out


def test_fit_budget_keeps_all_under_limit():
    from common.context_compressor import fit_budget

    msgs = _turns("问甲", "答甲", "问乙", "答乙")
    overflow, kept = fit_budget(msgs, token_budget=100, count_fn=len)
    assert overflow == []
    assert kept == msgs


def test_fit_budget_keeps_newest_when_over():
    from common.context_compressor import fit_budget

    msgs = _turns("AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD")
    overflow, kept = fit_budget(msgs, token_budget=20, count_fn=len)
    assert kept == msgs[-2:]
    assert overflow == msgs[:-2]
    assert sum(len(m["content"]) for m in kept) <= 20


def test_fit_budget_keeps_newest_even_if_it_exceeds_budget():
    from common.context_compressor import fit_budget

    msgs = [_msg("assistant", "X" * 50)]
    overflow, kept = fit_budget(msgs, token_budget=10, count_fn=len)
    assert overflow == []
    assert kept == msgs


def test_compress_noop_when_under_budget():
    from common.context_compressor import compress_history

    summarizer = FakeLLM("不应调用")
    msgs = _turns("问", "答")
    out = compress_history(
        msgs,
        token_budget=100,
        count_fn=len,
        summarizer=lambda _m: summarizer.invoke([]).content,
    )
    assert out == msgs
    assert summarizer.calls == 0


def test_compress_summarizes_overflow_and_keeps_recent():
    from common.context_compressor import compress_history

    seen = []

    def summarizer(overflow):
        seen.append(overflow)
        return "旧轮次讨论了四君子汤"

    msgs = _turns("AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE", "FFFFFFFF")
    out = compress_history(
        msgs,
        token_budget=40,
        count_fn=len,
        summarizer=summarizer,
    )
    assert seen, "超预算时应调用摘要"
    overflow = seen[0]
    assert overflow[0]["content"] == "AAAAAAAA"
    assert overflow[-1]["content"] != "FFFFFFFF"
    assert any("【对话摘要】" in m["content"] and "四君子汤" in m["content"] for m in out)
    assert out[-1]["content"] == "FFFFFFFF"
    assert not any(m["content"] == "AAAAAAAA" for m in out)


def test_compress_llm_failure_drops_overflow_not_recent():
    from common.context_compressor import compress_history

    msgs = _turns("AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD")

    def boom(_overflow):
        raise RuntimeError("llm down")

    out = compress_history(
        msgs,
        token_budget=20,
        count_fn=len,
        summarizer=boom,
    )
    assert out[-1]["content"] == "DDDDDDDD"
    assert not any("【对话摘要】" in m["content"] for m in out)
    assert not any(m["content"] == "AAAAAAAA" for m in out)


def test_testing_env_does_not_call_real_llm(monkeypatch):
    from common.context_compressor import compress_history

    monkeypatch.setenv("TESTING", "1")
    called = {"n": 0}

    def fake_create():
        called["n"] += 1
        raise AssertionError("must not create LLM in TESTING")

    monkeypatch.setattr("common.context_compressor.create_llm", fake_create)
    msgs = _turns("AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD")
    out = compress_history(msgs, token_budget=20, count_fn=len)
    assert called["n"] == 0
    assert out[-1]["content"] == "DDDDDDDD"


def test_ltrim_is_not_token_budget():
    """LTRIM 按条数；Compressor 按 token。同一条很长的回答会先触发 token 裁剪。"""
    from common.context_compressor import fit_budget

    short = [_msg("user", "好"), _msg("assistant", "嗯")]
    long = [_msg("user", "问"), _msg("assistant", "答" * 40)]
    assert len(short) == len(long) == 2
    o_short, k_short = fit_budget(short, token_budget=20, count_fn=len)
    o_long, k_long = fit_budget(long, token_budget=20, count_fn=len)
    assert o_short == [] and k_short == short
    assert k_long[-1]["content"] == "答" * 40
    assert o_long == long[:1]


def test_compress_disabled_falls_back_to_count(monkeypatch):
    from common.context_compressor import compress_history

    monkeypatch.setenv("CONTEXT_COMPRESS", "0")
    monkeypatch.setenv("GRAPH_HISTORY_LIMIT", "2")
    msgs = _turns("AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD")
    out = compress_history(msgs, token_budget=20, count_fn=len)
    assert out == msgs[-2:]
