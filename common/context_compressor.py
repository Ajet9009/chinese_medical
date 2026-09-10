"""W2 short-term memory: pack history by token budget, not by message count.

Redis LTRIM (REDIS_HISTORY_MAX) still caps the hot List by *count*.
This module packs that window into token_budget: keep newest turns that
fit_budget allows; older turns become one LLM summary. System prompt, KG
context, and the current question are not counted here.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Sequence

from common.env_loader import load_app_env

load_app_env()

logger = logging.getLogger("context_compressor")

SUMMARY_PREFIX = "【对话摘要】"
CountFn = Callable[[str], int]
Summarizer = Callable[[list[dict[str, str]]], str]

_SUMMARY_FALLBACK = """你是中医问答系统的对话摘要器。
把较早的多轮问答压成一段短摘要（不超过 120 字）。
必须保留：方剂/药材/症状/证型名称、已给出的结论、尚未确认的追问。
只输出摘要正文，不要标题、不要寒暄。"""


def _testing() -> bool:
    return os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes")


def _compress_enabled() -> bool:
    return os.getenv("CONTEXT_COMPRESS", "1").strip().lower() not in ("0", "false", "no")


def default_token_budget() -> int:
    return max(1, int(os.getenv("TOKEN_BUDGET", "2000")))


def graph_history_limit() -> int:
    return max(1, int(os.getenv("GRAPH_HISTORY_LIMIT", "6")))


def count_tokens(text: str) -> int:
    raw = text or ""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        n = len(enc.encode(raw))
        return max(1, n) if raw else 0
    except Exception:
        return len(raw)


def _msg_tokens(msg: dict[str, str], count_fn: CountFn) -> int:
    return count_fn(str(msg.get("content") or ""))


def fit_budget(
    messages: Sequence[dict[str, str]],
    token_budget: int,
    count_fn: CountFn | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split into (overflow_older, kept_recent) so kept fits token_budget.

    Walk from newest to oldest. The newest message is always kept, even if it
    alone exceeds the budget (current-turn packing still needs some context).
    """
    items = [dict(m) for m in messages]
    if not items:
        return [], []
    count_fn = count_fn or count_tokens
    budget = max(0, int(token_budget))
    kept_rev: list[dict[str, str]] = []
    used = 0
    for msg in reversed(items):
        cost = max(0, _msg_tokens(msg, count_fn))
        if kept_rev and used + cost > budget:
            break
        kept_rev.append(msg)
        used += cost
        if cost > budget and len(kept_rev) == 1:
            break
    kept = list(reversed(kept_rev))
    overflow = items[: len(items) - len(kept)]
    return overflow, kept


def create_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL"),
        temperature=0,
        streaming=False,
    )


def _default_summarizer(overflow: list[dict[str, str]]) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from common.langfuse_manager import fetch_prompt

    lines = []
    for item in overflow:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{item.get('content', '')}")
    prompt = fetch_prompt("context_compress", _SUMMARY_FALLBACK)
    llm = create_llm()
    raw = str(
        llm.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content="\n".join(lines)),
            ]
        ).content
    ).strip()
    return raw.splitlines()[0].strip() if raw else ""


def _pack_with_summary(
    summary: str,
    recent: list[dict[str, str]],
    budget: int,
    count_fn: CountFn,
) -> list[dict[str, str]]:
    summary_msg = {"role": "assistant", "content": f"{SUMMARY_PREFIX}{summary}"}
    used = _msg_tokens(summary_msg, count_fn)
    if used > budget:
        return recent
    kept_rev: list[dict[str, str]] = []
    for msg in reversed(recent):
        cost = _msg_tokens(msg, count_fn)
        if kept_rev and used + cost > budget:
            break
        if not kept_rev and used + cost > budget:
            break
        kept_rev.append(msg)
        used += cost
    kept_rev.reverse()
    return [summary_msg] + kept_rev


def compress_history(
    messages: Sequence[dict[str, str]],
    token_budget: int | None = None,
    count_fn: CountFn | None = None,
    summarizer: Summarizer | None = None,
) -> list[dict[str, str]]:
    items = [dict(m) for m in messages]
    if not items:
        return []
    if not _compress_enabled():
        return items[-graph_history_limit() :]
    budget = token_budget if token_budget is not None else default_token_budget()
    count_fn = count_fn or count_tokens
    overflow, kept = fit_budget(items, budget, count_fn)
    if not overflow:
        return kept

    fn = summarizer
    if fn is None and _testing():
        return kept
    if fn is None:
        fn = _default_summarizer
    try:
        summary = (fn(overflow) or "").strip()
    except Exception as exc:
        logger.warning("历史摘要失败，已按 token 保留最近轮次: %s", exc)
        return kept
    if not summary:
        return kept
    return _pack_with_summary(summary, kept, budget, count_fn)
