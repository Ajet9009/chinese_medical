"""SSE helpers: only stream tokens from the final-answer nodes."""

from __future__ import annotations

from typing import Any

ANSWER_STREAM_NODES = frozenset({"answer_generation", "general_response"})


def is_answer_token_event(event: dict[str, Any]) -> bool:
    if event.get("event") != "on_chat_model_stream":
        return False
    node = (event.get("metadata") or {}).get("langgraph_node")
    return node in ANSWER_STREAM_NODES


def token_text_from_event(event: dict[str, Any]) -> str:
    chunk = (event.get("data") or {}).get("chunk")
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if content is None and isinstance(chunk, dict):
        content = chunk.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
        return "".join(parts)
    return str(content or "")
