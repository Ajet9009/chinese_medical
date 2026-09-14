"""OpenAI 兼容 LLM 工厂：按请求选 DeepSeek / 通义 / 豆包。

对齐 grid-qa 的「下拉选模型 + 每问传入 model_type」。
各家都走 ChatOpenAI(base_url)，不依赖 response_format。
未配置的厂商在 list 里 configured=false；真正调用时缺 key 会 ValueError。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from common.env_loader import load_app_env

load_app_env()

KNOWN_PROVIDERS = ("deepseek", "qwen", "doubao")


@dataclass(frozen=True)
class LlmSpec:
    id: str
    label: str
    api_key: str
    base_url: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


def _env(*names: str, default: str = "") -> str:
    for name in names:
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    return default


def _default_spec() -> LlmSpec:
    return LlmSpec(
        id="",
        label="默认模型",
        api_key=_env("MODEL_API_KEY"),
        base_url=_env("MODEL_BASE_URL", default="https://api.deepseek.com"),
        model=_env("MODEL_NAME", default="deepseek-chat"),
    )


def _named_spec(provider: str) -> LlmSpec:
    if provider == "deepseek":
        return LlmSpec(
            id="deepseek",
            label="DeepSeek",
            api_key=_env("DEEPSEEK_API_KEY", "MODEL_API_KEY"),
            base_url=_env("DEEPSEEK_BASE_URL", "MODEL_BASE_URL", default="https://api.deepseek.com"),
            model=_env("DEEPSEEK_MODEL", "MODEL_NAME", default="deepseek-chat"),
        )
    if provider == "qwen":
        return LlmSpec(
            id="qwen",
            label="通义千问",
            api_key=_env("QWEN_API_KEY", "DASHSCOPE_API_KEY"),
            base_url=_env(
                "QWEN_BASE_URL",
                "DASHSCOPE_BASE_URL",
                default="https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            model=_env("QWEN_LLM_MODEL", "QWEN_MODEL", default="qwen-plus"),
        )
    if provider == "doubao":
        return LlmSpec(
            id="doubao",
            label="豆包",
            api_key=_env("DOUBAO_API_KEY", "ARK_API_KEY"),
            base_url=_env(
                "DOUBAO_BASE_URL",
                "ARK_BASE_URL",
                default="https://ark.cn-beijing.volces.com/api/v3",
            ),
            model=_env("DOUBAO_LLM_ENDPOINT_ID", "DOUBAO_MODEL", default=""),
        )
    raise ValueError(f"未知模型: {provider}（支持: deepseek | qwen | doubao）")


def normalize_provider(name: str | None) -> str:
    raw = (name or "").strip().lower()
    if raw in ("", "default", "auto"):
        return ""
    if raw not in KNOWN_PROVIDERS:
        raise ValueError(f"未知模型: {raw}（支持: deepseek | qwen | doubao）")
    return raw


def resolve_llm(name: str | None) -> LlmSpec:
    pid = normalize_provider(name)
    spec = _default_spec() if pid == "" else _named_spec(pid)
    if not spec.api_key:
        hint = "MODEL_API_KEY" if pid == "" else f"{spec.label} 的 API Key"
        raise ValueError(f"未配置 {hint}")
    if pid == "doubao" and not spec.model:
        raise ValueError("未配置 DOUBAO_LLM_ENDPOINT_ID（豆包需推理接入点）")
    return spec


def list_llm_providers() -> list[dict[str, Any]]:
    items = [_default_spec(), *(_named_spec(p) for p in KNOWN_PROVIDERS)]
    return [
        {
            "id": it.id,
            "label": it.label,
            "configured": it.configured and (it.id != "doubao" or bool(it.model)),
            "model": it.model,
        }
        for it in items
    ]


def get_chat_model(
    provider: str | None = None,
    *,
    streaming: bool = False,
    temperature: float = 0,
    max_tokens: int | None = None,
):
    """按厂商构造 ChatOpenAI。provider 空则用 MODEL_*。"""
    from langchain_openai import ChatOpenAI

    spec = resolve_llm(provider)
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "api_key": spec.api_key,
        "base_url": spec.base_url,
        "temperature": temperature,
        "streaming": streaming,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _chunk_text_step_01(chunk: Any) -> str:
    """步骤：01 从 ChatOpenAI.stream 的 chunk 取出纯文本。由 collect_chat_text_step_01 调用。"""
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


def _iter_llm_stream_step_03(llm: Any, messages: Any, config: Any | None) -> Any:
    """步骤：03 调用 llm.stream；有 config 就传入，测试替身不接受时回退。由 collect_chat_text_step_01 调用。"""
    stream_fn = getattr(llm, "stream", None)
    if not callable(stream_fn):
        return None
    if config is None:
        return stream_fn(messages)
    try:
        return stream_fn(messages, config=config)
    except TypeError:
        return stream_fn(messages)


def _iter_llm_astream_step_04(llm: Any, messages: Any, config: Any | None) -> Any:
    """步骤：04 调用 llm.astream；有 config 就传入。由 collect_chat_text_astream_step_02 调用。"""
    astream_fn = getattr(llm, "astream", None)
    if not callable(astream_fn):
        return None
    if config is None:
        return astream_fn(messages)
    try:
        return astream_fn(messages, config=config)
    except TypeError:
        return astream_fn(messages)


def collect_chat_text_step_01(
    llm: Any, messages: Any, *, streaming: bool = True, config: Any | None = None
) -> str:
    """步骤：01 优先 llm.stream 拼接完整回答；streaming=False 时只用 invoke。

    把 LangGraph 的 config 传给 stream，才能冒出 on_chat_model_stream 给 SSE。
    """
    if streaming:
        chunks = _iter_llm_stream_step_03(llm, messages, config)
        if chunks is not None:
            parts = [_chunk_text_step_01(chunk) for chunk in chunks]
            return "".join(parts).strip()
    resp = llm.invoke(messages)
    return str(getattr(resp, "content", resp) or "").strip()


async def collect_chat_text_astream_step_02(
    llm: Any, messages: Any, config: Any | None = None
) -> str:
    """步骤：02 异步 astream 拼完整回答，并把 config 传给模型以驱动 SSE token。"""
    stream = _iter_llm_astream_step_04(llm, messages, config)
    if stream is not None:
        parts: list[str] = []
        async for chunk in stream:
            parts.append(_chunk_text_step_01(chunk))
        return "".join(parts).strip()
    return collect_chat_text_step_01(llm, messages, config=config)
