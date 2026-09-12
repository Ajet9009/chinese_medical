#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Langfuse 可观测性管理器：封装客户端初始化、trace 创建、采样、flush。

零侵入设计：通过 LangChain CallbackHandler 自动拦截 LLM 调用和 LangGraph 节点执行，
无需修改任何节点代码。所有操作 try-except 包裹，失败不影响主业务流程。

用法:
    mgr = LangfuseManager()
    handler = mgr.create_handler(
        trace_name="POST:/ask",
        metadata={"request_id": "...", "session_id": "...", ...},
    )
    # 传入 LangGraph config
    _graph.invoke(initial, config={"callbacks": [handler]})
"""

from __future__ import annotations

import logging
import os
import random
import uuid
from typing import Any

logger = logging.getLogger("langfuse.integration")

# ── 模型价格（元 / 1M tokens）—— 用于成本核算 ──
INPUT_PRICE_PER_1M = 1.0    # 输入价格
OUTPUT_PRICE_PER_1M = 3.0   # 输出价格


def calc_cost(input_tokens: int, output_tokens: int) -> float:
    """按 deepseek 价格计算单次调用成本（元）。"""
    return input_tokens / 1_000_000 * INPUT_PRICE_PER_1M + \
        output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M


class NoOpSpan:
    """空操作 span，用于 Langfuse 禁用时的占位。"""
    def __enter__(self) -> "NoOpSpan":
        return self
    def __exit__(self, *args: Any) -> None:
        pass
    def end(self) -> None:
        pass


class _NoOpHandler:
    """空操作 callback handler。"""
    def __getattr__(self, _name: str) -> Any:
        return lambda *args, **kwargs: None


class _TraceCallbackHandler:
    """自定义 CallbackHandler — 兼容 langchain v1.x + langfuse v2/v4。

    直接调用 langfuse 原生 API（trace/span/generation）记录追踪数据。
    错误通过 logging.warning 输出到控制台，不会静默吞掉。
    """

    raise_error = False
    run_inline = False
    ignore_llm = False
    ignore_chain = False
    ignore_chat_model = False
    ignore_agent = False
    ignore_retriever = False

    def __init__(self, client: Any, trace_id: str):
        self._client = client
        self._trace_id = trace_id
        self._node_spans: dict[str, Any] = {}
        self._generations: dict[str, Any] = {}
        self._start_times: dict[str, float] = {}
        self._first_token_times: dict[str, float] = {}

    @property
    def trace_id(self) -> str:
        """暴露 trace_id，供外部评分关联。"""
        return self._trace_id
        self._first_token_times: dict[str, float] = {}

    # ── 节点 Span（由 main.py 通过 astream_events 的 event['name'] 驱动）──

    def start_node_span(self, name: str, input: Any = None) -> None:
        """创建节点 span，name 为真实节点名（intent_recognition 等）。

        由 main.py 在 astream_events 循环中调用，因为 callback 的 serialized
        拿不到 LangGraph 节点名（name=None）。
        """
        try:
            span = self._client.span(
                trace_id=self._trace_id,
                name=str(name),
                input=self._truncate(input) if input else "",
            )
            self._node_spans[str(name)] = span
            logger.info("[Langfuse] 节点 span 开始: %s", name)
        except Exception as exc:
            logger.warning("[Langfuse] 节点 span 创建失败 name=%s: %s", name, exc)

    def end_node_span(self, name: str, output: Any = None) -> None:
        """结束节点 span。"""
        span = self._node_spans.pop(str(name), None)
        if span is not None:
            try:
                span.update(output=self._truncate(output) if output else "").end()
                logger.info("[Langfuse] 节点 span 结束: %s", name)
            except Exception as exc:
                logger.warning("[Langfuse] 节点 span 结束失败 name=%s: %s", name, exc)

    # ── Chain 回调（LangGraph 节点名拿不到，留空实现，由 astream_events 驱动）──

    def on_chain_start(
        self, serialized: dict, inputs: dict, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any
    ) -> None:
        pass  # 节点名由 main.py 通过 astream_events 的 event['name'] 获取

    def on_chain_end(self, outputs: dict, *, run_id: Any, **kwargs: Any) -> None:
        pass  # 同上

    # ── Chat Model（ChatOpenAI）──

    def on_chat_model_start(
        self, serialized: dict, messages: list, *, run_id: Any,
        parent_run_id: Any = None, invocation_params: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if serialized is None:
            serialized = {}
        import time
        self._start_times[str(run_id)] = time.time()
        params = invocation_params or {}
        model = params.get("model_name", params.get("model", ""))
        logger.info("[Langfuse] chat model start: model=%s", model)
        try:
            gen = self._client.generation(
                trace_id=self._trace_id,
                name=str(serialized.get("name", "ChatOpenAI")),
                model=str(model),
                input=self._truncate([str(m) for m in (messages or [])], 3000),
            )
            self._generations[str(run_id)] = gen
        except Exception as exc:
            logger.warning("[Langfuse] generation 创建失败 model=%s: %s", model, exc)

    def on_chat_model_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        if response is None:
            return
        gen = self._generations.pop(str(run_id), None)
        start_time = self._start_times.pop(str(run_id), None)
        if gen is not None:
            import time
            try:
                resp_meta = getattr(response, "response_metadata", {}) or {}
                token_usage = resp_meta.get("token_usage", {}) or {}
                llm_output = getattr(response, "llm_output", {}) or {}
                if not token_usage:
                    token_usage = llm_output.get("token_usage", {}) or {}

                input_tokens = token_usage.get("prompt_tokens", 0) or 0
                output_tokens = token_usage.get("completion_tokens", 0) or 0
                logger.info(
                    "[Langfuse] chat model end: input=%s output=%s total=%s tokens",
                    input_tokens, output_tokens,
                    token_usage.get("total_tokens", 0),
                )

                cost = calc_cost(input_tokens, output_tokens)
                gen.update(
                    output=self._truncate(str(getattr(response, "content", response))),
                    usage={
                        "input": input_tokens,
                        "output": output_tokens,
                        "total": token_usage.get("total_tokens"),
                        "unit": "TOKENS",
                    },
                    cost_details={
                        "input": input_tokens / 1_000_000 * INPUT_PRICE_PER_1M,
                        "output": output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M,
                        "total": cost,
                    },
                )
                latency = None
                if start_time:
                    latency = (time.time() - start_time) * 1000
                # TTFT：首 token 时间（流式才有）
                completion_start_time = self._first_token_times.pop(str(run_id), None)
                gen.end(latency=latency, completion_start_time=completion_start_time)
            except Exception as exc:
                logger.warning("[Langfuse] generation 结束失败: %s", exc)

    def on_llm_new_token(
        self, token: str, *, run_id: Any, **kwargs: Any
    ) -> None:
        """流式输出每个 token 时触发，首次记录 TTFT。"""
        import time
        if str(run_id) not in self._first_token_times:
            self._first_token_times[str(run_id)] = time.time()

    # ── LLM 兼容（非 chat model 走这里）──

    def on_llm_start(
        self, serialized: dict, prompts: list, *, run_id: Any, **kwargs: Any
    ) -> None:
        logger.info("[Langfuse] llm start: %s", serialized.get("name", "llm"))
        self.on_chat_model_start(serialized, prompts, run_id=run_id, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        logger.info("[Langfuse] llm end")
        self.on_chat_model_end(response, run_id=run_id, **kwargs)

    # ── 工具 ──

    @staticmethod
    def _truncate(obj: Any, max_len: int = 2000) -> Any:
        try:
            text = str(obj)
            if len(text) > max_len:
                return text[:max_len] + "...[TRUNCATED]"
            return text
        except Exception:
            return str(type(obj))


class LangfuseManager:
    """Langfuse 客户端管理器。

    协议：
    - LANGFUSE_ENABLED=false 或初始化失败 → 全局禁用
    - LANGFUSE_SAMPLE_RATE 控制采样（0~1，默认 0.1）
    - TESTING=true → 跳过真实初始化
    """

    def __init__(self) -> None:
        self._enabled = self._check_enabled()
        self._client = None
        self._sample_rate = self._parse_sample_rate()

        if self._enabled and not os.getenv("TESTING"):
            try:
                import langfuse  # noqa: F401
                # 验证关键密钥存在
                if not os.getenv("LANGFUSE_SECRET_KEY") or not os.getenv("LANGFUSE_PUBLIC_KEY"):
                    logger.warning(
                        "Langfuse 密钥缺失（LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY），"
                        "已禁用 tracing。"
                    )
                    self._enabled = False
                else:
                    self._client = langfuse.Langfuse(
                        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                    )
                    logger.info("Langfuse 客户端已初始化 (host=%s)", os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))
            except ImportError as exc:
                from common.obs import degraded

                degraded("langfuse_init", exc)
                logger.warning("langfuse 包未安装，已禁用 tracing。pip install langfuse")
                self._enabled = False
            except Exception as exc:
                from common.obs import degraded

                degraded("langfuse_init", exc)
                logger.warning("Langfuse 初始化失败: %s，已禁用 tracing。", exc)
                self._enabled = False

    # ------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------

    @staticmethod
    def _check_enabled() -> bool:
        val = os.getenv("LANGFUSE_ENABLED", "true").strip().lower()
        if val in ("false", "0", "no", "off"):
            logger.info("Langfuse 已通过 LANGFUSE_ENABLED=false 禁用")
            return False
        return True

    @staticmethod
    def _parse_sample_rate() -> float:
        try:
            rate = float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
            return max(0.0, min(1.0, rate))
        except (ValueError, TypeError):
            logger.warning("LANGFUSE_SAMPLE_RATE 解析失败，使用默认值 0.1")
            return 0.1

    def _should_sample(self) -> bool:
        return self.should_sample()

    # ------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------

    def is_enabled(self) -> bool:
        """全局启用 + 客户端可用（不含采样判断）。"""
        if not self._enabled:
            return False
        if self._client is None:
            return False
        return True

    def should_sample(self) -> bool:
        """固定概率采样判断（正常请求）。"""
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        return random.random() < self._sample_rate

    def create_handler(
        self,
        trace_name: str = "POST:/ask",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        force: bool = False,
    ) -> Any | None:
        """创建 Langfuse 自定义 CallbackHandler（兼容 langchain v1.x）。

        Args:
            force: True 跳过采样，强制创建（用于异常/重试/低分场景）。

        自动创建 trace 并注入 metadata/tags。若未启用返回 None。
        """
        if not self.is_enabled() or self._client is None:
            return None
        if not force and not self.should_sample():
            return None

        try:
            meta = dict(metadata or {})
            meta.setdefault("request_id", str(uuid.uuid4()))
            meta.setdefault("session_id", str(uuid.uuid4()))
            meta.setdefault("service_version", os.getenv("SERVICE_VERSION", "1.0.0"))

            trace_tags = tags or []
            if "tcm-qa" not in trace_tags:
                trace_tags.append("tcm-qa")

            trace = self._client.trace(
                name=trace_name,
                metadata=meta,
                tags=trace_tags,
            )

            handler = _TraceCallbackHandler(
                client=self._client,
                trace_id=trace.id,
            )

            logger.info(
                "Langfuse trace 已创建 name=%s request_id=%s force=%s",
                trace_name,
                meta.get("request_id", "?"),
                force,
            )
            return handler

        except Exception as exc:
            from common.obs import degraded

            degraded("langfuse_handler", exc)
            logger.warning("创建 Langfuse handler 失败: %s", exc)
            return None

    def record_summary_span(
        self,
        trace_name: str,
        metadata: dict[str, Any] | None,
        summary: dict[str, Any],
    ) -> None:
        """强制采样：记录异常/重试/低分的摘要 trace。

        即使未命中采样也强制记录，只包含一个摘要 span（不重跑图）。
        """
        if not self.is_enabled() or self._client is None:
            return
        try:
            meta = dict(metadata or {})
            meta.setdefault("request_id", str(uuid.uuid4()))
            meta.setdefault("session_id", str(uuid.uuid4()))
            meta.setdefault("service_version", os.getenv("SERVICE_VERSION", "1.0.0"))
            meta["sampling"] = "forced"

            trace = self._client.trace(
                name=trace_name,
                metadata=meta,
                tags=["tcm-qa", "forced-sampling"],
                input={"user_question": summary.get("question", "")},
            )
            span = trace.span(
                name="summary",
                output=self._sanitize_summary(summary),
            )
            span.end()  # 必须 end，span 才会结束并上报
            logger.info(
                "[Langfuse] 强制采样摘要 trace name=%s reason=%s",
                trace_name,
                summary.get("reason", "unknown"),
            )
        except Exception as exc:
            logger.warning("[Langfuse] 记录摘要 trace 失败: %s", exc)

    @staticmethod
    def _sanitize_summary(summary: dict[str, Any]) -> dict[str, Any]:
        """摘要脱敏 + 截断，避免过长。"""
        from common.sanitizer import sanitize
        clean: dict[str, Any] = {}
        for k, v in summary.items():
            if isinstance(v, str):
                clean[k] = sanitize(v[:2000])
            else:
                clean[k] = v
        return clean

    def flush(self, timeout: float = 5.0) -> None:
        """刷新 Langfuse 缓冲区，确保数据上报完成。超时 5 秒。"""
        if not self._enabled or self._client is None:
            return
        try:
            self._client.flush()
            logger.info("Langfuse flush 完成")
        except Exception as exc:
            from common.obs import degraded

            degraded("langfuse_flush", exc)
            logger.warning("Langfuse flush 异常: %s", exc)

    def get_prompt(self, name: str, fallback: str, label: str = "production") -> str:
        """从 Langfuse 获取提示词，失败则返回 fallback。

        Args:
            name: Langfuse 中创建的 prompt 名称。
            fallback: 未启用或拉取失败时使用的本地硬编码提示词。
            label: prompt 标签，默认 "production"。

        Returns:
            提示词文本。优先 Langfuse 版本，不可用时返回 fallback。
        """
        if not self._enabled or self._client is None:
            return fallback
        try:
            prompt = self._client.get_prompt(
                name=name,
                label=label,
                fetch_timeout_seconds=5,   # 超时保护，防止网络阻塞
            )
            logger.info("[Langfuse] 使用远程 prompt: %s (label=%s)", name, label)
            return prompt.prompt  # type: ignore[union-attr]
        except Exception as exc:
            logger.warning("[Langfuse] 获取 prompt '%s' 失败，使用本地 fallback: %s", name, exc)
            return fallback

    @property
    def client(self):
        """暴露原始 Langfuse 客户端（高级用法）。"""
        return self._client


# ── 全局单例（模块级别懒加载）──

_langfuse_mgr: LangfuseManager | None = None


def get_langfuse_manager() -> LangfuseManager:
    """获取全局 LangfuseManager 单例。"""
    global _langfuse_mgr
    if _langfuse_mgr is None:
        _langfuse_mgr = LangfuseManager()
    return _langfuse_mgr


# ── 便捷函数：从 Langfuse 拉取 prompt ──


def fetch_prompt(name: str, fallback: str, label: str = "production", **variables: str) -> str:
    """从 Langfuse 获取提示词，不可用时返回本地 fallback。

    Langfuse 中的 prompt 是模板（含 {entity_types} 等占位符），
    拉取后用 .format(**variables) 替换占位符。fallback 同样处理。

    用法（在节点文件中替换硬编码字符串）:
        from common.langfuse_manager import fetch_prompt
        PROMPT = fetch_prompt("intent_recognition", "...模板...",
                              entity_types=desc, relation_types=desc2)
    """
    prompt = get_langfuse_manager().get_prompt(name, fallback, label)
    if variables:
        try:
            prompt = prompt.format(**variables)
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("[Langfuse] prompt '%s' 变量替换失败，使用原样: %s", name, exc)
    return prompt


# ============================================================
# main() — 直接运行验证 Langfuse 集成
# ============================================================


def main():
    """直接运行 python common/langfuse_manager.py 验证 Langfuse 集成。"""
    import os
    import sys
    import time
    import uuid
    from pathlib import Path

    # 配置日志：控制台输出 WARNING+ 级别（Langfuse 错误可见）
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 确保 .env 加载
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")

    print("=" * 64)
    print("  Langfuse 集成验证")
    print("=" * 64)
    print()

    # 1) 配置摘要
    secret = os.getenv("LANGFUSE_SECRET_KEY", "")
    public = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    enabled = os.getenv("LANGFUSE_ENABLED", "true")
    rate = os.getenv("LANGFUSE_SAMPLE_RATE", "1.0")

    print("  配置:")
    print(f"    LANGFUSE_ENABLED    = {enabled}")
    print(f"    LANGFUSE_SAMPLE_RATE = {rate}")
    print(f"    LANGFUSE_HOST        = {host}")
    print(f"    LANGFUSE_SECRET_KEY  = {'sk-lf-***' + secret[-4:] if len(secret) > 20 else '(未设置)'}")
    print(f"    LANGFUSE_PUBLIC_KEY  = {'pk-lf-***' + public[-4:] if len(public) > 20 else '(未设置)'}")
    print()

    # 2) 初始化
    print("  初始化 LangfuseManager...", end=" ", flush=True)
    mgr = LangfuseManager()
    print("就绪。")

    if not mgr.is_enabled():
        print()
        print("  ⚠  Langfuse 未启用（检查 LANGFUSE_ENABLED、密钥和采样率）")
        if mgr._client is None:
            print("  → 客户端初始化失败（密钥缺失或网络不通）")
        elif not mgr._enabled:
            print("  → LANGFUSE_ENABLED=false")
        else:
            print("  → 未命中采样（当前采样率={:.0%}）".format(mgr._sample_rate))
        print()
        print("  提示：设 LANGFUSE_SAMPLE_RATE=1.0 确保命中采样")
        print(f"{'═' * 64}")
        return

    print(f"  [OK] Langfuse 已启用 (host={host}, sample_rate={mgr._sample_rate:.0%})")
    print()

    # 3) 创建 trace + 模拟 LLM 调用
    print("  创建 trace + 模拟调用...")
    trace_name = "验证:langfuse_manager.main"
    metadata = {
        "request_id": str(uuid.uuid4()),
        "session_id": str(uuid.uuid4()),
        "user_id": "test_user",
        "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
    }
    handler = mgr.create_handler(
        trace_name=trace_name,
        metadata=metadata,
        tags=["tcm-qa", "test"],
    )

    if handler is None:
        print("  [FAIL] 创建 handler 失败")
        return

    print("  [OK] handler 已创建")
    print()

    # 用 OpenAI 兼容接口模拟一次 LLM 调用（产生 Generation span）
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.getenv("MODEL_NAME", "deepseek-chat"),
            api_key=os.getenv("MODEL_API_KEY"),
            base_url=os.getenv("MODEL_BASE_URL"),
            temperature=0,
            max_tokens=50,
        )
        messages = [
            SystemMessage(content="你是一个测试助手，回答简短。"),
            HumanMessage(content="说'你好，Langfuse集成成功'"),
        ]
        print("  调用 LLM...", end=" ", flush=True)
        t0 = time.time()
        resp = llm.invoke(messages, config={"callbacks": [handler]})
        elapsed = (time.time() - t0) * 1000
        print(f"完成 ({elapsed:.0f}ms)")
        print(f"  LLM 回复: {resp.content.strip()[:80]}")
        print()
    except Exception as exc:
        print(f"  [WARN] LLM 调用失败: {exc}")
        print()

    # 4) flush
    print("  Flush 数据到 Langfuse...", end=" ", flush=True)
    mgr.flush(timeout=10.0)
    print("完成。")
    print()

    # 5) 查看地址
    print(f"  在 Langfuse UI 查看 Trace:")
    print(f"  {host.rstrip('/')}/project → Traces → 搜索 '{trace_name}'")
    print()
    print(f"{'═' * 64}")
    print("  验证完成。")
    print(f"{'═' * 64}")


if __name__ == "__main__":
    main()
