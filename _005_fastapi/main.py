#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FastAPI 服务：向外暴露中医知识问答接口。

启动: python -m _005_fastapi.main
      或 uvicorn _005_fastapi.main:app --reload

POST /ask         {"question": "四君子汤有什么功效？"}
POST /ask/stream  {"question": "四君子汤有什么功效？"}  → SSE 流式
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

# 日志配置：控制台输出 INFO+，含时间戳和 logger 名
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from _004_langgraph_more_nodes.graph import build_graph
from common.langfuse_manager import LangfuseManager
from common.eval_manager import get_eval_manager

_graph = build_graph()


def _warmup_faiss() -> None:
    """预热 FAISS + BGE 模型，避免首次请求卡顿 9s。"""
    try:
        from _004_langgraph_more_nodes.entity_normalization import _get_faiss_store
        _get_faiss_store()
        logging.getLogger("main").info("FAISS/BGE 预热完成")
    except Exception as exc:
        logging.getLogger("main").warning("FAISS 预热失败（不影响服务，首次请求会慢）: %s", exc)


def _init_memory(app: FastAPI) -> None:
    """初始化记忆系统（失败不阻塞主流程）。"""
    try:
        from _008_memory.factory import get_memory_manager
        app.state.memory_mgr = get_memory_manager()
        logging.getLogger("main").info("MemoryManager 初始化完成")
    except Exception as exc:
        app.state.memory_mgr = None
        logging.getLogger("main").warning("MemoryManager 初始化失败（记忆功能降级）: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用生命周期：启动时初始化 Langfuse + 预热 FAISS + 记忆，关闭时 flush。"""
    app.state.langfuse_mgr = LangfuseManager()  # noqa: F841
    app.state.memory_mgr = None
    # 预热 FAISS/BGE + 记忆（放线程池，避免阻塞启动）
    await asyncio.to_thread(_warmup_faiss)
    await asyncio.to_thread(_init_memory, app)
    yield
    try:
        app.state.langfuse_mgr.flush(timeout=5.0)
    except Exception:
        pass


app = FastAPI(
    title="中医知识图谱问答系统",
    description="基于 LangGraph + Neo4j 的中医知识问答 API",
    version="0.2.0",
    lifespan=_lifespan,
)

# 节点名称 → 用户可见标签
_NODE_LABELS = {
    "intent_recognition":  "识别意图",
    "entity_extraction":   "抽取中医实体",
    "entity_normalization":"匹配标准实体",
    "cypher_generation":   "生成查询语句",
    "cypher_executor":     "执行图谱查询",
    "answer_generation":   "生成最终回答",
    "general_response":    "生成回答",
}

# ── 请求/响应模型 ──


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")


class MatchedEntityOut(BaseModel):
    id: str
    type: str
    name: str
    score: float


class AskResponse(BaseModel):
    question: str
    intent: str
    intent_reason: str = ""
    is_zhongyi_intent: bool
    user_entities: dict[str, list[str]] = Field(default_factory=dict)
    matched_entities: dict[str, list[MatchedEntityOut]] = Field(default_factory=dict)
    cypher_queries: list[str] = Field(default_factory=list)
    kg_context: str = ""
    answer: str = ""
    elapsed_ms: float = 0.0
    trace_id: str = ""


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="关联的 trace_id")
    feedback: str = Field(..., description="thumbs_up 或 thumbs_down")


def _build_response(
    question: str, final_state: dict, elapsed_ms: float, trace_id: str = ""
) -> AskResponse:
    matched_keys = [
        "matched_symptoms", "matched_diseases", "matched_formulas",
        "matched_herbs", "matched_effects", "matched_sources",
    ]
    matched: dict[str, list[MatchedEntityOut]] = {}
    for k in matched_keys:
        raw = final_state.get(k, []) or []
        matched[k] = [MatchedEntityOut(**e) for e in raw]

    user_keys = [
        "user_input_symptoms", "user_input_diseases", "user_input_formulas",
        "user_input_herbs", "user_input_effects", "user_input_sources",
    ]
    user_entities: dict[str, list[str]] = {}
    for k in user_keys:
        user_entities[k] = final_state.get(k, []) or []

    return AskResponse(
        question=question,
        intent=final_state.get("intent", "general"),
        intent_reason=final_state.get("intent_reason", ""),
        is_zhongyi_intent=final_state.get("is_zhongyi_intent", False),
        user_entities=user_entities,
        matched_entities=matched,
        cypher_queries=final_state.get("cypher_queries", []) or [],
        kg_context=final_state.get("neo4j_answer", "") or "",
        answer=final_state.get("final_answer", "") or "",
        elapsed_ms=round(elapsed_ms, 1),
        trace_id=trace_id,
    )


# ── 记忆辅助 ──


def _get_identity(request: Request) -> tuple[str, str]:
    """获取 (user_id, session_id)。"""
    user_id = request.headers.get("X-User-ID", "anonymous")
    session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
    return user_id, session_id


def _read_memory(request: Request, question: str) -> str:
    """读取记忆上下文，拼接到问题（失败返回原问题）。"""
    mgr = getattr(request.app.state, "memory_mgr", None)
    if mgr is None:
        return question
    try:
        user_id, session_id = _get_identity(request)
        ctx = mgr.get_context(user_id, session_id, question)
        # get_context 返回完整上下文（含"当前输入"），只取记忆部分
        if ctx and "【当前输入】" in ctx:
            ctx = ctx.split("【当前输入】")[0].strip()
        if ctx:
            return f"{question}\n\n[用户历史记忆]\n{ctx}"
    except Exception as exc:
        logging.getLogger("main").warning("读取记忆失败: %s", exc)
    return question


def _write_memory(request: Request, question: str, answer: str) -> None:
    """写入用户消息 + assistant 回答（失败不阻塞主流程）。"""
    mgr = getattr(request.app.state, "memory_mgr", None)
    if mgr is None:
        return
    try:
        user_id, session_id = _get_identity(request)
        mgr.add_message(user_id, session_id, "user", question)
        if answer:
            mgr.add_message(user_id, session_id, "assistant", answer)
    except Exception as exc:
        logging.getLogger("main").warning("写入记忆失败: %s", exc)


# ── Langfuse 辅助 ──


def _make_handler(request: Request, trace_name: str, force: bool = False):
    """从请求创建 Langfuse handler（未启用返回 None）。force=True 跳过采样。"""
    mgr: LangfuseManager = request.app.state.langfuse_mgr
    if not mgr.is_enabled():
        return None

    user_id, session_id = _get_identity(request)
    metadata = {
        "request_id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
    }
    return mgr.create_handler(trace_name=trace_name, metadata=metadata, force=force)


def _force_sample_summary(
    request: Request, trace_name: str, reason: str, summary: dict[str, Any]
) -> None:
    """异常/重试/低分时，即使未命中采样也强制记录摘要 trace。"""
    mgr: LangfuseManager = request.app.state.langfuse_mgr
    user_id, session_id = _get_identity(request)
    mgr.record_summary_span(
        trace_name=trace_name,
        metadata={"session_id": session_id, "user_id": user_id},
        summary={"reason": reason, **summary},
    )


def _check_force_sample(final_state: dict, elapsed_ms: float = 0.0) -> str | None:
    """检查 final_state + 耗时，判断是否需要强制采样。返回原因或 None。

    触发条件：重试 / 无 Cypher / 空回答 / 慢请求（>30s）。
    """
    # 慢请求：耗时超过 30 秒
    if elapsed_ms > 30_000:
        return "slow_request"
    # 重试：Cypher 校验失败重试过
    if final_state.get("cypher_retry_count", 0) > 0:
        return "cypher_retry"
    # 低分：中医意图但没生成 Cypher，或最终回答为空
    if final_state.get("is_zhongyi_intent") and not final_state.get("cypher_queries"):
        return "no_cypher"
    if not final_state.get("final_answer"):
        return "empty_answer"
    return None


# ── 普通接口 ──


@app.get("/health")
def health(request: Request):
    mem = getattr(request.app.state, "memory_mgr", None)
    memory = mem.health() if mem is not None and hasattr(mem, "health") else None
    return {"status": "ok", "memory": memory}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    t0 = time.time()
    # 读取记忆，拼接到问题
    enhanced_question = _read_memory(request, payload.question)
    initial = {"user_question": enhanced_question}
    handler = _make_handler(request, "POST:/ask")
    config = {"callbacks": [handler]} if handler else None
    try:
        final_state = _graph.invoke(initial, config=config) if config else _graph.invoke(initial)
    except Exception as exc:
        # 异常强制采样：即使未命中采样也记录异常现场
        _force_sample_summary(request, "POST:/ask", "exception",
                              {"error": str(exc), "question": payload.question})
        raise HTTPException(status_code=500, detail=f"图执行失败: {exc}")

    elapsed = (time.time() - t0) * 1000

    # 重试/低分/慢请求强制采样
    reason = _check_force_sample(final_state, elapsed)
    if reason:
        _force_sample_summary(request, "POST:/ask", reason,
                              {"question": payload.question, "elapsed_ms": elapsed})

    # 自动评分（命中采样的 trace）
    if handler is not None:
        get_eval_manager().score_auto(handler.trace_id, final_state)

    # 写入记忆（用户消息 + 回答）
    answer = final_state.get("final_answer", "") or ""
    _write_memory(request, payload.question, answer)

    # 显式 flush Langfuse，确保 trace/span/generation 立即上报
    request.app.state.langfuse_mgr.flush(timeout=5.0)

    trace_id = handler.trace_id if handler is not None else ""
    return _build_response(payload.question, final_state, elapsed, trace_id)


# ── 用户反馈 ──


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    """接收用户 👍/👎 反馈，写回 Langfuse score。"""
    evm = get_eval_manager()
    value = 1.0 if payload.feedback == "thumbs_up" else 0.0
    evm.score_trace(
        trace_id=payload.trace_id,
        name="user_feedback",
        value=value,
        data_type="NUMERIC",
    )
    return {"status": "ok", "trace_id": payload.trace_id, "feedback": payload.feedback}


# ── SSE 流式接口 ──


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request):
    """SSE 流式问答：逐节点推送进度，最终返回完整结果。"""

    handler = _make_handler(request, "POST:/ask/stream")
    config = {"callbacks": [handler]} if handler else None

    # 读取记忆，拼接到问题
    enhanced_question = _read_memory(request, payload.question)

    async def event_stream() -> AsyncGenerator[str, None]:
        t0 = time.time()
        initial = {"user_question": enhanced_question}
        queue: asyncio.Queue = asyncio.Queue()

        # ── 生产者：只跑一遍图，聚合最终 state，把事件放进队列 ──
        async def producer() -> None:
            try:
                final_state = dict(initial)
                async for event in _graph.astream_events(initial, version="v2", config=config):
                    await queue.put(("event", event))
                    # 聚合节点输出到最终 state（无需再 invoke 第二遍）
                    if event.get("event") == "on_chain_end" and event.get("name") in _NODE_LABELS:
                        output = event.get("data", {}).get("output", {}) or {}
                        if isinstance(output, dict):
                            final_state.update(output)
                await queue.put(("final", final_state))
            except Exception as exc:
                await queue.put(("error", exc))
            finally:
                await queue.put(("end", None))

        producer_task = asyncio.create_task(producer())

        # ── 消费者：从队列取事件，yield SSE ──
        try:
            while True:
                msg_type, item = await queue.get()

                if msg_type == "end":
                    break

                if msg_type == "error":
                    # 异常强制采样
                    _force_sample_summary(request, "POST:/ask/stream", "exception",
                                          {"error": str(item), "question": payload.question})
                    yield _sse("error", {"error": str(item)})
                    break

                if msg_type == "event":
                    event = item
                    kind = event.get("event", "")
                    node_name = event.get("name", "")

                    # 节点开始
                    if kind == "on_chain_start" and node_name in _NODE_LABELS:
                        if handler is not None:
                            handler.start_node_span(node_name, input=event.get("data", {}).get("input"))
                        yield _sse("progress", {
                            "node": node_name,
                            "label": _NODE_LABELS[node_name],
                            "status": "running",
                        })
                        await asyncio.sleep(0)

                    # 节点结束
                    elif kind == "on_chain_end" and node_name in _NODE_LABELS:
                        output = event.get("data", {}).get("output", {}) or {}
                        if handler is not None:
                            handler.end_node_span(node_name, output=output)
                        # 精简 output（列表/长文本截断）
                        short: dict[str, Any] = {}
                        for k, v in output.items():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                short[k] = [f"{x.get('name','?')}({x.get('score',0):.2f})" for x in v[:5]]
                            elif isinstance(v, list):
                                short[k] = v[:3]
                            elif isinstance(v, str) and len(v) > 200:
                                short[k] = v[:200] + "..."
                            else:
                                short[k] = v
                        yield _sse("progress", {
                            "node": node_name,
                            "label": _NODE_LABELS[node_name],
                            "status": "done",
                            "output": short,
                        })
                        await asyncio.sleep(0)

                if msg_type == "final":
                    final_state = item
                    elapsed = (time.time() - t0) * 1000
                    # 重试/低分/慢请求强制采样
                    reason = _check_force_sample(final_state, elapsed)
                    if reason:
                        _force_sample_summary(request, "POST:/ask/stream", reason,
                                              {"question": payload.question, "elapsed_ms": elapsed})
                    # 自动评分（命中采样的 trace）
                    if handler is not None:
                        get_eval_manager().score_auto(handler.trace_id, final_state)
                    # 写入记忆（用户消息 + 回答）
                    answer = final_state.get("final_answer", "") or ""
                    _write_memory(request, payload.question, answer)
                    # 显式 flush Langfuse
                    request.app.state.langfuse_mgr.flush(timeout=5.0)
                    trace_id = handler.trace_id if handler is not None else ""
                    resp = _build_response(payload.question, final_state, elapsed, trace_id)
                    yield _sse("done", resp.model_dump())
        finally:
            await producer_task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── main() ──


def main():
    import uvicorn
    uvicorn.run("_005_fastapi.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
