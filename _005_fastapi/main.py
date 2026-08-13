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

_graph = build_graph()


def _warmup_faiss() -> None:
    """预热 FAISS + BGE 模型，避免首次请求卡顿 9s。"""
    try:
        from _004_langgraph_more_nodes.entity_normalization import _get_faiss_store
        _get_faiss_store()
        logging.getLogger("main").info("FAISS/BGE 预热完成")
    except Exception as exc:
        logging.getLogger("main").warning("FAISS 预热失败（不影响服务，首次请求会慢）: %s", exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用生命周期：启动时初始化 Langfuse + 预热 FAISS，关闭时 flush。"""
    app.state.langfuse_mgr = LangfuseManager()  # noqa: F841
    # 预热 FAISS/BGE（放线程池，避免阻塞启动）
    await asyncio.to_thread(_warmup_faiss)
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


def _build_response(question: str, final_state: dict, elapsed_ms: float) -> AskResponse:
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
    )


# ── Langfuse 辅助 ──


def _make_handler(request: Request, trace_name: str):
    """从请求创建 Langfuse handler（未启用返回 None）。"""
    mgr: LangfuseManager = request.app.state.langfuse_mgr
    if not mgr.is_enabled():
        return None

    session_id = request.headers.get("X-Session-ID", str(uuid.uuid4()))
    metadata = {
        "request_id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": "anonymous",
    }
    return mgr.create_handler(trace_name=trace_name, metadata=metadata)


# ── 普通接口 ──


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    t0 = time.time()
    initial = {"user_question": payload.question}
    handler = _make_handler(request, "POST:/ask")
    config = {"callbacks": [handler]} if handler else None
    try:
        final_state = _graph.invoke(initial, config=config) if config else _graph.invoke(initial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图执行失败: {exc}")
    elapsed = (time.time() - t0) * 1000
    return _build_response(payload.question, final_state, elapsed)


# ── SSE 流式接口 ──


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request):
    """SSE 流式问答：逐节点推送进度，最终返回完整结果。"""

    handler = _make_handler(request, "POST:/ask/stream")
    config = {"callbacks": [handler]} if handler else None

    async def event_stream() -> AsyncGenerator[str, None]:
        t0 = time.time()
        initial = {"user_question": payload.question}
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
                    resp = _build_response(initial["user_question"], final_state, elapsed)
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
