#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FastAPI 服务：中医知识问答 + 会话。

启动: python -m _005_fastapi.main
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
from functools import lru_cache
from typing import Any, Literal, AsyncGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from _005_fastapi.deps import get_current_user, get_users, require_admin
from _005_fastapi.sse import is_answer_token_event, token_text_from_event
from common.conversation_store import ConversationStore, Message, default_store
from common.context_compressor import compress_history
from common.env_loader import load_app_env
from common.langfuse_manager import LangfuseManager
from common.redis_client import get_redis, redis_status
from common.user_store import User

load_app_env()

_graph = None

_NODE_LABELS = {
    "standalone_query": "消解指代",
    "intent_recognition": "识别意图",
    "entity_extraction": "抽取中医实体",
    "entity_normalization": "匹配标准实体",
    "cypher_generation": "生成查询语句",
    "cypher_executor": "执行图谱查询",
    "answer_generation": "生成最终回答",
    "general_response": "生成回答",
}


def get_graph():
    global _graph
    if _graph is None:
        from _004_langgraph_more_nodes.graph import build_graph
        _graph = build_graph()
    return _graph


@lru_cache(maxsize=1)
def get_store() -> ConversationStore:
    return default_store(redis_client=get_redis())


def _warmup_faiss() -> None:
    try:
        from _004_langgraph_more_nodes.entity_normalization import _get_faiss_store
        _get_faiss_store()
        logging.getLogger("main").info("FAISS/BGE 预热完成")
    except Exception as exc:
        logging.getLogger("main").warning("FAISS 预热失败（不影响服务，首次请求会慢）: %s", exc)


def _testing() -> bool:
    return os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.langfuse_mgr = LangfuseManager()
    secret = os.getenv("JWT_SECRET") or ""
    if secret in ("", "please-change-this-to-a-random-long-string") or len(secret) < 16:
        logging.getLogger("main").warning("JWT_SECRET 仍是占位或过短，生产环境请改成随机长串")
    try:
        admin = get_users().seed_admin()
        get_store().attach_orphans(admin.id)
    except Exception as exc:
        logging.getLogger("main").warning("种子管理员失败: %s", exc)
    if not _testing():
        await asyncio.to_thread(_warmup_faiss)
    yield
    try:
        app.state.langfuse_mgr.flush(timeout=5.0)
    except Exception:
        pass


app = FastAPI(
    title="中医知识图谱问答系统",
    description="基于 LangGraph + Neo4j 的中医知识问答 API",
    version="0.3.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    conversation_id: str | None = Field(default=None, description="会话 id，空则新建")
    omit_user_message: bool = Field(
        default=False,
        description="重新生成/编辑后重提：不再写入一条用户消息",
    )


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
    conversation_id: str = ""
    search_question: str = ""


class ConversationCreate(BaseModel):
    title: str = ""


class ConversationPatch(BaseModel):
    title: str = Field(..., min_length=1, max_length=80)


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    details: dict[str, Any] | None = None
    created_at: str


class IdsIn(BaseModel):
    ids: list[str] = Field(default_factory=list)


class MessagePatch(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=2000)
    feedback: Literal["like", "dislike"] | None = None


class FavoriteCreate(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    answer: str = Field(..., min_length=1, max_length=8000)


class FavoriteOut(BaseModel):
    id: str
    query: str
    answer: str
    created_at: str


class LoginIn(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class LoginOut(BaseModel):
    token: str
    id: str
    username: str
    role: str


class MeOut(BaseModel):
    id: str
    username: str
    role: str
    status: str


class PasswordIn(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


class ResetPasswordIn(BaseModel):
    password: str = Field(..., min_length=6, max_length=128)


class AdminUserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    role: Literal["user", "admin"] = "user"


class AdminUserOut(BaseModel):
    id: str
    username: str
    role: str
    status: str
    created_at: str


class AdminUserPatch(BaseModel):
    status: Literal["active", "inactive"] | None = None
    role: Literal["user", "admin"] | None = None


def _build_response(
    question: str,
    final_state: dict,
    elapsed_ms: float,
    conversation_id: str = "",
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
        conversation_id=conversation_id,
        search_question=final_state.get("search_question", "") or "",
    )


def _assistant_details(resp: AskResponse) -> dict[str, Any]:
    return {
        "is_zhongyi_intent": resp.is_zhongyi_intent,
        "intent_reason": resp.intent_reason,
        "elapsed_ms": resp.elapsed_ms,
        "matched_entities": {
            k: [e.model_dump() for e in v] for k, v in resp.matched_entities.items()
        },
        "cypher_queries": resp.cypher_queries,
        "search_question": resp.search_question,
    }


def _conv_out(c) -> ConversationOut:
    return ConversationOut(
        id=c.id,
        title=c.title,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


def _msg_out(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        details=m.details,
        created_at=m.created_at,
    )


def _prepare_turn(payload: AskRequest, user: User) -> tuple[str, list[dict[str, str]]]:
    store = get_store()
    if payload.omit_user_message:
        if not payload.conversation_id:
            raise HTTPException(status_code=400, detail="重新生成需要 conversation_id")
        conv = store.get(payload.conversation_id, user_id=user.id)
        if conv is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return conv.id, compress_history(store.get_recent_for_llm(conv.id))
    try:
        conv = store.ensure(
            payload.conversation_id,
            payload.question,
            user_id=user.id,
            is_admin=False,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在")
    history = compress_history(store.get_recent_for_llm(conv.id))
    store.add_message(conv.id, "user", payload.question)
    return conv.id, history


def _save_assistant(conversation_id: str, resp: AskResponse) -> None:
    try:
        get_store().add_message(
            conversation_id,
            "assistant",
            resp.answer or "",
            details=_assistant_details(resp),
        )
    except Exception as exc:
        logging.getLogger("main").warning("写入助手消息失败: %s", exc)


def _initial_state(question: str, history: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "user_question": question,
        "messages": history,
    }


def _make_handler(request: Request, trace_name: str, session_id: str, user_id: str = "anonymous"):
    mgr: LangfuseManager = request.app.state.langfuse_mgr
    if not mgr.is_enabled():
        return None
    metadata = {
        "request_id": str(uuid.uuid4()),
        "session_id": session_id,
        "user_id": user_id,
    }
    return mgr.create_handler(trace_name=trace_name, metadata=metadata)


@app.get("/health")
def health():
    return {"status": "ok", "redis": redis_status()}


def _admin_user_out(u: User) -> AdminUserOut:
    return AdminUserOut(
        id=u.id,
        username=u.username,
        role=u.role,
        status=u.status,
        created_at=u.created_at,
    )


@app.post("/auth/login", response_model=LoginOut)
def login(payload: LoginIn, request: Request):
    from common.login_limit import login_allowed
    from common.security import create_access_token

    ip = request.client.host if request.client else "unknown"
    if not login_allowed(ip):
        raise HTTPException(status_code=429, detail="登录过于频繁，请稍后再试")
    user = get_users().authenticate(payload.username, payload.password)
    if user is None:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    get_users().write_log(user.username, "登录", "用户登录")
    token = create_access_token(user.id, user.username, user.role)
    return LoginOut(token=token, id=user.id, username=user.username, role=user.role)


@app.get("/auth/me", response_model=MeOut)
def auth_me(user: User = Depends(get_current_user)):
    return MeOut(id=user.id, username=user.username, role=user.role, status=user.status)


@app.post("/auth/password")
def change_password(payload: PasswordIn, user: User = Depends(get_current_user)):
    from common.security import verify_password

    if not verify_password(payload.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="原密码不正确")
    try:
        get_users().set_password(user.id, payload.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    get_users().write_log(user.username, "改密码", "用户修改密码")
    return {"ok": True}


@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations(keyword: str = "", user: User = Depends(get_current_user)):
    return [
        _conv_out(c)
        for c in get_store().list_conversations(keyword, user_id=user.id)
    ]


@app.post("/conversations", response_model=ConversationOut)
def create_conversation(payload: ConversationCreate, user: User = Depends(get_current_user)):
    conv = get_store().create(title=payload.title or "新对话", user_id=user.id)
    return _conv_out(conv)


@app.post("/conversations/batch-delete")
def batch_delete_conversations(payload: IdsIn, user: User = Depends(get_current_user)):
    n = get_store().soft_delete_many(payload.ids, user_id=user.id)
    return {"deleted": n}


@app.patch("/conversations/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: str,
    payload: ConversationPatch,
    user: User = Depends(get_current_user),
):
    try:
        conv = get_store().rename(
            conversation_id, payload.title, user_id=user.id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _conv_out(conv)


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, user: User = Depends(get_current_user)):
    try:
        get_store().soft_delete(conversation_id, user_id=user.id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在")
    return None


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
def list_messages(conversation_id: str, user: User = Depends(get_current_user)):
    try:
        msgs = get_store().get_messages(
            conversation_id, user_id=user.id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在")
    return [_msg_out(m) for m in msgs]


@app.post("/conversations/{conversation_id}/messages/batch-delete")
def batch_delete_messages(
    conversation_id: str,
    payload: IdsIn,
    user: User = Depends(get_current_user),
):
    try:
        n = get_store().delete_messages(
            conversation_id, payload.ids, user_id=user.id
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": n}


@app.patch("/conversations/{conversation_id}/messages/{message_id}", response_model=MessageOut)
def patch_message(
    conversation_id: str,
    message_id: str,
    payload: MessagePatch,
    user: User = Depends(get_current_user),
):
    store = get_store()
    try:
        if store.get(conversation_id, user_id=user.id) is None:
            raise KeyError(conversation_id)
        if payload.content is not None:
            store.update_message_content(conversation_id, message_id, payload.content)
            store.delete_messages_after(conversation_id, message_id)
        if payload.feedback is not None:
            store.patch_message_details(conversation_id, message_id, {"feedback": payload.feedback})
        msg = store.get_message(conversation_id, message_id)
        if msg is None:
            raise KeyError(message_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="消息不存在")
    return _msg_out(msg)


@app.get("/favorites", response_model=list[FavoriteOut])
def list_favorites(keyword: str = "", user: User = Depends(get_current_user)):
    return [
        FavoriteOut(id=f.id, query=f.query, answer=f.answer, created_at=f.created_at)
        for f in get_store().list_favorites(keyword, user_id=user.id, is_admin=False)
    ]


@app.post("/favorites", response_model=FavoriteOut)
def create_favorite(payload: FavoriteCreate, user: User = Depends(get_current_user)):
    fav = get_store().add_favorite(payload.query, payload.answer, user_id=user.id)
    return FavoriteOut(id=fav.id, query=fav.query, answer=fav.answer, created_at=fav.created_at)


@app.delete("/favorites/{favorite_id}", status_code=204)
def delete_favorite(favorite_id: str, user: User = Depends(get_current_user)):
    try:
        get_store().delete_favorite(favorite_id, user_id=user.id)
    except KeyError:
        raise HTTPException(status_code=404, detail="收藏不存在")
    return None


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request, user: User = Depends(get_current_user)) -> AskResponse:
    t0 = time.time()
    conv_id, history = _prepare_turn(payload, user)
    handler = _make_handler(request, "POST:/ask", conv_id, user.id)
    config = {"callbacks": [handler]} if handler else None
    initial = _initial_state(payload.question, history)
    try:
        graph = get_graph()
        final_state = graph.invoke(initial, config=config) if config else graph.invoke(initial)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图执行失败: {exc}")
    elapsed = (time.time() - t0) * 1000
    resp = _build_response(payload.question, final_state, elapsed, conv_id)
    _save_assistant(conv_id, resp)
    return resp


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request, user: User = Depends(get_current_user)):
    conv_id, history = _prepare_turn(payload, user)
    handler = _make_handler(request, "POST:/ask/stream", conv_id, user.id)
    config = {"callbacks": [handler]} if handler else None
    initial = _initial_state(payload.question, history)

    async def event_stream() -> AsyncGenerator[str, None]:
        t0 = time.time()
        queue: asyncio.Queue = asyncio.Queue()
        yield _sse("session", {"conversation_id": conv_id})

        async def producer() -> None:
            try:
                final_state = dict(initial)
                graph = get_graph()
                async for event in graph.astream_events(initial, version="v2", config=config):
                    await queue.put(("event", event))
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
                    if is_answer_token_event(event):
                        text = token_text_from_event(event)
                        if text:
                            yield _sse("token", {"text": text})
                            await asyncio.sleep(0)
                    if kind == "on_chain_start" and node_name in _NODE_LABELS:
                        if handler is not None:
                            handler.start_node_span(node_name, input=event.get("data", {}).get("input"))
                        yield _sse("progress", {
                            "node": node_name,
                            "label": _NODE_LABELS[node_name],
                            "status": "running",
                        })
                        await asyncio.sleep(0)
                    elif kind == "on_chain_end" and node_name in _NODE_LABELS:
                        output = event.get("data", {}).get("output", {}) or {}
                        if handler is not None:
                            handler.end_node_span(node_name, output=output)
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
                    resp = _build_response(payload.question, final_state, elapsed, conv_id)
                    _save_assistant(conv_id, resp)
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


_PROMPT_CATALOG = [
    ("intent_recognition", "意图识别"),
    ("entity_extraction", "实体抽取"),
    ("cypher_generation", "Cypher 生成"),
    ("answer_generation", "图谱回答"),
    ("general_response", "通用回答"),
    ("context_compress", "对话摘要"),
]


@app.get("/admin/feedbacks")
def admin_feedbacks(
    feedback: str = "",
    user: User = Depends(require_admin),
):
    items = get_store().list_feedbacks(feedback)
    users = {u.id: u.username for u in get_users().list_users()}
    for item in items:
        item["username"] = users.get(item.get("user_id") or "", "")
    return {"total": len(items), "items": items}


@app.get("/admin/prompts")
def admin_prompts(_admin: User = Depends(require_admin)):
    from common.langfuse_manager import fetch_prompt, get_langfuse_manager

    mgr = get_langfuse_manager()
    enabled = bool(mgr.is_enabled())
    items = []
    for name, label in _PROMPT_CATALOG:
        source = "fallback"
        if enabled:
            sentinel = "__cm_local_fallback__"
            text = fetch_prompt(name, sentinel)
            source = "fallback" if text == sentinel else "langfuse"
        items.append({"name": name, "label": label, "source": source})
    return {"langfuse_enabled": enabled, "items": items}


@app.get("/admin/import-status")
def admin_import_status(user: User = Depends(require_admin)):
    faiss_index = os.getenv("FAISS_INDEX_PATH") or ""
    faiss_meta = os.getenv("FAISS_METADATA_PATH") or ""
    neo4j_ok = False
    neo4j_error = ""
    labels: dict[str, int] = {}
    try:
        from common.neo4j_manager import ALLOWED_LABELS, Neo4jManager

        mgr = Neo4jManager()
        try:
            neo4j_ok = True
            with mgr.driver.session(database=mgr.database) as session:
                for label in sorted(ALLOWED_LABELS):
                    row = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()
                    labels[label] = int(row["c"] if row else 0)
        finally:
            mgr.close()
    except Exception as exc:
        neo4j_error = str(exc)
    return {
        "redis": redis_status(),
        "neo4j": {"ok": neo4j_ok, "error": neo4j_error, "labels": labels},
        "faiss": {
            "index_path": faiss_index,
            "index_exists": bool(faiss_index and os.path.isfile(faiss_index)),
            "metadata_path": faiss_meta,
            "metadata_exists": bool(faiss_meta and os.path.isfile(faiss_meta)),
        },
    }


@app.get("/admin/users", response_model=list[AdminUserOut])
def admin_list_users(user: User = Depends(require_admin)):
    return [_admin_user_out(u) for u in get_users().list_users()]


@app.post("/admin/users", response_model=AdminUserOut)
def admin_create_user(payload: AdminUserCreate, user: User = Depends(require_admin)):
    try:
        created = get_users().create(payload.username, payload.password, role=payload.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    get_users().write_log(user.username, "用户管理", f"新建用户 {created.username}")
    return _admin_user_out(created)


@app.patch("/admin/users/{user_id}", response_model=AdminUserOut)
def admin_patch_user(
    user_id: str,
    payload: AdminUserPatch,
    user: User = Depends(require_admin),
):
    store = get_users()
    try:
        target = store.get(user_id)
        if target is None:
            raise KeyError(user_id)
        notes = []
        if payload.status is not None:
            target = store.set_status(user_id, payload.status, actor_id=user.id)
            notes.append(f"状态={payload.status}")
        if payload.role is not None:
            target = store.set_role(user_id, payload.role, actor_id=user.id)
            notes.append(f"角色={payload.role}")
    except KeyError:
        raise HTTPException(status_code=404, detail="用户不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if notes:
        store.write_log(user.username, "用户管理", f"更新 {target.username}：{'，'.join(notes)}")
    return _admin_user_out(target)


@app.delete("/admin/users/{user_id}", status_code=204)
def admin_delete_user(user_id: str, user: User = Depends(require_admin)):
    store = get_users()
    target = store.get(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        store.delete_user(user_id, actor_id=user.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    store.write_log(user.username, "用户管理", f"删除用户 {target.username}")
    return None


@app.post("/admin/users/{user_id}/reset-password")
def admin_reset_password(
    user_id: str,
    payload: ResetPasswordIn,
    user: User = Depends(require_admin),
):
    store = get_users()
    target = store.get(user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        store.set_password(user_id, payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    store.write_log(user.username, "用户管理", f"重置 {target.username} 的密码")
    return {"ok": True}


@app.get("/admin/logs")
def admin_list_logs(limit: int = 50, offset: int = 0, user: User = Depends(require_admin)):
    return get_users().list_logs(limit=limit, offset=offset)


def main():
    import uvicorn
    uvicorn.run("_005_fastapi.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
