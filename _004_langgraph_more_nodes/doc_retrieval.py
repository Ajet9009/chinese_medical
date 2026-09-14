"""文献检索节点：混合检索 + CRAG v1（一次改写重检索）。"""

from __future__ import annotations

import os
from typing import Any, Callable

from common.doc_rag import format_doc_context
from common.env_loader import load_app_env
from common.langfuse_manager import fetch_prompt
from common.rag.pipeline import retrieve_with_crag

try:
    from .state import GraphState, question_for_retrieval
except ImportError:
    from state import GraphState, question_for_retrieval

load_app_env()

SearchFn = Callable[[str], list[dict[str, Any]]]
RewriteFn = Callable[[str], str]

_CRAG_REWRITE_FALLBACK = """你是中医文献检索查询改写助手。把用户提问改写成更规范、信息更完整、适合向量与关键词检索的问句（保留方剂/药材/证型等术语，去掉口语）。
只输出改写后的问句，不要解释、不要引号。"""


def _rag_enabled() -> bool:
    return os.getenv("DOC_RAG_ENABLE", "1").strip().lower() not in ("0", "false", "no")


def _crag_enabled() -> bool:
    return os.getenv("CRAG_ENABLE", "1").strip().lower() not in ("0", "false", "no")


def _testing() -> bool:
    return os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes")


def _default_search(question: str, viewer_dept: str = "", viewer_role: str = "user") -> list[dict[str, Any]]:
    from common.doc_store import search_documents

    return search_documents(question, viewer_dept=viewer_dept, viewer_role=viewer_role)


def _default_rewrite(question: str, provider: str | None = None) -> str:
    q = (question or "").strip()
    if not q:
        return q
    if _testing():
        return q
    if os.getenv("CRAG_REWRITE_ENABLE", "1").strip().lower() in ("0", "false", "no"):
        return q
    from langchain_core.messages import HumanMessage, SystemMessage
    from common.llm import get_chat_model

    try:
        llm = get_chat_model(provider)
        prompt = fetch_prompt("crag_rewrite", _CRAG_REWRITE_FALLBACK)
        out = str(llm.invoke([SystemMessage(content=prompt), HumanMessage(content=q)]).content).strip()
        return out or q
    except Exception as exc:
        from common.obs import degraded

        degraded("crag_rewrite", exc)
        return q


def _empty_crag() -> dict[str, Any]:
    return {
        "doc_chunks": [],
        "doc_context": "",
        "crag_grade": "",
        "crag_action": "",
        "crag_confidence": "",
    }


def make_doc_retrieval_node(search_fn: SearchFn | None = None, rewrite_fn: RewriteFn | None = None):
    def doc_retrieval_node(state: GraphState) -> dict[str, Any]:
        if not _rag_enabled():
            return _empty_crag()
        question = question_for_retrieval(state)
        dept = str(state.get("viewer_dept") or "")
        role = str(state.get("viewer_role") or "user")
        fn = search_fn or (lambda q, _d=dept, _r=role: _default_search(q, _d, _r))
        # 普通意图只做一次检索，不跑 CRAG 改写，避免闲聊多一次 LLM
        if not state.get("is_zhongyi_intent"):
            try:
                hits = fn(question) if question else []
            except Exception as exc:
                from common.obs import degraded

                degraded("doc_search", exc)
                hits = []
            if hits:
                from common.rag.crag import GRADE_INCORRECT, grade as crag_grade_fn

                top1 = float(hits[0].get("score") or 0.0)
                grade, _ = crag_grade_fn(top1, len(hits), rerank_ok=True)
                if grade == GRADE_INCORRECT:
                    hits = []
            return {
                "doc_chunks": hits,
                "doc_context": format_doc_context(hits),
                "crag_grade": "",
                "crag_action": "",
                "crag_confidence": "",
            }
        if not _crag_enabled():
            try:
                hits = fn(question) if question else []
            except Exception as exc:
                from common.obs import degraded

                degraded("doc_search", exc)
                hits = []
            return {
                "doc_chunks": hits,
                "doc_context": format_doc_context(hits),
                "crag_grade": "",
                "crag_action": "",
                "crag_confidence": "",
            }
        if rewrite_fn is not None:
            rw = rewrite_fn
        else:
            provider = str(state.get("llm_provider") or "") or None

            def _rewrite(q: str) -> str:
                return _default_rewrite(q, provider)

            rw = _rewrite
        try:
            out = retrieve_with_crag(question, fn, rewrite_fn=rw)
        except Exception as exc:
            from common.obs import degraded

            degraded("doc_search", exc)
            return {
                "doc_chunks": [],
                "doc_context": "",
                "crag_grade": "incorrect",
                "crag_action": "normal",
                "crag_confidence": "medium",
            }
        hits = out.get("hits") or []
        return {
            "doc_chunks": hits,
            "doc_context": format_doc_context(hits),
            "crag_grade": out.get("crag_grade") or "",
            "crag_action": out.get("crag_action") or "",
            "crag_confidence": out.get("crag_confidence") or "",
        }

    return doc_retrieval_node


def doc_retrieval_node(state: GraphState) -> dict[str, Any]:
    return make_doc_retrieval_node()(state)
