"""Identify whether a question should enter the TCM knowledge-graph flow."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from .state import (
        KG_ENTITY_TYPES,
        KG_RELATION_TYPES,
        GraphState,
        question_for_retrieval,
    )
except ImportError:
    from state import (
        KG_ENTITY_TYPES,
        KG_RELATION_TYPES,
        GraphState,
        question_for_retrieval,
    )

from common.env_loader import load_app_env
from common.langfuse_manager import fetch_prompt

load_app_env()

logger = logging.getLogger("intent")

_ENTITY_CONTEXT = "\n".join(
    f"- {name}: {description}" for name, description in KG_ENTITY_TYPES.items()
)
_RELATION_CONTEXT = "\n".join(
    f"- {name}: {description}" for name, description in KG_RELATION_TYPES.items()
)

_INTENT_FALLBACK = """你是中医知识图谱问答系统的路由器，只判断是否进入中医知识图谱流程。

系统目的（why）：仅当用户的问题属于中医知识图谱可处理范围时，才继续执行实体抽取、BGE 向量匹配标准实体和 Cypher 查询；不相关问题不能进入该流程。

图谱实体范围：
{entity_types}
图谱关系范围：
{relation_types}

判定规则（how）：
1. 用户明确咨询中医、中药、方剂、药材、针灸、经络、辨证、养生、功效、禁忌、典籍、医籍、医案、本草或针灸著作时，设为 true。
2. 用户询问某个症状或疾病，并明确希望了解中医治疗、调理、方药、针灸或辨证时，设为 true。
3. 询问某部中医文献「是什么 / 讲什么 / 内容 / 作者」时，设为 true（例如十四经发挥、素问、伤寒论）。
4. 仅有泛健康症状但未出现中医方向，例如“头痛怎么办”，因无法确认中医意图，设为 false。
5. 天气、编程、数学、闲聊、新闻，以及与上述图谱实体、中医文献无关的问题，设为 false。

限制：不要回答用户问题；不要抽取实体；不要调用或模拟 BGE；不要生成 Cypher；不要给出诊断或用药建议。
只输出一行 JSON，不要 Markdown、不要解释，格式严格为：
{{"is_zhongyi_intent": true}} 或 {{"is_zhongyi_intent": false}}。"""

_JSON_PATTERN = re.compile(r"\{[^{}]*is_zhongyi_intent[^{}]*\}", re.IGNORECASE | re.DOTALL)
_ZHONGYI_HINTS = (
    "中医", "中药", "方剂", "药材", "针灸", "经络", "辨证", "养生",
    "功效", "禁忌", "草药", "性味", "归经", "伤寒", "本草",
)
# 仅这些可纠正模型把典籍问句判成普通（避免「千金/局方/汤头」误伤）
_LITERATURE_RESCUE = (
    "十四经", "素问", "灵枢", "难经", "伤寒论", "金匮要略", "本草纲目",
    "甲乙经", "千金要方", "脾胃论", "脉经", "温病", "临证指南",
    "医案", "医籍", "典籍",
)
_BOOK_ASK = re.compile(r"(是什么|讲什么|讲的是|内容|作者|哪部|什么书|简介|出自)")
_KB_TITLE_TTL = 30.0
_kb_titles_at = 0.0
_kb_titles: tuple[str, ...] = ()
_kb_titles_loaded = False


def _get_intent_prompt() -> str:
    return fetch_prompt(
        "intent_recognition",
        _INTENT_FALLBACK,
        entity_types=_ENTITY_CONTEXT,
        relation_types=_RELATION_CONTEXT,
    )


def create_llm(provider: str | None = None):
    from common.llm import get_chat_model

    return get_chat_model(provider)


def _content_text(raw: Any) -> str:
    if raw is None:
        return ""
    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or ""))
            else:
                parts.append(str(getattr(block, "text", "") or ""))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "是"):
        return True
    if text in ("false", "0", "no", "否"):
        return False
    return None


def parse_intent_flag(raw_text: str) -> bool | None:
    """从普通模型文本里取出 is_zhongyi_intent。解析不到返回 None。"""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    match = _JSON_PATTERN.search(text)
    if match:
        candidates.append(match.group())

    for chunk in candidates:
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "is_zhongyi_intent" in obj:
            return _coerce_bool(obj.get("is_zhongyi_intent"))
    return None


def keyword_zhongyi_intent(question: str) -> bool:
    """解析失败时的弱规则：有明确中医线索才进图谱。"""
    return any(token in (question or "") for token in _ZHONGYI_HINTS)


def literature_zhongyi_intent(question: str) -> bool:
    """典籍/医籍书名线索，用于纠正模型把「某书是什么」判成普通。"""
    return any(token in (question or "") for token in _LITERATURE_RESCUE)


def _testing() -> bool:
    return os.getenv("TESTING", "").strip().lower() in ("1", "true", "yes")


def _load_kb_titles() -> tuple[str, ...]:
    """知识库已上传文档的书名/章节名，用于把「某书是什么」判成中医。"""
    global _kb_titles_at, _kb_titles, _kb_titles_loaded
    if _testing():
        return ()
    now = time.monotonic()
    if _kb_titles_loaded and now - _kb_titles_at < _KB_TITLE_TTL:
        return _kb_titles
    titles: list[str] = []
    try:
        from common.knowledge_service import get_knowledge_service

        rows = get_knowledge_service().list_documents(page=1, size=500, user_role="admin")
        for item in rows.get("list") or []:
            stem = Path(str(item.get("docName") or "")).stem
            for part in stem.split("_"):
                part = part.strip()
                if len(part) >= 4:
                    titles.append(part)
    except Exception as exc:
        from common.obs import degraded

        degraded("intent_kb_titles", exc)
        logger.warning("读取知识库书名失败: %s", exc)
        _kb_titles_loaded = True
        _kb_titles_at = now
        return _kb_titles
    _kb_titles = tuple(dict.fromkeys(titles))
    _kb_titles_at = now
    _kb_titles_loaded = True
    return _kb_titles


def kb_title_zhongyi_intent(question: str, titles: tuple[str, ...] | None = None) -> bool:
    """问句同时像在问书，且出现已入库题名（≥6 字，或带「是什么」等）。"""
    q = (question or "").strip()
    if len(q) < 2:
        return False
    asking_book = _BOOK_ASK.search(q) is not None
    for title in titles if titles is not None else _load_kb_titles():
        if len(title) < 4 or title not in q:
            continue
        if len(title) >= 6 or asking_book:
            return True
    return False


def resolve_zhongyi_intent(
    question: str,
    llm_flag: bool | None,
    titles: tuple[str, ...] | None = None,
) -> tuple[bool, str]:
    """典籍线索/入库书名可纠正模型把文献问句判成普通；短关键词只在模型未给出判定时兜底。"""
    if kb_title_zhongyi_intent(question, titles):
        return True, "知识库文献"
    if literature_zhongyi_intent(question):
        return True, "中医典籍"
    if llm_flag is None:
        if keyword_zhongyi_intent(question):
            return True, "中医关键词"
        return False, "关键词未命中"
    if llm_flag:
        return True, "模型判定中医"
    return False, "模型判定普通"


def make_intent_recognition_node(llm):
    """Build an injectable node; injection keeps tests independent of the network."""

    def intent_recognition_node(state: GraphState) -> dict[str, Any]:
        question = question_for_retrieval(state)
        if not question:
            raise ValueError("state.user_question 不能为空")
        messages = [
            SystemMessage(content=_get_intent_prompt()),
            HumanMessage(content=f"用户问题：{question}"),
        ]
        flag: bool | None = None
        try:
            raw = llm.invoke(messages)
            flag = parse_intent_flag(_content_text(raw))
        except Exception as exc:
            from common.obs import degraded

            degraded("intent_invoke", exc)
            logger.warning("意图识别调用失败，改用关键词: %s", exc)
            zhongyi, reason = resolve_zhongyi_intent(question, None)
            return {"is_zhongyi_intent": zhongyi, "intent_reason": reason}

        if flag is None:
            from common.obs import degraded

            degraded("intent_parse", msg="unparsed_intent_json")
            logger.warning("意图识别未解析到 JSON，改用关键词/书名兜底")

        zhongyi, reason = resolve_zhongyi_intent(question, flag)
        return {"is_zhongyi_intent": zhongyi, "intent_reason": reason}

    return intent_recognition_node


def intent_recognition_node(state: GraphState) -> dict[str, Any]:
    """Default node used by the graph; the LLM is initialized only when invoked."""
    return make_intent_recognition_node(create_llm(state.get("llm_provider")))(state)
