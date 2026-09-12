"""指代消解：把依赖上文的追问改写成独立问句。"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from common.env_loader import load_app_env

try:
    from .state import GraphState, history_as_text
except ImportError:
    from state import GraphState, history_as_text

load_app_env()

_STANDALONE_FALLBACK = """你是中医问答系统的指代消解器。
根据对话历史，把当前问题改写成不依赖上下文、可单独检索的完整问题。
若当前问题本身已经完整，原样输出。
只输出改写后的问题，不要解释、不要引号。"""


def create_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL"),
        temperature=0,
        streaming=False,
    )


def make_standalone_query_node(llm):
    def standalone_query_node(state: GraphState) -> dict[str, Any]:
        question = str(state.get("user_question", "")).strip()
        if not question:
            raise ValueError("state.user_question 不能为空")
        history = list(state.get("messages") or [])
        if not history:
            return {"search_question": question}

        hist = history_as_text(state)
        prompt = f"""## 对话历史
{hist}

## 当前问题
{question}

请输出独立问句。"""
        try:
            raw = str(llm.invoke([
                SystemMessage(content=_STANDALONE_FALLBACK),
                HumanMessage(content=prompt),
            ]).content).strip()
        except Exception:
            return {"search_question": question}

        rewritten = raw.splitlines()[0].strip().strip("\"'“”")
        if not rewritten:
            rewritten = question
        return {"search_question": rewritten}

    return standalone_query_node


def standalone_query_node(state: GraphState) -> dict[str, Any]:
    question = str(state.get("user_question", "")).strip()
    if not question:
        raise ValueError("state.user_question 不能为空")
    if not (state.get("messages") or []):
        return {"search_question": question}
    return make_standalone_query_node(create_llm())(state)
