"""通用回答节点：非中医问题直接由 LLM 回答。

why: 意图识别为 general 时走此节点，不进入实体抽取/Cypher生成流程
how: 角色设定为中医知识助手，中医相关优先中医角度，无关问题简洁常规回答
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from .state import GraphState, history_as_text
except ImportError:
    from state import GraphState, history_as_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common.env_loader import load_app_env  # noqa: E402
load_app_env()

# ============================================================
# 提示词
# ============================================================

from common.langfuse_manager import fetch_prompt  # noqa: E402

_GENERAL_FALLBACK = """你是一名专业的中医知识助手，回答时请尽量基于中医理论和术语来解释。

要求：
- 优先从中医角度（如症状、方剂、中药材、功效、经络、辨证论治、典籍等）进行回答。
- 如果问题与中医无关，请直接给出简洁的常规回答，不要强行套用中医。
- 回答要准确、简洁，避免无关内容。
- 输出时只给出最终答案，不要解释你是如何推理的。"""


def _get_general_prompt() -> str:
    """运行时从 Langfuse 拉取 prompt（SDK 缓存 60s）。"""
    return fetch_prompt("general_response", _GENERAL_FALLBACK)

# ============================================================
# LLM 工厂
# ============================================================


def create_llm():
    """创建项目通用的 OpenAI 兼容 LLM。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL"),
        temperature=0,
        streaming=True,
    )


# ============================================================
# 节点构建
# ============================================================


def make_general_response_node(llm):
    """构建可注入 LLM 的通用回答节点。"""

    def general_response_node(state: GraphState) -> dict[str, Any]:
        question = str(state.get("user_question", "")).strip()
        if not question:
            raise ValueError("state.user_question 不能为空")

        hist = history_as_text(state)
        human = question
        if hist:
            human = f"## 对话历史\n{hist}\n\n## 当前问题\n{question}"
        messages = [
            SystemMessage(content=_get_general_prompt()),
            HumanMessage(content=human),
        ]
        answer = str(llm.invoke(messages).content).strip()

        return {"final_answer": answer}

    return general_response_node


def general_response_node(state: GraphState) -> dict[str, Any]:
    """默认节点（首次调用时惰性初始化 LLM）。"""
    return make_general_response_node(create_llm())(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================


def _print_state(title: str, state: dict, highlight_keys: set[str] | None = None):
    """打印 state 快照。"""
    highlight_keys = highlight_keys or set()
    print(f"\n  ┌─ {title} " + "─" * 50)
    keys = [
        ("user_question", "用户问题"),
        ("final_answer",  "最终回答"),
    ]
    for key, label in keys:
        val = state.get(key)
        if val is None or val == "":
            display = "(空)"
        else:
            display = str(val)[:120]
        marker = " ◀── 变更" if key in highlight_keys else ""
        print(f"  │  {label + ' (' + key + ')':<30s} = {display}{marker}")
        if key == "final_answer" and len(str(val)) > 120:
            for chunk_start in range(120, min(len(str(val)), 480), 120):
                print(f"  │  {'':30s}   {str(val)[chunk_start:chunk_start+120]}")
    print("  └" + "─" * 60)


def main():
    """直接运行 python -m _004_langgraph_more_nodes.general_response 测试节点。"""
    test_cases = [
        "今天天气怎么样？",
        "帮我写一个 Python 脚本",
        "感冒了应该注意什么？",
        "推荐一本好看的小说",
        "头痛怎么办？",
    ]

    print("=" * 64)
    print("  通用回答节点 · 自测")
    print("=" * 64)

    print("\n初始化 LLM...", end=" ", flush=True)
    llm = create_llm()
    print("就绪。")

    node = make_general_response_node(llm)

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'─' * 64}")
        print(f"  [{i}/{len(test_cases)}] 输入: {question}")

        before = {"user_question": question}
        update = node(before)
        after = {**before, **update}

        _print_state("before", before, {"user_question"})
        _print_state("after ", after, {"final_answer"})

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
