"""最终回答节点：结合知识图谱执行结果生成精准回答。

why: 仅有 Cypher 执行结果不够，需 LLM 将结构化数据转为自然语言回答
how: neo4j_answer (KG上下文) + user_question → LLM 生成最终回答 → state.final_answer
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

_ANSWER_FALLBACK = """你是专业的中医知识助手。根据知识图谱查询结果回答用户问题。

## 要求
1. 基于提供的图谱数据回答，数据中没有的信息请说明"暂未查到"
2. 使用中医术语，如症状、方剂、药材、功效、经络、辨证论治、典籍等
3. 回答简洁、准确，避免无关内容
4. 只输出最终答案，不要解释推理过程"""


def _get_answer_prompt() -> str:
    """运行时从 Langfuse 拉取 prompt（SDK 缓存 60s）。"""
    return fetch_prompt("answer_generation", _ANSWER_FALLBACK)


def _build_prompt(state: GraphState) -> str:
    kg_context = state.get("neo4j_answer", "") or "(无图谱数据)"
    question = state.get("user_question", "")
    hist = history_as_text(state)
    hist_block = f"## 对话历史\n{hist}\n\n" if hist else ""
    return f"""{hist_block}## 知识图谱查询结果
{kg_context}

## 当前问题
{question}

请基于以上图谱数据回答用户问题。"""

# ============================================================
# LLM 工厂
# ============================================================


def create_llm():
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


def make_answer_generation_node(llm):
    """构建最终回答节点。"""

    def answer_generation_node(state: GraphState) -> dict[str, Any]:
        question = str(state.get("user_question", "")).strip()
        if not question:
            raise ValueError("state.user_question 不能为空")

        prompt = _build_prompt(state)
        messages = [
            SystemMessage(content=_get_answer_prompt()),
            HumanMessage(content=prompt),
        ]
        answer = str(llm.invoke(messages).content).strip()
        return {"final_answer": answer}

    return answer_generation_node


def answer_generation_node(state: GraphState) -> dict[str, Any]:
    return make_answer_generation_node(create_llm())(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================


def main():
    """直接运行 python answer_generation.py 测试节点。"""
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    PARENT = os.path.dirname(HERE)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    test_state: dict[str, Any] = {
        "user_question": "四君子汤有什么功效？",
        "neo4j_answer": """-- 查询: MATCH (n:Formula {id: 'E100'})-[r]->(m) RETURN ...
name | type(r) | labels(m) | m.name
四君子汤 | HAS_EFFECT | ['Effect'] | 补气健脾
四君子汤 | HAS_INGREDIENT | ['Herb'] | 人参
四君子汤 | HAS_INGREDIENT | ['Herb'] | 白术
四君子汤 | HAS_INGREDIENT | ['Herb'] | 茯苓
四君子汤 | HAS_INGREDIENT | ['Herb'] | 甘草""",
    }

    print("=" * 64)
    print("  最终回答节点 · 自测")
    print("=" * 64)

    print("\n初始化 LLM...", end=" ", flush=True)
    llm = create_llm()
    print("就绪。")

    node = make_answer_generation_node(llm)

    print(f"\n用户问题: {test_state['user_question']}")
    print(f"KG 上下文: {len(test_state['neo4j_answer'])} 字符")

    update = node(test_state)
    answer = update.get("final_answer", "")

    print(f"\n最终回答:\n{answer}")

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
