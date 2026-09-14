"""最终回答节点：结合知识图谱执行结果生成精准回答。

why: 仅有 Cypher 执行结果不够，需 LLM 将结构化数据转为自然语言回答
how: neo4j_answer (KG上下文) + user_question → LLM 生成最终回答 → state.final_answer
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

try:
    from .state import GraphState, history_as_text
except ImportError:
    from state import GraphState, history_as_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common.env_loader import load_app_env  # noqa: E402
from common.llm import collect_chat_text_astream_step_02, collect_chat_text_step_01  # noqa: E402
load_app_env()

# ============================================================
# 提示词
# ============================================================

from common.langfuse_manager import fetch_prompt  # noqa: E402

_ANSWER_FALLBACK = """你是专业的中医知识助手。根据知识图谱和文献摘录回答用户问题。

## 要求
1. 优先依据图谱数据；文献仅作补充。两者都没有的信息必须说明未查到，不要编造
2. 使用中医术语，如症状、方剂、药材、功效、经络、辨证论治、典籍等
3. 若使用了文献，可点明出处文件名，不要编造未给出的书名或页码
4. 回答简洁、准确，避免无关内容
5. 只输出最终答案，不要解释推理过程"""


def _get_answer_prompt() -> str:
    """运行时从 Langfuse 拉取 prompt（SDK 缓存 60s）。"""
    return fetch_prompt("answer_generation", _ANSWER_FALLBACK)


def _build_prompt(state: GraphState) -> str:
    kg_context = state.get("neo4j_answer", "") or "(无图谱数据)"
    doc_context = (state.get("doc_context") or "").strip() or "(无文献摘录)"
    question = state.get("user_question", "")
    hist = history_as_text(state)
    hist_block = f"## 对话历史\n{hist}\n\n" if hist else ""
    return f"""{hist_block}## 知识图谱查询结果
{kg_context}

## 文献摘录
{doc_context}

## 当前问题
{question}

请基于以上图谱数据与文献摘录回答用户问题。不要使用未出现的依据。"""

# ============================================================
# LLM 工厂
# ============================================================


def create_llm(provider: str | None = None):
    from common.llm import get_chat_model

    return get_chat_model(provider, streaming=True)


# ============================================================
# 节点构建
# ============================================================


def _prepare_answer_ctx_step_01(state: GraphState) -> dict[str, Any]:
    """步骤：01 校验问题并组装 prompt。拒答短路时 early 为节点返回值。"""
    from common.doc_rag import (
        UNCERTAIN_PREFIX,
        kg_context_empty,
        should_refuse,
        REFUSE_ANSWER,
    )

    question = str(state.get("user_question", "")).strip()
    if not question:
        raise ValueError("state.user_question 不能为空")

    kg = state.get("neo4j_answer")
    chunks = state.get("doc_chunks") or []
    if should_refuse(
        kg,
        chunks,
        crag_grade=state.get("crag_grade"),
        crag_action=state.get("crag_action"),
        crag_confidence=state.get("crag_confidence"),
    ):
        return {
            "early": {"final_answer": REFUSE_ANSWER, "refused": True, "citation_ok": False},
        }

    prompt = _build_prompt(state)
    kg_empty = kg_context_empty(kg)
    ambiguous = (state.get("crag_grade") or "") == "ambiguous"
    if ambiguous and kg_empty:
        prompt += f"\n证据有限，请在答案开头写「{UNCERTAIN_PREFIX}」"

    return {
        "early": None,
        "messages": [
            SystemMessage(content=_get_answer_prompt()),
            HumanMessage(content=prompt),
        ],
        "kg_empty": kg_empty,
        "ambiguous": ambiguous,
        "chunks": chunks,
        "may_replace_with_refuse": bool(chunks) and kg_empty,
    }


def _finalize_answer_step_02(ctx: dict[str, Any], answer: str) -> dict[str, Any]:
    """步骤：02 不确定前缀与引用校验。由同步/异步生成路径共用。"""
    from common.doc_rag import REFUSE_ANSWER, UNCERTAIN_PREFIX
    from common.rag.cite import verify_citations

    if ctx.get("ambiguous") and ctx.get("kg_empty") and not answer.startswith(UNCERTAIN_PREFIX):
        answer = UNCERTAIN_PREFIX + answer

    citation_ok = True
    chunks = ctx.get("chunks") or []
    if chunks:
        checked = verify_citations(answer, chunks)
        citation_ok = bool(checked.get("ok"))
        if checked.get("rewrite_needed") and ctx.get("kg_empty"):
            return {"final_answer": REFUSE_ANSWER, "refused": True, "citation_ok": False}

    return {"final_answer": answer, "refused": False, "citation_ok": citation_ok}


def make_answer_generation_node(llm):
    """步骤：01 构建最终回答节点。同步 invoke + 异步 astream，供 /ask 与 /ask/stream。"""

    def _sync(state: GraphState, config: Any = None) -> dict[str, Any]:
        ctx = _prepare_answer_ctx_step_01(state)
        if ctx.get("early"):
            return ctx["early"]
        answer = collect_chat_text_step_01(
            llm,
            ctx["messages"],
            streaming=not ctx["may_replace_with_refuse"],
            config=config,
        )
        return _finalize_answer_step_02(ctx, answer)

    async def _async(state: GraphState, config: Any = None) -> dict[str, Any]:
        ctx = _prepare_answer_ctx_step_01(state)
        if ctx.get("early"):
            return ctx["early"]
        if ctx["may_replace_with_refuse"]:
            answer = collect_chat_text_step_01(
                llm, ctx["messages"], streaming=False, config=config
            )
        else:
            answer = await collect_chat_text_astream_step_02(
                llm, ctx["messages"], config=config
            )
        return _finalize_answer_step_02(ctx, answer)

    return RunnableLambda(_sync, afunc=_async)


def _answer_entry_sync_step_03(state: GraphState, config: Any = None) -> dict[str, Any]:
    """步骤：03 图同步入口。由 /ask 的 graph.invoke 调用。"""
    return make_answer_generation_node(create_llm(state.get("llm_provider"))).invoke(
        state, config=config
    )


async def _answer_entry_async_step_04(state: GraphState, config: Any = None) -> dict[str, Any]:
    """步骤：04 图异步入口。由 /ask/stream 的 astream_events 调用。"""
    return await make_answer_generation_node(create_llm(state.get("llm_provider"))).ainvoke(
        state, config=config
    )


answer_generation_node = RunnableLambda(
    _answer_entry_sync_step_03, afunc=_answer_entry_async_step_04
)


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

    update = node.invoke(test_state)
    answer = update.get("final_answer", "")

    print(f"\n最终回答:\n{answer}")

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
