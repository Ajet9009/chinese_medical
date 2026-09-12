"""意图识别节点：LLM 判断用户问题是否属于中医领域。

why: 识别中医意图后才能进入实体抽取 → BGE 向量匹配标准实体 → 生成 Cypher 查询
how: 提示词内置知识图谱全部实体/关系类型说明，LLM 分类后设置 is_zhongyi_intent
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

try:
    from .state import GraphState, KG_ENTITY_TYPES, KG_RELATION_TYPES, question_for_retrieval
except ImportError:
    from state import GraphState, KG_ENTITY_TYPES, KG_RELATION_TYPES, question_for_retrieval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common.env_loader import load_app_env  # noqa: E402
load_app_env()

# ============================================================
# 意图识别提示词（包含知识图谱实体/关系类型说明）
# ============================================================

_ENTITY_TYPE_DESC = "\n".join(
    f"  - {name}：{desc}" for name, desc in KG_ENTITY_TYPES.items()
)

_RELATION_TYPE_DESC = "\n".join(
    f"  - {name}：{desc}" for name, desc in KG_RELATION_TYPES.items()
)

from common.langfuse_manager import fetch_prompt  # noqa: E402

_INTENT_FALLBACK = """你是中医知识问答系统的意图识别器。

## 背景
本系统内置中医知识图谱，包含以下实体类型：
{entity_types}

关系类型：
{relation_types}

## 任务
判断用户问题是否属于中医领域。

## 归类规则

**归类为 tcm**（中医相关）：
- 问题涉及上述任意一种实体类型（药材、方剂、症状、疾病、功效、出处、分类、经络）
- 中医养生、食疗、针灸、推拿、气功等传统医学内容
- 中药配伍、方剂加减、辨证论治等专业内容
- 中西医结合的咨询

**归类为 general**（普通问题）：
- 日常闲聊、无关话题
- 西医/现代医学问题（不含中医视角）
- 编程、数学、天气等非医学问题
- 与上述实体类型完全无关的问题

## 输出格式（严格 JSON）
只输出一行 JSON，不要额外文字：
{{"intent": "<tcm 或 general>", "reason": "<简短理由，≤30字>"}}"""


def _get_intent_prompt() -> str:
    """运行时从 Langfuse 拉取 prompt（SDK 缓存 60s，改 prompt 后自动生效）。"""
    return fetch_prompt(
        "intent_recognition",
        _INTENT_FALLBACK,
        entity_types=_ENTITY_TYPE_DESC,
        relation_types=_RELATION_TYPE_DESC,
    )


# ============================================================
# LLM 工厂（惰性初始化 + 依赖注入，便于测试）
# ============================================================


def create_llm():
    """创建项目通用的 OpenAI 兼容 LLM。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL"),
        temperature=0,
    )


# ============================================================
# 解析
# ============================================================

_JSON_PATTERN = re.compile(r"\{[^{}]*\"intent\"[^{}]*\}", re.DOTALL)


def _parse_intent_json(raw_text: str) -> dict[str, str]:
    """从 LLM 文本输出中提取 JSON，容错处理。"""
    text = raw_text.strip()

    # 1. 尝试直接解析整段文本
    try:
        obj = json.loads(text)
        if "intent" in obj:
            return obj
    except json.JSONDecodeError:
        pass

    # 2. 正则匹配第一个含 intent 的 JSON 对象
    match = _JSON_PATTERN.search(text)
    if match:
        try:
            obj = json.loads(match.group())
            if "intent" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 3. 兜底：通过关键词判断
    text_lower = text.lower()
    if "tcm" in text_lower or "中医" in text:
        return {"intent": "tcm", "reason": "关键词兜底识别"}
    return {"intent": "general", "reason": "关键词兜底识别"}


def _normalize_result(raw: dict[str, str]) -> dict[str, Any]:
    """规范化并校验解析结果。"""
    intent = str(raw.get("intent", "general")).strip().lower()
    if intent not in ("tcm", "general"):
        intent = "general"
    reason = str(raw.get("reason", "")).strip()[:60]
    if not reason:
        reason = "未提供理由"
    return {"intent": intent, "reason": reason}


# ============================================================
# 节点构建
# ============================================================


def make_intent_recognition_node(llm):
    """构建可注入 LLM 的意图识别节点（测试时注入假 LLM）。"""

    def intent_recognition_node(state: GraphState) -> dict[str, Any]:
        question = question_for_retrieval(state)
        if not question:
            raise ValueError("state.user_question 不能为空")

        messages = [
            SystemMessage(content=_get_intent_prompt()),
            HumanMessage(content=f"用户问题：{question}"),
        ]

        raw_text = str(llm.invoke(messages).content)
        parsed = _parse_intent_json(raw_text)
        result = _normalize_result(parsed)

        return {
            "intent": result["intent"],
            "intent_reason": result["reason"],
            "is_zhongyi_intent": result["intent"] == "tcm",
        }

    return intent_recognition_node


def intent_recognition_node(state: GraphState) -> dict[str, Any]:
    """默认节点（首次调用时惰性初始化 LLM）。"""
    return make_intent_recognition_node(create_llm())(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================


def _print_state(title: str, state: dict, highlight_keys: set[str] | None = None):
    """打印 state 快照。"""
    highlight_keys = highlight_keys or set()
    print(f"\n  ┌─ {title} " + "─" * 50)
    keys = [
        ("user_question",     "用户问题"),
        ("intent",            "意图"),
        ("intent_reason",     "识别理由"),
        ("is_zhongyi_intent", "是否中医"),
    ]
    for key, label in keys:
        val = state.get(key)
        if val is None or val == "":
            display = "(空)"
        elif isinstance(val, bool):
            display = "True ✓ 中医" if val else "False ✗ 普通"
        else:
            display = str(val)[:80]
        marker = " ◀── 变更" if key in highlight_keys else ""
        print(f"  │  {label + ' (' + key + ')':<30s} = {display}{marker}")
    print("  └" + "─" * 60)


def main():
    """直接运行 python -m _004_langgraph_more_nodes.intent_recognition 测试节点。"""
    test_cases = [
        "四君子汤有什么功效？",
        "人参的性味归经是什么？",
        "今天天气怎么样？",
        "帮我写一个 Python 脚本",
        "头痛是什么原因？",
    ]

    print("=" * 64)
    print("  意图识别节点 · 自测")
    print("=" * 64)

    print("\n初始化 LLM...", end=" ", flush=True)
    llm = create_llm()
    print("就绪。")

    node = make_intent_recognition_node(llm)

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'─' * 64}")
        print(f"  [{i}/{len(test_cases)}] 输入: {question}")

        before = {"user_question": question}
        update = node(before)
        after = {**before, **update}

        _print_state("before", before, {"user_question"})
        _print_state("after ", after, set(update.keys()))

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
