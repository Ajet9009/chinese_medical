"""实体抽取节点：从用户问题中抽取六类中医实体。

why: 承接中医意图，按 6 类实体抽取后用 BGE 匹配标准实体 → 生成 Cypher
how: LLM 按 Symptom/Disease/Formula/Herb/Effect/Source 六类输出严格 JSON
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from .state import GraphState
except ImportError:
    from state import GraphState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, "common", ".env"))

# ============================================================
# 六类实体字段映射
# ============================================================

CATEGORY_FIELDS = [
    ("symptoms", "Symptom", "症状（如咳嗽、发热、腹痛、头痛）"),
    ("diseases", "Disease", "疾病（如感冒、肺炎、肾虚、阴虚火旺）"),
    ("formulas", "Formula", "方剂（如四君子汤、桂枝汤、麻黄汤）"),
    ("herbs",    "Herb",    "药材（如人参、黄芪、麻黄、桂枝）"),
    ("effects",  "Effect",  "功效（如补气、活血、祛湿、止痛）"),
    ("sources",  "Source",  "出处（如《本草纲目》《伤寒论》《金匮要略》）"),
]

# JSON key → state key 映射
_FIELD_TO_STATE_KEY = {
    "symptoms":  "user_input_symptoms",
    "diseases":  "user_input_diseases",
    "formulas":  "user_input_formulas",
    "herbs":     "user_input_herbs",
    "effects":   "user_input_effects",
    "sources":   "user_input_sources",
}

# ============================================================
# 提示词
# ============================================================

_CATEGORY_DESC = "\n".join(
    f"{i}. {label}（{desc}）" for i, (_, label, desc) in enumerate(CATEGORY_FIELDS, 1)
)

from common.langfuse_manager import fetch_prompt  # noqa: E402

_ENTITY_EXTRACTION_FALLBACK = """你是中医实体抽取器。从用户问题中抽取以下六类中医实体。

## 实体类别
{category_desc}

## 抽取规则
1. 只抽取问题中明确出现的中医相关词，不要凭空生成
2. 每个实体词应尽量短（2-8字），是独立的中医概念单元
3. 同一实体不要跨类别重复
4. 如果某类实体不存在，该字段返回空列表 []

## 示例
问题：感冒咳嗽用麻黄还是桂枝
输出：
{{"symptoms": ["咳嗽"], "diseases": ["感冒"], "formulas": [], "herbs": ["麻黄", "桂枝"], "effects": [], "sources": []}}

问题：四君子汤补气养血的功效出自哪里
输出：
{{"symptoms": [], "diseases": [], "formulas": ["四君子汤"], "herbs": [], "effects": ["补气", "养血"], "sources": []}}

问题：肚子疼吃什么药
输出：
{{"symptoms": ["肚子疼"], "diseases": [], "formulas": [], "herbs": [], "effects": [], "sources": []}}

## 输出格式（严格 JSON，只输出一行）
{{"symptoms": [...], "diseases": [...], "formulas": [...], "herbs": [...], "effects": [...], "sources": [...]}}"""


def _get_extraction_prompt() -> str:
    """运行时从 Langfuse 拉取 prompt（SDK 缓存 60s）。"""
    return fetch_prompt(
        "entity_extraction",
        _ENTITY_EXTRACTION_FALLBACK,
        category_desc=_CATEGORY_DESC,
    )

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
    )


# ============================================================
# 解析
# ============================================================

_JSON_PATTERN = re.compile(r"\{[^{}]*\"symptoms\"[^{}]*\"sources\"[^{}]*\}", re.DOTALL)
_EXPECTED_KEYS = frozenset(_FIELD_TO_STATE_KEY.keys())  # symptoms, diseases, ...


def _parse_entities_json(raw_text: str) -> dict[str, list[str]]:
    """从 LLM 文本中提取六类实体 JSON，容错。"""
    text = raw_text.strip()

    def _extract(obj: dict) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for key in _EXPECTED_KEYS:
            val = obj.get(key, [])
            result[key] = val if isinstance(val, list) else []
        return result

    # 1. 直接解析
    try:
        obj = json.loads(text)
        return _extract(obj)
    except json.JSONDecodeError:
        pass

    # 2. 正则匹配含 symptoms...sources 的 JSON 对象
    match = _JSON_PATTERN.search(text)
    if match:
        try:
            obj = json.loads(match.group())
            return _extract(obj)
        except json.JSONDecodeError:
            pass

    # 3. 兜底：全部空
    return {k: [] for k in _EXPECTED_KEYS}


def _clean_entities(categories: dict[str, list[str]]) -> dict[str, list[str]]:
    """清洗每类实体：去空、去重、去过长。"""
    result: dict[str, list[str]] = {}
    for key, items in categories.items():
        seen: set[str] = set()
        clean: list[str] = []
        for item in items:
            s = str(item).strip()
            if not s or len(s) > 20:
                continue
            low = s.lower()
            if low in seen:
                continue
            seen.add(low)
            clean.append(s)
        result[key] = clean
    return result


# ============================================================
# 节点构建
# ============================================================


def make_entity_extraction_node(llm):
    """构建可注入 LLM 的六类实体抽取节点。"""

    def entity_extraction_node(state: GraphState) -> dict[str, Any]:
        question = str(state.get("user_question", "")).strip()
        if not question:
            raise ValueError("state.user_question 不能为空")

        messages = [
            SystemMessage(content=_get_extraction_prompt()),
            HumanMessage(content=f"用户问题：{question}"),
        ]

        raw_text = str(llm.invoke(messages).content)
        categories = _parse_entities_json(raw_text)
        cleaned = _clean_entities(categories)

        # 映射回 state 字段
        return {_FIELD_TO_STATE_KEY[k]: v for k, v in cleaned.items()}

    return entity_extraction_node


def entity_extraction_node(state: GraphState) -> dict[str, Any]:
    """默认节点（首次调用时惰性初始化 LLM）。"""
    return make_entity_extraction_node(create_llm())(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================


def _print_state(title: str, state: dict, highlight_keys: set[str] | None = None):
    """打印 state 快照。"""
    highlight_keys = highlight_keys or set()
    print(f"\n  ┌─ {title} " + "─" * 50)
    print(f"  │  用户问题: {state.get('user_question', '(空)')}")
    for json_key, state_key in _FIELD_TO_STATE_KEY.items():
        val = state.get(state_key, [])
        display = "[" + ", ".join(str(x) for x in val) + "]" if val else "[] (无)"
        marker = " ◀──" if state_key in highlight_keys else ""
        print(f"  │  {json_key:<12s} = {display}{marker}")
    print("  └" + "─" * 60)


def main():
    """直接运行 python entity_extraction.py 测试节点。"""
    test_cases = [
        "肚子疼吃什么药",
        "有什么药汤可以治疗气虚",
        "四君子汤有什么功效？",
        "感冒咳嗽用麻黄还是桂枝",
        "人参的性味归经出自《本草纲目》",
        "肾虚怎么调理",
        "补气养血吃什么好",
        "今天天气怎么样",
    ]

    print("=" * 64)
    print("  实体抽取节点（六类）· 自测")
    print("=" * 64)

    print("\n初始化 LLM...", end=" ", flush=True)
    llm = create_llm()
    print("就绪。")

    node = make_entity_extraction_node(llm)

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'─' * 64}")
        print(f"  [{i}/{len(test_cases)}] 输入: {question}")

        before: dict[str, Any] = {"user_question": question}
        for state_key in _FIELD_TO_STATE_KEY.values():
            before[state_key] = []
        update = node(before)
        after = {**before, **update}

        _print_state("before", before, {"user_question"})
        _print_state("after ", after, set(update.keys()))

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
