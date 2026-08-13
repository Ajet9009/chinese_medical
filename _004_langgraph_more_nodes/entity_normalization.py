"""实体标准化节点：FAISS + BGE 向量匹配用户实体到知识图谱标准实体。

why: 承接实体抽取节点，将口语化实体匹配到 KG 标准实体 → 后续生成 Cypher
how: 用 FaissEntityStore 逐类搜索，动态阈值 + 骤降截断 + min_score=0.6
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

try:
    from .state import GraphState, MatchedEntity
except ImportError:
    from state import GraphState, MatchedEntity

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, "common", ".env"))

# ============================================================
# 六类映射：user_input_* → matched_* → KG 类型
# ============================================================

_CATEGORY_MAP: list[tuple[str, str, str]] = [
    # (user_input_state_key, matched_state_key, kg_entity_type)
    ("user_input_symptoms", "matched_symptoms", "Symptom"),
    ("user_input_diseases", "matched_diseases", "Disease"),
    ("user_input_formulas", "matched_formulas", "Formula"),
    ("user_input_herbs",    "matched_herbs",    "Herb"),
    ("user_input_effects",  "matched_effects",  "Effect"),
    ("user_input_sources",  "matched_sources",  "Source"),
]

# ============================================================
# FAISS Store 单例（惰性加载，避免每次调用重载模型）
# ============================================================

_faiss_store: Any = None  # FaissEntityStore | None


def _get_faiss_store():
    """惰性加载 FAISS 实体索引。"""
    global _faiss_store
    if _faiss_store is not None:
        return _faiss_store

    from common.faiss_vector_store import FaissEntityStore

    model_path = os.getenv("EMBEDDING_MODEL_PATH")
    index_path = os.getenv("FAISS_INDEX_PATH")
    metadata_path = os.getenv("FAISS_METADATA_PATH")
    if not all((model_path, index_path, metadata_path)):
        raise ValueError(
            "请在 common/.env 设置 EMBEDDING_MODEL_PATH、FAISS_INDEX_PATH、FAISS_METADATA_PATH"
        )

    _faiss_store = FaissEntityStore(model_path, index_path, metadata_path).load()
    return _faiss_store


# ============================================================
# 动态阈值 + 骤降截断
# ============================================================

MIN_SCORE = float(os.getenv("MATCH_MIN_SCORE", "0.6"))
TOP_K = int(os.getenv("MATCH_TOP_K", "3"))
DROP_RATIO = float(os.getenv("MATCH_DROP_RATIO", "0.6"))  # 骤降比例阈值


def _apply_threshold(results: list[dict], min_score: float, drop_ratio: float) -> list[dict]:
    """对已排序结果应用动态阈值：骤降截断 + 最低分过滤。

    Args:
        results: 按 score 降序排列的匹配结果。
        min_score: 绝对最低分阈值（< min_score 的全部丢弃）。
        drop_ratio: 骤降比例，score[i] / score[i-1] < drop_ratio 时从 i 截断。

    Returns:
        截断并过滤后的结果列表。
    """
    # 1. 最低分过滤
    filtered = [r for r in results if r["score"] >= min_score]
    if not filtered:
        return []

    # 2. 骤降检测：从第 2 条开始，判断相对前一条是否骤降
    keep = [filtered[0]]
    for i in range(1, len(filtered)):
        if filtered[i]["score"] / keep[-1]["score"] < drop_ratio:
            break  # 骤降，舍弃当前及之后全部
        keep.append(filtered[i])

    return keep


# ============================================================
# 匹配逻辑
# ============================================================

def _match_entities(
    store,
    raw_entities: list[str],
    kg_type: str,
    top_k: int,
    min_score: float,
    drop_ratio: float,
) -> list[MatchedEntity]:
    """将一组口语化实体匹配到 KG 标准实体。

    为每个输入实体搜索 top_k 条候选，合并后按 score 降序排列，
    经动态阈值截断后返回 MatchedEntity 列表。

    Args:
        store: FaissEntityStore 实例。
        raw_entities: 用户输入的口语化实体列表。
        kg_type: 期望匹配的 KG 实体类型（如 "Symptom"）。
        top_k: FAISS 搜索返回条数。
        min_score: 绝对最低分。
        drop_ratio: 骤降比例。

    Returns:
        MatchedEntity 列表（去重）。
    """
    # 收集所有候选
    candidates: list[dict] = []
    for entity_str in raw_entities:
        # 带类型前缀搜索，提高同类匹配精度
        query = f"{kg_type}: {entity_str}"
        hits = store.search(query, top_k=top_k, min_score=0.0)
        for h in hits:
            candidates.append({
                "id": h["id"],
                "type": h["type"],
                "name": h["name"],
                "score": h["score"],
            })

    # 按 score 降序排
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 去重（按 id）
    seen: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)

    # 动态阈值截断
    return _apply_threshold(unique, min_score, drop_ratio)


# ============================================================
# 节点构建
# ============================================================


def make_entity_normalization_node():
    """构建实体标准化节点（无 LLM，纯向量匹配）"""

    def entity_normalization_node(state: GraphState) -> dict[str, Any]:
        store = _get_faiss_store()
        result: dict[str, Any] = {}

        for user_key, matched_key, kg_type in _CATEGORY_MAP:
            raw_entities = state.get(user_key, []) or []
            if not raw_entities:
                result[matched_key] = []
                continue

            matched = _match_entities(
                store, raw_entities, kg_type,
                top_k=TOP_K, min_score=MIN_SCORE, drop_ratio=DROP_RATIO,
            )
            result[matched_key] = matched

        return result

    return entity_normalization_node


def entity_normalization_node(state: GraphState) -> dict[str, Any]:
    """默认节点（惰性初始化 FAISS）。"""
    return make_entity_normalization_node()(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================

_CATEGORY_LABELS = {
    "matched_symptoms": "症状",
    "matched_diseases": "疾病",
    "matched_formulas": "方剂",
    "matched_herbs":   "药材",
    "matched_effects": "功效",
    "matched_sources": "出处",
}


def _print_state(title: str, state: dict):
    """打印 state 快照。"""
    print(f"\n  ┌─ {title} " + "─" * 50)
    # 用户输入
    for user_key, matched_key, kg_type in _CATEGORY_MAP:
        raw = state.get(user_key, []) or []
        matched = state.get(matched_key, []) or []
        label = _CATEGORY_LABELS.get(matched_key, matched_key)
        raw_display = "[" + ", ".join(raw) + "]" if raw else "[]"
        if matched:
            match_display = " → ".join(
                f"{m['name']}({m['score']:.3f})" for m in matched
            )
        else:
            match_display = "(无匹配)"
        print(f"  │  {label:<6s}  输入={raw_display:<30s} 匹配={match_display}")
    print("  └" + "─" * 60)


def main():
    """直接运行 python entity_normalization.py 测试节点。"""
    import sys

    HERE = os.path.dirname(os.path.abspath(__file__))
    PARENT = os.path.dirname(HERE)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    test_cases: list[dict] = [
        {
            "user_input_symptoms": ["咳嗽", "肚子疼"],
            "user_input_diseases": ["感冒"],
            "user_input_formulas": ["四君子汤"],
            "user_input_herbs":    ["人参", "黄芪"],
            "user_input_effects":  ["补气", "活血"],
            "user_input_sources":  ["本草纲目"],
        },
        {
            "user_input_symptoms": ["头痛"],
            "user_input_diseases": ["肾虚"],
            "user_input_formulas": ["桂枝汤"],
            "user_input_herbs":    ["麻黄", "桂枝"],
            "user_input_effects":  ["祛湿"],
            "user_input_sources":  ["伤寒论"],
        },
    ]

    print("=" * 64)
    print("  实体标准化节点（FAISS + BGE）· 自测")
    print("=" * 64)

    print("\n初始化 FAISS...", end=" ", flush=True)
    try:
        node = make_entity_normalization_node()
        print("就绪。")
    except Exception as exc:
        print(f"\n  错误: {exc}")
        return

    for i, simulated_state in enumerate(test_cases, 1):
        print(f"\n{'─' * 64}")
        print(f"  [{i}/{len(test_cases)}]")

        _print_state("before", simulated_state)
        update = node(simulated_state)
        after = {**simulated_state, **update}
        _print_state("after ", after)

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
