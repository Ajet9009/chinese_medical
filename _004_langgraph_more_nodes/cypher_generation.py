"""Cypher 生成节点：根据匹配实体 + Neo4j schema 生成查询语句。

why: 标准实体无法概括所有关系，需通过图数据库探索关联实体
how: LLM 结合 matched_* 实体 + 实时 schema → 生成 Cypher → EXPLAIN 校验 → 纠正重试
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from .state import GraphState, question_for_retrieval
except ImportError:
    from state import GraphState, question_for_retrieval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from common.env_loader import load_app_env  # noqa: E402
load_app_env()

MAX_RETRIES = 3  # 每条 Cypher 最多校验重试次数

# ============================================================
# 匹配实体 → 提示词摘要
# ============================================================

_MATCHED_LABELS = [
    ("matched_symptoms", "Symptom"),
    ("matched_diseases", "Disease"),
    ("matched_formulas", "Formula"),
    ("matched_herbs",   "Herb"),
    ("matched_effects", "Effect"),
    ("matched_sources", "Source"),
]


def _matched_summary(state: GraphState) -> str:
    """将 state 中的 matched_* 实体格式化为提示词片段。"""
    lines: list[str] = []
    for state_key, _kg_type in _MATCHED_LABELS:
        entities = state.get(state_key, []) or []
        if not entities:
            continue
        items = ", ".join(f"{e['name']}(id:{e['id']})" for e in entities)
        lines.append(f"  {state_key}: {items}")
    return "\n".join(lines) if lines else "  (无匹配实体)"


# ============================================================
# 提示词模板
# ============================================================

from common.langfuse_manager import fetch_prompt  # noqa: E402

_CYPHER_FALLBACK = """你是 Neo4j Cypher 查询生成专家。根据给定的标准实体和图 schema，生成查询语句。

## 规则
1. 使用提供的实体 id 作为查询起点（不是 name），如 `WHERE n.id = 'E123'`
2. 每个实体至少探索一跳关系，找出关联的节点和关系
3. 关系方向和类型必须与 schema 中的 pairs 一致
4. RETURN 需包含关联节点的 id、type(labels)、name 以及关系类型
5. 生成 1-2 条 Cypher，每条独立可执行，每条尽量简短
6. 输出必须是严格 JSON 字符串数组，不得包含解释、注释或中文

## 输出格式
["cypher语句1", "cypher语句2", ...]

示例：
["MATCH (n:Herb {id: 'E1'})-[r]->(m) RETURN n.name, type(r), labels(m), m.name, m.id LIMIT 20",
 "MATCH (n:Herb {id: 'E1'})<-[r]-(m) RETURN n.name, type(r), labels(m), m.name, m.id LIMIT 20"]"""


def _get_cypher_prompt() -> str:
    """运行时从 Langfuse 拉取 prompt（SDK 缓存 60s）。"""
    return fetch_prompt("cypher_generation", _CYPHER_FALLBACK)


# ── schema 缓存（图结构很少变，缓存 5 分钟避免每次查询 Neo4j）──

_schema_cache: dict[str, Any] = {"data": None, "ts": 0}
SCHEMA_TTL = 300  # 秒


def _get_cached_schema() -> dict:
    """获取 schema，带 5 分钟内存缓存。"""
    now = time.time()
    if _schema_cache["data"] is None or now - _schema_cache["ts"] > SCHEMA_TTL:
        from common.neo4j_manager import Neo4jManager
        mgr = Neo4jManager()
        try:
            _schema_cache["data"] = mgr.get_schema_metadata()
        finally:
            mgr.close()
        _schema_cache["ts"] = now
    return _schema_cache["data"]


def _build_prompt(state: GraphState, schema_meta: dict) -> str:
    matched = _matched_summary(state)
    schema_json = json.dumps(schema_meta, ensure_ascii=False, indent=2)
    return f"""## 图 Schema
{schema_json}

## 匹配到的标准实体
{matched}

## 用户问题
{question_for_retrieval(state)}

请生成 Cypher 查询语句（JSON 数组格式）。"""


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
        max_tokens=512,  # 限制输出长度，Cypher 生成不需超长输出
    )


# ============================================================
# 解析 + 校验
# ============================================================

_JSON_ARRAY_PATTERN = re.compile(r"\[.*\]", re.DOTALL)


def _parse_cypher_array(raw_text: str) -> list[str]:
    """从 LLM 输出中提取 Cypher 字符串数组。"""
    text = raw_text.strip()
    # 1. 直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [str(s).strip() for s in obj if str(s).strip()]
    except json.JSONDecodeError:
        pass
    # 2. 正则提取 JSON 数组
    match = _JSON_ARRAY_PATTERN.search(text)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, list):
                return [str(s).strip() for s in obj if str(s).strip()]
        except json.JSONDecodeError:
            pass
    return []


def _validate_and_fix(llm, cypher: str, schema_meta: dict) -> str | None:
    """EXPLAIN 校验 + 错误修正，最多 MAX_RETRIES 次。返回有效 Cypher 或 None。"""
    from common.neo4j_manager import Neo4jManager

    mgr = Neo4jManager()
    try:
        for attempt in range(MAX_RETRIES):
            result = mgr.validate_cypher(cypher)
            if result["valid"]:
                return cypher  # 通过
            if attempt == MAX_RETRIES - 1:
                break  # 最后一次失败
            # 喂错误给 LLM 修正
            fix_prompt = f"""以下 Cypher 执行 EXPLAIN 时报错，请修正。

## 图 Schema
{json.dumps(schema_meta, ensure_ascii=False, indent=2)}

## 错误的 Cypher
{cypher}

## 错误信息
{result['error']}

请只输出修正后的 Cypher 语句（一行，不要引号包裹，不要解释）："""
            try:
                resp = llm.invoke([
                    SystemMessage(content="你是 Cypher 修正专家，只输出修正后的 Cypher 语句。"),
                    HumanMessage(content=fix_prompt),
                ])
                cypher = str(resp.content).strip().strip("`").strip(";").strip()
                # 去掉 markdown 代码块包裹
                if cypher.startswith("cypher"):
                    cypher = cypher[6:].strip()
            except Exception:
                break
        return None  # 修正失败
    finally:
        mgr.close()


# ============================================================
# 节点构建
# ============================================================


def make_cypher_generation_node(llm):
    """构建 Cypher 生成节点。"""

    def cypher_generation_node(state: GraphState) -> dict[str, Any]:
        # 1) 获取 schema（带缓存）
        schema_meta = _get_cached_schema()

        # 2) LLM 生成（流式，降低 TTFT）
        prompt = _build_prompt(state, schema_meta)
        messages = [
            SystemMessage(content=_get_cypher_prompt()),
            HumanMessage(content=prompt),
        ]
        raw_parts: list[str] = []
        for chunk in llm.stream(messages):
            raw_parts.append(str(chunk.content))
        raw = "".join(raw_parts)
        candidates = _parse_cypher_array(raw)

        # 3) 逐条校验 + 修正
        valid_queries: list[str] = []
        for c in candidates:
            fixed = _validate_and_fix(llm, c, schema_meta)
            if fixed:
                valid_queries.append(fixed)

        return {"cypher_queries": valid_queries}

    return cypher_generation_node


def cypher_generation_node(state: GraphState) -> dict[str, Any]:
    return make_cypher_generation_node(create_llm())(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================


def main():
    """直接运行 python cypher_generation.py 测试节点。"""
    import sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    PARENT = os.path.dirname(HERE)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    # 模拟 state（含匹配实体）
    test_state: dict[str, Any] = {
        "user_question": "四君子汤有什么功效？",
        "matched_symptoms": [],
        "matched_diseases": [],
        "matched_formulas": [
            {"id": "E100", "type": "Formula", "name": "四君子汤", "score": 0.92},
        ],
        "matched_herbs": [
            {"id": "E200", "type": "Herb", "name": "人参", "score": 0.88},
            {"id": "E201", "type": "Herb", "name": "白术", "score": 0.85},
        ],
        "matched_effects": [
            {"id": "E300", "type": "Effect", "name": "补气", "score": 0.90},
        ],
        "matched_sources": [],
    }

    print("=" * 64)
    print("  Cypher 生成节点 · 自测")
    print("=" * 64)

    print("\n初始化 LLM...", end=" ", flush=True)
    llm = create_llm()
    print("就绪。")

    node = make_cypher_generation_node(llm)

    print("\n输入实体:")
    for sk, _ in _MATCHED_LABELS:
        ents = test_state.get(sk, [])
        if ents:
            names = [f"{e['name']}({e['id']})" for e in ents]
            print(f"  {sk}: {', '.join(names)}")

    print("\n生成 Cypher...")
    update = node(test_state)
    queries = update.get("cypher_queries", [])

    if queries:
        print(f"\n✓ 生成 {len(queries)} 条有效 Cypher:")
        for i, q in enumerate(queries):
            print(f"  [{i+1}] {q}")
    else:
        print("\n✗ 未生成有效 Cypher")

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
