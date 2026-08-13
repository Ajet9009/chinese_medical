"""Cypher 执行节点：执行 Cypher 查询，结果去重去噪后填充 state.neo4j_answer。

why: 仅有 Cypher 无法获取内容，需通过 Neo4j 执行得到实际数据
how: execute_cypher_batch → 提取 records → 去重去空格 → 拼成精简上下文
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

try:
    from .state import GraphState
except ImportError:
    from state import GraphState

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, "common", ".env"))

# ============================================================
# 结果格式化
# ============================================================


def _format_records(records: list[dict]) -> str:
    """将 Cypher 返回的 records 列表压缩为精简纯文本。

    规则：
    - 去重（完全相同的 record 只保留一条）
    - 去空格（value 字符串首尾去空白）
    - 每条 record 一行，字段用 | 分隔
    - 最多 200 条，超出截断并标注
    """
    if not records:
        return ""

    # 去重
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        sig = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            unique.append(rec)

    max_rows = 200
    truncated = len(unique) > max_rows
    rows = unique[:max_rows]

    # 提取表头
    columns = list(rows[0].keys()) if rows else []

    lines: list[str] = []
    lines.append(" | ".join(columns))
    lines.append("-" * 40)
    for row in rows:
        vals = [str(row.get(c, "")).strip() for c in columns]
        lines.append(" | ".join(vals))
    if truncated:
        lines.append(f"... (截断，共 {len(unique)} 条，显示前 {max_rows} 条)")

    return "\n".join(lines)


def _build_context(exec_results: list[dict]) -> str:
    """将 execute_cypher_batch 的返回结果拼成精简上下文。"""
    parts: list[str] = []
    for r in exec_results:
        if not r["ok"]:
            continue
        records = r.get("records", [])
        if not records:
            continue
        cypher_preview = r["cypher"][:80]
        parts.append(f"-- 查询: {cypher_preview}")
        parts.append(_format_records(records))
        parts.append("")
    return "\n".join(parts).strip()


# ============================================================
# 节点构建（无 LLM）
# ============================================================


def make_cypher_executor_node():
    """构建 Cypher 执行节点。"""

    def cypher_executor_node(state: GraphState) -> dict[str, Any]:
        queries = state.get("cypher_queries", []) or []
        if not queries:
            return {"neo4j_answer": ""}

        from common.neo4j_manager import Neo4jManager

        mgr = Neo4jManager()
        try:
            results = mgr.execute_cypher_batch(queries)
        finally:
            mgr.close()

        context = _build_context(results)
        return {"neo4j_answer": context}

    return cypher_executor_node


def cypher_executor_node(state: GraphState) -> dict[str, Any]:
    return make_cypher_executor_node()(state)


# ============================================================
# main() — 直接运行本文件进行节点测试
# ============================================================


def main():
    """直接运行 python cypher_executor.py 测试节点。"""
    HERE = os.path.dirname(os.path.abspath(__file__))
    PARENT = os.path.dirname(HERE)
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)

    test_state: dict[str, Any] = {
        "cypher_queries": [
            "MATCH (n:Herb) RETURN n.name, labels(n) LIMIT 5",
            "MATCH (n:Formula) RETURN n.name, labels(n) LIMIT 3",
        ],
    }

    print("=" * 64)
    print("  Cypher 执行节点 · 自测")
    print("=" * 64)

    node = make_cypher_executor_node()
    update = node(test_state)
    result = update.get("neo4j_answer", "")

    print(f"\n✓ 执行 {len(test_state['cypher_queries'])} 条 Cypher")
    print(f"  结果长度: {len(result)} 字符\n")
    print(result[:2000] if len(result) > 2000 else result)

    print(f"\n{'═' * 64}")
    print("  测试完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
