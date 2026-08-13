#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""导出 Neo4j 图 schema 元数据 → data/neo4j_metadata.json。

用途：作为提示词上下文指导 LLM 精准生成 Cypher 语句。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from common.neo4j_manager import Neo4jManager  # noqa: E402


def main() -> None:
    out_path = ROOT / "data" / "neo4j_metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mgr = Neo4jManager()
    try:
        meta = mgr.get_schema_metadata()
    finally:
        mgr.close()

    out_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 统计
    nodes = meta["nodes"]
    rels = meta["relationships"]
    total_nodes = sum(v["count"] for v in nodes.values())
    total_rels = sum(v["count"] for v in rels.values())

    print(f"Schema 元数据已导出 → {out_path}")
    print(f"  节点类型: {len(nodes)} 种, 共 {total_nodes} 个")
    print(f"  关系类型: {len(rels)} 种, 共 {total_rels} 条")
    for label, info in nodes.items():
        print(f"    ({label}) x{info['count']} 属性={info['properties']}")
    for rt, info in rels.items():
        pairs_str = ", ".join(f"({p['from']})→({p['to']})" for p in info["pairs"])
        print(f"    [:{rt}] x{info['count']} {pairs_str}")


if __name__ == "__main__":
    main()
