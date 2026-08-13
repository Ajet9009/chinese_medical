#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 data/neo4j 的图谱数据导入 Neo4j。

读取:
  - data/neo4j/entities.json   实体字段 "entities"
  - data/neo4j/relations.json  关系字段 "relations"

特性:
  - 进度日志: 每批打印 已插入/总数/百分比
  - 断点续传: checkpoint 记录已插入数据，重跑跳过

用法:
  python import_neo4j.py                       # 全量导入
  python import_neo4j.py --limit 100            # 仅测试 100 条
  python import_neo4j.py --no-checkpoint        # 忽略断点重新全量
  python import_neo4j.py --skip-constraints     # 不建唯一约束
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 使 `import common.*` 生效

from common.neo4j_manager import Neo4jManager  # noqa: E402

DATA_DIR = ROOT / "data" / "neo4j"
DEFAULT_ENTITIES = DATA_DIR / "entities.json"
DEFAULT_RELATIONS = DATA_DIR / "relations.json"
DEFAULT_CHECKPOINT = DATA_DIR / "import_checkpoint.json"

log = logging.getLogger("import_neo4j")


def _load_list(path: Path) -> list[dict]:
    """读取 JSON 文件。兼容两种形态：顶层数组 / 顶层对象含 entities|relations 键。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("entities", "relations"):
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"无法识别的 JSON 结构: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="导入知识图谱到 Neo4j")
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES, help="实体 JSON 路径")
    ap.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS, help="关系 JSON 路径")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="断点文件路径")
    ap.add_argument("--no-checkpoint", action="store_true", help="忽略断点重新全量")
    ap.add_argument("--batch", type=int, default=500, help="每批条数")
    ap.add_argument("--limit", type=int, default=None, help="仅导入前 N 条实体/关系（测试用）")
    ap.add_argument("--skip-constraints", action="store_true", help="不建唯一约束")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    entities = _load_list(args.entities)
    relations = _load_list(args.relations)
    if args.limit:
        entities, relations = entities[: args.limit], relations[: args.limit]
    print(f"读取图谱: 实体 {len(entities)} 条，关系 {len(relations)} 条")
    print(f"来源: {args.entities.name} + {args.relations.name}")

    mgr = Neo4jManager()
    try:
        t0 = time.time()
        if not args.skip_constraints:
            mgr.create_constraints()

        stats = mgr.batch_import(
            entities,
            relations,
            checkpoint_path=None if args.no_checkpoint else args.checkpoint,
            batch_size=args.batch,
            on_progress=lambda m: print(m, flush=True),
        )
        print(f"\n导入完成，耗时 {time.time() - t0:.1f}s")
        print(f"  新插实体   {stats['entities']}")
        print(f"  新插关系   {stats['relations']}")
        print(f"  跳过(已插) {stats['skipped']}")
    finally:
        mgr.close()


if __name__ == "__main__":
    main()
