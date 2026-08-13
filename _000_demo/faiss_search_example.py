"""可直接运行的 FAISS 实体匹配示例。

用法：
    python tests/faiss_search_example.py
    python tests/faiss_search_example.py "治疗咳嗽的中药" --top-k 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "common" / ".env")

from common.faiss_vector_store import FaissEntityStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="用向量相似度匹配 Neo4j 标准实体")
    parser.add_argument("question", nargs="?", default="气虚")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()

    required = ("EMBEDDING_MODEL_PATH", "FAISS_INDEX_PATH", "FAISS_METADATA_PATH")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise SystemExit(f"common/.env 缺少配置: {', '.join(missing)}")

    store = FaissEntityStore(
        os.environ["EMBEDDING_MODEL_PATH"],
        os.environ["FAISS_INDEX_PATH"],
        os.environ["FAISS_METADATA_PATH"],
    ).load()
    matches = store.search(args.question, top_k=args.top_k, min_score=args.min_score)

    print(f"问题: {args.question}")
    print(f"匹配结果: {len(matches)} 条")
    for rank, entity in enumerate(matches, start=1):
        print(
            f"{rank}. {entity.get('name', '')} "
            f"[{entity.get('type', '')}] score={entity['score']:.4f} "
            f"id={entity.get('id', '')}"
        )


if __name__ == "__main__":
    main()
