"""Build FAISS vectors and an entity lookup file from all Neo4j entities."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / "common" / ".env")

from common.faiss_vector_store import FaissEntityStore  # noqa: E402
from common.neo4j_manager import Neo4jManager  # noqa: E402


def main() -> None:
    model_path = os.getenv("EMBEDDING_MODEL_PATH")
    index_path = os.getenv("FAISS_INDEX_PATH")
    metadata_path = os.getenv("FAISS_METADATA_PATH")
    if not all((model_path, index_path, metadata_path)):
        raise ValueError(
            "请在 common/.env 设置 EMBEDDING_MODEL_PATH、FAISS_INDEX_PATH 和 FAISS_METADATA_PATH"
        )

    manager = Neo4jManager()
    try:
        store = FaissEntityStore(model_path, index_path, metadata_path)
        count = store.build_from_neo4j(manager)
        print(f"FAISS 实体索引构建完成: {count} 个实体")
        print(f"向量索引: {index_path}")
        print(f"实体对照: {metadata_path}")
    finally:
        manager.close()


if __name__ == "__main__":
    main()
