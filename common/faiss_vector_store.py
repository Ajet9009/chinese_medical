"""FAISS index for matching user questions to Neo4j standard entities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def entity_text(entity: dict[str, Any]) -> str:
    """Build the text embedded for an entity; names are the primary signal."""
    name = str(entity.get("name") or "").strip()
    entity_type = str(entity.get("type") or "").strip()
    return f"{entity_type}: {name}" if entity_type else name


class FaissEntityStore:
    """Persistent cosine-similarity search over Neo4j entities."""

    def __init__(
        self,
        model_path: str,
        index_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.index_path.with_suffix(".json")
        self._model = None
        self._index = None
        self.entities: list[dict[str, Any]] = []

    def _load_dependencies(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("请安装 faiss-cpu 和 sentence-transformers") from exc
        return faiss, SentenceTransformer

    def _get_model(self):
        if self._model is None:
            _, sentence_transformer = self._load_dependencies()
            self._model = sentence_transformer(self.model_path)
        return self._model

    def build(self, entities: list[dict[str, Any]]) -> None:
        if not entities:
            raise ValueError("实体列表不能为空")
        faiss, _ = self._load_dependencies()
        vectors = self._get_model().encode(
            [entity_text(entity) for entity in entities],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype="float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self.entities = list(entities)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.entities, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> "FaissEntityStore":
        faiss, _ = self._load_dependencies()
        self._index = faiss.read_index(str(self.index_path))
        self.entities = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self._index.ntotal != len(self.entities):
            raise ValueError("FAISS 索引与实体元数据数量不一致")
        return self

    def search(self, question: str, top_k: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
        if not question or not question.strip():
            return []
        if self._index is None:
            self.load()
        vector = self._get_model().encode([question], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self._index.search(np.asarray(vector, dtype="float32"), min(top_k, len(self.entities)))
        return [
            {**self.entities[i], "score": float(score)}
            for score, i in zip(scores[0], indices[0])
            if i >= 0 and float(score) >= min_score
        ]

    def build_from_neo4j(self, manager) -> int:
        entities = manager.get_all_entities()
        self.build(entities)
        return len(entities)
