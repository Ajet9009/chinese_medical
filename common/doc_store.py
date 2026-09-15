"""Separate FAISS index for document chunks (not entity matching).

Hybrid search: dense FAISS + in-process BM25 + RRF + MMR. No Milvus / cloud rerank.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from common.doc_chunking import build_document_chunks_step_04
from common.env_loader import load_app_env
from common.query_expansion import expand_zhongyi_query_step_01
from common.rag.mmr import mmr
from common.rag.rrf import rrf_fuse

load_app_env()
logger = logging.getLogger("doc_store")

EncodeFn = Callable[[list[str]], np.ndarray]
_ALLOWED_UPLOAD = {".md", ".txt", ".pdf"}


@lru_cache(maxsize=100000)
def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(w for w in jieba.cut(text or "") if w.strip())


def _chunk_key(hit: dict[str, Any]) -> str:
    return f"{hit.get('doc_id', '')}#{hit.get('chunk_idx', 0)}"


def _chunk_search_blob_step_01(chunk: dict[str, Any]) -> str:
    """步骤 01：合并子块正文、父块标题和繁简字段，供稀疏检索与规则重排使用。"""
    fields: list[str] = []
    for key in (
        "text",
        "text_simplified",
        "text_traditional",
        "title",
        "parent_title",
        "title_simplified",
        "title_traditional",
        "doc_name",
        "doc_type",
    ):
        value = chunk.get(key)
        if value:
            fields.append(str(value))
    for item in chunk.get("section_path") or []:
        if item:
            fields.append(str(item))

    out: list[str] = []
    seen: set[str] = set()
    for field in fields:
        clean = field.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return "\n".join(out)


def _query_terms_step_02(question: str) -> list[str]:
    """步骤 02：把原问句扩展为可用于短语和分词命中的检索词。"""
    terms: list[str] = []
    seen: set[str] = set()
    for variant in expand_zhongyi_query_step_01(question):
        for term in (variant, *_tokenize(variant)):
            clean = str(term or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                terms.append(clean)
    return terms


def rule_rerank_hits_step_03(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 03：按标题短语、正文短语和 token 覆盖给召回结果做轻量规则重排。"""
    terms = _query_terms_step_02(question)
    if not terms:
        return list(hits)

    ranked: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits):
        item = dict(hit)
        title_blob = "\n".join(
            str(item.get(key) or "")
            for key in ("title", "parent_title", "title_simplified", "title_traditional", "doc_name")
        )
        search_blob = _chunk_search_blob_step_01(item)
        title_hits = sum(1 for term in terms if len(term) >= 2 and term in title_blob)
        body_hits = sum(1 for term in terms if len(term) >= 2 and term in search_blob)
        base_score = float(item.get("score") or 0.0)
        rule_score = min(0.28, title_hits * 0.12 + body_hits * 0.03)
        item["base_score"] = base_score
        item["rule_score"] = round(rule_score, 6)
        item["rerank_score"] = round(base_score + rule_score, 6)
        item["score"] = min(1.0, item["rerank_score"])
        item["_rule_rank"] = rank
        ranked.append(item)

    ranked.sort(
        key=lambda x: (
            -float(x.get("rerank_score") or 0.0),
            -float(x.get("rule_score") or 0.0),
            int(x.get("_rule_rank") or 0),
        )
    )
    for item in ranked:
        item.pop("_rule_rank", None)
    return ranked


def _bge_encode(model_path: str) -> EncodeFn:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)

    def _encode(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype="float32",
        )

    return _encode


class FaissDocStore:
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        encode_fn: EncodeFn | None = None,
        model_path: str | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._encode_fn = encode_fn
        self.model_path = model_path or os.getenv("EMBEDDING_MODEL_PATH") or ""
        self._index = None
        self.chunks: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encode_fn is None:
            if not self.model_path:
                raise ValueError("未配置 EMBEDDING_MODEL_PATH，无法编码文档")
            self._encode_fn = _bge_encode(self.model_path)
        return self._encode_fn(texts)

    def build_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise ValueError("文档块不能为空")
        import faiss

        texts = [str(c.get("text") or "") for c in chunks]
        vectors = np.asarray(self._encode(texts), dtype="float32")
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self.chunks = list(chunks)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._rebuild_bm25()

    def load(self) -> "FaissDocStore":
        import faiss

        self._index = faiss.read_index(str(self.index_path))
        self.chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self._index.ntotal != len(self.chunks):
            raise ValueError("文档 FAISS 与元数据数量不一致")
        self._rebuild_bm25()
        return self

    def _rebuild_bm25(self) -> None:
        """步骤 01：用正文和结构化 metadata 重建 BM25 倒排语料。"""
        texts = [_chunk_search_blob_step_01(c) for c in self.chunks]
        self._tokenized = [list(_tokenize(t)) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search_bm25(self, question: str, top_k: int = 20) -> list[dict[str, Any]]:
        """步骤 02：用 query 扩展后的繁简变体执行 BM25 与短语兜底检索。"""
        q = (question or "").strip()
        if not q:
            return []
        if self._bm25 is None:
            if not self.chunks:
                if self.metadata_path.is_file():
                    self.load()
                else:
                    return []
            else:
                self._rebuild_bm25()
        if self._bm25 is None:
            return []
        variants = expand_zhongyi_query_step_01(q) or [q]
        score_sets = [
            np.asarray(self._bm25.get_scores(list(_tokenize(variant))), dtype="float64")
            for variant in variants
            if list(_tokenize(variant))
        ]
        scores = np.maximum.reduce(score_sets) if score_sets else np.zeros(len(self.chunks))
        if scores.size and float(scores.max()) <= 0:
            terms = set(_query_terms_step_02(q))
            scores = np.asarray(
                [
                    float(sum(1 for term in terms if len(term) >= 2 and term in _chunk_search_blob_step_01(chunk)))
                    for chunk in self.chunks
                ],
                dtype="float64",
            )
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[: max(1, top_k)]
        hits: list[dict[str, Any]] = []
        for idx, score in ranked:
            if float(score) <= 0:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            hits.append(item)
        return hits

    def mixed_search(self, question: str, top_k: int = 4, min_score: float = 0.0) -> list[dict[str, Any]]:
        """步骤 03：融合 dense/BM25 后执行规则重排和 MMR 多样性筛选。"""
        q = (question or "").strip()
        if not q:
            return []
        cand = max(top_k * 4, 20)
        dense = self.search(q, top_k=cand, min_score=0.0)
        sparse = self.search_bm25(q, top_k=cand)
        key_fn = _chunk_key
        rrf_k = int(os.getenv("RRF_K", "60"))
        dense_w = float(os.getenv("RRF_DENSE_WEIGHT", "1.0"))
        sparse_w = float(os.getenv("RRF_SPARSE_WEIGHT", "1.0"))
        fused = rrf_fuse([dense, sparse], key_fn=key_fn, k=rrf_k, weights=[dense_w, sparse_w])
        dense_scores = {key_fn(h): float(h.get("score") or 0.0) for h in dense}
        sparse_keys = {key_fn(h) for h in sparse}
        for hit in fused:
            hit["rrf_score"] = float(hit.get("score") or 0.0)
            key = key_fn(hit)
            if key in dense_scores:
                hit["score"] = dense_scores[key]
            elif key in sparse_keys:
                hit["score"] = 0.45
        fused = rule_rerank_hits_step_03(q, fused)
        mmr_on = os.getenv("MMR_ENABLE", "1").strip().lower() not in ("0", "false", "no")
        lam = float(os.getenv("MMR_LAMBDA", "0.5"))
        if mmr_on and len(fused) > top_k:
            fused = mmr(fused, top_k, lambda_=lam)
        else:
            fused = fused[:top_k]
        kept: list[dict[str, Any]] = []
        for hit in fused:
            key = key_fn(hit)
            if float(hit.get("score") or 0.0) >= min_score or key in sparse_keys:
                kept.append(hit)
        return kept[:top_k]

    def search(self, question: str, top_k: int = 4, min_score: float = 0.35) -> list[dict[str, Any]]:
        if not question or not question.strip():
            return []
        if self._index is None:
            if not self.index_path.is_file():
                return []
            self.load()
        if not self.chunks:
            return []
        import faiss  # noqa: F401

        vector = np.asarray(self._encode([question]), dtype="float32")
        k = min(max(1, top_k), len(self.chunks))
        scores, indices = self._index.search(vector, k)
        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < min_score:
                continue
            item = dict(self.chunks[int(idx)])
            item["score"] = float(score)
            hits.append(item)
        return hits


def default_doc_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent
    index = Path(os.getenv("DOC_FAISS_INDEX_PATH") or root / "data" / "faiss" / "docs.index")
    meta = Path(os.getenv("DOC_FAISS_METADATA_PATH") or root / "data" / "faiss" / "docs.json")
    return index, meta


def default_source_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    return Path(os.getenv("DOC_SOURCE_DIR") or root / "corpus")


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("未安装 pypdf，跳过 PDF: %s", path)
            return ""
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return ""


def ingest_directory(source_dir: Path | None = None, store: FaissDocStore | None = None) -> int:
    """步骤 01：读取文献目录并用结构化父子分块重建文献索引。"""
    folder = Path(source_dir) if source_dir else default_source_dir()
    if not folder.is_dir():
        raise FileNotFoundError(f"文献目录不存在: {folder}")
    files = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"} and p.is_file()
    )
    size = int(os.getenv("DOC_CHUNK_SIZE", "400"))
    overlap = int(os.getenv("DOC_CHUNK_OVERLAP", "80"))
    chunks: list[dict[str, Any]] = []
    for path in files:
        body = read_document(path)
        doc_id = path.stem
        doc_type = path.parent.name if path.parent != folder else ""
        chunks.extend(
            build_document_chunks_step_04(
                doc_id=doc_id,
                doc_name=path.name,
                body=body,
                doc_type=doc_type,
                size=size,
                overlap=overlap,
            )
        )
    if not chunks:
        raise ValueError("目录中没有可分块的文本")
    if store is None:
        index, meta = default_doc_paths()
        store = FaissDocStore(index, meta)
    store.build_chunks(chunks)
    return len(chunks)


_store: FaissDocStore | None = None


def get_doc_store() -> FaissDocStore | None:
    global _store
    if _store is not None:
        return _store
    index, meta = default_doc_paths()
    if not index.is_file() or not meta.is_file():
        return None
    try:
        _store = FaissDocStore(index, meta).load()
    except Exception as exc:
        from common.obs import degraded

        degraded("doc_index", exc)
        logger.warning("文档索引加载失败: %s", exc)
        return None
    return _store


def search_documents(
    question: str,
    viewer_dept: str = "",
    viewer_role: str = "user",
) -> list[dict[str, Any]]:
    store = get_doc_store()
    if store is None:
        return []
    top_k = int(os.getenv("DOC_TOP_K", "4"))
    min_score = float(os.getenv("DOC_MIN_SCORE", "0.35"))
    hybrid = os.getenv("DOC_HYBRID_ENABLE", "1").strip().lower() not in ("0", "false", "no")
    if hybrid:
        hits = store.mixed_search(question, top_k=max(top_k * 2, top_k), min_score=min_score)
    else:
        hits = store.search(question, top_k=top_k, min_score=min_score)
    try:
        from common.knowledge_acl import acl_ok
        from common.knowledge_service import get_knowledge_service

        blocked = get_knowledge_service().blocked_ids()
    except Exception as exc:
        from common.obs import degraded

        degraded("doc_acl", exc)
        blocked = set()
        acl_ok = None  # type: ignore
    kept: list[dict[str, Any]] = []
    for hit in hits:
        if hit.get("doc_id") in blocked:
            continue
        if acl_ok is not None and not acl_ok(
            hit.get("dept"), hit.get("allowed_roles"), viewer_dept, viewer_role
        ):
            continue
        kept.append(hit)
    return kept[:top_k]


def save_uploaded_file(
    filename: str,
    data: bytes,
    dest_dir: Path | None = None,
) -> Path:
    raw = (filename or "").replace("\\", "/")
    if (not raw) or ".." in raw or "/" in raw:
        raise ValueError("非法文件名")
    name = Path(raw).name
    suffix = Path(name).suffix.lower()
    if suffix not in _ALLOWED_UPLOAD:
        raise ValueError("仅支持 md/txt/pdf")
    max_bytes = int(os.getenv("DOC_UPLOAD_MAX_BYTES", str(8 * 1024 * 1024)))
    if len(data) > max_bytes:
        raise ValueError("文件过大")
    folder = Path(dest_dir) if dest_dir else default_source_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = (folder / name).resolve()
    if path.parent != folder.resolve():
        raise ValueError("非法文件名")
    path.write_bytes(data)
    return path


def reset_doc_store() -> None:
    global _store
    _store = None
