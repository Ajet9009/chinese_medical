"""文献向量库后端工厂。

当前默认仍为 FAISS；Milvus 迁移先通过显式后端名阻断误用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

EncodeFn = Callable[[list[str]], np.ndarray]


def doc_vector_backend_name_step_01(raw: str | None = None) -> str:
    """步骤 01：读取并校验文献向量库后端名。"""
    name = (raw or os.getenv("DOC_VECTOR_BACKEND") or "faiss").strip().lower()
    if name in {"", "local"}:
        return "faiss"
    if name in {"faiss", "milvus"}:
        return name
    raise ValueError("DOC_VECTOR_BACKEND 仅支持 faiss / milvus")


def create_doc_store_step_02(
    index_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    encode_fn: EncodeFn | None = None,
    model_path: str | None = None,
) -> Any:
    """步骤 02：按后端名创建文献向量库实例。"""
    backend = doc_vector_backend_name_step_01()
    if backend == "milvus":
        raise NotImplementedError("Milvus 文献后端尚未接入；当前请继续使用 FAISS，迁移前需补集合 schema 与回填脚本。")

    from common.doc_store import FaissDocStore, default_doc_paths

    default_index, default_meta = default_doc_paths()
    return FaissDocStore(
        index_path=index_path or default_index,
        metadata_path=metadata_path or default_meta,
        encode_fn=encode_fn,
        model_path=model_path,
    )
