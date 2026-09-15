import pytest


def test_doc_vector_backend_defaults_to_faiss(monkeypatch, tmp_path):
    from common.doc_store import FaissDocStore
    from common.doc_vector_backend import create_doc_store_step_02

    monkeypatch.delenv("DOC_VECTOR_BACKEND", raising=False)

    store = create_doc_store_step_02(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=lambda texts: [],
    )

    assert isinstance(store, FaissDocStore)


def test_doc_vector_backend_rejects_unimplemented_milvus(monkeypatch, tmp_path):
    from common.doc_vector_backend import create_doc_store_step_02

    monkeypatch.setenv("DOC_VECTOR_BACKEND", "milvus")

    with pytest.raises(NotImplementedError, match="Milvus"):
        create_doc_store_step_02(
            index_path=tmp_path / "docs.index",
            metadata_path=tmp_path / "docs.json",
        )
