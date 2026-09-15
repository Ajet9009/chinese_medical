def test_bge_rerank_uses_injected_model_scores():
    from common.rag.reranker import rerank_hits_step_02

    class FakeReranker:
        def predict(self, pairs):
            assert pairs[0][0] == "麻黄汤主治"
            return [0.1, 0.9]

    hits = [
        {"doc_id": "low", "text": "桂枝汤解肌发表。", "score": 0.8},
        {"doc_id": "high", "text": "麻黄汤发汗解表。", "score": 0.7},
    ]

    ranked = rerank_hits_step_02("麻黄汤主治", hits, model=FakeReranker(), weight=0.3)

    assert ranked[0]["doc_id"] == "high"
    assert ranked[0]["cross_score"] == 0.9
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]


def test_bge_reranker_stays_disabled_without_model_path(monkeypatch):
    from common.rag.reranker import load_bge_reranker_step_01

    monkeypatch.delenv("DOC_RERANK_MODEL_PATH", raising=False)

    assert load_bge_reranker_step_01() is None
