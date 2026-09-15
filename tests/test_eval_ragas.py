"""RAGAS 嵌入评测集生成与文献 RAG 离线打分。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from common.doc_rag import REFUSE_ANSWER
from common.doc_store import FaissDocStore
from common.eval_ragas import (
    RAGAS_DATASET_COLUMNS,
    generate_testset_step_13,
    hash_encode_step_04,
    list_metrics_step_01,
    load_testset_step_15,
    pick_reference_contexts_step_11,
    retrieve_for_eval_step_16,
    run_rag_eval_step_20,
    save_ragas_dataset_step_27,
    save_testset_step_14,
    score_sample_step_18,
    to_ragas_dataset_step_26,
)
from tests.fakes import FakeRedis
from tests.test_auth import _login


def test_metrics_are_english_with_chinese():
    metrics = list_metrics_step_01()
    ids = {m["id"] for m in metrics}
    assert ids == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "answer_semantic_similarity",
    }
    for item in metrics:
        name = item["name"]
        assert "（" in name and name.endswith("）"), name
        latin = name.split("（", 1)[0]
        assert any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in latin), name
        zh = name.split("（", 1)[1].rstrip("）")
        assert zh, name


def test_generate_testset_uses_embeddings_and_golden(tmp_path):
    chunks = [
        {
            "doc_id": "sijunzi",
            "doc_name": "sijunzi.md",
            "chunk_idx": 0,
            "text": "四君子汤出自太平惠民和剂局方。组成为人参、白术、茯苓、甘草。功效益气健脾。",
            "title": "四君子汤",
        },
        {
            "doc_id": "weather",
            "doc_name": "weather.md",
            "chunk_idx": 0,
            "text": "今日晴转多云，气温适宜出行，与方剂无关。",
            "title": "天气",
        },
    ]
    golden = [
        {
            "query": "四君子汤有什么功效？",
            "expect": ["益气", "健脾"],
            "category": "方剂",
            "source": "seed",
            "relevant_docs": ["sijunzi.md"],
        },
        {
            "query": "今天上证指数多少？",
            "expect": ["未查到"],
            "category": "拒答",
            "source": "seed",
        },
        {
            "query": "待标注问句",
            "expect": [],
            "category": "用户反馈",
            "source": "feedback",
        },
    ]
    payload = generate_testset_step_13(
        golden_items=golden,
        chunks=chunks,
        encode_fn=hash_encode_step_04,
        extra_per_doc=1,
    )
    queries = [it["user_input"] for it in payload["items"]]
    assert "四君子汤有什么功效？" in queries
    assert "今天上证指数多少？" in queries
    assert "待标注问句" not in queries
    sijunzi = next(
        it
        for it in payload["items"]
        if "四君子" in it["user_input"] and it["source"] != "embedding-synth"
    )
    assert sijunzi["reference_contexts"]
    assert any("益气健脾" in t or "四君子汤" in t for t in sijunzi["reference_contexts"])
    refuse = next(it for it in payload["items"] if it["category"] == "拒答")
    assert refuse["reference_contexts"] == []
    assert any(it.get("source") == "embedding-synth" for it in payload["items"])
    path = tmp_path / "ragas_testset.json"
    save_testset_step_14(payload, path)
    loaded = load_testset_step_15(path)
    assert len(loaded["items"]) == len(payload["items"])


def test_rag_eval_retrieval_scores(tmp_path):
    chunks = [
        {
            "doc_id": "sijunzi",
            "doc_name": "sijunzi.md",
            "chunk_idx": 0,
            "text": "四君子汤由人参、白术、茯苓、甘草组成，功效益气健脾。",
            "title": "四君子汤",
        },
        {
            "doc_id": "noise",
            "doc_name": "noise.md",
            "chunk_idx": 0,
            "text": "今日股市波动与中医文献无关。",
            "title": "杂讯",
        },
    ]
    golden = [
        {
            "query": "四君子汤有什么功效？",
            "expect": ["益气", "健脾"],
            "category": "方剂",
            "relevant_docs": ["sijunzi.md"],
        },
        {
            "query": "今天上证指数多少？",
            "expect": ["未查到"],
            "category": "拒答",
        },
    ]
    payload = generate_testset_step_13(
        golden_items=golden,
        chunks=chunks,
        encode_fn=hash_encode_step_04,
        extra_per_doc=0,
    )
    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=hash_encode_step_04,
    )
    store.build_chunks(chunks)
    report = run_rag_eval_step_20(
        payload,
        encode_fn=hash_encode_step_04,
        store=store,
        use_crag=False,
        top_k=2,
    )
    names = [m["name"] for m in report["metrics"]]
    assert "Faithfulness（忠实度）" in names
    assert "Context Recall（上下文召回度）" in names
    hit = next(r for r in report["items"] if "四君子" in r["user_input"])
    assert hit["retrieved_n"] >= 1
    assert (hit["scores"]["context_recall"] or 0) >= 0.5
    assert (hit["scores"]["faithfulness"] or 0) > 0.3
    assert "retrieved_contexts" in hit
    assert "response" in hit
    samples = to_ragas_dataset_step_26(report)
    assert samples
    assert set(samples[0]) == set(RAGAS_DATASET_COLUMNS)
    ds = tmp_path / "ragas_dataset.json"
    save_ragas_dataset_step_27(samples, ds)
    loaded = json.loads(ds.read_text(encoding="utf-8"))
    assert isinstance(loaded, list)
    assert set(loaded[0]) == set(RAGAS_DATASET_COLUMNS)
    means = {m["id"]: m["score"] for m in report["metrics"]}
    assert means["faithfulness"] is not None
    assert means["context_precision"] is not None


def test_score_refuse_without_contexts():
    item = {
        "user_input": "今天上证指数多少？",
        "reference": "未查到",
        "reference_contexts": [],
        "expect": ["未查到"],
        "category": "拒答",
    }
    scores = score_sample_step_18(item, [], REFUSE_ANSWER, hash_encode_step_04)
    assert scores["faithfulness"] == 1.0
    assert scores["context_precision"] == 1.0
    assert scores["context_recall"] == 1.0
    assert scores["answer_correctness"] == 1.0


def test_hash_encode_normalized():
    mat = hash_encode_step_04(["四君子汤", "四君子汤"])
    assert mat.shape[1] == 64
    assert np.allclose(np.linalg.norm(mat[0]), 1.0, atol=1e-5)
    assert np.allclose(mat[0], mat[1])


def test_pick_reference_prefers_expect_keywords():
    chunks = [
        {
            "doc_id": "weather",
            "doc_name": "weather.md",
            "chunk_idx": 0,
            "text": "今日晴转多云，气温适宜出行。",
        },
        {
            "doc_id": "huangqi",
            "doc_name": "herb.md",
            "chunk_idx": 0,
            "text": "黄芪补气升阳，固表止汗。",
        },
    ]
    vecs = hash_encode_step_04([c["text"] for c in chunks])
    refs = pick_reference_contexts_step_11(
        "黄芪有什么功效？",
        chunks,
        vecs,
        hash_encode_step_04,
        None,
        top_k=1,
        expect=["黄芪"],
    )
    assert refs
    assert "黄芪" in refs[0]


def test_pick_reference_missing_docs_returns_empty():
    chunks = [
        {
            "doc_id": "sijunzi",
            "doc_name": "sijunzi.md",
            "chunk_idx": 0,
            "text": "四君子汤益气健脾。",
        }
    ]
    vecs = hash_encode_step_04([c["text"] for c in chunks])
    refs = pick_reference_contexts_step_11(
        "四君子汤有什么功效？",
        chunks,
        vecs,
        hash_encode_step_04,
        ["missing.md"],
        top_k=1,
    )
    assert refs == []


def test_load_app_env_keeps_process_env(monkeypatch, tmp_path):
    import os

    golden = str(tmp_path / "golden.json")
    testset = str(tmp_path / "ragas_testset.json")
    monkeypatch.setenv("GOLDEN_QA_PATH", golden)
    monkeypatch.setenv("RAGAS_TESTSET_PATH", testset)
    from common.env_loader import load_app_env
    from common.eval_golden import golden_path
    from common.eval_ragas import testset_path_step_02

    load_app_env()
    assert os.environ["GOLDEN_QA_PATH"] == golden
    assert os.environ["RAGAS_TESTSET_PATH"] == testset
    assert golden_path() == tmp_path / "golden.json"
    assert testset_path_step_02() == tmp_path / "ragas_testset.json"


def test_crag_clears_incorrect_hits(tmp_path, monkeypatch):
    monkeypatch.setenv("CRAG_HIGH", "1.5")
    monkeypatch.setenv("CRAG_LOW", "1.1")
    chunks = [
        {
            "doc_id": "sijunzi",
            "doc_name": "sijunzi.md",
            "chunk_idx": 0,
            "text": "四君子汤由人参、白术、茯苓、甘草组成，功效益气健脾。",
            "title": "四君子汤",
        }
    ]
    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=hash_encode_step_04,
    )
    store.build_chunks(chunks)
    hits = retrieve_for_eval_step_16(store, "四君子汤有什么功效？", use_crag=True)
    assert hits == []
    kept = retrieve_for_eval_step_16(store, "四君子汤有什么功效？", use_crag=False)
    assert kept


def test_repo_corpus_generate_smoke():
    from common.eval_ragas import load_corpus_chunks_step_10

    chunks = load_corpus_chunks_step_10()
    assert chunks, "corpus/ 评测摘录应纳入版本库"
    names = {str(c.get("doc_name") or "") for c in chunks}
    for required in (
        "fangji_tangtou.md",
        "bencao_shennong.md",
        "dianji_shanghanlun.md",
        "yian_linzheng.md",
        "qita_piweilun.md",
    ):
        assert required in names, required
    payload = generate_testset_step_13(
        chunks=chunks,
        encode_fn=hash_encode_step_04,
        extra_per_doc=0,
    )
    assert payload["items"]
    assert any(it.get("reference_contexts") for it in payload["items"])
    assert any(m["name"] == "Faithfulness（忠实度）" for m in payload["metrics"])
    cats = {str(it.get("category") or "") for it in payload["items"]}
    assert {"方剂", "本草", "证候", "典籍", "医案", "其他", "拒答"} <= cats


def test_repo_ragas_testset_schema():
    from common.eval_ragas import load_testset_step_15, testset_path_step_02

    path = testset_path_step_02()
    assert path.is_file(), "eval/ragas_testset.json 应纳入版本库"
    data = load_testset_step_15(path)
    assert data["items"]
    assert all("（" in m["name"] and m["name"].endswith("）") for m in data["metrics"])
    assert any(it.get("user_input") for it in data["items"])


def test_repo_ragas_dataset_official_file():
    from common.eval_ragas import dataset_path_step_24, load_ragas_dataset_step_28

    path = dataset_path_step_24()
    assert path.is_file(), "eval/ragas_dataset.json 应纳入版本库（RAGAS EvaluationDataset 五列）"
    samples = load_ragas_dataset_step_28(path)
    assert samples
    for sample in samples:
        assert set(sample.keys()) == set(RAGAS_DATASET_COLUMNS)


def test_ragas_dataset_official_columns():
    payload = generate_testset_step_13(
        golden_items=[
            {
                "query": "四君子汤有什么功效？",
                "expect": ["益气"],
                "category": "方剂",
                "relevant_docs": ["sijunzi.md"],
            }
        ],
        chunks=[
            {
                "doc_id": "sijunzi",
                "doc_name": "sijunzi.md",
                "chunk_idx": 0,
                "text": "四君子汤益气健脾。",
                "title": "四君子汤",
            }
        ],
        encode_fn=hash_encode_step_04,
        extra_per_doc=0,
    )
    samples = to_ragas_dataset_step_26(payload)
    assert samples
    for sample in samples:
        assert list(sample.keys()) == list(RAGAS_DATASET_COLUMNS)
        assert isinstance(sample["retrieved_contexts"], list)
        assert isinstance(sample["reference_contexts"], list)
        assert isinstance(sample["user_input"], str)
        assert isinstance(sample["reference"], str)
        assert isinstance(sample["response"], str)


def test_repo_golden_expect_is_corpus_substring():
    from common.eval_golden import load_golden
    from common.eval_ragas import load_corpus_chunks_step_10

    chunks = load_corpus_chunks_step_10()
    by_doc: dict[str, str] = {}
    for row in chunks:
        name = str(row.get("doc_name") or "")
        by_doc[name] = by_doc.get(name, "") + str(row.get("text") or "")
    for item in load_golden():
        if str(item.get("category") or "") == "拒答":
            continue
        docs = [str(x) for x in (item.get("relevant_docs") or [])]
        blob = "".join(by_doc.get(name, "") for name in docs)
        assert blob, item.get("query")
        for needle in item.get("expect") or []:
            assert str(needle) in blob, f"{item.get('query')} expect={needle!r}"


def test_repo_ragas_dataset_non_refuse_filled():
    from common.eval_ragas import load_ragas_dataset_step_28

    samples = load_ragas_dataset_step_28()
    assert samples
    non_refuse = [s for s in samples if s.get("response") != REFUSE_ANSWER]
    refuse = [s for s in samples if s.get("response") == REFUSE_ANSWER]
    assert non_refuse
    assert refuse
    for sample in non_refuse:
        assert sample["retrieved_contexts"], sample["user_input"]
        assert str(sample["response"] or "").strip(), sample["user_input"]
        assert sample["reference_contexts"], sample["user_input"]
        assert str(sample["reference"] or "").strip(), sample["user_input"]
    for sample in refuse:
        assert sample["reference_contexts"] == []
        assert "上证" in sample["user_input"]


def test_official_evaluate_skips_without_key(monkeypatch):
    from common.eval_ragas import ragas_official_prereq_step_31, run_official_evaluate_step_35

    monkeypatch.setenv("MODEL_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    reason = ragas_official_prereq_step_31()
    if "未安装 ragas" in (reason or ""):
        pytest.skip(reason)
    assert reason
    with pytest.raises(RuntimeError, match="未配置|未安装"):
        run_official_evaluate_step_35(
            [
                {
                    "user_input": "人參味如何？",
                    "retrieved_contexts": ["人參 味甘小寒。主補五臟"],
                    "response": "人參 味甘小寒。主補五臟",
                    "reference": "人參 味甘小寒。主補五臟",
                    "reference_contexts": ["人參 味甘小寒。主補五臟"],
                }
            ],
            encode_fn=hash_encode_step_04,
        )


def test_official_evaluate_wires_from_list(monkeypatch):
    from common.eval_ragas import (
        collect_official_metrics_step_33,
        ragas_official_prereq_step_31,
        run_official_evaluate_step_35,
    )

    monkeypatch.setenv("MODEL_API_KEY", "sk-test-not-used")
    reason = ragas_official_prereq_step_31()
    if reason and "未安装 ragas" in reason:
        pytest.skip(reason)
    metrics = collect_official_metrics_step_33()
    assert metrics
    names = {getattr(m, "name", type(m).__name__) for m in metrics}
    assert "faithfulness" in names

    class _FakeResult:
        def to_pandas(self):
            import pandas as pd

            return pd.DataFrame(
                [
                    {
                        "faithfulness": 1.0,
                        "answer_relevancy": 0.5,
                        "context_precision": 1.0,
                        "context_recall": 1.0,
                        "factual_correctness": 0.9,
                        "semantic_similarity": 0.8,
                    }
                ]
            )

    def _fake_evaluate(*_args, **_kwargs):
        assert _kwargs.get("metrics") or _args
        return _FakeResult()

    samples = [
        {
            "user_input": "人參味如何？",
            "retrieved_contexts": ["人參 味甘小寒。主補五臟"],
            "response": "人參 味甘小寒。主補五臟",
            "reference": "人參 味甘小寒。主補五臟",
            "reference_contexts": ["人參 味甘小寒。主補五臟"],
        }
    ]
    report = run_official_evaluate_step_35(
        samples,
        encode_fn=hash_encode_step_04,
        llm=object(),
        embeddings=object(),
        evaluate_fn=_fake_evaluate,
    )
    assert report["answer_mode"] == "official"
    by_id = {m["id"]: m["score"] for m in report["metrics"]}
    assert by_id["faithfulness"] == 1.0
    assert by_id["answer_semantic_similarity"] == 0.8


@pytest.fixture
def ragas_client(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(tmp_path / "c.sqlite"))
    monkeypatch.setenv("GOLDEN_QA_PATH", str(tmp_path / "golden.json"))
    monkeypatch.setenv("RAGAS_TESTSET_PATH", str(tmp_path / "ragas_testset.json"))
    monkeypatch.setenv("RAGAS_DATASET_PATH", str(tmp_path / "ragas_dataset.json"))
    monkeypatch.setenv("RAGAS_REPORT_PATH", str(tmp_path / "ragas_report.json"))
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    monkeypatch.setenv("JWT_SECRET", "test-secret-w4-please-use-32bytes!!")
    monkeypatch.setenv("ADMIN_USERNAME", "admin")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin123")

    import common.redis_client as redis_mod

    redis_mod.reset_redis_client()
    from common.obs import reset_obs

    reset_obs()

    import _005_fastapi.deps as deps
    import _005_fastapi.main as main

    orig_get_store = main.get_store
    orig_get_store.cache_clear()
    deps.get_users.cache_clear()

    from common.conversation_store import ConversationStore
    from common.knowledge_service import reset_knowledge_service
    from common.user_store import UserStore

    db = tmp_path / "c.sqlite"
    store = ConversationStore(db, redis_client=FakeRedis())
    users = UserStore(db)
    users.seed_admin()
    monkeypatch.setattr(main, "get_store", lambda: store)
    monkeypatch.setattr(deps, "get_users", lambda: users)
    monkeypatch.setattr(main, "get_users", lambda: users)
    reset_knowledge_service()

    with TestClient(main.app) as c:
        yield c, tmp_path

    orig_get_store.cache_clear()
    cached = getattr(deps.get_users, "cache_clear", None)
    if cached:
        cached()


def test_admin_ragas_requires_admin(ragas_client):
    c, tmp_path = ragas_client
    h = _login(c)
    body = c.get("/admin/ragas", headers=h).json()
    assert body["metrics"]
    assert all("（" in m["name"] for m in body["metrics"])
    assert Path(body["testset"]["path"]) == tmp_path / "ragas_testset.json"
    assert body["dataset"]["columns"] == list(RAGAS_DATASET_COLUMNS)
    assert body["report"] is None

    c.post(
        "/admin/users",
        json={"username": "herb", "password": "herb1234", "role": "user"},
        headers=h,
    )
    herb = _login(c, "herb", "herb1234")
    assert c.get("/admin/ragas", headers=herb).status_code == 403
    assert c.get("/admin/ragas").status_code in (401, 403)
