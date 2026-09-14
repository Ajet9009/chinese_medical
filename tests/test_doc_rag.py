"""W3 document RAG: chunking, mix with KGQA, refuse when both empty."""

from __future__ import annotations

import numpy as np

from tests.fakes import FakeLLM


def _encode(texts: list[str]) -> np.ndarray:
    dim = 64
    mat = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        for ch in text:
            mat[i, ord(ch) % dim] += 1.0
        norm = np.linalg.norm(mat[i])
        if norm:
            mat[i] /= norm
    return mat


def test_chunk_text_overlaps_and_keeps_order():
    from common.doc_rag import chunk_text

    text = "甲" * 30 + "乙" * 30
    chunks = chunk_text(text, size=20, overlap=5)
    assert len(chunks) >= 3
    assert chunks[0].startswith("甲")
    reconstructed = chunks[0]
    for part in chunks[1:]:
        reconstructed += part[5:]
    assert "甲" * 30 in reconstructed
    assert "乙" * 30 in reconstructed


def test_kg_context_empty():
    from common.doc_rag import kg_context_empty

    assert kg_context_empty("") is True
    assert kg_context_empty("   ") is True
    assert kg_context_empty("(无图谱数据)") is True
    assert kg_context_empty("四君子汤 | HAS_EFFECT | 补气健脾") is False


def test_should_refuse_only_when_both_empty():
    from common.doc_rag import should_refuse

    docs = [{"doc_name": "a.md", "text": "四君子汤益气健脾", "score": 0.8, "chunk_idx": 0}]
    weak = [{"doc_name": "a.md", "text": "天气转阴", "score": 0.1, "chunk_idx": 0}]
    assert should_refuse("", []) is True
    assert should_refuse("(无图谱数据)", []) is True
    assert should_refuse("四君子汤 HAS_EFFECT 补气健脾", []) is False
    assert should_refuse("", docs) is False
    assert should_refuse("四君子汤 HAS_EFFECT 补气健脾", weak) is False
    assert should_refuse("", weak) is True
    assert should_refuse("", weak, crag_grade="incorrect", crag_action="refused") is True
    assert should_refuse("四君子汤 HAS_EFFECT 补气健脾", weak, crag_grade="incorrect") is False


def test_format_doc_context_cites_source():
    from common.doc_rag import format_doc_context

    text = format_doc_context(
        [
            {
                "doc_name": "四君子汤.md",
                "chunk_idx": 1,
                "text": "出自《太平惠民和剂局方》。",
                "score": 0.91,
            }
        ]
    )
    assert "四君子汤.md" in text
    assert "#1" in text
    assert "太平惠民和剂局方" in text


def test_answer_refuses_without_calling_llm():
    from _004_langgraph_more_nodes.answer_generation import make_answer_generation_node
    from common.doc_rag import REFUSE_ANSWER

    llm = FakeLLM("不该生成这段")
    node = make_answer_generation_node(llm)
    out = node.invoke(
        {
            "user_question": "某某不存在的古方有何功效？",
            "neo4j_answer": "",
            "doc_chunks": [],
            "doc_context": "",
        }
    )
    assert out["final_answer"] == REFUSE_ANSWER
    assert out["refused"] is True
    assert llm.calls == 0


def test_answer_prompt_mixes_kg_and_docs():
    from _004_langgraph_more_nodes.answer_generation import make_answer_generation_node

    captured = []

    class Cap:
        def invoke(self, messages):
            captured.append(messages)
            return type("R", (), {"content": "益气健脾"})()

    node = make_answer_generation_node(Cap())
    out = node.invoke(
        {
            "user_question": "四君子汤有什么功效？",
            "neo4j_answer": "四君子汤 | HAS_EFFECT | 补气健脾",
            "doc_context": "[文献: 四君子汤.md#0] 四君子汤益气健脾。",
            "doc_chunks": [
                {
                    "doc_name": "四君子汤.md",
                    "chunk_idx": 0,
                    "text": "四君子汤益气健脾。",
                    "score": 0.9,
                }
            ],
        }
    )
    assert out["refused"] is False
    human = str(captured[0][-1].content)
    assert "补气健脾" in human
    assert "四君子汤.md" in human


def test_doc_retrieval_node_uses_search_question():
    from _004_langgraph_more_nodes.doc_retrieval import make_doc_retrieval_node

    seen = []

    def search(q: str):
        seen.append(q)
        return [
            {
                "doc_name": "四君子汤.md",
                "chunk_idx": 0,
                "text": "人参、白术、茯苓、甘草。",
                "score": 0.88,
            }
        ]

    node = make_doc_retrieval_node(search_fn=search)
    out = node(
        {
            "user_question": "它由哪些药组成？",
            "search_question": "四君子汤由哪些药组成？",
            "is_zhongyi_intent": True,
        }
    )
    assert seen == ["四君子汤由哪些药组成？"]
    assert out["doc_chunks"][0]["doc_name"] == "四君子汤.md"
    assert "四君子汤.md" in out["doc_context"]
    assert out["crag_grade"] == "correct"
    assert out["crag_action"] == "normal"
    assert out["crag_confidence"] == "high"


def test_faiss_doc_store_roundtrip(tmp_path):
    from common.doc_store import FaissDocStore

    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=_encode,
    )
    store.build_chunks(
        [
            {
                "doc_id": "sijunzi",
                "doc_name": "四君子汤.md",
                "chunk_idx": 0,
                "text": "四君子汤由人参、白术、茯苓、甘草组成，功效益气健脾。",
            },
            {
                "doc_id": "other",
                "doc_name": "天气.md",
                "chunk_idx": 0,
                "text": "今日晴转多云，气温适宜出行。",
            },
        ]
    )
    hits = store.search("四君子汤由哪些药组成", top_k=2, min_score=0.01)
    assert hits
    assert hits[0]["doc_name"] == "四君子汤.md"


def test_rrf_fuse_shared_hit_ranks_first():
    from common.rag.rrf import rrf_fuse

    dense = [{"key": 1}, {"key": 2}, {"key": 3}]
    sparse = [{"key": 2}, {"key": 4}]
    fused = rrf_fuse([dense, sparse], key_fn=lambda h: h["key"])
    assert [h["key"] for h in fused][0] == 2
    scores = [h["score"] for h in fused]
    assert scores == sorted(scores, reverse=True)


def test_mmr_keeps_diverse_chunks():
    from common.rag.mmr import mmr

    candidates = [
        {"text": "四君子汤由人参、白术、茯苓、甘草组成。", "score": 0.9},
        {"text": "四君子汤由人参白术茯苓甘草组成。", "score": 0.88},
        {"text": "桂枝汤解肌发表，调和营卫。", "score": 0.7},
    ]
    picked = mmr(candidates, topk=2, lambda_=0.5)
    texts = [c["text"] for c in picked]
    assert any("四君子" in t for t in texts)
    assert any("桂枝" in t for t in texts)


def test_bm25_ranks_term_match(tmp_path):
    from common.doc_store import FaissDocStore

    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=_encode,
    )
    store.build_chunks(
        [
            {
                "doc_id": "gui",
                "doc_name": "桂枝汤.md",
                "chunk_idx": 0,
                "text": "桂枝汤解肌发表调和营卫。",
            },
            {
                "doc_id": "sijunzi",
                "doc_name": "四君子汤.md",
                "chunk_idx": 0,
                "text": "四君子汤人参白术茯苓甘草益气健脾。",
            },
        ]
    )
    hits = store.search_bm25("四君子汤人参白术", top_k=2)
    assert hits
    assert hits[0]["doc_name"] == "四君子汤.md"


def test_mixed_search_fuses_dense_and_bm25(tmp_path):
    from common.doc_store import FaissDocStore

    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=_encode,
    )
    store.build_chunks(
        [
            {
                "doc_id": "sijunzi",
                "doc_name": "四君子汤.md",
                "chunk_idx": 0,
                "text": "四君子汤由人参、白术、茯苓、甘草组成，功效益气健脾。",
            },
            {
                "doc_id": "other",
                "doc_name": "天气.md",
                "chunk_idx": 0,
                "text": "今日晴转多云，气温适宜出行。",
            },
        ]
    )
    hits = store.mixed_search("四君子汤由哪些药组成", top_k=2, min_score=0.0)
    assert hits
    assert hits[0]["doc_name"] == "四君子汤.md"
    assert "rrf_score" in hits[0]


def test_crag_grade_buckets():
    from common.rag import crag

    assert crag.grade(0.85, 5, high=0.72, low=0.3)[0] == crag.GRADE_CORRECT
    assert crag.grade(0.1, 5, high=0.72, low=0.3)[0] == crag.GRADE_INCORRECT
    assert crag.grade(0.0, 0, high=0.72, low=0.3)[0] == crag.GRADE_INCORRECT
    assert crag.grade(0.45, 5, high=0.72, low=0.3)[0] == crag.GRADE_AMBIGUOUS
    assert crag.grade(0.95, 5, rerank_ok=False)[0] == crag.GRADE_AMBIGUOUS
    assert crag.confidence_of(crag.GRADE_CORRECT, False) == "high"
    assert crag.confidence_of(crag.GRADE_INCORRECT, True) == "refused"


def test_crag_rewrites_then_recovers():
    from common.rag.pipeline import retrieve_with_crag

    seen = []

    def search(q: str):
        seen.append(q)
        if "四君子汤" in q:
            return [
                {
                    "doc_id": "a",
                    "doc_name": "a.md",
                    "chunk_idx": 0,
                    "text": "四君子汤益气健脾。",
                    "score": 0.9,
                }
            ]
        return [
            {
                "doc_id": "x",
                "doc_name": "x.md",
                "chunk_idx": 0,
                "text": "无关天气",
                "score": 0.05,
            }
        ]

    out = retrieve_with_crag("它有什么用", search, rewrite_fn=lambda q: "四君子汤功效")
    assert seen == ["它有什么用", "四君子汤功效"]
    assert out["crag_grade"] == "correct"
    assert out["crag_action"] == "rewritten"
    assert out["hits"][0]["doc_name"] == "a.md"


def test_crag_rewrite_still_incorrect_refuses_docs():
    from common.rag.pipeline import retrieve_with_crag

    def search(q: str):
        return [
            {
                "doc_id": "w",
                "doc_name": "天气.md",
                "chunk_idx": 0,
                "text": "今日多云",
                "score": 0.05,
            }
        ]

    out = retrieve_with_crag("火星有什么功效", search, rewrite_fn=lambda q: q + " 改写")
    assert out["crag_action"] == "refused"
    assert out["crag_confidence"] == "refused"
    assert out["hits"] == []


def test_citation_overlap_ok_and_fail():
    from common.rag.cite import verify_citations

    chunks = [{"doc_name": "四君子汤.md", "chunk_idx": 0, "text": "四君子汤益气健脾，出自局方。"}]
    ok = verify_citations("四君子汤益气健脾。", chunks)
    assert ok["ok"] is True
    assert ok["rewrite_needed"] is False
    bad = verify_citations("水星绕太阳公转，与中药无关。", chunks)
    assert bad["ok"] is False
    assert bad["rewrite_needed"] is True


def test_answer_citation_fail_refuses_without_kg():
    from _004_langgraph_more_nodes.answer_generation import make_answer_generation_node
    from common.doc_rag import REFUSE_ANSWER

    class Cap:
        def invoke(self, messages):
            return type("R", (), {"content": "水星绕太阳公转，与中药无关。"})()

    node = make_answer_generation_node(Cap())
    out = node.invoke(
        {
            "user_question": "四君子汤有什么功效？",
            "neo4j_answer": "",
            "doc_context": "[文献: 四君子汤.md#0] 四君子汤益气健脾。",
            "doc_chunks": [
                {
                    "doc_name": "四君子汤.md",
                    "chunk_idx": 0,
                    "text": "四君子汤益气健脾。",
                    "score": 0.9,
                }
            ],
            "crag_grade": "correct",
        }
    )
    assert out["refused"] is True
    assert out["final_answer"] == REFUSE_ANSWER
    assert out["citation_ok"] is False


def test_answer_citation_fail_keeps_kg_answer():
    from _004_langgraph_more_nodes.answer_generation import make_answer_generation_node

    class Cap:
        def invoke(self, messages):
            return type("R", (), {"content": "水星绕太阳公转。"})()

    node = make_answer_generation_node(Cap())
    out = node.invoke(
        {
            "user_question": "四君子汤有什么功效？",
            "neo4j_answer": "四君子汤 | HAS_EFFECT | 补气健脾",
            "doc_context": "[文献: 四君子汤.md#0] 四君子汤益气健脾。",
            "doc_chunks": [
                {
                    "doc_name": "四君子汤.md",
                    "chunk_idx": 0,
                    "text": "四君子汤益气健脾。",
                    "score": 0.9,
                }
            ],
            "crag_grade": "incorrect",
        }
    )
    assert out["refused"] is False
    assert "水星" in out["final_answer"]
    assert out["citation_ok"] is False


def test_answer_ambiguous_prefixes_uncertainty():
    from _004_langgraph_more_nodes.answer_generation import make_answer_generation_node
    from common.doc_rag import UNCERTAIN_PREFIX

    class Cap:
        def invoke(self, messages):
            return type("R", (), {"content": "四君子汤益气健脾。"})()

    node = make_answer_generation_node(Cap())
    out = node.invoke(
        {
            "user_question": "四君子汤有什么功效？",
            "neo4j_answer": "",
            "doc_context": "[文献: 四君子汤.md#0] 四君子汤益气健脾。",
            "doc_chunks": [
                {
                    "doc_name": "四君子汤.md",
                    "chunk_idx": 0,
                    "text": "四君子汤益气健脾。",
                    "score": 0.45,
                }
            ],
            "crag_grade": "ambiguous",
        }
    )
    assert out["refused"] is False
    assert out["final_answer"].startswith(UNCERTAIN_PREFIX)


def test_save_uploaded_document_rejects_path_escape(tmp_path):
    from common.doc_store import save_uploaded_file

    dest = tmp_path / "corpus"
    dest.mkdir()
    path = save_uploaded_file("四君子汤.md", "# 四君子汤\n益气健脾".encode("utf-8"), dest_dir=dest)
    assert path.name == "四君子汤.md"
    assert path.read_text(encoding="utf-8").startswith("# 四君子汤")
    try:
        save_uploaded_file("../evil.md", b"x", dest_dir=dest)
        assert False, "should reject traversal"
    except ValueError:
        pass
