from common.doc_chunking import build_document_chunks_step_04, clean_markdown_step_01


def test_clean_markdown_removes_front_matter():
    raw = "---\ntitle: 伤寒论\n---\n# 太阳病\n桂枝汤主之。"

    cleaned = clean_markdown_step_01(raw)

    assert "title:" not in cleaned
    assert "桂枝汤主之" in cleaned


def test_build_document_chunks_preserves_parent_metadata():
    body = "# 汤头歌诀\n\n### 麻黃湯\n治寒傷營、無汗。\n\n### 桂枝湯\n治風傷衛、有汗。"

    chunks = build_document_chunks_step_04("tangtou", "tangtou.md", body, doc_type="方剂")

    assert len(chunks) >= 2
    mah = next(item for item in chunks if "麻黃湯" in item["text"])
    assert mah["parent_title"] == "麻黃湯"
    assert mah["doc_type"] == "方剂"
    assert "麻黄汤" in mah["text_simplified"]
    assert mah["parent_id"].startswith("tangtou#")
