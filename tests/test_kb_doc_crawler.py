"""Open TCM knowledge-base crawler converters (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from _001_crawler.kb_doc_crawler import (
    AuthRequiredError,
    html_to_markdown_step_13,
    load_sources_step_2,
    parse_ctext_payload_step_12,
    parse_wikisource_payload_step_11,
    render_doc_step_18,
    sanitize_filename_step_16,
    split_markdown_chapters_step_20,
    write_parts_step_15,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kb_crawler" / "wiki_sample.html"


def test_sanitize_filename():
    assert sanitize_filename_step_16("伤寒论:*?") == "伤寒论___"
    assert sanitize_filename_step_16("foo/bar.md") == "bar.md"
    assert sanitize_filename_step_16("") == "unnamed"
    assert sanitize_filename_step_16("..") == "unnamed"
    assert sanitize_filename_step_16("CON") == "unnamed"


def test_html_to_markdown_strips_wiki_chrome():
    html = FIXTURE.read_text(encoding="utf-8")
    md = html_to_markdown_step_13(html)
    assert "目录应被去掉" not in md
    assert "编辑" not in md
    assert "[1]" not in md
    assert "## 辨太阳病" in md
    assert "脉浮" in md
    assert "- 桂枝汤主之。" in md
    assert "## 辨阳明病" in md
    assert "<script>" not in html_to_markdown_step_13('<div class="mw-parser-output"><script>alert(1)</script><p>正文</p></div>')
    assert "alert" not in html_to_markdown_step_13('<script>alert(1)</script>')


def test_split_only_when_over_max_bytes():
    md = "## 甲\n\n短\n\n## 乙\n\n也短"
    one = split_markdown_chapters_step_20("伤寒论", md, max_bytes=10_000)
    assert len(one) == 1
    assert one[0][0] == "伤寒论"

    parts = split_markdown_chapters_step_20("伤寒论", md, max_bytes=20)
    names = [p[0] for p in parts]
    assert "甲" in names
    assert "乙" in names


def test_split_by_size_when_no_headings():
    body = "甲段。\n\n" + ("乙" * 40) + "\n\n丙段。"
    parts = split_markdown_chapters_step_20("素问", body, max_bytes=30)
    assert len(parts) >= 2
    assert all(len(p[1].encode("utf-8")) <= 30 for p in parts)


def test_render_and_write_parts(tmp_path):
    item = {
        "id": "shanghanlun",
        "title": "伤寒论",
        "docType": "典籍",
        "license": "公有领域",
    }
    body = html_to_markdown_step_13(FIXTURE.read_text(encoding="utf-8"))
    paths = write_parts_step_15(
        item=item,
        chapter_title="伤寒论",
        markdown=body,
        source_url="https://zh.wikisource.org/wiki/傷寒論",
        out_dir=tmp_path,
        max_bytes=4096,
        fetched_at="2026-09-13T00:00:00+00:00",
    )
    assert paths
    text = paths[0].read_text(encoding="utf-8")
    assert text.startswith("---")
    assert 'docType: "典籍"' in text
    assert "zh.wikisource.org" in text
    assert paths[0].name.startswith("shanghanlun_")
    assert paths[0].parent.name == "典籍"


def test_same_title_writes_distinct_files(tmp_path):
    body = "上古天真论正文。"
    paths = []
    for item_id in ("suwen-wiki", "suwen-ctext"):
        paths.extend(
            write_parts_step_15(
                item={
                    "id": item_id,
                    "title": "黄帝内经素问·上古天真论",
                    "docType": "典籍",
                    "license": "公有领域",
                },
                chapter_title="黄帝内经素问·上古天真论",
                markdown=body,
                source_url="https://example.test/" + item_id,
                out_dir=tmp_path,
                max_bytes=4096,
                fetched_at="2026-09-13T00:00:00+00:00",
            )
        )
    names = {p.name for p in paths}
    assert len(names) == 2
    assert any(n.startswith("suwen-wiki_") for n in names)
    assert any(n.startswith("suwen-ctext_") for n in names)


def test_render_doc_front_matter():
    out = render_doc_step_18(
        title="伤寒论",
        chapter="辨太阳病",
        doc_type="典籍",
        source_url="https://example.test",
        license_text="CC BY-SA 4.0",
        fetched_at="2026-09-13T00:00:00+00:00",
        body="脉浮。",
    )
    assert "# 伤寒论 · 辨太阳病" in out
    assert 'license: "CC BY-SA 4.0"' in out


def test_parse_wikisource_payload():
    html = FIXTURE.read_text(encoding="utf-8")
    title, md = parse_wikisource_payload_step_11(
        {"parse": {"title": "傷寒論", "text": html}}
    )
    assert title == "傷寒論"
    assert "脉浮" in md


def test_parse_ctext_payload_and_auth():
    title, md = parse_ctext_payload_step_12(
        {"title": "上古天真论", "fulltext": ["昔在黄帝", "成而登天"]}
    )
    assert title == "上古天真论"
    assert "昔在黄帝" in md
    with pytest.raises(AuthRequiredError):
        parse_ctext_payload_step_12({"subsections": ["ctp:x/a"], "fulltext": []})
    with pytest.raises(AuthRequiredError):
        parse_ctext_payload_step_12({"error": "ERR_REQUIRES_AUTHENTICATION"})


def test_load_sources():
    items = load_sources_step_2(
        Path(__file__).resolve().parents[1] / "_001_crawler" / "kb_sources.json"
    )
    assert items
    assert all("id" in it and "source" in it for it in items)
    kinds = {it["source"] for it in items}
    assert "wikisource" in kinds
    assert "ctext" in kinds
    by_type: dict[str, int] = {}
    for it in items:
        by_type[it["docType"]] = by_type.get(it["docType"], 0) + 1
    assert set(by_type) <= set(("方剂", "本草", "典籍", "医案", "其他"))
    assert all(n <= 10 for n in by_type.values())


def test_load_state_rejects_corrupt(tmp_path):
    from _001_crawler.kb_doc_crawler import load_state_step_3, save_state_step_24

    path = tmp_path / "_state.json"
    save_state_step_24(path, {"done": ["a"], "skipped": ["b"]})
    loaded = load_state_step_3(path)
    assert loaded["done"] == ["a"]
    assert loaded["skipped"] == ["b"]
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_state_step_3(path)
