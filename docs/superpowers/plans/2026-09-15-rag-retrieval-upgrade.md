# RAG Retrieval Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the document RAG stack with simplified/traditional normalization, structured parent-child chunking, reranking, query expansion, and a FAISS-default vector backend abstraction with optional Milvus.

**Architecture:** Keep Neo4j as the graph source of truth and keep FAISS as the default document vector backend. Add focused helper modules around the existing `common/doc_store.py` path so current API behavior remains compatible while richer chunk metadata and rerank signals improve retrieval quality.

**Tech Stack:** Python 3.10+, FastAPI, LangGraph, FAISS, BM25, jieba, optional OpenCC, optional sentence-transformers reranker, optional Milvus.

## Global Constraints

- Follow `C:/Users/Administrator/.agent/agent-coding-conventions.md`: modified/new named functions need `_step_XX` suffix unless they are framework hooks or public compatibility APIs.
- All new/modified functions and classes need Chinese docstrings; complex logic needs Chinese comments.
- Do not read or commit `.env` or `common/.env`; update only `.env.example` for configuration.
- Keep FAISS as the default vector backend; Milvus is optional and must not affect default startup.
- Do not migrate Neo4j graph entities or relationships into Milvus.
- HyDE and sub-question splitting stay disabled by default.
- Run tests with the local available Python executable: `D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest ...`.

---

## File Structure

- Create `common/text_normalization.py`
  - Owns simplified/traditional conversion with optional OpenCC fallback.
  - Owns deterministic query variants.
- Create `common/query_expansion.py`
  - Owns low-cost Chinese medicine synonym and alias expansion.
  - No LLM calls.
- Create `common/doc_chunking.py`
  - Owns Markdown cleanup, section parsing, parent-child chunk generation, and compatibility fixed-window fallback.
- Create `common/rag/rerank.py`
  - Owns rule rerank and optional BGE reranker adapter.
- Create `common/vector_backends.py`
  - Owns backend protocol plus FAISS and optional Milvus document vector backends.
- Modify `common/doc_store.py`
  - Integrates query variants, normalized BM25 text, vector backend abstraction, rule/model rerank, and parent-child chunk metadata.
- Modify `common/eval_ragas.py`
  - Uses structured chunking for evaluation corpus.
- Modify `scripts/eval_ragas.py`
  - No behavior change expected; it should benefit through shared chunking.
- Modify `.env.example`
  - Adds normalization, rerank, and vector backend config.
- Modify `README.md`
  - Documents new retrieval path and commands.
- Test `tests/test_rag_query_processing.py`
  - Covers normalization and query expansion.
- Test `tests/test_doc_chunking.py`
  - Covers structured parent-child chunking.
- Test `tests/test_doc_rerank.py`
  - Covers default rule rerank and optional model fallback behavior.
- Modify `tests/test_doc_rag.py`
  - Covers integrated `FaissDocStore` retrieval behavior.
- Modify `tests/test_eval_ragas.py`
  - Confirms generated evaluation chunks keep parent metadata and refusal samples remain correct.

---

### Task 1: Query Normalization And Expansion

**Files:**
- Create: `common/text_normalization.py`
- Create: `common/query_expansion.py`
- Create: `tests/test_rag_query_processing.py`

**Interfaces:**
- Produces: `normalize_text_step_01(text: str, target: str = "simplified") -> str`
- Produces: `query_variants_step_02(query: str) -> list[str]`
- Produces: `expand_zhongyi_query_step_01(query: str) -> list[str]`
- Consumed by: `common/doc_store.py` and `common/rag/rerank.py`

- [ ] **Step 1: Write failing tests for繁简 query variants**

```python
from common.query_expansion import expand_zhongyi_query_step_01
from common.text_normalization import normalize_text_step_01, query_variants_step_02


def test_query_variants_include_simplified_and_traditional():
    variants = query_variants_step_02("人参味如何、主补什么？")

    assert "人参味如何、主补什么？" in variants
    assert any("人參" in item or "主補" in item for item in variants)


def test_normalize_text_converts_common_medical_terms():
    assert normalize_text_step_01("人參主補五臟", "simplified") == "人参主补五脏"
    assert normalize_text_step_01("麻黄汤治什么", "traditional") == "麻黃湯治什麼"


def test_expand_zhongyi_query_adds_aliases_without_llm():
    variants = expand_zhongyi_query_step_01("麻黄汤治什么？")

    assert any("麻黃湯" in item for item in variants)
    assert any("主治" in item or "功效" in item for item in variants)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_rag_query_processing.py -q
```

Expected: FAIL with `ModuleNotFoundError` or missing function names.

- [ ] **Step 3: Implement normalization and expansion**

Create `common/text_normalization.py`:

```python
"""中医 RAG 文本繁简归一工具。"""

from __future__ import annotations

from functools import lru_cache


_S2T_FALLBACK = str.maketrans(
    {
        "医": "醫",
        "药": "藥",
        "汤": "湯",
        "气": "氣",
        "证": "證",
        "伤": "傷",
        "脏": "臟",
        "补": "補",
        "为": "為",
        "无": "無",
        "发": "發",
        "泻": "瀉",
        "与": "與",
        "参": "參",
        "么": "麼",
    }
)
_T2S_FALLBACK = {v: k for k, v in _S2T_FALLBACK.items()}


@lru_cache(maxsize=4)
def _opencc_converter_step_01(target: str):
    """步骤 01：懒加载 OpenCC 转换器；缺依赖时返回 None。

    参数：target 为 simplified 或 traditional。
    返回：OpenCC 实例或 None。
    异常：不向外抛依赖缺失。
    副作用：缓存转换器，避免重复加载。
    """
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    config = "t2s" if target == "simplified" else "s2t"
    return OpenCC(config)


def normalize_text_step_01(text: str, target: str = "simplified") -> str:
    """步骤 01：把文本归一为简体或繁体。

    参数：text 为原始文本；target 支持 simplified / traditional。
    返回：转换后的文本。
    异常：target 非法时抛 ValueError。
    副作用：无。
    """
    raw = str(text or "")
    if target not in {"simplified", "traditional"}:
        raise ValueError("target 仅支持 simplified / traditional")
    converter = _opencc_converter_step_01(target)
    if converter is not None:
        return converter.convert(raw)
    table = _T2S_FALLBACK if target == "simplified" else _S2T_FALLBACK
    return raw.translate(str.maketrans(table))


def query_variants_step_02(query: str) -> list[str]:
    """步骤 02：生成原文、简体、繁体 query 变体并去重。

    参数：query 为用户检索问句。
    返回：按优先级排序的 query 变体。
    异常：无。
    副作用：无。
    """
    raw = str(query or "").strip()
    if not raw:
        return []
    variants = [
        raw,
        normalize_text_step_01(raw, "simplified"),
        normalize_text_step_01(raw, "traditional"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
```

Create `common/query_expansion.py`:

```python
"""中医检索 query 低成本扩展。"""

from __future__ import annotations

from common.text_normalization import query_variants_step_02


_TERM_EXPANSIONS = {
    "治什么": ("主治", "功效", "适应证"),
    "治什麼": ("主治", "功效", "適應證"),
    "味如何": ("气味", "性味"),
    "味如何": ("氣味", "性味"),
    "主补": ("主补", "補", "主治"),
    "主補": ("主補", "補", "主治"),
}


def expand_zhongyi_query_step_01(query: str) -> list[str]:
    """步骤 01：生成中医术语和繁简兼容 query 变体。

    参数：query 为检索问句。
    返回：去重后的扩展问句列表，首项为原始问句。
    异常：无。
    副作用：无。
    """
    variants = query_variants_step_02(query)
    expanded = list(variants)
    for item in variants:
        for needle, replacements in _TERM_EXPANSIONS.items():
            if needle not in item:
                continue
            for repl in replacements:
                expanded.append(item.replace(needle, repl))
    out: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        clean = str(item or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out
```

- [ ] **Step 4: Run tests to verify GREEN**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_rag_query_processing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add common/text_normalization.py common/query_expansion.py tests/test_rag_query_processing.py
git commit -m "【需求】20260915-草药通-RAG查询繁简归一`n【修改内容】新增繁简归一与中医术语扩展工具及测试"
```

---

### Task 2: Structured Parent-Child Chunking

**Files:**
- Create: `common/doc_chunking.py`
- Create: `tests/test_doc_chunking.py`
- Modify: `common/doc_store.py`
- Modify: `common/eval_ragas.py`

**Interfaces:**
- Consumes: `normalize_text_step_01`
- Produces: `build_document_chunks_step_04(doc_id: str, doc_name: str, body: str, doc_type: str = "") -> list[dict[str, Any]]`
- Produces metadata fields: `parent_id`, `parent_title`, `section_path`, `text_simplified`, `text_traditional`, `parent_text_preview`

- [ ] **Step 1: Write failing tests for parent-child chunks**

```python
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
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_chunking.py -q
```

Expected: FAIL because `common.doc_chunking` does not exist.

- [ ] **Step 3: Implement structured chunking**

Create `common/doc_chunking.py` with:

```python
"""文献结构化父子分块。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from common.doc_rag import chunk_text
from common.text_normalization import normalize_text_step_01


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def clean_markdown_step_01(text: str) -> str:
    """步骤 01：清理 YAML front matter 和空白噪声。

    参数：text 为原始文档。
    返回：可用于切块的正文。
    异常：无。
    副作用：无。
    """
    raw = str(text or "").replace("\r\n", "\n")
    raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.DOTALL)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def split_sections_step_02(text: str) -> list[dict[str, Any]]:
    """步骤 02：按 Markdown 标题切父块；无标题时整篇作为父块。

    参数：text 为已清洗正文。
    返回：父块列表，包含 title、section_path、text。
    异常：无。
    副作用：无。
    """
    matches = list(_HEADING_RE.finditer(text or ""))
    if not matches:
        body = (text or "").strip()
        return [{"title": "", "section_path": [], "text": body}] if body else []
    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        body = text[start:end].strip()
        if body:
            sections.append(
                {
                    "title": title,
                    "section_path": [item[1] for item in stack],
                    "text": body,
                }
            )
    return sections


def split_child_texts_step_03(text: str, size: int = 400, overlap: int = 80) -> list[str]:
    """步骤 03：优先按段落切子块，超长段落用字符窗口兜底。

    参数：text 为父块正文；size/overlap 为最大长度和重叠。
    返回：子块文本列表。
    异常：无。
    副作用：无。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(chunk_text(para, size=size, overlap=overlap))
            continue
        candidate = f"{buf}\n\n{para}".strip() if buf else para
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = para
    if buf:
        chunks.append(buf)
    return chunks


def _parent_id_step_04(doc_id: str, section_path: list[str]) -> str:
    """步骤 04：生成稳定父块 ID。由 build_document_chunks_step_04 调用。"""
    raw = " / ".join(section_path)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}#{digest}"


def build_document_chunks_step_04(
    doc_id: str,
    doc_name: str,
    body: str,
    doc_type: str = "",
    size: int = 400,
    overlap: int = 80,
) -> list[dict[str, Any]]:
    """步骤 04：把单篇文档构造成带父子 metadata 的子块。

    参数：doc_id/doc_name 标识文档；body 为正文；doc_type 为知识库类型。
    返回：可入向量库的子块 metadata 列表。
    异常：无。
    副作用：无。
    """
    cleaned = clean_markdown_step_01(body)
    sections = split_sections_step_02(cleaned)
    chunks: list[dict[str, Any]] = []
    for section in sections:
        parent_title = str(section.get("title") or "")
        section_path = [str(x) for x in section.get("section_path") or []]
        parent_text = str(section.get("text") or "")
        parent_id = _parent_id_step_04(doc_id, section_path or [parent_title or doc_name])
        for text in split_child_texts_step_03(parent_text, size=size, overlap=overlap):
            idx = len(chunks)
            chunks.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "chunk_idx": idx,
                    "parent_id": parent_id,
                    "parent_title": parent_title,
                    "section_path": section_path,
                    "text": text,
                    "text_simplified": normalize_text_step_01(text, "simplified"),
                    "text_traditional": normalize_text_step_01(text, "traditional"),
                    "title_simplified": normalize_text_step_01(parent_title, "simplified"),
                    "title_traditional": normalize_text_step_01(parent_title, "traditional"),
                    "parent_text_preview": parent_text[:240],
                }
            )
    return chunks
```

- [ ] **Step 4: Integrate chunking into ingestion**

Modify `common/doc_store.py` so `ingest_directory` calls `build_document_chunks_step_04` instead of raw `chunk_text`. Preserve `doc_type` by using the source directory name when available:

```python
from common.doc_chunking import build_document_chunks_step_04
```

Inside the file loop:

```python
doc_type = path.parent.name if path.parent != folder else ""
chunks.extend(
    build_document_chunks_step_04(
        doc_id=path.stem,
        doc_name=path.name,
        body=body,
        doc_type=doc_type,
        size=size,
        overlap=overlap,
    )
)
```

Modify `common/eval_ragas.py::load_corpus_chunks_step_10` to call the same helper so evaluation uses production-like chunks.

- [ ] **Step 5: Run chunking and existing doc tests**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_chunking.py tests\test_doc_rag.py tests\test_eval_ragas.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add common/doc_chunking.py common/doc_store.py common/eval_ragas.py tests/test_doc_chunking.py tests/test_doc_rag.py tests/test_eval_ragas.py
git commit -m "【需求】20260915-草药通-RAG结构化分块`n【修改内容】新增 Markdown 父子分块和繁简检索字段，并接入文献入库与评测语料生成"
```

---

### Task 3: Rule Rerank

**Files:**
- Create: `common/rag/rerank.py`
- Create: `tests/test_doc_rerank.py`
- Modify: `common/doc_store.py`

**Interfaces:**
- Produces: `rule_rerank_step_01(query: str, hits: list[dict[str, Any]], top_n: int | None = None) -> list[dict[str, Any]]`
- Consumes: `query_variants_step_02`, `expand_zhongyi_query_step_01`

- [ ] **Step 1: Write failing rerank tests**

```python
from common.rag.rerank import rule_rerank_step_01


def test_rule_rerank_promotes_title_match():
    hits = [
        {"text": "桂枝湯治風傷衛。", "parent_title": "桂枝湯", "score": 0.9},
        {"text": "麻黃湯 治寒傷營、無汗。", "parent_title": "麻黃湯", "score": 0.7},
    ]

    out = rule_rerank_step_01("麻黄汤治什么？", hits)

    assert out[0]["parent_title"] == "麻黃湯"
    assert out[0]["rerank_score"] > out[1]["rerank_score"]


def test_rule_rerank_keeps_original_order_for_ties():
    hits = [
        {"text": "甲", "score": 0.5},
        {"text": "乙", "score": 0.5},
    ]

    out = rule_rerank_step_01("无关问题", hits)

    assert [item["text"] for item in out] == ["甲", "乙"]
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_rerank.py -q
```

Expected: FAIL because `common.rag.rerank` does not exist.

- [ ] **Step 3: Implement rule rerank**

Create `common/rag/rerank.py`:

```python
"""文献检索结果重排。"""

from __future__ import annotations

import os
from typing import Any

import jieba

from common.query_expansion import expand_zhongyi_query_step_01
from common.text_normalization import query_variants_step_02


def _tokens_step_01(text: str) -> set[str]:
    """步骤 01：切词并过滤空 token。由规则重排调用。"""
    return {w for w in jieba.cut(str(text or "")) if w.strip()}


def rule_rerank_step_01(
    query: str,
    hits: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """步骤 01：按标题、正文、关键词覆盖和原始分数做轻量重排。

    参数：query 为检索问句；hits 为候选块；top_n 限制重排数量。
    返回：带 rerank_score 的候选块。
    异常：无。
    副作用：无。
    """
    if not hits:
        return []
    variants = expand_zhongyi_query_step_01(query)
    normalized = set()
    for item in variants:
        normalized.update(query_variants_step_02(item))
    q_tokens = set()
    for item in normalized:
        q_tokens.update(_tokens_step_01(item))
    limit = top_n or len(hits)
    head = list(hits[:limit])
    tail = list(hits[limit:])
    scored: list[tuple[int, dict[str, Any]]] = []
    for idx, hit in enumerate(head):
        text = str(hit.get("text") or "")
        title = " ".join(
            str(hit.get(key) or "")
            for key in ("parent_title", "title_simplified", "title_traditional", "doc_name")
        )
        text_blob = " ".join(
            str(hit.get(key) or "")
            for key in ("text", "text_simplified", "text_traditional")
        )
        title_hit = 1.0 if any(v and v in title for v in normalized) else 0.0
        body_hit = 1.0 if any(v and v in text_blob for v in normalized) else 0.0
        h_tokens = _tokens_step_01(text_blob)
        coverage = len(q_tokens & h_tokens) / max(1, len(q_tokens))
        base = float(hit.get("rrf_score") or hit.get("score") or 0.0)
        score = base + title_hit * 2.0 + body_hit * 1.0 + coverage * 0.8
        copied = dict(hit)
        copied["rerank_score"] = round(float(score), 6)
        scored.append((idx, copied))
    scored.sort(key=lambda pair: (-float(pair[1].get("rerank_score") or 0.0), pair[0]))
    return [item for _, item in scored] + tail


def rerank_enabled_step_02() -> bool:
    """步骤 02：读取规则/模型重排总开关。"""
    return os.getenv("RERANK_ENABLE", "1").strip().lower() not in ("0", "false", "no")
```

- [ ] **Step 4: Integrate rule rerank into `mixed_search`**

In `common/doc_store.py`, after RRF fusion and before MMR, add:

```python
from common.rag.rerank import rerank_enabled_step_02, rule_rerank_step_01
```

Then:

```python
if rerank_enabled_step_02() and fused:
    top_n = int(os.getenv("RERANK_TOP_N", "20"))
    fused = rule_rerank_step_01(q, fused, top_n=top_n)
```

- [ ] **Step 5: Run rerank and doc tests**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_rerank.py tests\test_doc_rag.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```powershell
git add common/rag/rerank.py common/doc_store.py tests/test_doc_rerank.py tests/test_doc_rag.py
git commit -m "【需求】20260915-草药通-RAG规则重排`n【修改内容】新增标题、实体与关键词覆盖的规则 rerank 并接入混合检索"
```

---

### Task 4: Optional BGE Reranker

**Files:**
- Modify: `common/rag/rerank.py`
- Modify: `tests/test_doc_rerank.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `model_rerank_step_03(query: str, hits: list[dict[str, Any]], model_path: str | None = None) -> list[dict[str, Any]]`
- Produces: `rerank_hits_step_04(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing tests for model fallback**

Append to `tests/test_doc_rerank.py`:

```python
def test_rerank_hits_uses_rule_when_model_missing(monkeypatch):
    from common.rag.rerank import rerank_hits_step_04

    monkeypatch.setenv("RERANK_ENABLE", "1")
    monkeypatch.setenv("RERANK_MODEL_PATH", "")
    hits = [
        {"text": "桂枝湯治風傷衛。", "parent_title": "桂枝湯", "score": 0.9},
        {"text": "麻黃湯 治寒傷營、無汗。", "parent_title": "麻黃湯", "score": 0.7},
    ]

    out = rerank_hits_step_04("麻黄汤治什么？", hits)

    assert out[0]["parent_title"] == "麻黃湯"
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_rerank.py::test_rerank_hits_uses_rule_when_model_missing -q
```

Expected: FAIL because `rerank_hits_step_04` does not exist.

- [ ] **Step 3: Implement optional model rerank**

Add to `common/rag/rerank.py`:

```python
from functools import lru_cache


@lru_cache(maxsize=2)
def _load_reranker_step_03(model_path: str):
    """步骤 03：懒加载本地 reranker 模型。

    参数：model_path 为本地模型目录。
    返回：CrossEncoder 实例。
    异常：依赖或模型错误向外抛，由调用方降级。
    副作用：加载模型到内存。
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_path)


def model_rerank_step_03(
    query: str,
    hits: list[dict[str, Any]],
    model_path: str | None = None,
) -> list[dict[str, Any]]:
    """步骤 03：使用本地 BGE/CrossEncoder reranker 重排候选块。

    参数：query 为检索问句；hits 为候选块；model_path 可覆盖环境变量。
    返回：带 model_rerank_score 的候选块。
    异常：模型加载失败时抛出，外层负责降级。
    副作用：首次调用会加载模型。
    """
    path = (model_path or os.getenv("RERANK_MODEL_PATH") or "").strip()
    if not path:
        return hits
    model = _load_reranker_step_03(path)
    pairs = [(query, str(hit.get("text") or "")) for hit in hits]
    scores = model.predict(pairs)
    out: list[dict[str, Any]] = []
    for hit, score in zip(hits, scores):
        copied = dict(hit)
        copied["model_rerank_score"] = float(score)
        out.append(copied)
    out.sort(key=lambda item: -float(item.get("model_rerank_score") or 0.0))
    return out


def rerank_hits_step_04(query: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """步骤 04：统一 rerank 入口；模型缺失或失败时自动规则降级。

    参数：query 为检索问句；hits 为候选块。
    返回：重排后的候选块。
    异常：不抛模型错误，保证主问答路径可用。
    副作用：可能加载本地 reranker。
    """
    if not rerank_enabled_step_02():
        return hits
    top_n = int(os.getenv("RERANK_TOP_N", "20"))
    ruled = rule_rerank_step_01(query, hits, top_n=top_n)
    model_path = (os.getenv("RERANK_MODEL_PATH") or "").strip()
    if not model_path:
        return ruled
    try:
        return model_rerank_step_03(query, ruled[:top_n], model_path=model_path) + ruled[top_n:]
    except Exception as exc:
        from common.obs import degraded

        degraded("rerank_model", exc)
        return ruled
```

Update `common/doc_store.py` to call `rerank_hits_step_04` instead of `rule_rerank_step_01` directly.

- [ ] **Step 4: Document environment config**

Append to `.env.example` near RAG settings:

```env
# RAG rerank：默认规则重排；配置本地 CrossEncoder/BGE reranker 后启用模型重排
RERANK_ENABLE=1
RERANK_MODEL_PATH=
RERANK_TOP_N=20
```

- [ ] **Step 5: Run tests**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_rerank.py tests\test_doc_rag.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```powershell
git add common/rag/rerank.py common/doc_store.py tests/test_doc_rerank.py .env.example
git commit -m "【需求】20260915-草药通-RAG可选模型重排`n【修改内容】新增本地 BGE reranker 可选入口，缺模型时自动降级为规则重排"
```

---

### Task 5: Vector Backend Abstraction With FAISS Default

**Files:**
- Create: `common/vector_backends.py`
- Modify: `common/doc_store.py`
- Create or modify: `tests/test_doc_vector_backend.py`
- Modify: `.env.example`

**Interfaces:**
- Produces: `DocumentVectorBackend` protocol.
- Produces: `FaissDocumentBackend`.
- Produces: `MilvusDocumentBackend` optional scaffold with lazy imports.
- Produces: `create_document_backend_step_03(...) -> DocumentVectorBackend`

- [ ] **Step 1: Write failing backend tests**

```python
from pathlib import Path

from common.doc_store import FaissDocStore
from common.eval_ragas import hash_encode_step_04


def test_faiss_doc_store_remains_default_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_VECTOR_BACKEND", "faiss")
    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=hash_encode_step_04,
    )
    chunks = [
        {
            "doc_id": "a",
            "doc_name": "a.md",
            "chunk_idx": 0,
            "text": "人參 味甘小寒。主補五臟。",
            "text_simplified": "人参 味甘小寒。主补五脏。",
            "parent_title": "人參",
        }
    ]

    store.build_chunks(chunks)
    hits = store.search("人参主补什么", top_k=1, min_score=0.0)

    assert hits
    assert hits[0]["doc_id"] == "a"
```

- [ ] **Step 2: Run tests to verify baseline**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_vector_backend.py -q
```

Expected: FAIL if the new test file imports missing backend helpers; if the direct compatibility test passes, add a second test importing `create_document_backend_step_03` to force RED.

- [ ] **Step 3: Implement backend abstraction**

Create `common/vector_backends.py`:

```python
"""文献向量后端抽象。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol


class DocumentVectorBackend(Protocol):
    """文献向量后端协议。"""

    chunks: list[dict[str, Any]]

    def build_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """步骤：接口，构建文献块索引。"""

    def load(self):
        """步骤：接口，加载已有索引。"""

    def search(self, question: str, top_k: int = 4, min_score: float = 0.35) -> list[dict[str, Any]]:
        """步骤：接口，语义检索。"""


class MilvusDocumentBackend:
    """Milvus 文献向量后端占位实现。"""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        """步骤：01 初始化 Milvus 后端配置，实际连接延后。"""
        self.chunks: list[dict[str, Any]] = []

    def build_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """步骤：02 构建 Milvus 文献向量索引。"""
        raise NotImplementedError("Milvus 文献后端将在后续任务接入；当前默认使用 FAISS")

    def load(self):
        """步骤：03 加载 Milvus 文献向量索引。"""
        raise NotImplementedError("Milvus 文献后端将在后续任务接入；当前默认使用 FAISS")

    def search(self, question: str, top_k: int = 4, min_score: float = 0.35) -> list[dict[str, Any]]:
        """步骤：04 Milvus 语义检索。"""
        raise NotImplementedError("Milvus 文献后端将在后续任务接入；当前默认使用 FAISS")


def create_document_backend_step_03(
    index_path: str | Path,
    metadata_path: str | Path,
    encode_fn=None,
    model_path: str | None = None,
) -> DocumentVectorBackend:
    """步骤 03：按 DOC_VECTOR_BACKEND 创建文献向量后端。

    参数：index_path/metadata_path 为 FAISS 文件；encode_fn/model_path 为编码器配置。
    返回：文献向量后端实例。
    异常：未知后端抛 ValueError。
    副作用：无。
    """
    backend = os.getenv("DOC_VECTOR_BACKEND", "faiss").strip().lower()
    if backend == "faiss":
        from common.doc_store import FaissDocStore

        return FaissDocStore(index_path, metadata_path, encode_fn=encode_fn, model_path=model_path)
    if backend == "milvus":
        return MilvusDocumentBackend()
    raise ValueError(f"未知 DOC_VECTOR_BACKEND: {backend}")
```

Do not force `FaissDocStore` to depend on this factory in the same task if it introduces circular imports. Keep the factory as a compatibility entrypoint and document FAISS as default.

- [ ] **Step 4: Document backend config**

Add to `.env.example`:

```env
# 文献向量后端：默认 faiss；milvus 为可选后端，Neo4j 图谱不迁移
DOC_VECTOR_BACKEND=faiss
```

- [ ] **Step 5: Run backend tests**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_vector_backend.py tests\test_doc_rag.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```powershell
git add common/vector_backends.py common/doc_store.py tests/test_doc_vector_backend.py .env.example
git commit -m "【需求】20260915-草药通-RAG向量后端抽象`n【修改内容】新增文献向量后端工厂，保留 FAISS 默认并预留 Milvus 可选入口"
```

---

### Task 6: Evaluation, Docs, And Regression Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `eval/ragas_dataset.json`
- Modify: `eval/ragas_testset.json`
- Possibly modify: `eval/golden_qa.json` only if tests require category alignment

**Interfaces:**
- Consumes all previous tasks.
- Produces updated docs and regenerated evaluation artifacts.

- [ ] **Step 1: Add regression tests for real retrieval examples**

Extend `tests/test_doc_rag.py` with a small in-memory store using `hash_encode_step_04`:

```python
def test_mixed_search_simplified_query_hits_traditional_chunk(tmp_path):
    from common.doc_store import FaissDocStore
    from common.eval_ragas import hash_encode_step_04

    store = FaissDocStore(
        index_path=tmp_path / "docs.index",
        metadata_path=tmp_path / "docs.json",
        encode_fn=hash_encode_step_04,
    )
    store.build_chunks(
        [
            {
                "doc_id": "bencao",
                "doc_name": "bencao.md",
                "chunk_idx": 0,
                "parent_title": "人參",
                "text": "人參 味甘小寒。主補五臟。",
                "text_simplified": "人参 味甘小寒。主补五脏。",
                "text_traditional": "人參 味甘小寒。主補五臟。",
            },
            {
                "doc_id": "noise",
                "doc_name": "noise.md",
                "chunk_idx": 0,
                "parent_title": "杂项",
                "text": "今日天气晴。",
                "text_simplified": "今日天气晴。",
                "text_traditional": "今日天氣晴。",
            },
        ]
    )

    hits = store.mixed_search("人参主补什么？", top_k=1, min_score=0.0)

    assert hits[0]["doc_id"] == "bencao"
```

- [ ] **Step 2: Run regression test to verify RED or current weakness**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_doc_rag.py::test_mixed_search_simplified_query_hits_traditional_chunk -q
```

Expected before full integration: may FAIL if normalized fields are not used in BM25/rerank. After integration it must PASS.

- [ ] **Step 3: Update `common/doc_store.py` to search normalized text**

Ensure `_rebuild_bm25` tokenizes a combined normalized blob:

```python
texts = [
    " ".join(
        str(c.get(key) or "")
        for key in ("text", "text_simplified", "text_traditional", "parent_title", "title_simplified", "title_traditional")
    )
    for c in self.chunks
]
```

Ensure `mixed_search` uses query variants from `expand_zhongyi_query_step_01`; merge dense/BM25 hits across variants with RRF.

- [ ] **Step 4: Regenerate RAGAS artifacts**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe scripts\eval_ragas.py all --no-crag
```

Expected: writes `eval/ragas_testset.json`, `eval/ragas_dataset.json`, and `eval/ragas_report.json`. `eval/ragas_report.json` remains gitignored.

- [ ] **Step 5: Update README**

Document:

```markdown
文献 RAG 检索链路：繁简归一 + 术语扩展 + dense/BM25/标题召回 + RRF + rerank + MMR + CRAG。
默认后端是 FAISS；`DOC_VECTOR_BACKEND=milvus` 为可选后端，Neo4j 图谱不迁移。
```

Add commands:

```bash
python scripts/eval_ragas.py all --no-crag
python scripts/eval_ragas.py official
```

- [ ] **Step 6: Run targeted tests**

Run:

```powershell
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests\test_rag_query_processing.py tests\test_doc_chunking.py tests\test_doc_rerank.py tests\test_doc_vector_backend.py tests\test_doc_rag.py tests\test_eval_ragas.py -q
```

Expected: PASS.

- [ ] **Step 7: Run full test suite**

Run:

```powershell
New-Item -ItemType Directory -Force tmp_codex_pytest | Out-Null
$env:TEMP=(Resolve-Path tmp_codex_pytest).Path
$env:TMP=$env:TEMP
D:\DevAPP\anaconda3\envs\ctm_kg\python.exe -m pytest tests -q --basetemp=tmp_codex_pytest\pytest --cache-clear -o cache_dir=tmp_codex_pytest\cache
```

Expected: all tests pass.

- [ ] **Step 8: Clean temporary test directory**

Run:

```powershell
$target=(Resolve-Path tmp_codex_pytest).Path
$root=(Resolve-Path .).Path
if ($target.StartsWith($root + [System.IO.Path]::DirectorySeparatorChar)) {
  Remove-Item -LiteralPath $target -Recurse -Force
} else {
  throw "refuse to remove outside workspace: $target"
}
```

- [ ] **Step 9: Commit Task 6**

```powershell
git add README.md .env.example common/doc_store.py eval/ragas_dataset.json eval/ragas_testset.json tests/test_doc_rag.py
git commit -m "【需求】20260915-草药通-RAG检索增强验收`n【修改内容】更新文档与 RAGAS 评测产物，补充简繁检索回归测试并完成全量验证"
```

---

## Self-Review

- Spec coverage: all confirmed design requirements map to tasks: normalization Task 1, parent-child chunks Task 2, rule/model rerank Tasks 3-4, vector backend abstraction Task 5, docs/eval verification Task 6.
- Incomplete-marker scan: no incomplete markers are present; Milvus is explicitly scoped as optional scaffold behavior with FAISS default preserved.
- Type consistency: produced functions use `_step_XX` naming and are referenced consistently by later tasks.
- Risk control: default path does not require OpenCC, reranker model, or Milvus.
