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
    raw = str(text or "")
    matches = list(_HEADING_RE.finditer(raw))
    if not matches:
        body = raw.strip()
        return [{"title": "", "section_path": [], "text": body}] if body else []

    sections: list[dict[str, Any]] = []
    stack: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        body = raw[start:end].strip()
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
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", str(text or "")) if p.strip()]
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
        raw_key = " / ".join(section_path or [parent_title or doc_name])
        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()[:10]
        parent_id = f"{doc_id}#{digest}"
        for child in split_child_texts_step_03(parent_text, size=size, overlap=overlap):
            text = child
            if parent_title and parent_title not in text:
                text = f"{parent_title}\n{text}"
            idx = len(chunks)
            chunks.append(
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "chunk_idx": idx,
                    "parent_id": parent_id,
                    "title": parent_title,
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
