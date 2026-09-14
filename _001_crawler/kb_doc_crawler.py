#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公开中医文献爬虫：写成知识库可上传的 Markdown。

只拉取白名单源（维基文库公有领域整理、ctext 匿名可取的章节），
不爬中医百科详情页。原文多属公有领域；维基文库整理条款为 CC BY-SA 4.0，
文件头会保留出处，上传知识库时请勿删 YAML。

用法:
  python -m _001_crawler.kb_doc_crawler --limit 2
  python -m _001_crawler.kb_doc_crawler --per-type 10
  python -m _001_crawler.kb_doc_crawler --force

落盘 data/kb_crawl/{docType}/{id}_{书名}_{章节}.md 后，到知识库页按类型分批上传
（每次最多 5 个文件）。本脚本不调用 /document/upload。

执行顺序（函数名 = 原名_step_N；wiki / ctext 为分支）:
  1 main_step_1
  2 load_sources_step_2
  3 load_state_step_3
  4 upload_max_bytes_step_4
  5 worker_step_5
  6 crawl_item_step_6
  7 fetch_wikisource_step_7 / 8 fetch_ctext_step_8
  9 _http_get_step_9
  10 pace_wait_step_10
  11 parse_wikisource_payload_step_11 / 12 parse_ctext_payload_step_12
  13 html_to_markdown_step_13
  14 wikisource_url_step_14
  15 write_parts_step_15
  16 sanitize_filename_step_16
  17 prefix_size_step_17
  18 render_doc_step_18
  19 yaml_scalar_step_19
  20 split_markdown_chapters_step_20
  21 _split_by_size_step_21
  22 split_utf8_step_22
  23 _write_md_step_23
  24 save_state_step_24
  25 append_error_step_25
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, Tag

HERE = Path(__file__).resolve().parent  # 本模块目录 _001_crawler
ROOT = HERE.parent  # 仓库根目录
DEFAULT_SOURCES = HERE / "kb_sources.json"  # 白名单书目
DEFAULT_OUT = ROOT / "data" / "kb_crawl"  # Markdown 输出根目录
DEFAULT_WORKERS = 3  # 默认并发线程数
MAX_WORKERS = 4  # 并发上限，避免打爆源站
REQUEST_INTERVAL = 0.5  # 全进程两次请求的最小间隔（秒）
DOC_TYPES = ("方剂", "本草", "典籍", "医案", "其他")  # 知识库文档类型
ALLOWED_SOURCES = ("wikisource", "ctext")  # 仅允许这两种来源
WIKI_API = "https://zh.wikisource.org/w/api.php"
CTEXT_API = "https://api.ctext.org/gettext"
HEADERS = {
    "User-Agent": (
        "chinese_medical-kb-crawler/1.0 (non-commercial TCM RAG; "
        "https://github.com/Ajet9009/chinese_medical)"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}
# 维基文库页面里要剥掉的导航、目录、脚注、编辑链等
NOISE_SELECTORS = (
    "table.infobox",
    "table.navbox",
    ".mw-editsection",
    "#toc",
    ".toc",
    ".sister-projects",
    ".mw-empty-elt",
    "sup.reference",
    ".noprint",
    "style",
    "script",
    ".hatnote",
    ".ambox",
)
# Windows 保留设备名，不能当文件名
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_lock = threading.Lock()  # 保护限速时间戳、进度统计、状态落盘
_stats = {"ok": 0, "err": 0, "skip": 0}  # 本次运行计数
_last_fetch = 0.0  # 上次发请求的 monotonic 时间


class AuthRequiredError(RuntimeError):
    """HTTP 401/403 或 ctext 需登录；记入 skipped，下次默认不再抓。"""


def upload_max_bytes_step_4() -> int:
    """第 4 步：读取知识库单文件上传上限（字节），默认 8MB，至少 1KB。"""
    raw = os.getenv("DOC_UPLOAD_MAX_BYTES", str(8 * 1024 * 1024))
    try:
        return max(1024, int(raw))
    except ValueError:
        return 8 * 1024 * 1024


def pace_wait_step_10(interval: float = REQUEST_INTERVAL) -> None:
    """第 10 步：全进程串行限速，保证两次 HTTP 间隔不少于 interval 秒。"""
    global _last_fetch
    with _lock:
        now = time.monotonic()
        delay = _last_fetch + interval - now
        if delay > 0:
            time.sleep(delay)
        _last_fetch = time.monotonic()


def sanitize_filename_step_16(name: str) -> str:
    """第 16 步：去掉路径穿越、Windows 非法字符和保留名，截断到 80 字。"""
    cleaned = (name or "").replace("\x00", "")
    cleaned = cleaned.replace("\\", "/").split("/")[-1]  # 只保留最后一段，防 ../
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned).strip(" .")
    stem = cleaned.upper().split(".")[0]
    if (not cleaned) or cleaned in {".", ".."} or stem in _WIN_RESERVED:
        return "unnamed"
    return cleaned[:80]


def yaml_scalar_step_19(value: str) -> str:
    """第 19 步：YAML 标量用 JSON 双引号转义，避免书名里的冒号/引号弄坏 front matter。"""
    return json.dumps(str(value).replace("\r", " ").replace("\n", " ").strip(), ensure_ascii=False)


def load_sources_step_2(path: Path) -> list[dict[str, Any]]:
    """第 2 步：读 kb_sources.json 白名单，丢掉无 id 或不在允许来源里的行。"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("kb_sources.json 必须是数组")
    items: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict) or not str(row.get("id") or "").strip():
            continue
        source = str(row.get("source") or "").strip()
        if source not in ALLOWED_SOURCES:
            continue
        doc_type = str(row.get("docType") or "典籍")
        if doc_type not in DOC_TYPES:
            doc_type = "其他"
        items.append({**row, "id": str(row["id"]).strip(), "source": source, "docType": doc_type})
    return items


def html_to_markdown_step_13(html: str) -> str:
    """第 13 步：剥掉维基导航噪声，把标题/段落/列表转成 Markdown。"""
    soup = BeautifulSoup(html or "", "html.parser")
    for sel in NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    root = soup.select_one(".mw-parser-output") or soup.body or soup
    if not isinstance(root, Tag):
        return soup.get_text("\n", strip=True)
    lines: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"]):
        if not isinstance(el, Tag):
            continue
        # 跳过嵌套块，避免同一段文字被父、子各输出一次
        parent_block = el.find_parent(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote"])
        if parent_block is not None and parent_block is not el:
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        name = el.name or ""
        if name.startswith("h") and name[1:].isdigit():
            lines.append("#" * int(name[1]) + " " + text)
            lines.append("")
        elif name == "li":
            lines.append("- " + text)
        else:
            lines.append(text)
            lines.append("")
    return "\n".join(lines).strip()


def split_utf8_step_22(text: str, max_bytes: int) -> list[str]:
    """第 22 步：按 UTF-8 字节硬切，回退到字符边界，避免切开汉字。"""
    raw = (text or "").encode("utf-8")
    if max_bytes < 1:
        return [text] if text else []
    out: list[str] = []
    i = 0
    while i < len(raw):
        j = min(i + max_bytes, len(raw))
        if j < len(raw):
            # 10xxxxxx 是续字节，往回找到字符起点
            while j > i and (raw[j] & 0xC0) == 0x80:
                j -= 1
        if j == i:
            j = min(i + 1, len(raw))
        out.append(raw[i:j].decode("utf-8"))
        i = j
    return [p for p in out if p]


def split_markdown_chapters_step_20(
    title: str,
    markdown: str,
    max_bytes: int,
) -> list[tuple[str, str]]:
    """第 20 步：未超限则整篇一份；超限先按二级标题拆章，单章仍超限再交给第 21 步。"""
    body = (markdown or "").strip()
    if not body:
        return []
    if len(body.encode("utf-8")) <= max_bytes:
        return [(title, body)]

    parts = re.split(r"(?m)^(## .+)$", body)
    preamble = parts[0].strip()  # ## 之前的前言
    headings = parts[1:]  # [标题, 正文, 标题, 正文, ...]
    if len(headings) < 2:
        return _split_by_size_step_21(title, body, max_bytes)

    chunks: list[tuple[str, str]] = []
    i = 0
    while i < len(headings):
        heading = headings[i].strip()
        text = headings[i + 1].strip() if i + 1 < len(headings) else ""
        i += 2
        chapter = re.sub(r"^##\s+", "", heading).strip() or "未标题"
        piece = heading + ("\n\n" + text if text else "")
        if preamble and not chunks:
            piece = preamble + "\n\n" + piece
            preamble = ""
        if len(piece.encode("utf-8")) <= max_bytes:
            chunks.append((chapter, piece))
        else:
            chunks.extend(_split_by_size_step_21(chapter, piece, max_bytes))
    if preamble:
        chunks.insert(0, ("前言", preamble))
    return chunks or [(title, body)]


def _split_by_size_step_21(title: str, body: str, max_bytes: int) -> list[tuple[str, str]]:
    """第 21 步：按空行分段拼装；单段仍超限则第 22 步按字节切开。"""
    paras = re.split(r"\n\s*\n", body.strip())
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    idx = 1

    def flush() -> None:
        """把当前缓冲写成一份，后续份用 书名_序号。"""
        nonlocal idx, buf
        if not buf:
            return
        label = title if idx == 1 and not out else f"{title}_{idx}"
        out.append((label, "\n\n".join(buf).strip()))
        idx += 1
        buf = []

    for para in paras:
        candidate = "\n\n".join(buf + [para]) if buf else para
        if len(candidate.encode("utf-8")) > max_bytes and buf:
            flush()
            candidate = para
        if len(candidate.encode("utf-8")) > max_bytes:
            for piece in split_utf8_step_22(para, max_bytes):
                out.append((f"{title}_{idx}", piece.strip()))
                idx += 1
            buf = []
            continue
        buf.append(para)
    flush()
    return out or [(title, body)]


def render_doc_step_18(
    *,
    title: str,
    chapter: str,
    doc_type: str,
    source_url: str,
    license_text: str,
    fetched_at: str,
    body: str,
) -> str:
    """第 18 步：拼 YAML 头 + 一级标题 + 正文，供知识库上传。"""
    heading = title if not chapter or chapter == title else f"{title} · {chapter}"
    return "\n".join(
        [
            "---",
            f"title: {yaml_scalar_step_19(title)}",
            f"chapter: {yaml_scalar_step_19(chapter)}",
            f"docType: {yaml_scalar_step_19(doc_type)}",
            f"source: {yaml_scalar_step_19(source_url)}",
            f"license: {yaml_scalar_step_19(license_text)}",
            f"fetched_at: {yaml_scalar_step_19(fetched_at)}",
            "---",
            "",
            f"# {heading}",
            "",
            (body or "").strip(),
            "",
        ]
    )


def prefix_size_step_17(
    *,
    title: str,
    chapter: str,
    doc_type: str,
    source_url: str,
    license_text: str,
    fetched_at: str,
) -> int:
    """第 17 步：空正文时前缀（YAML + 一级标题）有多大，拆章预算要扣掉。"""
    empty = render_doc_step_18(
        title=title,
        chapter=chapter,
        doc_type=doc_type,
        source_url=source_url,
        license_text=license_text,
        fetched_at=fetched_at,
        body="",
    )
    return len(empty.encode("utf-8"))


def wikisource_url_step_14(page: str) -> str:
    """第 14 步：把 API 页名编成可点开的维基文库 URL，写入 YAML source。"""
    return "https://zh.wikisource.org/wiki/" + quote(page, safe="/")


def parse_wikisource_payload_step_11(payload: dict[str, Any]) -> tuple[str, str]:
    """第 11 步：从 parse API JSON 取出标题和 HTML，再转 Markdown。"""
    parsed = payload.get("parse") or {}
    title = str(parsed.get("title") or "").strip()
    html = parsed.get("text") or ""
    if isinstance(html, dict):
        html = html.get("*") or ""  # formatversion=1 时正文在 text["*"]
    md = html_to_markdown_step_13(str(html))
    if not md:
        raise ValueError("维基文库页面无正文")
    return title, md


def parse_ctext_payload_step_12(payload: dict[str, Any]) -> tuple[str, str]:
    """第 12 步：从 ctext gettext JSON 取 fulltext；只有子目没有正文则视为需登录。"""
    if payload.get("subsections") and not payload.get("fulltext"):
        raise AuthRequiredError("ctext 该 URN 需要登录才能展开子目")
    err = str(payload.get("error") or payload.get("err") or payload.get("code") or "")
    if "AUTH" in err.upper() or "authentication" in err.lower():
        raise AuthRequiredError(err or "ERR_REQUIRES_AUTHENTICATION")
    title = str(payload.get("title") or "").strip()
    fulltext = payload.get("fulltext") or []
    if isinstance(fulltext, str):
        paras = [fulltext]
    else:
        paras = [str(p).strip() for p in fulltext if str(p).strip()]
    if not paras:
        raise ValueError("ctext 未返回 fulltext")
    return title, "\n\n".join(paras)


def _http_get_step_9(session: requests.Session, url: str, params: dict[str, str]) -> requests.Response:
    """第 9 步：限速 GET；429/503 最多尝试 3 次（再试 2 次）；401/403 当成需登录。"""
    last: requests.Response | None = None
    for attempt in range(3):
        pace_wait_step_10()
        last = session.get(url, params=params, headers=HEADERS, timeout=30)
        if last.status_code in (429, 503):
            time.sleep(2 ** attempt)  # 1s、2s、4s 退避
            continue
        if last.status_code in (401, 403):
            raise AuthRequiredError(f"HTTP {last.status_code}")
        last.raise_for_status()
        return last
    assert last is not None
    last.raise_for_status()
    return last


def fetch_wikisource_step_7(session: requests.Session, page: str) -> tuple[str, str, str]:
    """第 7 步：拉维基文库 parse API，返回 (标题, Markdown, 页面 URL)。"""
    resp = _http_get_step_9(
        session,
        WIKI_API,
        {
            "action": "parse",
            "page": page,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
            "disablelimitreport": "1",
            "disableeditsection": "1",
        },
    )
    title, md = parse_wikisource_payload_step_11(resp.json())
    return title or page, md, wikisource_url_step_14(page)


def fetch_ctext_step_8(session: requests.Session, urn: str) -> tuple[str, str, str]:
    """第 8 步：匿名拉 ctext 章节 gettext，返回 (标题, 正文, 页面 URL)。"""
    resp = _http_get_step_9(session, CTEXT_API, {"urn": urn, "if": "zh"})
    title, md = parse_ctext_payload_step_12(resp.json())
    return title or urn, md, f"https://ctext.org/{urn.removeprefix('ctp:')}"


def load_state_step_3(path: Path) -> dict[str, Any]:
    """第 3 步：读断点 _state.json；文件损坏直接抛错，不重置，避免误重爬。"""
    if not path.exists():
        return {"done": [], "skipped": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"状态文件损坏: {path}")
    done = data.get("done") or []
    skipped = data.get("skipped") or []
    if not isinstance(done, list) or not isinstance(skipped, list):
        raise ValueError(f"状态文件字段无效: {path}")
    return {"done": [str(x) for x in done], "skipped": [str(x) for x in skipped]}


def save_state_step_24(path: Path, state: dict[str, Any]) -> None:
    """第 24 步：先写 .tmp 再 os.replace，保证状态文件原子更新。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_error_step_25(path: Path, item_id: str, exc: BaseException) -> None:
    """第 25 步：把失败/需登录记录追加进 _errors.jsonl。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": item_id,
        "error": f"{type(exc).__name__}: {exc}",
        "at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_parts_step_15(
    *,
    item: dict[str, Any],
    chapter_title: str,
    markdown: str,
    source_url: str,
    out_dir: Path,
    max_bytes: int,
    fetched_at: str,
) -> list[Path]:
    """第 15 步：按类型分子目录，拆章后写成不超过上传上限的 .md。"""
    book = str(item.get("title") or chapter_title)
    doc_type = str(item.get("docType") or "典籍")
    if doc_type not in DOC_TYPES:
        doc_type = "其他"
    item_id = sanitize_filename_step_16(str(item.get("id") or "item"))
    dest_dir = (out_dir / doc_type).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    if not dest_dir.is_relative_to(root):
        raise ValueError("输出路径越出目录")
    license_text = str(item.get("license") or "")
    # 正文预算 = 上传上限 - YAML/标题前缀 - 一点余量
    overhead = prefix_size_step_17(
        title=book,
        chapter=book,
        doc_type=doc_type,
        source_url=source_url,
        license_text=license_text,
        fetched_at=fetched_at,
    )
    budget = max(64, int(max_bytes) - overhead - 32)
    parts = split_markdown_chapters_step_20(book, markdown, max_bytes=budget)
    written: list[Path] = []
    for chapter, body in parts:
        rendered = render_doc_step_18(
            title=book,
            chapter=chapter,
            doc_type=doc_type,
            source_url=source_url,
            license_text=license_text,
            fetched_at=fetched_at,
            body=body,
        )
        encoded = rendered.encode("utf-8")
        if len(encoded) > max_bytes:
            # 加上前缀后仍超限：按实际章节名重算前缀，再走第 21 步（必要时第 22 步硬切）
            extra = prefix_size_step_17(
                title=book,
                chapter=chapter,
                doc_type=doc_type,
                source_url=source_url,
                license_text=license_text,
                fetched_at=fetched_at,
            )
            for sub_ch, sub_body in _split_by_size_step_21(chapter, body, max(64, max_bytes - extra - 32)):
                rendered = render_doc_step_18(
                    title=book,
                    chapter=sub_ch,
                    doc_type=doc_type,
                    source_url=source_url,
                    license_text=license_text,
                    fetched_at=fetched_at,
                    body=sub_body,
                )
                written.append(
                    _write_md_step_23(dest_dir, item_id, book, sub_ch, rendered, max_bytes)
                )
            continue
        written.append(_write_md_step_23(dest_dir, item_id, book, chapter, rendered, max_bytes))
    return written


def _write_md_step_23(
    dest_dir: Path,
    item_id: str,
    book: str,
    chapter: str,
    rendered: str,
    max_bytes: int,
) -> Path:
    """第 23 步：写成 {id}_{书名}_{章节}.md，文件名带 id 以免同书覆盖。"""
    encoded = rendered.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"拆分后仍超过 {max_bytes} 字节: {chapter}")
    fname = f"{item_id}_{sanitize_filename_step_16(book)}_{sanitize_filename_step_16(chapter)}.md"
    path = (dest_dir / fname).resolve()
    if not path.is_relative_to(dest_dir.resolve()):
        raise ValueError("非法文件名")
    path.write_text(rendered, encoding="utf-8")
    return path


def crawl_item_step_6(
    item: dict[str, Any],
    session: requests.Session,
    out_dir: Path,
    max_bytes: int,
) -> list[Path]:
    """第 6 步：按 source 走维基或 ctext，再落盘拆好的 Markdown。"""
    source = str(item.get("source") or "")
    if source == "wikisource":
        page = str(item.get("page") or "").strip()
        if not page:
            raise ValueError(f"{item.get('id')} 缺少 page")
        _title, md, url = fetch_wikisource_step_7(session, page)
    elif source == "ctext":
        urn = str(item.get("urn") or "").strip()
        if not urn:
            raise ValueError(f"{item.get('id')} 缺少 urn")
        _title, md, url = fetch_ctext_step_8(session, urn)
    else:
        raise ValueError(f"未知 source: {source}")
    fetched_at = datetime.now(timezone.utc).isoformat()
    return write_parts_step_15(
        item=item,
        chapter_title=str(item.get("title") or _title),
        markdown=md,
        source_url=url,
        out_dir=out_dir,
        max_bytes=max_bytes,
        fetched_at=fetched_at,
    )


def main_step_1() -> None:
    """第 1 步：解析参数，过滤已完成/已跳过条目，再并发抓取。"""
    ap = argparse.ArgumentParser(description="公开中医文献 → 知识库 Markdown")
    ap.add_argument("--sources", default=str(DEFAULT_SOURCES), help="白名单 JSON")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="输出目录")
    ap.add_argument("--limit", type=int, default=None, help="仅抓取前 N 条（跨类型）")
    ap.add_argument("--per-type", type=int, default=None, help="每种 docType 最多抓取 N 条")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数，上限 4")
    ap.add_argument("--force", action="store_true", help="忽略断点，覆盖已抓条目")
    args = ap.parse_args()

    sources_path = Path(args.sources)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_state.json"
    error_path = out_dir / "_errors.jsonl"
    items = load_sources_step_2(sources_path)
    # --force 清空断点，否则跳过 done / skipped
    state = {"done": [], "skipped": []} if args.force else load_state_step_3(state_path)
    done = set(state.get("done") or [])
    skipped = set(state.get("skipped") or [])
    todo = [it for it in items if str(it["id"]) not in done and str(it["id"]) not in skipped]
    if args.per_type is not None:
        if args.per_type < 0:
            raise SystemExit("--per-type 不能为负数")
        counts: dict[str, int] = {}
        capped: list[dict[str, Any]] = []
        for it in todo:
            doc_type = str(it.get("docType") or "其他")
            n = counts.get(doc_type, 0)
            if n >= args.per_type:
                continue
            counts[doc_type] = n + 1
            capped.append(it)
        todo = capped
    if args.limit:
        todo = todo[: args.limit]
    max_bytes = upload_max_bytes_step_4()
    workers = min(MAX_WORKERS, max(1, int(args.workers)))

    type_n: dict[str, int] = {}
    for it in todo:
        dt = str(it.get("docType") or "其他")
        type_n[dt] = type_n.get(dt, 0) + 1
    type_hint = "，".join(f"{k} {v}" for k, v in type_n.items()) or "无"
    print(f"[1/3] 书目 {len(items)} 条，本次 {len(todo)} 条（{type_hint}），输出 {out_dir}")
    if not todo:
        print("      无待抓取（已在 state 的 done/skipped 中）。")
        print("      请到知识库页按类型分批上传（每次最多 5 个 .md）。")
        return

    def worker_step_5(item: dict[str, Any]) -> None:
        """第 5 步：抓一条；成功写入 done，需登录写入 skipped，其它失败只记错误。"""
        item_id = str(item["id"])
        local = requests.Session()
        local.headers.update(HEADERS)
        try:
            paths = crawl_item_step_6(item, local, out_dir, max_bytes)
            with _lock:
                done.add(item_id)
                skipped.discard(item_id)
                state["done"] = sorted(done)
                state["skipped"] = sorted(skipped)
                save_state_step_24(state_path, state)
                _stats["ok"] += 1
            print(f"      [OK] {item.get('title')} -> {len(paths)} 文件")
        except AuthRequiredError as exc:
            append_error_step_25(error_path, item_id, exc)
            with _lock:
                skipped.add(item_id)
                state["done"] = sorted(done)
                state["skipped"] = sorted(skipped)
                save_state_step_24(state_path, state)
                _stats["skip"] += 1
            print(f"      [SKIP] {item.get('title')} 需要 ctext 登录，已跳过")
        except Exception as exc:  # noqa: BLE001
            # 失败不进 done，下次可重试
            append_error_step_25(error_path, item_id, exc)
            with _lock:
                _stats["err"] += 1
            print(f"      [ERR] {item.get('title')} -> {type(exc).__name__}: {exc}")

    print(f"[2/3] workers={workers} interval={REQUEST_INTERVAL}s")
    if workers == 1:
        for it in todo:
            worker_step_5(it)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(worker_step_5, it) for it in todo]
            for fut in as_completed(futs):
                fut.result()

    print(
        f"[3/3] 成功 {_stats['ok']}，跳过 {_stats['skip']}，失败 {_stats['err']}。"
        "请到知识库页按类型分批上传（每次最多 5 个 .md）。"
    )


if __name__ == "__main__":
    main_step_1()
