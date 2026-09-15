#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 data/kb_crawl 按类型原样摘录评测语料到 corpus/。

步骤：01 入口。只截取源文件连续字符，不改写医理。
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB = ROOT / "data" / "kb_crawl"
OUT = ROOT / "corpus"
PREVIEW = ROOT / "eval" / "_excerpt_preview.json"

# 每类：目标文件名关键字、必须出现在摘录中的原文（繁体优先）
SPECS = (
    {
        "doc_type": "方剂",
        "out_name": "fangji_tangtou.md",
        "file_key": "tangtou-fabiao",
        "needles": ("桂枝湯", "麻黃湯"),
        "title": "湯頭歌訣·發表之劑（摘錄）",
    },
    {
        "doc_type": "本草",
        "out_name": "bencao_shennong.md",
        "file_key": "shennongbencaojing",
        "needles": ("人參", "黃芪"),
        "title": "神農本草經（摘錄）",
    },
    {
        "doc_type": "典籍",
        "out_name": "dianji_shanghanlun.md",
        "file_key": "shanghanlun_",
        "needles": ("小柴胡湯", "往來寒熱"),
        "title": "傷寒論（摘錄）",
    },
    {
        "doc_type": "医案",
        "out_name": "yian_linzheng.md",
        "file_key": "linzheng-juan2",
        "needles": ("四君子湯",),
        "title": "臨證指南醫案（摘錄）",
    },
    {
        "doc_type": "其他",
        "out_name": "qita_piweilun.md",
        "file_key": "piweilun-shang",
        "needles": ("脾胃", "四君子湯"),
        "title": "脾胃論（摘錄）",
    },
)


def find_source_step_01(file_key: str) -> Path:
    """步骤 01：按文件名关键字定位 kb_crawl 源文件。"""
    hits = [p for p in KB.rglob("*.md") if file_key in p.name]
    if not hits:
        raise FileNotFoundError(f"找不到源文件: {file_key}")
    return sorted(hits, key=lambda p: len(p.name))[0]


def strip_frontmatter_step_02(text: str) -> tuple[str, str]:
    """步骤 02：剥 YAML 头，返回 (正文, 原 license 行)。"""
    license_line = "原文公有领域；维基文库整理 CC BY-SA 4.0"
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            head = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            for line in head.splitlines():
                if line.strip().startswith("license:"):
                    license_line = line.split(":", 1)[1].strip().strip('"').strip("'")
            return body, license_line
    return text, license_line


def pick_window_step_03(body: str, needles: tuple[str, ...], target: int = 1200) -> str:
    """步骤 03：取覆盖全部关键字的最短连续段，再扩到约 target 字，尽量落在换行处。"""
    loc_lists: list[list[int]] = []
    for needle in needles:
        found = [m.start() for m in re.finditer(re.escape(needle), body)]
        if not found:
            raise ValueError(f"源文不含关键字: {needle}")
        loc_lists.append(found)
    # 在各关键字出现位置中选覆盖跨度最短的一组
    best: tuple[int, int] | None = None
    for combo in itertools.product(*loc_lists):
        starts = list(combo)
        ends = [s + len(n) for s, n in zip(starts, needles)]
        span = (min(starts), max(ends))
        if best is None or (span[1] - span[0]) < (best[1] - best[0]):
            best = span
    assert best is not None
    if best[1] - best[0] > 1600:
        # 跨度过大则只保留距离最近的两个关键字
        pairs = []
        for i, a in enumerate(loc_lists):
            for j, b in enumerate(loc_lists):
                if j <= i:
                    continue
                for x in a:
                    for y in b:
                        lo, hi = (x, y + len(needles[j])) if x <= y else (y, x + len(needles[i]))
                        pairs.append((hi - lo, lo, hi))
        pairs.sort()
        best = (pairs[0][1], pairs[0][2])
    start, end = best
    span = end - start
    pad = max(0, (target - span) // 2)
    start = max(0, start - pad)
    end = min(len(body), end + pad)
    nl = body.rfind("\n", 0, start + 1)
    if nl >= 0 and start - nl < 200:
        start = nl + 1
    nl2 = body.find("\n", end)
    if nl2 >= 0 and nl2 - end < 200:
        end = nl2
    excerpt = body[start:end].strip()
    if len(excerpt) < 400:
        raise ValueError(f"摘录过短: {len(excerpt)}")
    if len(excerpt) > 1800:
        excerpt = excerpt[:1800]
        cut = excerpt.rfind("\n")
        if cut > 800:
            excerpt = excerpt[:cut].strip()
    covered = [n for n in needles if n in excerpt]
    if not covered:
        raise ValueError("扩窗后未覆盖任何关键字")
    return excerpt


def write_excerpt_step_04(spec: dict, source: Path, excerpt: str, license_line: str) -> Path:
    """步骤 04：写入 corpus 文件，头部只注明来源，正文为原文连续段。"""
    rel = source.relative_to(KB).as_posix()
    header = (
        f"# {spec['title']}\n\n"
        f"> 来源：`data/kb_crawl/{rel}`  \n"
        f"> 许可：{license_line}  \n"
        f"> 说明：评测用连续摘录，未改写原文。\n\n"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / spec["out_name"]
    path.write_text(header + excerpt.strip() + "\n", encoding="utf-8")
    return path


def main() -> int:
    """步骤：05 按 SPECS 摘录五类文献。"""
    if not KB.is_dir():
        print(f"缺少 {KB}，无法摘录", file=sys.stderr)
        return 1
    preview = []
    for spec in SPECS:
        source = find_source_step_01(spec["file_key"])
        raw = source.read_text(encoding="utf-8")
        body, license_line = strip_frontmatter_step_02(raw)
        excerpt = pick_window_step_03(body, spec["needles"])
        path = write_excerpt_step_04(spec, source, excerpt, license_line)
        preview.append(
            {
                "doc_type": spec["doc_type"],
                "out": path.name,
                "source": source.name,
                "chars": len(excerpt),
                "needles": list(spec["needles"]),
                "covered": [n for n in spec["needles"] if n in excerpt],
                "excerpt": excerpt,
            }
        )
        print(f"OK {spec['doc_type']} -> {path.name} ({len(excerpt)} 字) from {source.name}")
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.write_text(json.dumps(preview, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
