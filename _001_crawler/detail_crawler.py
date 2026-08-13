#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""详情页结构化爬虫

读取索引 Excel（方剂/中药）中的链接，抓取详情页，解析为结构化 JSON 保存。
每个详情页一个 .json 文件，保留 h2/h3 标题层级与段落内实体（wiki 内链），
为后续知识抽取与图谱构建奠定数据基础。

特性:
- 多线程: ThreadPoolExecutor，默认 5 个 worker
- 断点续爬: 已存在且非空的 .json 自动跳过，重跑无需担心覆盖
- 测试模式: --limit N 仅抓取前 N 条

JSON 结构:
  {
    "name": "羌活",          # 条目名称
    "type": "中药",           # 中药/方剂
    "url": "https://...",     # 详情页绝对链接
    "rel": "/wiki/羌活",      # 相对链接
    "meta": {"中药大类": "解表药", "功效分类": "发散风寒药"},  # Excel 其他列
    "sections": [             # h2 大节
      {"heading": "羌活的种植和炮制", "level": 2,
       "children": [           # h3 小节（字段名）
         {"heading": "来源", "level": 3,
          "blocks": [{"tag": "p", "text": "...", "entities": ["..."]}]}
       ]}
    ]
  }

用法:
  # 中药（主 Agent 负责）
  python detail_crawler.py --excel ../data/zhongyao_index.xlsx --out-dir ../data/中药 \
      --name-col 中药名称 --url-col 中药url --rel-col 相对链接 --type 中药 --limit 2

  # 方剂（子 Agent 负责）
  python detail_crawler.py --excel ../data/fangji_index.xlsx --out-dir ../data/方剂 \
      --name-col 方剂名称 --url-col 方剂url --rel-col 相对链接 --type 方剂 --limit 2

  # 全量（断点续爬）
  python detail_crawler.py --excel ../data/zhongyao_index.xlsx --out-dir ../data/中药 \
      --name-col 中药名称 --url-col 中药url --rel-col 相对链接 --type 中药
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://zhongyibaike.com"
DEFAULT_WORKERS = 5
REQUEST_INTERVAL = 0.15  # 秒，线程内限速
NOISE_RE = re.compile(r"^[\d\s#]+$")  # 纯数字/# 的模板残留段

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_lock = threading.Lock()
_stats = {"ok": 0, "err": 0}


def sanitize_filename(name: str) -> str:
    """清理文件名中 Windows 不允许的字符。"""
    return re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "unnamed"


def fetch_text(url: str) -> str:
    """GET 请求并返回文本，编码自动探测。"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def _extract_block(el) -> dict | None:
    """将单个内容元素（p/ul/ol/table 等）转为 block dict；噪点段返回 None。"""
    text = el.get_text("\n", strip=True)
    if not text or NOISE_RE.match(text):
        return None
    entities = [
        a.get_text(strip=True)
        for a in el.select("a[href^='/wiki/']")
        if a.get_text(strip=True)
    ]
    return {"tag": el.name, "text": text, "entities": entities}


def parse_detail(html: str, name: str, url: str, rel: str, meta: dict, type_label: str) -> dict:
    """解析详情页为结构化 JSON dict。

    页面结构（#content > div.col.s12.p_content > div.card-panel）:
        div.card-panel
          h2 '羌活的种植和炮制'        <- 大节标题
          h3 '来源'                    <- 小节标题（字段名）
          p '为伞形科植物羌活...'       <- 内容（可含 <a href="/wiki/..."> 实体）
          ul > li '①《药性论》：...'    <- 列表内容
    不同条目 panel 数量与 h2/h3 布局不固定，按「标题驱动」通用解析。
    """
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one("#content div.col.s12.p_content")
    if node is None:
        node = soup.select_one("#content")

    doc = {"name": name, "type": type_label, "url": url, "rel": rel,
           "meta": meta, "sections": []}
    cur_sec = None   # 当前 h2 大节
    cur_sub = None   # 当前 h3 小节

    for el in node.find_all(recursive=False):
        if el.get("class") and "card-panel" not in el.get("class"):
            continue  # 跳过 p_content 下的广告/脚本/非内容块
        for tag in el.find_all(["script", "style", "ins"]):
            tag.decompose()

        # 按标题元素拆分层级
        for child in el.find_all(recursive=False):
            if child.name in ("h1", "h2"):
                cur_sec = {"heading": child.get_text(strip=True), "level": 2,
                           "children": [], "blocks": []}
                doc["sections"].append(cur_sec)
                cur_sub = None
            elif child.name in ("h3", "h4"):
                if cur_sec is None:
                    cur_sec = {"heading": "", "level": 2, "children": [], "blocks": []}
                    doc["sections"].append(cur_sec)
                cur_sub = {"heading": child.get_text(strip=True), "level": 3, "blocks": []}
                cur_sec["children"].append(cur_sub)
            else:
                # 内容元素：追加到当前小节；无小节则追加到大节；都无则建匿名大节
                block = _extract_block(child)
                if block is None:
                    continue
                if cur_sub is not None:
                    cur_sub["blocks"].append(block)
                elif cur_sec is not None:
                    cur_sec["blocks"].append(block)
                else:
                    cur_sec = {"heading": "", "level": 2, "children": [], "blocks": [block]}
                    doc["sections"].append(cur_sec)
    return doc


def load_excel(path: Path, name_col: str, url_col: str, rel_col: str | None) -> list[dict]:
    df = pd.read_excel(path)
    exclude = {name_col, url_col}
    if rel_col:
        exclude.add(rel_col)
    meta_cols = [c for c in df.columns if c not in exclude]
    items = []
    for _, row in df.iterrows():
        name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        url = str(row[url_col]).strip() if pd.notna(row[url_col]) else ""
        if not name or not url:
            continue
        meta = {}
        for c in meta_cols:
            v = row[c]
            if pd.notna(v):
                meta[c] = str(v).strip()
        rel = str(row[rel_col]).strip() if rel_col and pd.notna(row[rel_col]) else ""
        items.append({"name": name, "url": url, "rel": rel, "meta": meta})
    return items


def crawl_one(item: dict, out_dir: Path, type_label: str) -> None:
    """抓取单条详情并写入 {out_dir}/{name}.json。"""
    url = item["url"]
    if not url.startswith("http"):
        url = BASE_URL + url
    fpath = out_dir / f"{sanitize_filename(item['name'])}.json"
    try:
        doc = parse_detail(fetch_text(url), item["name"], url,
                           item.get("rel", ""), item.get("meta", {}), type_label)
        fpath.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with _lock:
            _stats["ok"] += 1
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _stats["err"] += 1
            print(f"      [ERR] {item['name']} -> {exc.__class__.__name__}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="详情页结构化爬虫")
    ap.add_argument("--excel", required=True, help="索引 Excel 路径")
    ap.add_argument("--out-dir", required=True, help="json 输出目录")
    ap.add_argument("--name-col", required=True, help="Excel 中的名称列")
    ap.add_argument("--url-col", required=True, help="Excel 中的链接列")
    ap.add_argument("--rel-col", default=None, help="Excel 中的相对链接列（可选）")
    ap.add_argument("--type", default="条目", help="条目类型：中药/方剂")
    ap.add_argument("--limit", type=int, default=None, help="仅抓取前 N 条（测试用）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="线程数")
    args = ap.parse_args()

    excel = Path(args.excel)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 读取 {excel} ...")
    all_items = load_excel(excel, args.name_col, args.url_col, args.rel_col)
    print(f"      Excel 共 {len(all_items)} 条")

    # 断点续爬：已存在且非空的 json 视为完成
    todo = [
        it for it in all_items
        if not (out_dir / f"{sanitize_filename(it['name'])}.json").exists()
        or (out_dir / f"{sanitize_filename(it['name'])}.json").stat().st_size == 0
    ]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[2/3] 本次待抓取 {len(todo)} 条，线程数 {args.workers}，输出目录 {out_dir}")

    if todo:
        def worker(item: dict) -> None:
            time.sleep(REQUEST_INTERVAL)  # 线程内限速
            crawl_one(item, out_dir, args.type)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, todo))

    done_files = list(out_dir.glob("*.json"))
    print(f"[3/3] 完成。本次成功 {_stats['ok']}，失败 {_stats['err']}，目录内共 {len(done_files)} 个 json")

    print("\n===== 输出目录样例 =====")
    for f in sorted(done_files)[:3]:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
