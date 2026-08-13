#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方剂索引爬虫

抓取中医百科「中医方剂」索引页，提取所有方剂的【名称】与【URL】，写入 Excel。

特性:
- 多线程: ThreadPoolExecutor，默认 5 个 worker
- 断点续爬: state.json 记录已完成条目，重跑自动跳过
- 测试模式: --limit N 仅抓取前 N 条

用法:
  python fangji_index_crawler.py --limit 2            # 测试 2 条
  python fangji_index_crawler.py --verify --limit 2   # 测试 2 条并校验详情页可达性
  python fangji_index_crawler.py                      # 全量抓取（断点续爬）
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://zhongyibaike.com/wiki/%E4%B8%AD%E5%8C%BB%E6%96%B9%E5%89%82"
BASE_URL = "https://zhongyibaike.com"
DEFAULT_WORKERS = 5
REQUEST_INTERVAL = 0.1  # 秒，线程内限速

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "data"  # 爬虫数据统一存项目根 data/（已 gitignore）
STATE_FILE = OUTPUT_DIR / "fangji_state.json"
EXCEL_FILE = OUTPUT_DIR / "fangji_index.xlsx"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

_lock = threading.Lock()


def fetch_text(url: str) -> str:
    """GET 请求并返回文本，编码自动探测。"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_index(html: str) -> list[dict]:
    """解析索引页，返回 [{major, cat, name, url, rel}]，按页面顺序、按名称去重。

    页面结构:
        #content
          <h2>解表剂</h2>                  <- 方剂大类
          <ol>
            <li><strong>辛温解表</strong>   <- 功效分类
              <ul><li><a href="/wiki/麻黄汤">麻黄汤</a></li> ...</ul>
            </li>
          </ol>
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#content")
    if content is None:
        raise RuntimeError("页面结构变化：未找到 #content 区域")

    items, seen = [], set()
    major = ""  # 当前方剂大类（由 h2 决定，作用到其后的 ol）
    for el in content.find_all(["h2", "ol"]):
        if el.name == "h2":
            major = el.get_text(strip=True)
            continue
        for li in el.find_all("li", recursive=False):
            strong = li.find("strong")
            cat = strong.get_text(strip=True) if strong else ""
            for a in li.select("a[href^='/wiki/']"):
                name = a.get_text(strip=True)
                href = a.get("href", "")
                if not name or name in seen:
                    continue
                seen.add(name)
                items.append(
                    {
                        "major": major,
                        "cat": cat,
                        "name": name,
                        "url": href if href.startswith("http") else BASE_URL + href,
                        "rel": href,
                    }
                )
    return items


def verify_url(url: str) -> int:
    """校验详情页是否可达，返回 HTTP 状态码。"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    return resp.status_code


def load_state() -> set[str]:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text(encoding="utf-8")))
    return set()


def save_state(done: set[str]) -> None:
    STATE_FILE.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="方剂索引爬虫")
    ap.add_argument("--limit", type=int, default=None, help="仅抓取前 N 条（测试用）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="线程数")
    ap.add_argument("--verify", action="store_true", help="逐条校验详情页可达性")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] 抓取索引页 ...")
    items = parse_index(fetch_text(INDEX_URL))
    print(f"      索引页共 {len(items)} 条方剂")

    done = load_state()
    todo = [it for it in items if it["name"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[2/3] 本次待抓取 {len(todo)} 条，线程数 {args.workers}")

    def worker(item: dict) -> dict:
        time.sleep(REQUEST_INTERVAL)  # 线程内限速，避免请求过快
        if args.verify:
            try:
                item["status"] = verify_url(item["url"])
            except Exception as exc:  # noqa: BLE001
                item["status"] = f"ERR:{exc.__class__.__name__}"
        with _lock:
            done.add(item["name"])
        return item

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(worker, todo):
            results.append(res)
            if len(results) % 10 == 0:
                print(f"      已处理 {len(results)}/{len(todo)}")
                save_state(done)

    save_state(done)

    # Excel 仅输出本次已完成条目（按索引页顺序）
    rows = [
        {
            "方剂大类": it["major"],
            "功效分类": it["cat"],
            "方剂名称": it["name"],
            "方剂url": it["url"],
            "相对链接": it["rel"],
        }
        for it in items
        if it["name"] in done
    ]
    df = pd.DataFrame(rows)
    df.to_excel(EXCEL_FILE, index=False)
    print(f"[3/3] 写入 {EXCEL_FILE}，共 {len(df)} 行")

    print("\n===== 抓取结果预览 =====")
    print(df.to_string(index=False))
    if args.verify:
        print(f"\n状态码分布: {pd.Series([str(r.get('status')) for r in results]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
