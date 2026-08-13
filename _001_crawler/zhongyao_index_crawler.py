#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中药索引爬虫

抓取中医百科「中药大全」索引页，提取所有中药的【名称】与【URL】，写入 Excel。

特性:
- 多线程: ThreadPoolExecutor，默认 5 个 worker
- 断点续爬: state.json 记录已完成条目，重跑自动跳过
- 测试模式: --limit N 仅抓取前 N 条

用法:
  python zhongyao_index_crawler.py --limit 3             # 测试 3 条
  python zhongyao_index_crawler.py --verify --limit 3    # 测试 3 条并校验详情页可达性
  python zhongyao_index_crawler.py                       # 全量抓取（断点续爬）
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

INDEX_URL = "https://zhongyibaike.com/wiki/%E4%B8%AD%E8%8D%AF%E5%A4%A7%E5%85%A8"
BASE_URL = "https://zhongyibaike.com"
DEFAULT_WORKERS = 5
REQUEST_INTERVAL = 0.1  # 秒，线程内限速

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "data"  # 爬虫数据统一存项目根 data/（已 gitignore）
STATE_FILE = OUTPUT_DIR / "zhongyao_state.json"
EXCEL_FILE = OUTPUT_DIR / "zhongyao_index.xlsx"

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

    页面结构（#content 内单个 div.col.s12.p_content 承载整页索引）:
        <div class="col s12 p_content">
          <h2>解表药</h2>                <- 中药大类
          <ol>                            <- 含 strong 功效分类
            <li><strong>发散风寒药</strong>   <- 功效分类
              <ul><li><a href="/wiki/羌活">羌活</a></li> ...</ul>
            </li>
          </ol>
          <h2>温里药</h2>
          <ul>                            <- 无 strong 分类，直接列药
            <li><a href="/wiki/附子">附子</a></li> ...
          </ul>
        </div>

    两类结构在本页混合出现：ol 内每项含 <strong>功效分类>，ul 内无分类
    （功效分类留空）。解析时统一按 大类/分类 两级输出，便于后续入库。
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#content")
    if content is None:
        raise RuntimeError("页面结构变化：未找到 #content 区域")

    items, seen = [], set()
    major = ""  # 当前中药大类（由 h2 决定，作用到其后的 ol/ul）
    for el in content.find_all(["h2", "ol", "ul"]):
        if el.name == "h2":
            major = el.get_text(strip=True)
            continue
        if el.name == "ol":
            # ol：每个 li 为「strong 功效分类 + ul 药名列表」
            for li in el.find_all("li", recursive=False):
                strong = li.find("strong")
                cat = strong.get_text(strip=True) if strong else ""
                for a in li.select("a[href^='/wiki/']"):
                    _append(items, seen, major, cat, a)
        else:  # ul：无 strong，li 直接含药名链接
            for a in el.select("li a[href^='/wiki/']"):
                _append(items, seen, major, "", a)
    return items


def _append(items: list[dict], seen: set[str], major: str, cat: str, a) -> None:
    """按名称去重后追加一条记录（保持页面顺序，保留首次出现）。"""
    name = a.get_text(strip=True)
    href = a.get("href", "")
    if not name or name in seen:
        return
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
    ap = argparse.ArgumentParser(description="中药索引爬虫")
    ap.add_argument("--limit", type=int, default=None, help="仅抓取前 N 条（测试用）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="线程数")
    ap.add_argument("--verify", action="store_true", help="逐条校验详情页可达性")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] 抓取索引页 ...")
    items = parse_index(fetch_text(INDEX_URL))
    print(f"      索引页共 {len(items)} 条中药")

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
            "中药大类": it["major"],
            "功效分类": it["cat"],
            "中药名称": it["name"],
            "中药url": it["url"],
            "相对链接": it["rel"],
        }
        for it in items
        if it["name"] in done
    ]
    df = pd.DataFrame(rows).fillna("")  # 无功效分类的 ul 段统一留空串，避免 Excel 空单元格读出 NaN
    df.to_excel(EXCEL_FILE, index=False)
    print(f"[3/3] 写入 {EXCEL_FILE}，共 {len(df)} 行")

    print("\n===== 抓取结果预览 =====")
    print(df.to_string(index=False))
    if args.verify:
        print(f"\n状态码分布: {pd.Series([str(r.get('status')) for r in results]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
