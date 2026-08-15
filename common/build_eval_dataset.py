#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据回流脚本：从 Langfuse 拉取 Bad Case（强制采样 trace）构建评估数据集。

用法:
    python common/build_eval_dataset.py            # 拉取全部 forced-sampling trace
    python common/build_eval_dataset.py --name bad-cases-v2  # 指定数据集名
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(HERE / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from common.eval_manager import EvalManager  # noqa: E402
from common.sanitizer import sanitize  # noqa: E402


def fetch_bad_traces(evm: EvalManager, limit: int = 200) -> list:
    """从 Langfuse 拉取强制采样（Bad Case）的 trace。"""
    if not evm.is_available():
        return []
    traces = []
    try:
        resp = evm.client.fetch_traces(tags="forced-sampling", limit=limit)
        traces = list(resp.data)
    except Exception as exc:
        logging.warning("拉取 trace 失败: %s", exc)
    return traces


def extract_question(trace) -> str | None:
    """从 trace 的 input 提取用户问题。"""
    inp = getattr(trace, "input", None)
    if isinstance(inp, dict):
        return inp.get("user_question") or inp.get("question")
    if isinstance(inp, str):
        return inp
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="从 Langfuse 拉 Bad Case 建评估数据集")
    ap.add_argument("--name", default="bad-cases", help="数据集名")
    ap.add_argument("--limit", type=int, default=100, help="最多拉取 trace 数")
    args = ap.parse_args()

    evm = EvalManager()
    if not evm.is_available():
        print("Langfuse 不可用，请检查 common/.env 的密钥配置")
        return

    # 1) 拉取 Bad Case
    print(f"[1/3] 拉取 forced-sampling trace（最多 {args.limit} 条）...")
    traces = fetch_bad_traces(evm, limit=args.limit)
    print(f"      共 {len(traces)} 条 Bad Case")

    if not traces:
        print("      无 Bad Case，退出")
        return

    # 2) 清洗 + 构建数据集
    print(f"[2/3] 构建数据集 '{args.name}'...")
    evm.create_dataset(args.name, description="自动筛选的 Bad Case 评估集")

    added = 0
    skipped = 0
    for t in traces:
        question = extract_question(t)
        if not question:
            skipped += 1
            continue
        evm.add_dataset_item(
            dataset_name=args.name,
            input={"question": sanitize(question)},
            source_trace_id=getattr(t, "id", None),
            metadata={"source": "forced-sampling"},
        )
        added += 1

    # 3) flush
    evm.flush()
    print(f"[3/3] 完成：添加 {added} 条，跳过 {skipped} 条（无问题字段）")
    print(f"      在 Langfuse UI → Datasets 查看 '{args.name}'")


if __name__ == "__main__":
    main()
