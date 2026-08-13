#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回归实验：在评估数据集上跑问答，量化评分对比。

用法:
    python common/run_experiment.py                # 在 bad-cases 数据集跑实验
    python common/run_experiment.py --dataset bad-cases-v2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(HERE / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from common.eval_manager import EvalManager  # noqa: E402

# 全局图（懒加载，复用）
_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        from _004_langgraph_more_nodes.graph import build_graph
        _graph = build_graph()
    return _graph


def task_fn(item_input: Any) -> dict[str, Any]:
    """任务：跑图问答，返回最终 state。"""
    question = item_input.get("question", "") if isinstance(item_input, dict) else str(item_input)
    graph = _get_graph()
    initial = {"user_question": question}
    return graph.invoke(initial)


def eval_fn(trace_id: str, output: dict[str, Any], expected_output: Any) -> None:
    """评分：启发式评估输出质量。"""
    evm = EvalManager()

    # 1) 是否有回答
    answer = output.get("final_answer", "") or ""
    evm.score_trace(trace_id, "answer_nonempty", 1 if answer else 0)

    # 2) 是否生成 Cypher（中医问题应生成）
    cypher = output.get("cypher_queries", []) or []
    evm.score_trace(trace_id, "cypher_generated", 1 if cypher else 0)

    # 3) 回答长度合理性（避免过短）
    answer_score = 1 if len(answer) >= 10 else (0 if answer else 0.5)
    evm.score_trace(trace_id, "answer_length_ok", answer_score)


def main() -> None:
    ap = argparse.ArgumentParser(description="在评估数据集上跑回归实验")
    ap.add_argument("--dataset", default="bad-cases", help="数据集名")
    args = ap.parse_args()

    evm = EvalManager()
    if not evm.is_available():
        print("Langfuse 不可用，请检查 common/.env 配置")
        return

    print(f"在数据集 '{args.dataset}' 上跑实验...")
    results = evm.run_experiment(
        dataset_name=args.dataset,
        task_fn=task_fn,
        eval_fn=eval_fn,
    )

    print(f"\n实验完成：共 {len(results)} 条")
    print("在 Langfuse UI → Experiments 查看评分对比")
    print("对比指标：answer_nonempty / cypher_generated / answer_length_ok")


if __name__ == "__main__":
    main()
