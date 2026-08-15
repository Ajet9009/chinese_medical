#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据飞轮完整演示脚本：制造 Bad Case → 回流 → 基线实验 → 改 prompt → 优化实验对比。

用法:
    python _000_demo/demo_flywheel.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "common" / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from common.langfuse_manager import LangfuseManager  # noqa: E402
from common.eval_manager import EvalManager  # noqa: E402

# 制造 8 条不同领域的 Bad Case（慢请求/重试/低分场景）
BAD_CASES = [
    ("四君子汤有什么功效？", "slow_request", 35000),
    ("人参的性味归经是什么？", "slow_request", 42000),
    ("咳嗽应该吃什么中药？", "cypher_retry", 8000),
    ("肾虚用什么方剂调理？", "no_cypher", 5000),
    ("黄芪和党参有什么区别？", "slow_request", 31000),
    ("感冒了怎么辨证？", "empty_answer", 6000),
    ("补气养血吃什么好？", "cypher_retry", 7500),
    ("肚子疼吃什么药？", "slow_request", 38000),
]


def create_bad_cases() -> None:
    """制造 Bad Case（强制采样 trace）。"""
    mgr = LangfuseManager()
    for i, (question, reason, elapsed) in enumerate(BAD_CASES):
        mgr.record_summary_span(
            "POST:/ask",
            {"session_id": f"demo-{i}", "user_id": "anonymous"},
            {"reason": reason, "question": question, "elapsed_ms": elapsed},
        )
    mgr.flush(timeout=10.0)
    print(f"  制造 {len(BAD_CASES)} 条 Bad Case")


def run_experiment(run_label: str) -> None:
    """跑一次实验（复用 run_experiment 的 task_fn + eval_fn）。"""
    from _004_langgraph_more_nodes.graph import build_graph

    graph = build_graph()

    def task_fn(item_input):
        question = item_input.get("question", "") if isinstance(item_input, dict) else str(item_input)
        return graph.invoke({"user_question": question})

    def eval_fn(trace_id, output, expected_output):
        evm = EvalManager()
        answer = output.get("final_answer", "") or ""
        cypher = output.get("cypher_queries", []) or []
        evm.score_trace(trace_id, "answer_nonempty", 1 if answer else 0)
        evm.score_trace(trace_id, "cypher_generated", 1 if cypher else 0)
        # 回答质量：长度归一化（>100 字满分，体现优化效果）
        quality = min(1.0, len(answer) / 100.0)
        evm.score_trace(trace_id, "answer_quality", round(quality, 2))

    evm = EvalManager()
    print(f"\n  [{run_label}] 跑实验中...")
    run_name = "baseline" if "基线" in run_label else "optimized"
    results = evm.run_experiment("bad-cases", task_fn, eval_fn, run_name=run_name)
    print(f"  [{run_label}] 完成，共 {len(results)} 条")


def update_prompt(version: str) -> None:
    """更新 answer_generation prompt（模拟优化）。"""
    import os
    from langfuse import Langfuse

    client = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )

    prompt = """你是专业的中医知识助手。根据知识图谱查询结果回答用户问题。

## 要求
1. 基于提供的图谱数据回答，数据中没有的信息请说明"暂未查到"
2. 使用中医术语，如症状、方剂、药材、功效、经络、辨证论治、典籍等
3. 回答要**详细完整**，尽量覆盖功效、主治、组成、用法等多个维度
4. 只输出最终答案，不要解释推理过程"""

    client.create_prompt(name="answer_generation", type="text", prompt=prompt, labels=["production"])
    client.flush()
    print(f"  prompt 已更新（{version}）")


def main() -> None:
    print("=" * 60)
    print("  数据飞轮完整演示")
    print("=" * 60)

    # 1. 制造 Bad Case
    print("\n[1/5] 制造 Bad Case")
    create_bad_cases()

    # 2. 数据回流
    print("\n[2/5] 数据回流（构建数据集）")
    from common.build_eval_dataset import main as backflow_main
    backflow_main()

    # 3. 基线实验（当前 prompt）
    print("\n[3/5] 基线实验（Run #1）")
    run_experiment("基线")

    # 4. 优化 prompt
    print("\n[4/5] 优化 prompt")
    update_prompt("优化版：更详细")

    # 5. 优化实验（新 prompt）
    print("\n[5/5] 优化实验（Run #2）")
    run_experiment("优化")

    print("\n" + "=" * 60)
    print("  演示完成。Langfuse UI → Datasets → bad-cases → Runs")
    print("  对比两个 run 的 answer_quality 分数")
    print("=" * 60)


if __name__ == "__main__":
    main()
