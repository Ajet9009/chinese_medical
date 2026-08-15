#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM-as-Judge 评分对比演示：好回答 vs 坏回答，展示强烈评分差异。

用法:
    python _000_demo/demo_judge.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / "common" / ".env")

from common.eval_manager import EvalManager  # noqa: E402
from common.langfuse_manager import LangfuseManager  # noqa: E402

# 对比用例：同一问题，坏回答 vs 好回答
CASES = [
    {
        "question": "四君子汤有什么功效？",
        "bad": "补气。",
        "good": "四君子汤是益气健脾的基础方，功效为补气健脾。主治脾胃气虚证，症见面色萎白、语声低微、气短乏力、食少便溏。方由人参、白术、茯苓、甘草四味组成，其中人参甘温益气为君，白术健脾燥湿为臣，茯苓渗湿健脾为佐，甘草调和诸药为使。",
    },
    {
        "question": "人参的性味归经是什么？",
        "bad": "人参是补气的，具体不知道。",
        "good": "人参性温，味甘、微苦，归脾、肺、心、肾经。具有大补元气、复脉固脱、补脾益肺、生津养血、安神益智的功效。",
    },
    {
        "question": "咳嗽应该吃什么中药？",
        "bad": "吃点西药就行。",
        "good": "咳嗽需辨证论治：风寒咳嗽宜用三拗汤或止嗽散，风热咳嗽宜用桑菊饮，痰湿咳嗽宜用二陈汤，肺燥咳嗽宜用桑杏汤。具体用药需根据证型、兼症辨证选择。",
    },
]


def main() -> None:
    print("=" * 64)
    print("  LLM-as-Judge 评分对比演示")
    print("=" * 64)

    # 创建 trace 用于关联评分
    mgr = LangfuseManager()
    evm = EvalManager()

    handler = mgr.create_handler(trace_name="judge-demo", force=True)
    trace_id = handler.trace_id if handler else None
    if not trace_id:
        print("  Langfuse 不可用，仅本地演示")
        trace_id = "local-demo"

    print(f"\n  trace_id: {trace_id}\n")

    header = f"  {'问题':<24s} | {'维度':<20s} | {'坏回答':>6s} | {'好回答':>6s} | {'提升':>6s}"
    print(header)
    print("  " + "-" * 70)

    for case in CASES:
        question = case["question"]
        bad_result = evm.judge_answer(trace_id, question, case["bad"])
        good_result = evm.judge_answer(trace_id, question, case["good"])

        dims = ["accuracy", "completeness", "relevance"]
        for dim in dims:
            bad_score = (bad_result or {}).get(dim, 0)
            good_score = (good_result or {}).get(dim, 0)
            diff = good_score - bad_score
            print(
                f"  {question:<24s} | {dim:<20s} | {bad_score:>6.1f} | {good_score:>6.1f} | {diff:>+5.1f}"
            )
        print("  " + "-" * 70)

    mgr.flush(timeout=5.0)
    print("\n  演示完成。Langfuse UI → Traces → judge-demo 查看评分")


if __name__ == "__main__":
    main()
