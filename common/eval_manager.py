#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估闭环管理器：Score（评分）+ Dataset（数据集）+ Experiment（实验）。

数据飞轮闭环：
  生产 Trace → 自动评分 → 筛选 Bad Case → 构建数据集 → 回归实验 → 优化

用法:
    from common.eval_manager import EvalManager
    evm = EvalManager()
    evm.score_trace(trace_id, "cypher_success", 1)
    ds = evm.create_dataset("bad-cases")
    evm.add_dataset_item("bad-cases", {"question": "..."}, expected_output="...")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

logger = logging.getLogger("eval_manager")

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

# ── LLM-as-Judge 提示词 ──
_JUDGE_PROMPT = """请从中医专业知识角度，评估以下回答的质量。

## 用户问题
{question}

## 模型回答
{answer}

## 评分维度（每项 0-10 分）
1. accuracy（准确性）：回答是否符合中医理论，有无明显错误或幻觉
2. completeness（完整性）：是否充分回答了问题的核心诉求
3. relevance（相关性）：是否紧扣问题，无无关内容

## 输出格式（严格 JSON）
{{"accuracy": 8, "completeness": 7, "relevance": 9}}"""


class EvalManager:
    """Langfuse 评估闭环封装。"""

    def __init__(self) -> None:
        self._client = None
        self._llm = None
        try:
            if not os.getenv("LANGFUSE_SECRET_KEY") or not os.getenv("LANGFUSE_PUBLIC_KEY"):
                logger.warning("Langfuse 密钥缺失，评估功能不可用")
                return
            from common.langfuse_manager import make_langfuse_client

            self._client = make_langfuse_client()
        except Exception as exc:
            logger.warning("Langfuse 初始化失败，评估功能不可用: %s", exc)

    @property
    def client(self):
        return self._client

    def is_available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------
    # Score（评分）
    # ------------------------------------------------------------

    def score_trace(
        self,
        trace_id: str,
        name: str,
        value: float | str,
        data_type: str = "NUMERIC",
        comment: str | None = None,
    ) -> None:
        """给 trace 打一个评分。"""
        if not self.is_available():
            return
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                data_type=data_type,  # type: ignore[arg-type]
                comment=comment,
            )
        except Exception as exc:
            logger.warning("评分失败 name=%s: %s", name, exc)

    def score_auto(self, trace_id: str, final_state: dict[str, Any]) -> None:
        """请求结束后自动评分（启发式）。"""
        if not self.is_available() or not trace_id:
            return
        cypher_queries = final_state.get("cypher_queries", []) or []
        self.score_trace(trace_id, "cypher_success", 1 if cypher_queries else 0)
        self.score_trace(trace_id, "retry_count", final_state.get("cypher_retry_count", 0))
        self.score_trace(trace_id, "answer_nonempty", 1 if final_state.get("final_answer") else 0)

    def judge_answer(
        self,
        trace_id: str,
        question: str,
        answer: str,
    ) -> dict[str, float] | None:
        """LLM-as-Judge：用 LLM 评估回答质量（准确性/完整性/相关性）。

        主观质量启发式（长度）无法判断"对错"，用 LLM 做裁判，
        从三个维度打分（0-10），写入 Langfuse score。

        Returns:
            {"accuracy": x, "completeness": x, "relevance": x} 或 None。
        """
        import json
        import re

        llm = self._get_llm()
        if llm is None or not answer:
            return None

        prompt = _JUDGE_PROMPT.format(question=question, answer=answer[:2000])
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            raw = str(llm.invoke([
                SystemMessage(content="你是中医知识问答评审专家，只输出 JSON 评分。"),
                HumanMessage(content=prompt),
            ]).content)

            # 容错提取 JSON
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            scores = json.loads(m.group()) if m else {}

            accuracy = float(scores.get("accuracy", 0))
            completeness = float(scores.get("completeness", 0))
            relevance = float(scores.get("relevance", 0))

            self.score_trace(trace_id, "judge_accuracy", accuracy, "NUMERIC")
            self.score_trace(trace_id, "judge_completeness", completeness, "NUMERIC")
            self.score_trace(trace_id, "judge_relevance", relevance, "NUMERIC")
            return {"accuracy": accuracy, "completeness": completeness, "relevance": relevance}
        except Exception as exc:
            logger.warning("LLM-as-Judge 评分失败: %s", exc)
            return None

    def _get_llm(self):
        """惰性加载 LLM（复用项目 ChatOpenAI）。"""
        if self._llm is None:
            try:
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(
                    model=os.getenv("MODEL_NAME", "deepseek-chat"),
                    api_key=os.getenv("MODEL_API_KEY"),
                    base_url=os.getenv("MODEL_BASE_URL"),
                    temperature=0,
                )
            except Exception as exc:
                logger.warning("LLM 初始化失败: %s", exc)
                self._llm = None
        return self._llm

    # ------------------------------------------------------------
    # Dataset（数据集）
    # ------------------------------------------------------------

    def create_dataset(self, name: str, description: str = "") -> Any | None:
        """创建数据集（已存在则返回 None）。"""
        if not self.is_available():
            return None
        try:
            return self._client.create_dataset(name=name, description=description)
        except Exception as exc:
            logger.warning("创建数据集失败 name=%s: %s", name, exc)
            return None

    def add_dataset_item(
        self,
        dataset_name: str,
        input: Any,
        expected_output: Any = None,
        source_trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """向数据集添加一条测试用例（可关联来源 trace）。"""
        if not self.is_available():
            return
        try:
            self._client.create_dataset_item(
                dataset_name=dataset_name,
                input=input,
                expected_output=expected_output,
                source_trace_id=source_trace_id,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("添加数据集 item 失败: %s", exc)

    # ------------------------------------------------------------
    # Experiment（实验/回归测试）
    # ------------------------------------------------------------

    def run_experiment(
        self,
        dataset_name: str,
        task_fn: Callable[[Any], Any],
        eval_fn: Callable[[str, Any, Any], None],
        run_name: str = "",
    ) -> list[dict[str, Any]]:
        """在数据集上运行实验：每个 item 跑 task_fn + eval_fn。

        Args:
            dataset_name: 数据集名称。
            task_fn: 任务函数，输入 dataset item.input，返回输出。
            eval_fn: 评分函数，签名 (trace_id, output, expected_output)。
            run_name: 实验名（默认 dataset_name-run，用于对比多个版本）。

        Returns:
            每个 item 的结果列表。
        """
        if not self.is_available():
            return []
        results: list[dict[str, Any]] = []
        try:
            dataset = self._client.get_dataset(dataset_name)
            run_label = run_name or f"{dataset_name}-run"
            for item in dataset.items:
                # 观察模式：自动关联 dataset run
                with item.observe(run_name=run_label) as trace_id:
                    output = task_fn(item.input)
                    eval_fn(trace_id, output, item.expected_output)
                results.append({
                    "item_id": item.id,
                    "output": output,
                    "trace_id": trace_id,
                })
        except Exception as exc:
            logger.warning("实验失败: %s", exc)
        finally:
            self.flush()
        return results

    # ------------------------------------------------------------
    # flush
    # ------------------------------------------------------------

    def flush(self) -> None:
        if self.is_available():
            try:
                self._client.flush()
            except Exception as exc:
                logger.warning("flush 失败: %s", exc)


# ── 全局单例 ──

_eval_mgr: EvalManager | None = None


def get_eval_manager() -> EvalManager:
    global _eval_mgr
    if _eval_mgr is None:
        _eval_mgr = EvalManager()
    return _eval_mgr
