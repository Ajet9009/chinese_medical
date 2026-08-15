---
name: interview-tech-doc
description: 当用户提到"面试技术文档"、"面试准备"、"技术评审"、"可观测性落地"、"数据飞轮"、"企业级落地"等关键词时，加载并讲解数据飞轮与可观测性的面试技术文档，帮助理解企业生产级流程。触发场景：用户要应对面试提问、准备技术评审、梳理项目亮点、理解可观测性/评估/数据飞轮的落地实现。
---

# 面试技术文档技能

当用户提到"面试技术文档"或相关关键词时，执行以下步骤：

## 步骤

1. **读取一问一答主文档**：读取 `docs/数据飞轮/面试问答-数据飞轮.md`（企业生产级面试问答，优先用这份对练）。

2. **读取原理结构文档**（需要系统讲解时）：读取 `docs/数据飞轮/data-flywheel-interview.md`（原理认知→工程取舍→面试表达→实战案例）。

3. **读取配套技术文档**（需代码索引时）：读取 `docs/数据飞轮/observability-data-flywheel.md`。

4. **按"三件套"结构讲解**：
   - **原理认知**：先讲清"为什么"（LLM 概率性 → 需要可观测性/评估/数据飞轮）
   - **工程取舍**：每个设计决策的权衡（零侵入 vs 精细控制、动态采样 vs 全量、规则评分 vs LLM-as-Judge、为何要有 baseline）
   - **面试表达**：给出标准回答话术（可观测性/评估/成本控制/排查能力/baseline）
   - **实战案例**：引用项目真实数据（性能优化 25s→3s、评分对比 +3.7%、Judge 0 vs 10）

5. **结合项目代码**：指出关键实现位置，让用户能对照代码讲：
   - 动态采样：`common/langfuse_manager.py` 的 `create_handler(force)` / `should_sample`
   - 自动评分：`common/eval_manager.py` 的 `score_auto` / `judge_answer`
   - 数据回流：`common/build_eval_dataset.py`
   - 回归实验：`common/run_experiment.py` / `_000_demo/demo_flywheel.py`（baseline → optimized）
   - 记忆系统：`_008_memory/memory_manager.py`

6. **模拟面试问答**：按 `面试问答-数据飞轮.md` 的题目顺序扮演面试官；强调诚实边界（强制采样是摘要、Judge 不在每次线上请求、不背未验证的网上指标）。

## 关键原则

- 强调"落地"：不只讲概念，要落到代码和真实数据。
- 强调"取舍"：每个决策都有权衡，面试官看重的是"为什么这么选"。
- 强调"复用"：common 层可移植到任何 LLM 项目。
