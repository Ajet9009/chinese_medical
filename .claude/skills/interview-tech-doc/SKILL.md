---
name: interview-tech-doc
description: 当用户提到"面试技术文档"、"面试准备"、"技术评审"、"可观测性落地"、"数据飞轮"、"记忆系统"、"上下文管理"、"企业级落地"等关键词时，加载并讲解对应面试技术文档，帮助理解企业生产级流程。触发场景：用户要应对面试提问、准备技术评审、梳理项目亮点、理解可观测性/评估/数据飞轮/Agent记忆的落地实现。
---

# 面试技术文档技能

当用户提到"面试技术文档"或相关关键词时，按主题选择文档：

## A. 数据飞轮 / 可观测性

1. **一问一答**：`docs/数据飞轮/面试问答-数据飞轮.md`
2. **原理结构**：`docs/数据飞轮/data-flywheel-interview.md`
3. **代码索引**：`docs/数据飞轮/observability-data-flywheel.md`

## B. 记忆与上下文管理（优先本主题时）

1. **一问一答**：`docs/记忆AND上下文管理/面试问答-记忆与上下文.md`
2. **技术架构**：`docs/记忆AND上下文管理/memory-context-architecture.md`
3. **必须强调**：`ContextCompressor` / `fit_budget`（Token 预算摘要）与 Redis `LTRIM`（条数窗口）的区别——这是关键点，不可省略。

## 通用讲解结构（三件套）

- **原理认知**：先讲清"为什么"
- **工程取舍**：每个设计决策的权衡
- **面试表达**：标准话术 + 诚实边界
- **实战案例**：对照真实代码与演示（`demo_memory.py` / `demo_flywheel.py`）

## 代码锚点速查

- 动态采样：`common/langfuse_manager.py`
- 自动评分 / Judge：`common/eval_manager.py`
- 数据回流：`common/build_eval_dataset.py`
- 回归实验：`common/run_experiment.py` / `_000_demo/demo_flywheel.py`
- 记忆系统：`_008_memory/memory_manager.py` · `stores.py` · `backends.py` · `factory.py`
- API 接入：`_005_fastapi/main.py`（`_read_memory` / `_write_memory`）

## 关键原则

- 强调落地：落到代码，不编造未验证指标。
- 强调取舍：为什么这么选。
- 强调诚实边界：局限主动说。
