# 数据飞轮与可观测性 — 面试技术文档

> 三件套：**原理认知 → 工程取舍 → 面试表达 → 实战案例**
> 项目：草本通（中医药知识图谱问答系统）
> 技术栈：LangGraph + Neo4j + FAISS + Langfuse + FastAPI

---

## 一、原理认知（先讲清"为什么"）

### 1.1 LLM 应用的核心矛盾

传统软件开发有确定性（输入→输出可预测），LLM 应用是**概率性的**——同一个问题，模型可能给出不同质量的回答。

```
传统软件：单元测试 → 断言结果 → 通过/失败（确定性）
LLM 应用：发请求 → 回答像人话但可能错 → 怎么判断好不好？（概率性）
```

**所以 LLM 应用需要三件新东西**：
1. **可观测性**（Observability）：看不到模型内部，只能看输入输出 + 中间过程
2. **评估**（Evaluation）：怎么量化"回答好不好"
3. **数据飞轮**（Data Flywheel）：怎么用生产数据持续优化

### 1.2 数据飞轮的本质

```
        生产数据（真实用户问题）
              │
              ▼
        发现 Bad Case（答错的、慢的、用户不满的）
              │
              ▼
        固化成数据集（golden dataset）
              │
              ▼
        回归测试（每次改动跑分对比）
              │
              ▼
        分数提升 → 上线；下降 → 回滚
              │
              └──────── 回到生产，循环
```

**核心认知**：大模型上线只是开始，**数据飞轮才是优化的关键**。Langfuse 是"数据沉淀池"，把生产经验变成可复用的测试资产。

---

## 二、工程取舍（每个决策的权衡）

### 2.1 为什么用 Langfuse 而不是自己造轮子

| 方案 | 取舍 |
|------|------|
| 自己写日志 | 灵活但要做 UI、存储、聚合、检索，成本高 |
| Langfuse | 开源自托管，Trace/Span/Score/Dataset/Experiment 全有 |
| 商业方案（LangSmith） | 贵，数据出域 |

**取舍**：选择 Langfuse 自托管——数据私有 + 功能全 + 可二次开发。

### 2.2 零侵入 vs 侵入式埋点

**关键决策**：用 LangChain 的 callback 机制，不改节点代码。

```python
# 零侵入：通过 config 透传 callback，节点代码零改动
handler = _make_handler(request, "POST:/ask")
final_state = _graph.invoke(initial, config={"callbacks": [handler]})
```

**取舍**：零侵入牺牲了"每个节点精细控制"的灵活性，换来了"改观测不影响业务"的安全性。企业级优先保证业务稳定。

### 2.3 动态采样（成本 vs 覆盖率）

**矛盾**：全量采集 → 存储爆炸；固定 10% → 漏掉故障。

**取舍**：分层采样——正常请求 10%，异常/慢请求/重试/低分 100%。

```python
def _check_force_sample(final_state, elapsed_ms):
    if elapsed_ms > 30_000:      return "slow_request"   # 慢请求必记录
    if cypher_retry_count > 0:   return "cypher_retry"   # 重试必记录
    if 中医 and not cypher:      return "no_cypher"      # 失败必记录
    if not final_answer:         return "empty_answer"   # 空回答必记录
```

**坑**：Langfuse v2 不支持 trace 级 `sampled`（事后丢弃），所以只能"请求开始采样 + 事后 force 补记摘要"。

### 2.4 规则评分 vs LLM-as-Judge

| 评分方式 | 适用 | 局限 |
|---------|------|------|
| 规则型（answer_nonempty） | 确定性检查 | 只能判断"有没有"，不能判断"对不对" |
| 启发式（answer_quality 长度） | 粗略质量 | 长≠好，会误判 |
| **LLM-as-Judge** | 主观质量 | 用 LLM 判断准确性/完整性/相关性，接近人评 |

**取舍**：三层结合——规则型兜底（快、免费），LLM-as-Judge 精细（慢、耗 token），只在需要时调用 Judge。

### 2.5 记忆系统的信息冲突

**矛盾**：用户说"我不用那张银行卡了"，系统还记着旧信息 → 回答错误。

**取舍**：三元组 `(subject, predicate)` 唯一约束，写入前先删旧的，否定意图直接删除（遗忘）。

---

## 三、面试表达（怎么说才专业）

### 3.1 "你做过什么可观测性？" 标准回答

> "给中医问答系统接了 Langfuse 全链路追踪。一次请求对应一条 Trace，7 个 LangGraph 节点各一个 Span，每次 LLM 调用一个 Generation（记录 token、耗时、成本、TTFT）。用自定义 CallbackHandler 实现零侵入，通过 config 透传，不改任何节点代码。"

### 3.2 "怎么评估模型效果？" 标准回答

> "建立了数据飞轮闭环。生产环境自动评分（规则型 + LLM-as-Judge），动态采样筛出 Bad Case，回流成 golden dataset，每次改 prompt 或换模型都在固定数据集上跑回归实验，量化对比分数。提升才上线，退化就回滚——把'凭感觉调 prompt'变成'用数据说话'。"

### 3.3 "怎么控制观测成本？" 标准回答

> "动态采样：正常请求 10% 采样，异常/慢请求/重试/低分强制 100% 记录。既控制了存储成本，又不放过任何故障现场。"

### 3.4 "trace 写不进去怎么排查？"（展示系统化思维）

> "六层定位：服务日志（只读降级）→ 配置（缺 ClickHouse）→ 网络（镜像拉取）→ 认证（环境变量）→ 权限（volume）→ 进程状态（reload）。每层验证，逐层排除。"

---

## 四、实战案例（这个项目的真实数据）

### 4.1 性能瓶颈定位

**现象**：一次问答 38 秒，用户抱怨慢。

**定位过程**：Langfuse Trace 显示 cypher_generation 节点耗时 25 秒（占 80%），其中 LLM 生成了 2106 tokens（其他节点只有几十个）。

**优化**：精简 schema（去掉 properties）+ max_tokens=512 + 限制 Cypher 数量，耗时 25s → 3s。

### 4.2 优化效果量化（真实评分对比）

改 prompt 后，在两个 run 上对比：

| 指标 | 基线 | 优化后 | 变化 |
|------|------|--------|------|
| answer_nonempty | 1.000 | 1.000 | 持平 |
| answer_quality | 0.749 | **0.777** | ↑ +3.7% |
| cypher_generated | 1.000 | 0.900 | ↓ -10% |

**结论**：优化有得有失——回答更详细了（+3.7%），但部分题目 Cypher 生成退化了（-10%）。**这就是回归测试的价值：不靠感觉，靠数据发现副作用。**

### 4.3 LLM-as-Judge 三维度评分（真实对比数据）

**实现**（`common/eval_manager.py` 的 `judge_answer`）：

```python
def judge_answer(self, trace_id, question, answer):
    # 用 LLM 从准确性/完整性/相关性三维度打分（0-10）
    prompt = _JUDGE_PROMPT.format(question=question, answer=answer)
    raw = llm.invoke([...]).content
    scores = json.loads(提取 JSON)
    self.score_trace(trace_id, "judge_accuracy", scores["accuracy"])
    self.score_trace(trace_id, "judge_completeness", scores["completeness"])
    self.score_trace(trace_id, "judge_relevance", scores["relevance"])
```

**真实评分结果**（好回答 vs 坏回答对比强烈）：

| 问题 | 回答类型 | accuracy | completeness | relevance |
|------|---------|----------|--------------|-----------|
| 四君子汤功效 | ❌ 敷衍"补气。" | **0** | **0** | **0** |
| 四君子汤功效 | ✅ 详细完整 | **10** | **10** | **10** |
| 人参性味归经 | ❌ "具体不知道" | **6** | **1** | **4** |
| 人参性味归经 | ✅ 准确完整 | **9** | **7** | **9** |

**关键洞察**：启发式评分（长度）无法区分"补气。"和"补气健脾的完整解释"，但 LLM-as-Judge 能精准判断——敷衍回答 completeness=0，错误回答 accuracy 骤降，好回答三维度全 9-10。

### 4.4 三层评分体系

| 评分层 | 能判断 | 局限 | 成本 |
|--------|--------|------|------|
| 规则型（answer_nonempty） | 有没有回答 | 不能判断对错 | 免费 |
| 启发式（answer_quality 长度） | 详细度 | 长≠好 | 免费 |
| **LLM-as-Judge** | 准确性/完整性/相关性 | 耗 token | ~几百 token/次 |

**生产实践**：三层结合——规则型兜底（快免费），LLM-as-Judge 精细（只在回归实验时用，不在每次请求都调）。

---

## 五、可复用的代码结构

```
common/
├── langfuse_manager.py   # 追踪：trace/span/generation + 动态采样 + 成本
├── eval_manager.py       # 评估：评分 + LLM-as-Judge + 数据集 + 实验
├── build_eval_dataset.py # 数据回流：Bad Case → 数据集
└── run_experiment.py     # 回归实验：数据集跑分对比
_008_memory/
├── memory_manager.py     # 记忆：三层存储 + Token预算 + 遗忘
└── factory.py            # 工厂：依赖注入 + 降级
```

**复用原则**：所有可观测/评估逻辑封装在 `common/`，业务节点（`_004_langgraph_more_nodes/`）零改动。任何新项目都能直接复用这套 common 层。

---

## 六、一句话总结（电梯陈述）

> "LLM 应用是概率性的，所以我建立了可观测 + 评估 + 数据飞轮三件套：用 Langfuse 零侵入追踪每次请求，动态采样控成本，把生产 Bad Case 回流成数据集，用回归实验量化每次优化，实现数据驱动的持续迭代。"
