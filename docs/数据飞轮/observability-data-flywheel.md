# 可观测性与数据飞轮 — 企业级落地文档

> 面向面试/技术评审的完整实现说明。覆盖：Langfuse 全链路追踪、动态采样、成本核算、评估闭环、记忆系统。

---

## 一、整体架构（7 层）

```
┌─────────────────────────────────────────────────────────┐
│  前端 (Streamlit)                                        │
│  - 独立 user_id 隔离  - 👍👎 反馈  - 流式进度计时        │
└──────────────────────┬──────────────────────────────────┘
                       │ X-User-ID / X-Session-ID
┌──────────────────────▼──────────────────────────────────┐
│  API 层 (FastAPI)                                        │
│  /ask  /ask/stream  /feedback                           │
│  - 请求前后读写记忆  - 显式 flush 观测数据               │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  编排层 (LangGraph 7 节点)                               │
│  意图识别→实体抽取→向量匹配→Cypher生成→执行→回答        │
└──────────────────────┬──────────────────────────────────┘
                       │ callback / astream_events
┌──────────────────────▼──────────────────────────────────┐
│  可观测层 (Langfuse)                                     │
│  Trace → Span → Generation → Score → Dataset → Experiment │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  记忆层 (_008_memory)                                    │
│  短期 Redis + 长期向量库 + 用户画像                       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  数据层 (Neo4j + ClickHouse + PostgreSQL + Redis)        │
└─────────────────────────────────────────────────────────┘
```

---

## 二、可观测性（Langfuse）核心设计

### 1. 零侵入集成

**关键决策**：用自定义 `_TraceCallbackHandler`（继承 LangChain callback），通过 LangGraph 的 `config={"callbacks": [handler]}` 透传，**不改任何节点代码**。

```python
# _005_fastapi/main.py
handler = _make_handler(request, "POST:/ask")
config = {"callbacks": [handler]}
final_state = _graph.invoke(initial, config=config)
```

**为什么不用官方 CallbackHandler**：Langfuse v2 的 `langfuse.callback.langchain.CallbackHandler` 与 langchain v1.x 不兼容（`langchain.callbacks` 旧路径）。自定义 handler 直接调 langfuse 原生 API（`client.trace/span/generation`）。

### 2. 三层观测结构

| 对象 | 粒度 | 记录内容 |
|------|------|---------|
| Trace | 一次 API 请求 | request_id / session_id / user_id / version |
| Span | 一个图节点 | 节点名 / 输入 / 输出 / 耗时 |
| Generation | 一次 LLM 调用 | model / tokens / latency / cost / TTFT |

**节点名获取的坑**：LangGraph 节点执行时，callback 的 `serialized` 里 `name=None`（拿不到节点名）。节点名只能从 `astream_events` 的 `event['name']` 拿到。所以节点 Span 改在 main.py 的流式循环里用 `start_node_span(event['name'])` 手动创建。

### 3. 动态采样（分层采样）

**设计**：正常请求按比例采样（控成本），异常请求 100% 记录（不放过故障现场）。

```python
# common/langfuse_manager.py
def create_handler(self, ..., force=False):
    if not force and not self.should_sample():  # 10% 概率
        return None
    # force=True 跳过采样

def _check_force_sample(final_state, elapsed_ms):  # main.py
    if elapsed_ms > 30_000:      return "slow_request"
    if cypher_retry_count > 0:   return "cypher_retry"
    if 中医 and not cypher:      return "no_cypher"
    if not final_answer:         return "empty_answer"
```

**为什么不能"全量采集+事后过滤"**：Langfuse v2 不支持 trace 级 `sampled` 参数（事后丢弃）。所以采用"请求开始采样 + 事后强制补记摘要"的折中。

### 4. 成本核算

```python
INPUT_PRICE_PER_1M = 1.0    # 元
OUTPUT_PRICE_PER_1M = 3.0   # 元

gen.update(
    usage={"input": N, "output": M, "total": N+M, "unit": "TOKENS"},
    cost_details={"input": N/1M*1.0, "output": M/1M*3.0, "total": ...},
)
```

**坑**：Langfuse v2 的 `gen.update()` 没有 `cost` 参数，正确参数是 `cost_details`（dict）。

### 5. Prompt 管理

提示词从硬编码迁到 Langfuse，运行时拉取 + SDK 60s 缓存：

```python
def _get_intent_prompt():
    return fetch_prompt("intent_recognition", _FALLBACK, entity_types=..., relation_types=...)
```

**价值**：改 prompt 不用重启，60s 自动生效；支持版本回滚；A/B 测试。

---

## 三、数据飞轮闭环（5 步）

```
生产请求 → ① 自动评分 + ② 动态采样
   → Bad Case（forced-sampling）→ ③ 数据回流 → 数据集
   → ④ 回归实验 → 跑分对比 → 优化 Prompt/模型 → 回生产
   ← ⑤ 用户反馈（👍👎）补充 Bad Case
```

### ① 自动评分（三层体系）

```python
# common/eval_manager.py
def score_auto(self, trace_id, final_state):
    # 规则型（免费，请求结束自动打）
    self.score_trace(trace_id, "cypher_success", 1 if cypher_queries else 0)
    self.score_trace(trace_id, "retry_count", cypher_retry_count)
    self.score_trace(trace_id, "answer_nonempty", 1 if final_answer else 0)

def judge_answer(self, trace_id, question, answer):
    # LLM-as-Judge（精细，回归实验时用）
    # 三维度：accuracy / completeness / relevance（0-10）
    scores = {"accuracy": 8, "completeness": 7, "relevance": 9}
    self.score_trace(trace_id, "judge_accuracy", scores["accuracy"])
    self.score_trace(trace_id, "judge_completeness", scores["completeness"])
    self.score_trace(trace_id, "judge_relevance", scores["relevance"])
```

**三层评分体系**：

| 层 | 指标 | 能判断 | 成本 |
|----|------|--------|------|
| 规则型 | cypher_success / retry_count / answer_nonempty | 有没有、成不成功 | 免费 |
| 启发式 | answer_quality（长度归一化） | 详细度 | 免费 |
| LLM-as-Judge | judge_accuracy / completeness / relevance | 对不对、全不全、相不相关 | ~几百 token |

**真实对比**（好回答 vs 坏回答）：敷衍"补气。"→ 三维度全 0；详细完整回答 → 三维度全 9-10。差异强烈，一眼看出好坏。

### ② 动态采样（见上文）

### ③ 数据回流

```bash
python common/build_eval_dataset.py --name bad-cases
# fetch_traces(tags="forced-sampling") → 清洗 → create_dataset_item
```

### ④ 回归实验

```bash
python common/run_experiment.py --dataset bad-cases
# item.observe(run_name=...) → task_fn → eval_fn → score
```

### ⑤ 用户反馈

```python
# POST /feedback → score_trace(trace_id, "user_feedback", 1/0)
```

---

## 四、记忆系统（_008_memory）

### 三层存储

| 层 | 存储 | 实现 | 隔离维度 |
|----|------|------|---------|
| 短期 | Redis List（LPUSH+LTRIM 滑动窗口） | RedisShortTermStore | session_id |
| 长期 | JSON 向量库（numpy 余弦） | JsonLongTermStore | user_id |
| 画像 | 字典（接口可换 PG/Mongo） | DictProfileStore | user_id |

### 三个企业级特性

1. **Token 预算 + 摘要压缩**：超 4k 不截断，LLM 摘要压缩前半段。
2. **信息冲突/遗忘**：三元组 `(subject, predicate)` 唯一，新值覆盖，否定删除。
3. **防幻觉**：每条记忆带 `source` + `timestamp`，按 user_id 隔离。

---

## 五、面试问答要点

### Q1: 为什么 trace 写不进 ClickHouse？怎么排查的？

**排查链**（体现系统化思维）：
1. 查 Langfuse 日志 → "Read from postgres only"（只读降级）
2. 查 docker-compose → 无 ClickHouse（v2.95 架构变更，trace 从 PG 迁到 CH）
3. 拉 ClickHouse 镜像 → Docker Hub 被墙 → 配 Clash 代理
4. 认证失败 → 账号密码要用 `CLICKHOUSE_USER/PASSWORD` 环境变量（非 URL 内嵌）
5. 文件权限 → Windows bind mount 换 named volume
6. `--reload` 导致 SDK 状态混乱 → 去掉 reload

**体现**：分层定位（服务日志 → 配置 → 网络 → 认证 → 权限 → 进程状态）。

### Q2: 怎么控制观测成本？

动态采样：正常 10%，异常/慢请求/重试/低分 100%。既控存储，又不放过故障。

### Q3: 怎么评估模型效果？

数据飞轮：自动评分 → Bad Case 筛选 → 数据集 → 回归实验 → 量化对比。

### Q4: 怎么处理信息冲突（记忆系统）？

三元组 `(subject, predicate)` 唯一约束，写入前先删旧的，否定意图直接删除。

### Q5: 观测系统对业务零侵入怎么做？

自定义 callback handler + try-except 降级，初始化失败/上报失败都不影响主流程。

---

## 六、关键代码索引

| 能力 | 文件 | 关键函数 |
|------|------|---------|
| 动态采样 | common/langfuse_manager.py | create_handler(force) / should_sample / record_summary_span |
| 强制采样判断 | _005_fastapi/main.py | _check_force_sample |
| 自动评分 | common/eval_manager.py | score_auto |
| 数据回流 | common/build_eval_dataset.py | fetch_bad_traces / main |
| 回归实验 | common/run_experiment.py | task_fn / eval_fn |
| 记忆系统 | _008_memory/memory_manager.py | add_message / get_context |
| 记忆工厂 | _008_memory/factory.py | create_memory_manager |
