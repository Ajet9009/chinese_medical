# Agent 记忆与上下文管理 — 技术文档（企业生产级）

> 项目：草本通（中医药知识图谱问答）  
> 模块：`_008_memory` + FastAPI / Streamlit 接入  
> 原则：**对齐仓库真实实现**；不编造未发生的 SLA/压测数字。  
> 配套面试问答：同目录 `面试问答-记忆与上下文.md`  
> 提示词原文：见本文 **第 7 节**（与 `memory_manager.py` 对齐）

---

## 1. 原理认知：为什么要分层记忆？

LLM 无跨请求状态。若不做记忆：

- 多轮对话只能靠客户端回传全文 → 成本高、易泄露、难隔离。  
- 跨会话偏好（体质、忌口）无法复用。  
- 上下文无限增长 → Token 爆仓、延迟与费用失控。

因此拆成三层职责：

| 层 | 问题 | 存储 | 隔离维度 |
|----|------|------|----------|
| 短期 | 本轮对话刚说了什么 | Redis List | `user_id` + `session_id` |
| 长期 | 跨会话稳定事实 | Milvus 三元组向量 | `user_id` |
| 画像 | 结构化偏好快读 | MySQL KV | `user_id` |

另加：**Token 预算压缩**（拼 Prompt 时）与 **审计**（写变更可追溯）。

---

## 2. 请求时序（Ask 路径）

```
POST /ask
  Headers: X-User-ID, X-Session-ID
       │
       ▼
  _read_memory → MemoryManager.get_context
       │  长期检索 + 画像 + 短期窗口 → ContextCompressor.fit_budget
       ▼
  LangGraph（增强后的 user_question）
       │
       ├─ 失败 500 → 不写记忆
       └─ 成功 → _write_memory
              add_message(user)  → 短期同步 + 异步抽长期
              add_message(assistant) → 仅短期
```

代码：`_005_fastapi/main.py` 的 `_read_memory` / `_write_memory` / `_get_identity`。

---

## 3. 短期记忆

### 3.1 实现

- 类：`RedisShortTermStore`（`_008_memory/stores.py`）  
- Key：`mem:short:{user_id}:{session_id}`（`short_term_key`）  
- 滑动窗口：`LPUSH` + `LTRIM 0..N-1` + `EXPIRE`  
- 默认：N=`MEMORY_SHORT_ROUNDS`（20 **条消息**，不是 20 轮问答对）；TTL=`MEMORY_SHORT_TTL_SECONDS`（7 天）  
- 降级：`InMemoryShortTermStore`（进程内，重启丢）

### 3.2 为何 user_id + session_id

| 只用 session | 同 session 串用户 |
| 只用 user | 多会话揉成一个窗口 |
| 两者都要 | 会话内连续对话 + 用户级隔离 |

### 3.3 与「压缩」的边界

- **LTRIM**：按**条数**裁 Redis，防键无限长。  
- **ContextCompressor**：按 **Token 预算**裁/摘要即将进 Prompt 的短期文本（见第 6 节）。  
二者不是一回事，面试必须分清。

---

## 4. 长期记忆

### 4.1 数据模型

`MemoryTriple`：`subject | predicate | object` + `source` + `timestamp`。  
`source` 保留原文片段，解决「记忆幻觉不可追溯」。

### 4.2 写入路径

1. `add_message` 仅对 `role=user` 触发抽取。  
2. 默认 `async_extract=True`：有界线程池（`extract_workers` + `extract_queue_size`），满则丢弃并审计。  
3. `TripleExtractor`：LLM 输出 JSON，`action=set|delete`。  
4. `_apply_triple`：  
   - `delete` → Milvus 删 + 画像清空（偏好/身份/计划）  
   - `set` → **先删同 subject+predicate** 再 insert（冲突覆盖）  
5. BGE `embed_query` → `MilvusLongTermStore.add`  
6. 审计：`triple_upsert` / `forget` / `profile_upsert`

存储：`backends.MilvusLongTermStore`，collection `agent_long_term_memory`（独立，不占业务 `edurag`）。  
降级：`JsonLongTermStore`。

### 4.3 读取路径

`get_context`：`embed_query(当前问题)` → `search(user_id, vec, top_k)`，过滤表达式强制本用户。

---

## 5. 用户画像

- 表：`subjects_kg.agent_user_profile`（`user_id` + `pref_key` PK）  
- 写入：三元组中 `subject=用户` 且 `predicate ∈ {偏好,身份,计划}` 时 upsert  
- 读取：`get_all(user_id)` 拼 `【用户画像】`  
- 连接池：`_MySQLPool`；降级 `JsonProfileStore`

画像是长期事实的**结构化投影**，方便固定字段快读；语义检索仍靠 Milvus。

---

## 6. 关键点：上下文压缩（ContextCompressor）

文件：`_008_memory/memory_manager.py` → `ContextCompressor`。

### 6.1 为什么需要

短期窗口按条数保留（最多约 20 条），但单条很长时仍可能撑爆 Prompt。直接截断会丢掉关键语义。

### 6.2 策略（`fit_budget`）

1. 预算默认 `token_budget=4000`（tiktoken 计数）。  
2. 先扣掉 prefix（长期+画像）与 suffix（当前输入）占用。  
3. 短期全文若 `count(近期对话) <= budget_left` → 全放。  
4. 否则：尽量保留**最近**消息；更早段落交给 `_summarize`（LLM 摘要）。  
5. 摘要失败 → 字符截断保底。

### 6.3 工程取舍

| 方案 | 优点 | 缺点 | 我们的选择 |
|------|------|------|------------|
| 硬截断 | 快、零成本 | 丢语义 | 仅作摘要失败兜底 |
| 始终摘要 | 长度稳 | 每请求多一次 LLM | 否，仅超预算才摘要 |
| 条数窗口 + 条件摘要 | 多数请求零额外 LLM | 实现稍复杂 | **采用** |

---

## 7. 记忆模块提示词（对齐源码）

文件：`_008_memory/memory_manager.py`。记忆侧一共 **2 处调 LLM 的提示词**，外加 `get_context` 拼给下游 Agent 的**上下文模板**（不是 LLM 指令，但决定最终 Prompt 长什么样）。

### 7.1 长期三元组抽取（`_TRIPLE_EXTRACT_PROMPT`）

**何时用：** `add_message(role=user)` → `TripleExtractor.extract`（默认同步队列异步执行）。  
**输入占位：** `{content}` = 当前用户消息（经脱敏后）。

```text
从用户消息中提取值得长期记忆的事实三元组。

规则：
1. 只提取稳定、可复用的信息（偏好、身份、计划、重要事实），忽略寒暄。
2. 每个三元组格式：subject|predicate|object|action
   - action = "set"（新增/更新）或 "delete"（用户明确否定/不再适用）
3. 识别"信息冲突"与"遗忘"：
   - "我不喝咖啡了" → action=delete, subject=用户, predicate=偏好
   - "我改用尾号5678的卡" → 先 delete 旧卡，再 set 新卡
4. 无价值则返回空列表。

用户消息：
{content}

输出 JSON 数组：
[{"subject":"...","predicate":"...","object":"...","action":"set|delete"}]
```

**设计要点：** 用 `action=set|delete` 把遗忘/冲突写进提示词，而不是事后规则猜；输出强制 JSON 数组，解析失败则跳过（不挡主问答）。

### 7.2 短期超预算摘要（`ContextCompressor._summarize`）

**何时用：** `get_context` → `fit_budget` 发现短期对话超过剩余 Token 预算时。  
**输入：** 被挤出「最近对话」窗口的更早消息（`role: content` 逐行拼接）。

```text
以下是历史对话，请压缩为一段摘要，保留关键事实、用户偏好、未完成任务：

{history}

摘要：
```

**设计要点：** 只压「更早」段，最近原文尽量保留；摘要失败则字符截断保底（见第 6 节）。压缩后拼回 Prompt 的形态：

```text
[历史摘要]
{summary}

[最近对话]
{recent_text}
```

### 7.3 拼进 Ask 的上下文模板（`get_context` + `fit_budget`）

不调记忆侧 LLM，只组装字符串，作为增强后的「用户侧上下文」交给 LangGraph：

```text
【长期记忆】
- {subject} {predicate} {object}
...

【用户画像】
- {pref_key}: {value}
...

【近期对话】          # 未超预算时
{role}: {content}
...

【当前输入】
{current_input}
```

说明：`source` 字段存库可追溯，**当前不注入**上述模板；画像键来自三元组里 `subject=用户` 且 `predicate ∈ {偏好,身份,计划}` 的投影。

### 7.4 提示词速查

| 用途 | 常量 / 方法 | 触发频率 |
|------|-------------|----------|
| 抽长期事实 | `_TRIPLE_EXTRACT_PROMPT` | 每条成功写入的 user 消息（可异步） |
| 压短期历史 | `_summarize` 内联 prompt | **仅** Token 超预算 |
| 上下文骨架 | `get_context` 固定标题 | 每次 Ask 读记忆 |

---

## 8. 生产级能力与诚实边界

### 已具备

- 用户/会话隔离、TTL、滑动窗口  
- Redis / Milvus / MySQL 失败降级  
- PII 脱敏（`common.sanitizer`）  
- 异步抽取不挡主路径；有界队列  
- 冲突覆盖与遗忘；审计表  
- `/health` 暴露各层 ping  
- 单测：`tests/test_memory_isolation.py`

### 尚未（或仅雏形）

- Redis 集群/哨兵、写失败重试与幂等  
- 图执行失败时的对话落库（当前成功才写）  
- 字段级加密、用户「删除我的记忆」完整合规 API  
- 短期 Redis 层专项压测报告  

面试口径：**可上线的企业级雏形**，不是「大厂记忆中台全家桶」。

---

## 9. 代码索引

| 主题 | 文件 | 符号 |
|------|------|------|
| 短期键 / Redis 窗口 | `_008_memory/stores.py` | `short_term_key`, `RedisShortTermStore` |
| 压缩 | `_008_memory/memory_manager.py` | `ContextCompressor`, `fit_budget`, `_summarize` |
| 抽取提示词 / 冲突 | 同上 | `_TRIPLE_EXTRACT_PROMPT`, `TripleExtractor`, `_apply_triple` |
| 读写总控 | 同上 | `add_message`, `get_context` |
| Milvus / MySQL / 审计 | `_008_memory/backends.py` | `MilvusLongTermStore`, `MySQLProfileStore`, `MySQLAuditStore` |
| 工厂与降级 | `_008_memory/factory.py` | `create_memory_manager`, `_create_*` |
| API 接入 | `_005_fastapi/main.py` | `_read_memory`, `_write_memory`, `health` |
| 前端 Header | `_006_streamlit/app.py` | `X-User-ID`, `X-Session-ID` |
| 演示 / 测试 | `_000_demo/demo_memory.py`, `tests/test_memory_isolation.py` | — |

---

## 10. 配置要点（`common/.env`）

```
REDIS_* / MYSQL_* / MILVUS_*
MILVUS_MEMORY_COLLECTION=agent_long_term_memory
MEMORY_SHORT_ROUNDS / MEMORY_SHORT_TTL_SECONDS
MEMORY_ASYNC_EXTRACT / MEMORY_EXTRACT_WORKERS / MEMORY_EXTRACT_QUEUE_SIZE
MEMORY_LONG_BACKEND / MEMORY_PROFILE_BACKEND / MEMORY_AUDIT_BACKEND
EMBEDDING_MODEL_PATH   # BGE，长期检索与图谱实体匹配共用
```

模板：`common/.env.example`。

---

## 11. 验证命令

```bash
python _000_demo/demo_memory.py
python -m unittest tests.test_memory_isolation -v
curl http://127.0.0.1:8000/health
```
