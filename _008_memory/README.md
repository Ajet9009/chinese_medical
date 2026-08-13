# 企业级 Agent 记忆系统

分层记忆框架：短期（Redis）+ 长期（向量库）+ 画像（结构化 DB），含 Token 预算压缩与实体遗忘。

## 架构

```
add_message 写入
  ├─ 短期记忆：Redis 滑动窗口（LPUSH + LTRIM，最近 N 轮）
  ├─ 价值判断：LLM 提取三元组（忽略寒暄）
  ├─ 信息冲突：subject+predicate 唯一，新值覆盖旧值
  └─ 实体遗忘：action=delete 删除旧记忆

get_context 读取
  ├─ 长期记忆：向量检索 top_k 相关三元组
  ├─ 用户画像：结构化偏好
  ├─ 短期记忆：最近 N 轮对话
  └─ Token 预算：超阈值触发摘要压缩（不截断）
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `stores.py` | 三层存储抽象 + Redis/向量/画像实现 |
| `memory_manager.py` | MemoryManager 主类 + 三元组提取 + Token 预算 |

## 解决的两个核心问题

### 1. 幻觉（Hallucination）

- 每条长期记忆带 `source`（来源消息）+ `timestamp`，可追溯。
- 检索时按 `user_id` 隔离，避免串记忆。
- 只返回有来源的事实，不凭空生成。

### 2. 信息冲突（Information Conflict）

- 三元组以 `(subject, predicate)` 为唯一键。
- 写入新值前先删除旧值 → 不重复存储矛盾信息。
- 否定意图（"我不用那个卡了"）→ `action=delete` 直接删除。
- 冲突时取 `timestamp` 最新的。

## Token 预算管理

超阈值（默认 4k）不直接截断，而是：
1. 从后往前保留最近消息，直到剩余放得下
2. 前半段用 LLM 摘要压缩
3. 摘要失败降级为截断（保底）

## 使用示例

```python
from _008_memory.memory_manager import MemoryManager, MemoryConfig
from _008_memory.stores import RedisShortTermStore, LangChainLongTermStore, DictProfileStore

mgr = MemoryManager(
    short_term=RedisShortTermStore(redis_client),
    long_term=LangChainLongTermStore(vectorstore, embeddings),
    profile=DictProfileStore(),
    llm=llm,
    embeddings=embeddings,
    tokenizer=tiktoken_tokenizer,
    config=MemoryConfig(token_budget=4000),
)

# 写入
mgr.add_message("u1", "s1", "user", "我喝咖啡，不用尾号1234的卡了")

# 读取
context = mgr.get_context("u1", "s1", "给我推荐一家咖啡馆")
```

## 技术栈

- Python 3.10+
- LangChain（VectorStore + LLM 抽象）
- Redis（短期存储）
- tiktoken（Token 计算）
- Postgres / Mongo（用户画像，可替换）
