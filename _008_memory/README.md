# 企业级 Agent 记忆系统

分层记忆：短期 Redis + 长期 Milvus + 画像 MySQL + 审计；失败自动降级。

## 本地服务映射

| 层 | 服务 | 配置 | 说明 |
|----|------|------|------|
| 短期 | Redis `localhost:6379` | `REDIS_*` | key=`mem:short:{user}:{session}` + TTL |
| 画像 | MySQL `subjects_kg` | `MYSQL_*` | 表 `agent_user_profile`（自动建表） |
| 长期 | Milvus `19530` / db `itcast` | `MILVUS_*` | collection **`agent_long_term_memory`**（独立，不占用 `edurag`） |
| 审计 | MySQL | `MEMORY_AUDIT_BACKEND` | 表 `agent_memory_audit`（无完整 PHI，仅预览） |

降级：Redis→内存；Milvus→`data/memory/long_term.json`；MySQL→`profile.json`；审计→`memory_audit.jsonl`。

## 生产级特性

- **用户隔离**：短期键必须 `user_id + session_id`
- **有界抽取队列**：`ThreadPoolExecutor` + 信号量；队列满丢弃并审计
- **PII 脱敏**：写入前 `sanitize`
- **冲突覆盖 / 遗忘**：同 `subject+predicate` 覆盖；`action=delete` 删除
- **连接池**：MySQL Profile/Audit 复用连接
- **健康检查**：`GET /health` 返回各层后端与 ping 结果

## Cursor MCP（已写入）

- 用户级：`C:\Users\Administrator\.cursor\mcp.json`
- 项目级：`.cursor/mcp.json`

| MCP 名 | 用途 |
|--------|------|
| `local-mysql` | 查/改 `subjects_kg` |
| `local-redis` | Redis 命令 |
| `local-milvus` | Milvus（db=`itcast`） |

改完 mcp.json 后请在 Cursor：**Settings → MCP → Refresh**。

## 架构

```
add_message
  ├─ PII 脱敏
  ├─ Redis 短期（user+session）
  └─ 有界线程池抽三元组 → Milvus + MySQL 画像 + 审计

get_context
  ├─ Milvus 按 user 检索
  ├─ MySQL 画像
  └─ Redis 近期对话 + Token 预算压缩
```

## 演示 / 测试

```bash
python _000_demo/demo_memory.py
python -m unittest tests.test_memory_isolation -v
```
