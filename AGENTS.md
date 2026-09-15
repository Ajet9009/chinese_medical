# 草本通 — Agent 手册

给 Cursor / Claude Code 的项目共识。架构百科用 `README.md` 与 `docs/系统架构.md`。本机盘符与 Neo4j 启动路径写 `CLAUDE.local.md`（不入库）。改约定只改本文件。

## 锁栈

- Python 3.10+，conda 环境名 **grid-qa**，禁止另建 `venv/`。
- 主前端：Vue 3 + Vite + Pinia，端口 **5174**（避开电网 5173）。
- API：FastAPI + JWT，`127.0.0.1:8000`。`/ask` 与 `/ask/stream` 必须带登录，禁止示范裸 POST。
- 编排：LangGraph。拓扑只认 `_004_langgraph_more_nodes/graph.py`（九节点：`standalone_query` → `intent_recognition` → 中医抽取/归一/Cypher/执行 → `doc_retrieval` → 回答；非中医可走文献或 `general_response`）。
- 存储：Neo4j 4.4 管图；**实体 FAISS** 与 **文献 FAISS** 禁止混用；SQLite 管会话与知识登记；Redis 管热缓存。
- 产品对齐电网分层与完成度，**禁止抄** grid-qa 业务壳（两票、数字孪生、SCADA、Prometheus 全家桶、知识自进化）。

## 红线

- 不要臆造 `GraphState` 字段，以 `_004_langgraph_more_nodes/state.py` 为准。
- 不要读取或提交 `.env`、`common/.env`；改配置只动 `.env.example` 并说明用户本地要填的键。
- 先 Grep/Glob，再 Read 2–3 个文件；禁止无目标海读；禁止把 `data/logs/app.log` 整文件堆进主会话。
- 实体对齐：查询加类型前缀（如 `Symptom: 咳嗽`）；骤降 `score[i]/score[i-1] < 0.6` 且绝对分 ≥ 0.6。Cypher 逐条 `EXPLAIN`，最多 3 次修正。

## 短期记忆（两条，不要说成一件事）

- Redis **LTRIM**：按**条数**裁热缓存（`REDIS_HISTORY_MAX`）。失败 fail-open 回 SQLite，并 `degraded("redis*")`。
- **ContextCompressor**：按 **token** 决定进模型的历史。`token_budget`（默认 2000，环境变量 `TOKEN_BUDGET`）+ `fit_budget` 从最新往旧留；放不下则旧轮次 LLM 摘要，摘要失败降级。`CONTEXT_COMPRESS=0` 时退回 `GRAPH_HISTORY_LIMIT` **条数**截断。
- LTRIM 不管 token；Compressor 不管 Redis 列表长度。长期记忆在 `_008_memory/`，失败不挡问答。

## 命令

```bash
docker compose up -d redis
conda activate grid-qa
python -m uvicorn _005_fastapi.main:app --host 127.0.0.1 --port 8000
npm --prefix frontend run dev
pytest tests/ -q
```

默认登录 `admin` / `admin123`。根目录 `.env` 覆盖 `common/.env`。Neo4j 用本机 7687，不进 compose。6379 已被占用则改 `REDIS_URL`，勿再起一个 Redis。

## 入口

- 问答：`_005_fastapi/main.py`
- 图：`_004_langgraph_more_nodes/graph.py`
- token 压缩：`common/context_compressor.py`（问答路径）与 `_008_memory`（长期记忆路径）
