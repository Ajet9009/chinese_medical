# 草本通 — 中医药知识图谱问答系统

基于 **Neo4j 知识图谱 + LangGraph + 大语言模型** 的中医药智能问答系统。

将非结构化中医药文本（古籍、教材、百科）转化为结构化知识图谱，结合 LLM 实现自然语言问答，覆盖方剂、药材、症状、疾病、功效、经络、典籍等领域。

---

## ✨ 核心能力

- **意图识别**：LLM 判断问题是否属于中医领域，自动路由
- **知识库**：顶栏「知识库」上传方剂/本草/典籍/医案；解析、向量化后进入 FAISS+BM25 检索。科室 ACL 与版本回滚。管理员另有「知识治理」补录责任人、适用区域、有效期
- **实体抽取**：从自然语言问题中抽取六类中医实体（症状/疾病/方剂/药材/功效/出处）
- **实体抽取**：从自然语言问题中抽取六类中医实体（症状/疾病/方剂/药材/功效/出处）
- **向量匹配**：FAISS + BGE-large-zh-v1.5 将口语化实体匹配到知识图谱标准实体
- **Cypher 生成**：LLM 结合图 schema 生成查询语句，EXPLAIN 校验 + 错误自动修正
- **知识问答**：图查询结果 → LLM 生成自然语言回答
- **全链路追踪**：Langfuse 记录每次请求的 Trace/Span/Generation，含 token 用量、成本、TTFT
- **评测飞轮**：黄金集 `eval/golden_qa.json`（方剂/本草/证候/文献/拒答）。管理页点踩可「标为黄金集」待标注；`python scripts/eval_golden.py` 做离线关键词命中，CI 不打线上 LLM。Redis/索引等失败走 `obs.degraded`，挂在 `/health` 与系统运行状态。不是旁路评测子系统，也没有知识自进化

---

## 🏗️ 架构

```
用户问题
  │
  ▼
intent_recognition (LLM) ──→ 是否中医问题？
  │                              │
  │ 否                           │ 是
  ▼                              ▼
general_response (LLM)    entity_extraction (LLM)
  直接回答                  抽取六类实体
  │                              │
  ▼                              ▼
END                       entity_normalization (FAISS+BGE)
                          向量匹配标准实体
                              │
                              ▼
                          cypher_generation (LLM)
                          生成 Cypher + EXPLAIN 校验修正
                              │
                              ▼
                          cypher_executor (Neo4j)
                          执行查询，结果去重去噪
                              │
                              ▼
                          answer_generation (LLM)
                          结合图谱结果生成回答
                              │
                              ▼
                             END
```

---

## 🛠️ 技术栈

| 层 | 技术 |
|----|------|
| 图数据库 | Neo4j 4.4 |
| 向量检索 | FAISS + sentence-transformers (BGE-large-zh-v1.5) |
| 大模型 | DeepSeek (OpenAI 兼容接口) |
| 编排框架 | LangGraph 1.x + LangChain 1.x |
| API 服务 | FastAPI + Uvicorn（SSE 流式） |
| 前端 | Streamlit |
| 可观测性 | Langfuse（自托管） |
| 微调模型 | vllm + LoRA (Qwen2.5-1.5B) |

---

## 📁 目录结构

```
_000_demo/                  验证脚本（API 调用、Streamlit 旧版）
_001_crawler/               数据爬取（中药/方剂索引 + 详情页）
_002_extract_information/   知识抽取（LLM 实体关系抽取 + 规则层合并）
_003_create_neo4j_database/ Neo4j 入库 + FAISS 索引构建
_004_langgraph_more_nodes/  LangGraph 图节点（核心）
├── state.py                  GraphState 定义 + KG 实体/关系常量
├── graph.py                  图编排（8 节点 + 条件路由）
├── standalone_query.py       节点0: 多轮指代消解
├── intent_recognition.py     节点1: 意图识别
├── entity_extraction.py     节点2: 六类实体抽取
├── entity_normalization.py   节点3: FAISS+BGE 向量匹配
├── cypher_generation.py      节点4: Cypher 生成 + 校验修正
├── cypher_executor.py        节点5: 图查询执行
├── answer_generation.py      节点6: 最终回答（token 流式）
└── general_response.py       节点7: 非中医问题回答
_005_fastapi/               FastAPI 服务（/ask + 会话 CRUD + SSE）
_006_streamlit/             Streamlit 前端（遗留）
_007_fine_tune/             vllm 微调模型客户端（LoRA 多适配器）
frontend/                   Vue 3 Chat（主前端，端口 5174，避开电网项目 5173）
common/                     公共模块
├── env_loader.py             加载 common/.env 与根目录 .env
├── neo4j_manager.py          Neo4j 连接/导入/校验/执行/元数据
├── faiss_vector_store.py     FAISS 向量存储
├── redis_client.py           Redis 连接（失败返回 None）
├── conversation_store.py     SQLite 会话 + Redis List/LTRIM
├── langfuse_manager.py       Langfuse 客户端（追踪/采样/成本）
├── obs.py                    降级打点（内存计数，失败不抛）
├── eval_golden.py            黄金集校验 / 离线关键词命中
├── sanitizer.py              PII 脱敏
└── export_neo4j_metadata.py  导出图 schema
eval/golden_qa.json         中医黄金集（data/ 已 gitignore）
scripts/eval_golden.py       校验；--score-file 离线打分；--live 仅本机
docker-compose.yml          仅 Redis（Neo4j 用本机实例）
tests/                      单元测试
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Neo4j 4.4（本地 7687 端口）
- Docker（用于本仓库 Redis；Langfuse 可选）

### 第 1 步：启动 Redis（本仓库 compose，仅 Redis）

```bash
docker compose up -d redis
```

若本机 6379 已被其它 compose（例如 `E:\devapp\milvus_data`）占用，不要再起本服务，把 `.env` 的 `REDIS_URL` 指到已有 Redis。Langfuse 仍可用原有基础设施。

### 第 2 步：启动 Neo4j

```bash
cd E:\devapp\neo4j-community-4.4.41\bin
neo4j.bat console        # 或 neo4j.bat start
```

### 第 3 步：配置环境变量

复制 `.env.example` 为仓库根目录 `.env`（或 `common/.env`）并填写。根目录 `.env` 会覆盖 `common/.env`。

```bash
# 大模型
MODEL_API_KEY=sk-xxx
MODEL_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-v4-flash

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=xxx

# 向量检索
EMBEDDING_MODEL_PATH=path/to/bge-large-zh-v1.5
FAISS_INDEX_PATH=path/to/entities.index
FAISS_METADATA_PATH=path/to/entities.json

# 会话
REDIS_URL=redis://localhost:6379/0
CONVERSATION_DB_PATH=data/conversations.sqlite
REDIS_HISTORY_MAX=50
GRAPH_HISTORY_LIMIT=6
TOKEN_BUDGET=2000
DOC_RAG_ENABLE=1
DOC_SOURCE_DIR=corpus
KNOWLEDGE_DOCS_DIR=data/docs

# Langfuse（可观测性）
LANGFUSE_SECRET_KEY=sk-lf-xxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_ENABLED=true
LANGFUSE_SAMPLE_RATE=1.0
```

### 第 4 步：安装依赖

```bash
pip install -r requirements.txt
npm --prefix frontend install
```

### 第 5 步：启动服务

```bash
# 终端 1：API 服务（端口 8000）
python -m _005_fastapi.main

# 终端 2：Vue Chat（端口 5174，避开电网项目 5173）
npm --prefix frontend run dev
```

访问：
- 前端：http://localhost:5174
- API 文档：http://localhost:8000/docs
- Langfuse：http://localhost:3000（若已启动）

遗留 Streamlit：`python _006_streamlit/run.py`（端口 8501，不回传会话）。

---

## 📡 API 接口

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "四君子汤有什么功效？"}'
```

返回完整结构（意图、实体、匹配、Cypher、回答）。

### POST /ask/stream（SSE 流式）

```bash
curl -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "四君子汤有什么功效？"}'
```

SSE 事件：`session`（会话 id）→ `progress`（图节点）→ `token`（仅最终回答节点）→ `done`。可带 `conversation_id` 做多轮。

---

## 📊 可观测性（Langfuse）

- **Trace**：一次 API 请求一条，含 request_id/session_id
- **Span**：7 个图节点各一个，记录执行顺序 + 耗时
- **Generation**：每次 LLM 调用，含模型名、token 用量、成本、TTFT
- **Prompt 管理**：5 个提示词在 Langfuse UI 集中管理，改后 60s 自动生效
- **成本核算**：按 token 自动计算成本（输入 1元/1M，输出 3元/1M）

---

## 🧪 测试

```bash
# 全量单测
pytest tests/ -q

# Langfuse 集成测试
pytest tests/test_langfuse_integration.py -v

# 单节点测试（每个文件有 main()）
python -m _004_langgraph_more_nodes.intent_recognition
python -m _004_langgraph_more_nodes.cypher_generation

# 黄金集校验（不调 LLM）
python scripts/eval_golden.py

# API 调用验证
python _000_demo/demo_api.py "四君子汤有什么功效？"
```

---

## 📝 License

MIT
