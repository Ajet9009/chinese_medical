# RAG 检索增强设计

## 背景

当前草药通问答链路采用 Neo4j 图谱 + 文献 RAG + LangGraph 编排。Neo4j 负责中医实体、方剂组成、功效、病症关系；文献 RAG 负责典籍、本草、方剂、医案等原文依据。

现有 RAG 已有 FAISS dense 检索、BM25、RRF 融合、MMR 去冗余和 CRAG 改写重检索，但仍存在四类问题：

- 文献中繁体内容占比较高，简体用户问题在 BM25 和关键词命中上会损失召回。
- 固定 400 字 + 80 overlap 的字符切块容易截断章节、方剂条目、医案上下文。
- 当前 RRF + MMR 不是语义 rerank，标题、实体、章节命中未被系统性提权。
- 查询改写主要是指代消除和 CRAG 低质量改写，缺少低成本的中医术语/别名扩展。

## 目标

1. 提升简体用户问题对繁体文献的召回能力。
2. 将文献分块从固定字符窗口升级为结构化父子分块。
3. 增加双层 rerank：默认规则 rerank，可选本地 BGE reranker。
4. 增加低成本查询扩展：繁简归一、中医术语/别名扩展。
5. 保留 FAISS 作为默认文献向量后端，同时抽象出可选 Milvus 后端接口。
6. 保持 Neo4j 图谱职责不变，不将图谱知识迁移到 Milvus。

## 非目标

- 本次不把 Neo4j 实体关系迁移到 Milvus。
- 本次不默认开启 HyDE。
- 本次不默认开启子问题拆分。
- 本次不把问答系统定位为个体化诊疗或处方决策系统。
- 本次不强制引入外部 rerank API。

## 总体方案

保留当前主链路：

```text
用户问题
→ standalone_query 指代消除
→ intent_recognition 意图识别
→ entity_extraction / entity_normalization
→ cypher_generation / cypher_executor
→ doc_retrieval 文献检索
→ answer_generation 最终回答
```

增强后的文献检索链路：

```text
检索问句
→ 繁简归一
→ 中医术语/别名扩展
→ dense 召回
→ BM25 召回
→ 标题/章节召回
→ RRF 融合
→ 规则 rerank
→ 可选 BGE reranker
→ MMR 去冗余
→ 父子块上下文扩展
→ CRAG 判定，必要时改写重检索
```

## 繁简归一

入库时保留原文，同时生成检索字段：

- `text`: 原文，用于展示、引用和最终回答。
- `text_simplified`: 简体归一文本，用于 BM25 与关键词命中。
- `text_traditional`: 繁体归一文本，用于繁体 query 兼容。
- `title_simplified`: 标题简体归一。
- `title_traditional`: 标题繁体归一。

检索时对 query 生成变体：

- 原始 query。
- 简体 query。
- 繁体 query。
- 中医别名/术语扩展 query。

默认展示仍使用原文，避免破坏典籍原貌。

## 结构化父子分块

新增结构化分块流程：

```text
读取 Markdown / TXT / PDF
→ 清理 YAML front matter 和来源说明类噪声
→ 按 Markdown 标题切父块
→ 父块过长时按段落、条文、空行切子块
→ 子块仍过长时再按字符窗口兜底
→ 子块入向量索引
→ 父块 metadata 随子块保存
```

每个子块 metadata 至少包含：

- `doc_id`
- `doc_name`
- `doc_type`
- `chunk_idx`
- `parent_id`
- `parent_title`
- `section_path`
- `text`
- `text_simplified`
- `text_traditional`
- `parent_text_preview`

检索命中子块后，回答上下文默认返回子块原文；必要时可带相邻子块或父块摘要。

## 向量后端

新增文献向量后端抽象，但默认仍使用 FAISS。

配置：

```text
DOC_VECTOR_BACKEND=faiss
DOC_VECTOR_BACKEND=milvus
```

FAISS 实现沿用本地文件：

- `DOC_FAISS_INDEX_PATH`
- `DOC_FAISS_METADATA_PATH`

Milvus 实现作为可选后端，使用同一套 chunk metadata。Milvus 不承担图谱实体关系存储，只承担文献向量检索。

## Rerank 策略

默认启用规则 rerank，规则信号包括：

- 标题或章节命中。
- 标准实体命中。
- query 繁简归一命中。
- 关键词覆盖率。
- 文档类型权重。
- dense / BM25 / RRF 分数。

如果配置 `RERANK_MODEL_PATH`，则在规则 rerank 后启用本地 BGE reranker。

```text
RERANK_ENABLE=1
RERANK_MODEL_PATH=
RERANK_TOP_N=20
DOC_TOP_K=4
```

无 rerank 模型时系统必须正常运行，不允许因为缺模型影响主问答路径。

## 查询改写策略

默认开启低成本策略：

- 指代消除：沿用 `standalone_query`。
- 繁简归一：默认开启。
- 中医术语/别名扩展：默认开启。
- CRAG 低质量改写：沿用现有逻辑。

默认关闭高成本策略：

- HyDE。
- 子问题拆分。

HyDE 和子问题拆分后续仅在低召回、复杂多实体或对比型问题中配置化启用。

## 回答安全边界

回答继续遵守当前原则：

- 优先依据 Neo4j 图谱。
- 文献只作为补充依据。
- 图谱和文献都没有时拒答。
- 不编造书名、页码、剂量、处方禁忌。
- 面向中医知识科普和资料检索，不输出个体化诊疗结论。

## 测试策略

采用 TDD。

新增或扩展测试覆盖：

- 简体 query 能命中繁体文献原文。
- 繁体 query 能命中简体/繁体混合文献。
- Markdown 标题和段落切分能产生父子块。
- YAML front matter 不进入检索答案正文。
- 标题/实体命中的 chunk 在规则 rerank 后前置。
- 无 `RERANK_MODEL_PATH` 时规则 rerank 正常运行。
- FAISS 默认后端行为保持兼容。
- RAGAS dataset 重新生成后拒答样本仍为空检索。
- RAGAS answer relevancy 相比当前抽取式长块拼接有改善。

## 验收标准

1. `pytest tests -q` 通过。
2. `python scripts/eval_ragas.py all --no-crag` 可重新生成评测集和报告。
3. 简体问题 `人参味如何、主补什么？` 能召回包含 `人參 味甘小寒。主補五臟` 的原文块。
4. `麻黄汤治什么？` 能优先召回 `麻黃湯` 或 `麻黄汤` 标题所在的方剂块。
5. 后端不配置 rerank 模型、不配置 Milvus 时仍能用 FAISS 正常启动。
6. `.env.example` 说明新增配置项。

## 风险与缓解

- 风险：结构化分块改变 chunk 数量，可能影响既有索引文件。
  - 缓解：新增索引构建逻辑和测试，保留 FAISS 默认路径，必要时可重建。
- 风险：繁简转换依赖缺失。
  - 缓解：提供轻量 fallback，缺 OpenCC 时至少保留原文检索。
- 风险：BGE reranker 增加内存占用。
  - 缓解：默认不开模型 rerank，只有配置 `RERANK_MODEL_PATH` 时加载。
- 风险：HyDE 生成虚假内容污染检索。
  - 缓解：本次默认不启用 HyDE。
- 风险：Milvus 增加部署复杂度。
  - 缓解：Milvus 仅为可选后端，FAISS 继续作为默认稳定路径。
