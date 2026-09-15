# RAG 检索优化落地说明

更新时间：2026-09-15

## 结论

当前知识库可以支撑“中医问答”的基础检索，但更适合回答方剂、药材、功效、组成、出处、主治等事实型问题。若要覆盖用户资讯类、辨证建议类问题，还需要继续补齐证型、禁忌、适用人群、现代病名映射、来源依据和安全提醒。

## 繁简体

繁体字会影响纯向量和 BM25 的召回稳定性，尤其是“麻黄汤/麻黃湯”“证/證”“气/氣”等词。当前已在 query 和 chunk metadata 两侧做繁简归一：

- `common/text_normalization.py` 生成简体/繁体变体。
- `common/doc_chunking.py` 在子块 metadata 中写入 `text_simplified`、`text_traditional`、`title_simplified`、`title_traditional`。
- `common/doc_store.py` 的 BM25 会把正文、父标题、章节路径、繁简字段合并检索。

## 子块 metadata 与 FAISS 匹配

子块 metadata 存在两处：

- 命令行或目录入库：`data/faiss/docs.json` 保存 chunk metadata 列表。
- 管理页上传知识库：SQLite `kb_chunks` 保存父块和繁简 metadata；重建索引时再写入 `data/faiss/docs.json`。

FAISS 本身只保存向量，不保存业务字段。匹配方式是数组下标一一对应：

- `FaissDocStore.build_chunks()` 按 `chunks` 顺序编码文本并写入 FAISS。
- 同一顺序的 `chunks` 写入 `docs.json`。
- 检索时 FAISS 返回向量下标 `idx`，代码用 `self.chunks[idx]` 取回 metadata。
- 加载时会校验 `index.ntotal == len(chunks)`，避免向量数量和 metadata 数量错位。

## 分块策略

旧逻辑是固定字符窗口，容易切断方剂条目、章节标题和上下文。当前已改为父子分块：

- 先清理 Markdown front matter。
- 按 Markdown 标题切父块。
- 子块优先按段落聚合，超长段落才回退字符窗口。
- 子块保留 `parent_id`、`parent_title`、`section_path`、`parent_text_preview`。

父子分块目前仍以子块入向量库，父块 metadata 用于标题命中、解释上下文和后续 rerank。后续若要进一步增强，可增加“命中子块后扩展相邻子块/父块摘要”的上下文拼接。

## 是否迁移 Milvus

现阶段不建议立即迁移。当前文献 chunk 规模较小，FAISS 更简单、稳定、部署成本低。已经新增 `DOC_VECTOR_BACKEND` 后端工厂，默认仍是 `faiss`；显式配置 `milvus` 会给出未接入提示，避免误以为已经切库。

建议在以下条件满足后再迁移 Milvus：

- 文档 chunk 达到十万级以上，或需要多租户、多集合、在线增量删除。
- 需要分布式服务、可观测索引状态和独立向量库运维。
- 已设计好 Milvus collection schema、metadata 过滤字段和全量回填脚本。

## Query 改写

当前已加入低成本 query 扩展，包括繁简变体和常见中医术语替换。除了指代消除，后续可按优先级加入：

- 压缩：多轮问题先保留实体、证型、症状、禁忌等关键词，减少闲聊噪声。
- 子问题拆分：适合“组成、功效、禁忌一起问”的复合问题。
- HyDE：只建议在召回弱、但问题明确时启用；中医场景容易生成似是而非的假依据，需要保守开关。

## 召回与重排

当前文献检索路径已经包含：

- dense FAISS 召回。
- BM25 稀疏召回。
- RRF 融合。
- MMR 多样性筛选。
- 标题/父块/正文/繁简字段的规则重排。
- 可选 CrossEncoder/BGE reranker：配置 `DOC_RERANK_MODEL_PATH` 后启用，失败自动降级。

推荐下一步用 RAGAS/黄金集对比四组配置：dense only、hybrid、hybrid+rule、hybrid+rule+reranker。以 `context_recall` 和 `faithfulness` 为主指标，不只看相似度。
