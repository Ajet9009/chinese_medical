#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""中医知识图谱抽取器

读取详情页结构化 JSON（data/中药/*.json、data/方剂/*.json），
使用 langChain 管道链 `prompt | llm | parse` 抽取实体与关系。

输出:
  - data/neo4j/entities.json         去重后的实体节点（含 id，供 Neo4j 导入）
  - data/neo4j/relations.json        关系列表（引用实体 id，供 Neo4j 导入）
  - data/aipaca/alpaca.json  Alpaca 格式微调数据

特性:
- 多线程: ThreadPoolExecutor，默认 5 个 worker
- 断点续爬: 每页抽取结果缓存到 data/kg_cache/<type>/，重跑自动跳过
- 测试模式: --limit N

用法:
  # 中药（主 Agent）
  python kg_extractor.py --source-dir ../data/中药 --type 中药 --limit 2
  # 方剂
  python kg_extractor.py --source-dir ../data/方剂 --type 方剂 --limit 2
  # 全量（断点续爬）
  python kg_extractor.py --source-dir ../data/中药 --type 中药
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ============================================================
# 一、输出类型定义（parse 使用的类）
# ============================================================

EntityType = Literal["Symptom", "Disease", "Formula", "Herb", "Effect", "Source"]
RelationType = Literal[
    "TREATS_DISEASE",
    "ALLEVIATES_SYMPTOM",
    "HAS_EFFECT",
    "HAS_INGREDIENT",
    "HAS_SYMPTOM",
    "FROM_SOURCE",
]


class FormulaAttributes(BaseModel):
    """方剂属性字段（文本主要讲方剂时补充，值为空则省略）。"""

    出处: Optional[str] = None
    组成: Optional[str] = None
    功用: Optional[str] = None
    主治: Optional[str] = None
    用法: Optional[str] = None
    功效分类: Optional[str] = None


class HerbAttributes(BaseModel):
    """药材属性字段（文本主要讲药材时补充，值为空则省略）。"""

    性味: Optional[str] = None
    归经: Optional[str] = None
    功效: Optional[str] = None
    主治: Optional[str] = None
    用法用量: Optional[str] = None
    别名: Optional[str] = None
    出处: Optional[str] = None


class Entity(BaseModel):
    """知识图谱实体节点。

    - name: 实体名称
    - type: 实体类型（枚举，见 EntityType）
    - attributes: 方剂/药材的属性字段；Symptom/Disease/Effect/Source 等类型为 None
    """

    name: str
    type: EntityType
    attributes: Optional[Union[FormulaAttributes, HerbAttributes]] = None


class Relation(BaseModel):
    """知识图谱关系。

    - subject/subject_type: 起点实体的名称及类型
    - relation: 关系类型（枚举，见 RelationType）
    - object/object_type: 终点实体的名称及类型
    """

    subject: str
    subject_type: EntityType
    relation: RelationType
    object: str
    object_type: EntityType


class TCMKnowledgeGraph(BaseModel):
    """单次抽取的完整输出。

    若文本仅描述单个实体、未涉及其他实体或关系，应返回空结构：
    {"entities": [], "relations": []}
    """

    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


# ============================================================
# 二、配置与常量
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "kg_cache"           # 每页抽取结果缓存（断点续爬）
NEO4J_DIR = DATA_DIR / "neo4j"              # 入 Neo4j 的 JSON
ALPACA_DIR = DATA_DIR / "aipaca"   # Alpaca 微调数据

DEFAULT_WORKERS = 5
REQUEST_INTERVAL = 0.1

SYSTEM_TEMPLATE = """你是一个中医知识图谱抽取专家。请从给定的中医文本中抽取实体和关系。

【实体类型说明】
- Symptom：症状，如咳嗽、腹痛等
- Disease：疾病，如感冒、肺炎、肾虚等
- Formula：方剂，如四君子汤、桂枝汤等
- Herb：药材，如人参、黄芪、丁香等
- Effect：功效，如补气、活血、祛湿、止痛等
- Source：出处，如《本草纲目》《伤寒论》等

【关系类型说明】
- TREATS_DISEASE：方剂或药材治疗某种疾病
- ALLEVIATES_SYMPTOM：方剂或药材缓解某种症状
- HAS_EFFECT：方剂或药材具有某种功效
- HAS_INGREDIENT：方剂包含某种药材
- HAS_SYMPTOM：疾病包含某种症状
- FROM_SOURCE：方剂出自某文献或出处

【限制】
1. 如果文本中仅描述单个实体的信息、未涉及其他实体或关系，请不要抽取，返回空结构：{{"entities": [], "relations": []}}
2. 若文本涉及方剂或药材，请补充对应的属性字段（如功效、性味、剂量等）。
   如果文本主要是讲方剂的，请不要抽取药材的属性字段。
   如果文本主要是讲药材的，请不要抽取方剂的属性字段。
   如果值为空 null，则不必显示键的值。

所有输出必须严格符合以下 JSON 格式：
{format_instructions}

输入文本：
{text}"""

_lock = threading.Lock()
_stats = {"ok": 0, "err": 0, "empty": 0}


def _build_chain():
    """构建 langChain 管道链 prompt | llm | parse。"""
    load_dotenv(ROOT / "common" / ".env")
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("MODEL_API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL"),
        temperature=0,
        timeout=90,
        max_retries=2,
    )
    parser = JsonOutputParser(pydantic_object=TCMKnowledgeGraph)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_TEMPLATE), ("human", "{text}")]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


# 只送 LLM 的自由文本字段（结构字段交给规则层 build_kg.py，不喂 LLM 省 token）
KEEP_HEADING_KEYWORDS = ("主治", "功效", "功用", "注意", "禁忌", "用法", "药方", "名称")


def detail_to_text(doc: dict) -> str:
    """把详情页结构化 JSON 转成 LLM 输入文本。

    仅保留含自由文本医学信息的章节（主治/功效/功用/注意/禁忌/用法/药方/名称），
    结构字段（分类/组成/性味/归经/性状等）由规则层抽取，不喂 LLM 以省 token 提速。
    """
    lines = [f"名称：{doc.get('name', '')}"]
    for k, v in doc.get("meta", {}).items():
        if v:
            lines.append(f"{k}：{v}")

    def _keep(heading: str) -> bool:
        return any(kw in heading for kw in KEEP_HEADING_KEYWORDS)

    for sec in doc.get("sections", []):
        h2 = sec.get("heading", "")
        children = sec.get("children", [])
        keep_children = [c for c in children if _keep(c.get("heading", ""))]
        if not keep_children and not _keep(h2):
            continue  # 整章无自由文本字段，跳过
        if _keep(h2) or not children:
            lines.append(f"\n【{h2}】")
        for sub in keep_children:
            lines.append(f"\n{sub['heading']}：")
            for b in sub.get("blocks", []):
                if b.get("text"):
                    lines.append(b["text"])
        for b in sec.get("blocks", []):
            if b.get("text"):
                lines.append(b["text"])
    return "\n".join(lines)


# ============================================================
# 三、抽取主流程
# ============================================================

def process_one(item: dict, cache_dir: Path, chain) -> None:
    """抽取单条详情页，结果写入 cache（断点续爬）。"""
    name = item["name"]
    out = cache_dir / f"{name}.json"
    if out.exists() and out.stat().st_size > 0:
        return  # 已缓存
    try:
        doc = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        text = detail_to_text(doc)
        raw = chain.invoke({"text": text})  # JsonOutputParser 返回 dict
        graph = TCMKnowledgeGraph.model_validate(raw)  # 校验为 pydantic 对象

        def _dump_entity(e) -> dict:
            ed = e.model_dump()
            if ed.get("attributes"):
                # 值为空/None 的字段不输出（符合"值为空则省略"）
                ed["attributes"] = {k: v for k, v in ed["attributes"].items() if v is not None}
            return ed

        payload = {
            "name": name,
            "type": item["type"],
            "input": text,
            "entities": [_dump_entity(e) for e in graph.entities],
            "relations": [r.model_dump() for r in graph.relations],
        }
        with _lock:
            if not graph.entities and not graph.relations:
                _stats["empty"] += 1
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with _lock:
            _stats["ok"] += 1
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _stats["err"] += 1
            print(f"      [ERR] {name} -> {exc.__class__.__name__}: {exc}")


def summarize(cache_dir: Path) -> dict:
    """汇总 cache：实体去重加 id，映射关系，返回 {entities, relations}。"""
    entities_by_key: dict[tuple, dict] = {}
    relations: list[dict] = []

    for f in sorted(cache_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for e in data.get("entities", []):
            key = (e["type"], e["name"])
            if key not in entities_by_key:
                entities_by_key[key] = {
                    "id": f"E{len(entities_by_key) + 1}",
                    "type": e["type"],
                    "name": e["name"],
                    "attributes": e.get("attributes"),
                }
            else:
                # 合并属性（补齐空缺字段）
                exist = entities_by_key[key].get("attributes")
                new_attr = e.get("attributes")
                if new_attr:
                    if exist:
                        for k, v in new_attr.items():
                            exist.setdefault(k, v)
                    else:
                        entities_by_key[key]["attributes"] = new_attr
        for r in data.get("relations", []):
            relations.append(r)

    name_to_id = {v["name"]: v["id"] for v in entities_by_key.values()}
    rel_out = []
    seen_rel: set[tuple] = set()
    for r in relations:
        src_id = name_to_id.get(r["subject"], r["subject"])
        tgt_id = name_to_id.get(r["object"], r["object"])
        rel_key = (r["relation"], src_id, tgt_id)
        if rel_key in seen_rel:
            continue
        seen_rel.add(rel_key)
        rel_out.append({
            "relation": r["relation"],
            "subject": src_id,
            "subject_type": r.get("subject_type"),
            "object": tgt_id,
            "object_type": r.get("object_type"),
        })
    return {"entities": list(entities_by_key.values()), "relations": rel_out}


def build_alpaca(cache_dir: Path) -> list[dict]:
    """构建 Alpaca 微调数据：instruction / input / output。"""
    items = []
    for f in sorted(cache_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        output = {
            "entities": data.get("entities", []),
            "relations": data.get("relations", []),
        }
        items.append({
            "instruction": "你是一个中医知识图谱抽取专家。请从给定的中医文本中抽取实体和关系。"
                           "实体类型：Symptom症状、Disease疾病、Formula方剂、Herb药材、Effect功效、Source出处；"
                           "关系类型：TREATS_DISEASE、ALLEVIATES_SYMPTOM、HAS_EFFECT、HAS_INGREDIENT、HAS_SYMPTOM、FROM_SOURCE。"
                           "若文本仅描述单个实体或未涉及实体关系，返回空结构{\"entities\":[],\"relations\":[]}。"
                           "若涉及方剂或药材，补充对应属性字段（功效、性味、剂量等），值为空则省略。",
            "input": data.get("input", ""),
            "output": json.dumps(output, ensure_ascii=False),
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="中医知识图谱抽取器")
    ap.add_argument("--source-dir", required=True, help="详情 JSON 目录（data/中药 或 data/方剂）")
    ap.add_argument("--type", default="条目", help="条目类型：中药/方剂")
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条（测试用）")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="线程数")
    ap.add_argument("--no-summary", action="store_true", help="只抽取不汇总输出")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    cache_dir = CACHE_DIR / args.type
    cache_dir.mkdir(parents=True, exist_ok=True)
    NEO4J_DIR.mkdir(parents=True, exist_ok=True)
    ALPACA_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(source_dir.glob("*.json"))
    print(f"[1/3] 详情 JSON 共 {len(files)} 条，来源 {source_dir}")

    items = [
        {"name": f.stem, "path": str(f), "type": args.type}
        for f in files
        if not (cache_dir / f"{f.stem}.json").exists()
        or (cache_dir / f"{f.stem}.json").stat().st_size == 0
    ]
    if args.limit:
        items = items[: args.limit]
    print(f"[2/3] 本次待抽取 {len(items)} 条，线程数 {args.workers}")

    if items:
        chain = _build_chain()
        def worker(it: dict) -> None:
            time.sleep(REQUEST_INTERVAL)
            process_one(it, cache_dir, chain)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, items))

    cached = list(cache_dir.glob("*.json"))
    print(f"[3/3] 本次成功 {_stats['ok']}，失败 {_stats['err']}，空结果 {_stats['empty']}，缓存共 {len(cached)} 条")

    if not args.no_summary:
        kg = summarize(cache_dir)
        (NEO4J_DIR / "entities.json").write_text(
            json.dumps(kg["entities"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (NEO4J_DIR / "relations.json").write_text(
            json.dumps(kg["relations"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        alpaca = build_alpaca(cache_dir)
        (ALPACA_DIR / "alpaca.json").write_text(
            json.dumps(alpaca, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"      汇总: entities {len(kg['entities'])}，relations {len(kg['relations'])}，alpaca {len(alpaca)} 条")
        print(f"      neo4j  -> {NEO4J_DIR}")
        print(f"      alpaca -> {ALPACA_DIR / 'alpaca.json'}")


if __name__ == "__main__":
    main()
