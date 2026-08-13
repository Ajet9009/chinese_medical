#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""混合抽取构建知识图谱：规则层 + LLM 缓存层合并。

规则层（零 API，处理结构化字段）：
  - meta 大类/功效分类 -> (条目)-BELONGS_TO->(Class/SubClass)
  - 组成段 entities -> (方剂)-HAS_INGREDIENT->(中药)
  - 归经/经脉段 entities -> (中药)-HAS_MERIDIAN->(经络)
  - 性味段 -> 味/性 写入 Herb 节点属性

LLM 层（复用 data/kg_cache/<type>/ 断点缓存，未抽的条目自然缺省，
  后续 kg_extractor.py 补抽后重跑本脚本即可增量）。

输出（供 _003 Neo4j 导入）：
  - data/neo4j/entities.json   {id, type, name, attributes}
  - data/neo4j/relations.json  {relation, subject(id), subject_type, object(id), object_type}

用法:
  python build_kg.py                    # 合并 中药+方剂
  python build_kg.py --types 方剂       # 只合并方剂
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"

TYPE_MAP = {"中药": "Herb", "方剂": "Formula"}
SOURCES = {t: DATA / t for t in TYPE_MAP}
CACHES = {t: DATA / "kg_cache" / t for t in TYPE_MAP}
OUT_DIR = DATA / "neo4j"

REL_CLASS = "BELONGS_TO"
REL_INGREDIENT = "HAS_INGREDIENT"
REL_MERIDIAN = "HAS_MERIDIAN"

TASTE_CHARS = "辛甘酸苦咸淡涩"
NATURE_CHARS = "温热凉寒平"


def iter_blocks(doc: dict):
    """遍历所有 (heading, block) 对。"""
    for sec in doc.get("sections", []):
        for sub in sec.get("children", []):
            for b in sub.get("blocks", []):
                yield sub.get("heading", ""), b
        for b in sec.get("blocks", []):
            yield sec.get("heading", ""), b


def parse_sexing(text: str) -> tuple[list[str], list[str]]:
    """性味文本 -> (味列表, 性列表)。例：'辛苦，温。' -> (['辛','苦'], ['温'])"""
    tastes = [c for c in TASTE_CHARS if c in text]
    natures = [c for c in NATURE_CHARS if c in text]
    return tastes, natures


def rule_extract(doc: dict) -> tuple[list[dict], list[dict]]:
    """规则层抽取，返回 (entities, relations)。"""
    type_ = doc.get("type", "")
    node_type = TYPE_MAP.get(type_, "Herb")
    name = doc.get("name", "")
    entities: list[dict] = []
    relations: list[dict] = []

    def add_ent(ename: str, etype: str, attributes=None) -> None:
        entities.append({"name": ename, "type": etype, "attributes": attributes})

    def add_rel(subj, subj_type, rel, obj, obj_type) -> None:
        relations.append(
            {
                "subject": subj,
                "subject_type": subj_type,
                "relation": rel,
                "object": obj,
                "object_type": obj_type,
            }
        )

    # 0) 条目自身节点（方剂/中药）
    add_ent(name, node_type)

    # 1) 属于：大类/功效分类
    for k, v in doc.get("meta", {}).items():
        if not v:
            continue
        if "大类" in k:
            obj_type = "Class"
        elif "分类" in k:
            obj_type = "SubClass"
        else:
            continue
        add_ent(v, obj_type)
        add_rel(name, node_type, REL_CLASS, v, obj_type)

    # 2) 组成 -> 方剂含中药
    for heading, b in iter_blocks(doc):
        if "组成" not in heading:
            continue
        for ent in b.get("entities", []):
            add_ent(ent, "Herb")
            add_rel(name, "Formula", REL_INGREDIENT, ent, "Herb")

    # 3) 归经/经脉 -> 中药归经络
    for heading, b in iter_blocks(doc):
        if not ("归经" in heading or "经脉" in heading):
            continue
        for ent in b.get("entities", []):
            if "经" in ent:
                add_ent(ent, "Meridian")
                add_rel(name, node_type, REL_MERIDIAN, ent, "Meridian")

    # 4) 性味 -> Herb 属性 味/性
    for heading, b in iter_blocks(doc):
        if "性味" not in heading:
            continue
        tastes, natures = parse_sexing(b.get("text", ""))
        if tastes or natures:
            attrs = {}
            if tastes:
                attrs["味"] = "、".join(tastes)
            if natures:
                attrs["性"] = "、".join(natures)
            add_ent(name, node_type, attrs)

    return entities, relations


def merge(types: list[str]) -> tuple[list[dict], list[dict]]:
    """合并规则层 + LLM 缓存层，去重加 id。返回 (entities, relations)。"""
    entities_by_key: dict[tuple, dict] = {}
    relations: list[dict] = []

    def merge_entity(e: dict) -> None:
        key = (e["type"], e["name"])
        attrs = e.get("attributes")
        if key not in entities_by_key:
            entities_by_key[key] = {
                "id": f"E{len(entities_by_key) + 1}",
                "type": e["type"],
                "name": e["name"],
                "attributes": attrs,
            }
        elif attrs:
            exist = entities_by_key[key]["attributes"]
            if exist:
                for k, v in attrs.items():
                    exist.setdefault(k, v)
            else:
                entities_by_key[key]["attributes"] = attrs

    for t in types:
        src_dir = SOURCES[t]
        cache_dir = CACHES[t]
        if not src_dir.exists():
            print(f"  跳过 {t}：{src_dir} 不存在")
            continue
        files = sorted(src_dir.glob("*.json"))
        for f in files:
            doc = json.loads(f.read_text(encoding="utf-8"))
            # 规则层
            r_ents, r_rels = rule_extract(doc)
            # LLM 缓存层
            cache = cache_dir / f"{doc['name']}.json"
            l_ents, l_rels = [], []
            if cache.exists() and cache.stat().st_size > 0:
                data = json.loads(cache.read_text(encoding="utf-8"))
                l_ents = data.get("entities", [])
                l_rels = data.get("relations", [])
            for e in r_ents + l_ents:
                merge_entity(e)
            relations.extend(r_rels)
            relations.extend(l_rels)

    # 关系解析为 id（按 (类型, 名称) 定位，避免不同类型同名冲突）
    type_name_to_id = {(v["type"], v["name"]): v["id"] for v in entities_by_key.values()}

    def _resolve(r: dict, side: str) -> str:
        key = (r.get(f"{side}_type"), r[side])
        return type_name_to_id.get(key, r[side])

    rel_out: list[dict] = []
    seen: set[tuple] = set()
    for r in relations:
        src, tgt = _resolve(r, "subject"), _resolve(r, "object")
        key = (r["relation"], src, tgt)
        if key in seen:
            continue
        seen.add(key)
        rel_out.append(
            {
                "relation": r["relation"],
                "subject": src,
                "subject_type": r.get("subject_type"),
                "object": tgt,
                "object_type": r.get("object_type"),
            }
        )
    return list(entities_by_key.values()), rel_out


def stats(entities: list[dict], relations: list[dict]) -> None:
    from collections import Counter

    et = Counter(e["type"] for e in entities)
    rt = Counter(r["relation"] for r in relations)
    print("\n===== 实体类型分布 =====")
    for k, v in et.most_common():
        print(f"  {k}: {v}")
    print("===== 关系类型分布 =====")
    for k, v in rt.most_common():
        print(f"  {k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description="混合抽取构建知识图谱（规则层 + LLM 缓存）")
    ap.add_argument("--types", nargs="+", choices=list(TYPE_MAP), default=list(TYPE_MAP), help="合并类型")
    ap.add_argument("--no-stats", action="store_true", help="不打印统计")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"构建图谱：{args.types}")
    entities, relations = merge(args.types)
    (OUT_DIR / "entities.json").write_text(
        json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "relations.json").write_text(
        json.dumps(relations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"写入 {OUT_DIR}")
    print(f"  实体 {len(entities)}，关系 {len(relations)}")
    if not args.no_stats:
        stats(entities, relations)


if __name__ == "__main__":
    main()
