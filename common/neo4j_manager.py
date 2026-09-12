#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neo4j 图数据库连接管理 + 实体/关系批量导入。

职责：
  - 读取 common/.env 的 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD 建立连接
  - 为实体类型建 id 唯一约束（幂等，CREATE IF NOT EXISTS）
  - 批量插入实体（MERGE，按 type 分组 UNWIND）与关系（MERGE）
  - 进度日志：每批打印 已插入/总数/百分比
  - 断点续传：checkpoint 文件记录已插入的实体 id 与关系 key，
    程序再次执行时跳过已插入的数据

用法：
    from common.neo4j_manager import Neo4jManager
    mgr = Neo4jManager()
    mgr.create_constraints()
    mgr.batch_import(entities, relations, checkpoint_path=...)
    mgr.close()
"""

import json
import logging
import os
import time
from pathlib import Path

from neo4j import GraphDatabase

from common.env_loader import load_app_env

# 允许的实体类型（节点 Label）与关系类型——白名单，防 Cypher 注入
ALLOWED_LABELS = frozenset(
    {
        "Herb",
        "Formula",
        "Class",
        "SubClass",
        "Effect",
        "Symptom",
        "Disease",
        "Meridian",
        "Source",
    }
)
ALLOWED_RELATIONS = frozenset(
    {
        "BELONGS_TO",
        "HAS_INGREDIENT",
        "HAS_MERIDIAN",
        "TREATS_DISEASE",
        "ALLEVIATES_SYMPTOM",
        "HAS_EFFECT",
        "FROM_SOURCE",
        "HAS_SYMPTOM",
    }
)

log = logging.getLogger("neo4j_manager")


def _load_env(env_path: Path | None = None) -> None:
    """加载 common/.env 与仓库根 .env（根目录覆盖）。"""
    if env_path is not None:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    load_app_env()


def _clean_attributes(attributes) -> dict | None:
    """清洗属性：丢弃 None 与不可标量的值（dict/list），避免 SET 报错。"""
    if not attributes:
        return None
    clean = {k: v for k, v in attributes.items() if v is not None
             and isinstance(v, (str, int, float, bool))}
    return clean or None


def _validate_label(label: str) -> str:
    if label not in ALLOWED_LABELS:
        raise ValueError(f"非法实体类型（Label）: {label}")
    return label


def _validate_relation(rel: str) -> str:
    if rel not in ALLOWED_RELATIONS:
        raise ValueError(f"非法关系类型: {rel}")
    return rel


class Neo4jManager:
    """Neo4j 连接 + 批量导入管理器。"""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        env_path: Path | None = None,
    ) -> None:
        _load_env(env_path)
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        if not all([self.uri, self.user, self.password]):
            raise ValueError(
                "Neo4j 连接信息缺失：请检查 common/.env 的 NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD"
            )
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.database = database
        # 立即验证连通性，失败快速报错
        self.driver.verify_connectivity()
        log.info("已连接 Neo4j: %s (database=%s, user=%s)", self.uri, database, self.user)

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            log.info("Neo4j 连接已关闭")

    # ------------------------------------------------------------
    # 约束
    # ------------------------------------------------------------

    def create_constraints(self, labels: set[str] | None = None) -> None:
        """为实体类型建 id 唯一约束（IF NOT EXISTS，幂等）。"""
        labels = labels or ALLOWED_LABELS
        with self.driver.session(database=self.database) as s:
            for label in sorted(labels):
                _validate_label(label)
                cypher = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                )
                try:
                    s.run(cypher).consume()
                    log.info("约束就绪: %s.id 唯一", label)
                except Exception as exc:  # noqa: BLE001
                    # 约束失败不阻断导入（MERGE 单线程顺序导入仍可去重）
                    log.warning("建约束失败 %s: %s", label, exc)

    # ------------------------------------------------------------
    # 单条插入（供调试/增量使用）
    # ------------------------------------------------------------

    def create_entity(self, entity: dict) -> None:
        """插入单个实体：MERGE (n:Label {id}) ON CREATE/MATCH SET 属性。"""
        label = _validate_label(entity.get("type", ""))
        attrs = _clean_attributes(entity.get("attributes"))
        cypher = (
            f"MERGE (n:{label} {{id: $id}}) "
            f"SET n.name = $name "
            f"SET n += $attrs"
        )
        with self.driver.session(database=self.database) as s:
            s.run(cypher, id=str(entity["id"]), name=entity.get("name", ""),
                  attrs=attrs or {}).consume()

    def create_relation(self, rel: dict) -> None:
        """插入单个关系：先匹配两端节点再 MERGE 关系（幂等）。"""
        rtype = _validate_relation(rel.get("relation", ""))
        subj_label = _validate_label(rel.get("subject_type", ""))
        obj_label = _validate_label(rel.get("object_type", ""))
        cypher = (
            f"MATCH (a:{subj_label} {{id: $sid}}) "
            f"MATCH (b:{obj_label} {{id: $oid}}) "
            f"MERGE (a)-[r:{rtype}]->(b)"
        )
        with self.driver.session(database=self.database) as s:
            s.run(cypher, sid=str(rel["subject"]), oid=str(rel["object"])).consume()

    def get_all_entities(self) -> list[dict]:
        """Return every graph node in the standard entity shape used by vector search."""
        cypher = (
            "MATCH (n) "
            "RETURN labels(n) AS labels, n.id AS id, n.name AS name, "
            "properties(n) AS properties ORDER BY id(n)"
        )
        entities: list[dict] = []
        with self.driver.session(database=self.database) as session:
            for row in session.run(cypher):
                labels = list(row.get("labels") or [])
                entity_type = next((label for label in labels if label in ALLOWED_LABELS), labels[0] if labels else "Entity")
                properties = dict(row.get("properties") or {})
                properties.pop("id", None)
                properties.pop("name", None)
                entities.append({
                    "id": str(row.get("id", "")),
                    "type": entity_type,
                    "name": row.get("name") or "",
                    "attributes": properties,
                })
        return entities

    # ------------------------------------------------------------
    # 图 Schema 元数据（供 LLM 生成 Cypher 用）
    # ------------------------------------------------------------

    def get_schema_metadata(self, compact: bool = True) -> dict:
        """提取图 schema。

        compact=True（默认）：只返回 labels + 关系 from/to 对，供 Cypher 生成用。
        compact=False：返回完整 schema（含属性键 + 数量）。
        """
        with self.driver.session(database=self.database) as s:
            # 1) labels（一次查询）
            node_result = s.run(
                "CALL db.labels() YIELD label "
                "RETURN label ORDER BY label"
            )
            labels = [row["label"] for row in node_result if row["label"] in ALLOWED_LABELS]

            # 2) 关系类型（一次查询）
            rel_result = s.run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            )
            rel_types = [
                row["relationshipType"]
                for row in rel_result
                if row["relationshipType"] in ALLOWED_RELATIONS
            ]

            # 3) 关系 from/to label 对（每个关系类型一次查询）
            relationships: dict[str, list[list[str]]] = {}
            for rt in rel_types:
                rows = s.run(
                    f"MATCH (a)-[r:`{rt}`]->(b) "
                    "WITH DISTINCT labels(a) AS la, labels(b) AS lb "
                    "RETURN la, lb LIMIT 20"
                )
                seen: set[tuple[str, str]] = set()
                pairs: list[list[str]] = []
                for rrow in rows:
                    from_labels = [lbl for lbl in (rrow["la"] or []) if lbl in ALLOWED_LABELS]
                    to_labels = [lbl for lbl in (rrow["lb"] or []) if lbl in ALLOWED_LABELS]
                    for fl in (from_labels or ["?"]):
                        for tl in (to_labels or ["?"]):
                            key = (fl, tl)
                            if key not in seen:
                                seen.add(key)
                                pairs.append([fl, tl])
                relationships[rt] = pairs

        if compact:
            return {"labels": labels, "relationships": relationships}

        # 完整模式：补充属性键 + 数量
        with self.driver.session(database=self.database) as s:
            nodes: dict[str, dict] = {}
            for label in labels:
                rows = s.run(
                    f"MATCH (n:`{label}`) "
                    "WITH n LIMIT 500 "
                    "UNWIND keys(n) AS k "
                    "RETURN DISTINCT k ORDER BY k"
                )
                props = sorted(
                    [row["k"] for row in rows if row["k"] not in ("id",)]
                )
                cnt_row = s.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()
                count = cnt_row["c"] if cnt_row else 0
                nodes[label] = {"properties": props, "count": count}

            rel_full: dict[str, dict] = {}
            for rt in rel_types:
                cnt_row = s.run(
                    f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c"
                ).single()
                count = cnt_row["c"] if cnt_row else 0
                rel_full[rt] = {
                    "pairs": [{"from": p[0], "to": p[1]} for p in relationships[rt]],
                    "count": count,
                }

        return {"nodes": nodes, "relationships": rel_full}

    # ------------------------------------------------------------
    # Cypher 语法校验（EXPLAIN，不实际执行）
    # ------------------------------------------------------------

    def validate_cypher(self, cypher: str) -> dict:
        """用 EXPLAIN 校验 Cypher 语法，不实际执行。

        Returns:
            {"valid": True, "error": None}   → 语法正确
            {"valid": False, "error": "..."}  → 语法错误，含 Neo4j 原始错误信息
        """
        if not cypher or not cypher.strip():
            return {"valid": False, "error": "Cypher 语句为空"}
        try:
            with self.driver.session(database=self.database) as s:
                s.run(f"EXPLAIN {cypher}").consume()
            return {"valid": True, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"valid": False, "error": str(exc)}

    # ------------------------------------------------------------
    # 批量执行 Cypher
    # ------------------------------------------------------------

    def execute_cypher_batch(self, statements: list[str]) -> list[dict]:
        """逐条执行多条 Cypher 语句，每条返回结果或错误。

        Args:
            statements: Cypher 语句列表。

        Returns:
            [{ "index": 0, "cypher": "...", "ok": True, "records": [...], "summary": {...} },
             { "index": 1, "cypher": "...", "ok": False, "error": "..." }]
        """
        results: list[dict] = []
        for idx, stmt in enumerate(statements):
            entry: dict = {"index": idx, "cypher": stmt}
            if not stmt or not stmt.strip():
                entry["ok"] = False
                entry["error"] = "Cypher 语句为空"
                results.append(entry)
                continue
            try:
                with self.driver.session(database=self.database) as s:
                    r = s.run(stmt)
                    records = [dict(rec) for rec in r]
                    summary = r.consume()
                    entry["ok"] = True
                    entry["records"] = records
                    entry["summary"] = {
                        "counters": str(summary.counters) if summary else None,
                    }
            except Exception as exc:  # noqa: BLE001
                entry["ok"] = False
                entry["error"] = str(exc)
            results.append(entry)
        return results

    # ------------------------------------------------------------
    # 批量导入 + 断点续传
    # ------------------------------------------------------------

    @staticmethod
    def _rel_key(r: dict) -> str:
        return f"{r['relation']}|{r['subject']}|{r['object']}"

    def batch_import(
        self,
        entities: list[dict],
        relations: list[dict],
        checkpoint_path: Path | str | None = None,
        batch_size: int = 500,
        on_progress=None,
    ) -> dict:
        """批量导入实体与关系，支持断点续传。

        checkpoint_path 指向 JSON 文件，记录已插入的实体 id 与关系 key；
        重跑时跳过已插入的数据，每批提交后更新 checkpoint。
        返回 {'entities': 新插, 'relations': 新插, 'skipped': 已跳}。
        """
        entities = list(entities or [])
        relations = list(relations or [])
        done_ents: set[str] = set()
        done_rels: set[str] = set()
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        if checkpoint_path and checkpoint_path.exists():
            try:
                cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                done_ents = set(cp.get("entities", []))
                done_rels = set(cp.get("relations", []))
                log.info("读取断点：已插实体 %d，已插关系 %d", len(done_ents), len(done_rels))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("断点文件读取失败，将全新导入: %s", exc)

        stats = {"entities": 0, "relations": 0, "skipped": 0}

        def _report(stage: str, idx: int, total: int) -> None:
            pct = f"{idx / total * 100:.1f}%" if total else "100.0%"
            msg = f"[{stage}] {idx}/{total} ({pct})"
            if on_progress:
                on_progress(msg)
            else:
                print(msg)

        # ---------- 阶段一：实体 ----------
        todo = [e for e in entities if str(e.get("id")) not in done_ents]
        stats["skipped"] += len(entities) - len(todo)
        # 按 type 分组，同 label 用 UNWIND 批量 MERGE
        by_label: dict[str, list[dict]] = {}
        for e in todo:
            by_label.setdefault(_validate_label(e.get("type", "")), []).append(e)
        done_ents_now = 0
        for label, rows in by_label.items():
            total = len(rows)
            for start in range(0, total, batch_size):
                chunk = rows[start : start + batch_size]
                payload = [
                    {
                        "id": str(e["id"]),
                        "name": e.get("name", ""),
                        "attrs": _clean_attributes(e.get("attributes")) or {},
                    }
                    for e in chunk
                ]
                cypher = (
                    f"UNWIND $rows AS row "
                    f"MERGE (n:{label} {{id: row.id}}) "
                    f"SET n.name = row.name "
                    f"SET n += row.attrs"
                )
                with self.driver.session(database=self.database) as s:
                    s.run(cypher, rows=payload).consume()
                done_ents_now += len(chunk)
                done_ents.update(e["id"] for e in chunk)
                self._save_checkpoint(checkpoint_path, done_ents, done_rels)
                _report(f"实体:{label}", start + len(chunk), total)
        stats["entities"] = done_ents_now

        # ---------- 阶段二：关系 ----------
        todo = [r for r in relations if self._rel_key(r) not in done_rels]
        stats["skipped"] += len(relations) - len(todo)
        done_rels_now = 0
        total = len(todo)
        for start in range(0, total, batch_size):
            chunk = todo[start : start + batch_size]
            # 按 (关系, 起点label, 终点label) 分组 -> 静态 label 拼接 + UNWIND 批量
            groups: dict[tuple[str, str, str], list[tuple[str, str]]] = {}
            for r in chunk:
                rtype = _validate_relation(r.get("relation", ""))
                sl = _validate_label(r.get("subject_type", ""))
                ol = _validate_label(r.get("object_type", ""))
                groups.setdefault((rtype, sl, ol), []).append(
                    (str(r["subject"]), str(r["object"]))
                )
            for (rtype, sl, ol), pairs in groups.items():
                payload = [{"sid": sid, "oid": oid} for sid, oid in pairs]
                cypher = (
                    f"UNWIND $rows AS row "
                    f"MATCH (a:{sl} {{id: row.sid}}) "
                    f"MATCH (b:{ol} {{id: row.oid}}) "
                    f"MERGE (a)-[r:{rtype}]->(b)"
                )
                with self.driver.session(database=self.database) as s:
                    s.run(cypher, rows=payload).consume()
            done_rels_now += len(chunk)
            done_rels.update(self._rel_key(r) for r in chunk)
            self._save_checkpoint(checkpoint_path, done_ents, done_rels)
            _report("关系", start + len(chunk), total)
        stats["relations"] = done_rels_now

        log.info(
            "导入完成：实体 %d，关系 %d，跳过 %d",
            stats["entities"],
            stats["relations"],
            stats["skipped"],
        )
        return stats

    def _create_relation_pair(self, rtype: str, sl: str, ol: str, sid: str, oid: str) -> None:
        cypher = (
            f"MATCH (a:{sl} {{id: $sid}}) "
            f"MATCH (b:{ol} {{id: $oid}}) "
            f"MERGE (a)-[r:{rtype}]->(b)"
        )
        with self.driver.session(database=self.database) as s:
            s.run(cypher, sid=sid, oid=oid).consume()

    @staticmethod
    def _save_checkpoint(path: Path | None, ents: set[str], rels: set[str]) -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"entities": sorted(ents), "relations": sorted(rels)},
                       ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(path)


if __name__ == "__main__":
    neo4j = Neo4jManager()
    # print(neo4j.get_all_entities())
    res = neo4j.get_schema_metadata()
    print(res)