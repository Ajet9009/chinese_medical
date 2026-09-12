"""Shared state for the LangGraph question-answering workflow."""

from typing import Literal, TypedDict


class MatchedEntity(TypedDict, total=False):
    """FAISS 匹配到的标准实体。"""
    id: str
    type: str
    name: str
    score: float


# ── 知识图谱实体类型（与 data/neo4j/entities.json 一致）──
KG_ENTITY_TYPES = {
    "Herb": "药材（单味中药，如人参、黄芪、甘草）",
    "Formula": "方剂（多味药材组成的复方，如四君子汤、桂枝汤）",
    "Symptom": "症状（患者表现，如咳嗽、发热、腹痛）",
    "Disease": "疾病（中医病名/证型，如感冒、肾虚、阴虚火旺）",
    "Effect": "功效（药物或方剂的作用，如补气、活血、清热）",
    "Source": "出处（文献来源，如《本草纲目》《伤寒论》）",
    "Class": "大类（药材/方剂的上位分类，如解表药、补虚药）",
    "SubClass": "功效分类（大类下的细分，如发散风寒药）",
    "Meridian": "经络（药物归经，如肺经、肝经、脾经）",
}

# ── 知识图谱关系类型 ──
KG_RELATION_TYPES = {
    "TREATS_DISEASE": "治疗疾病",
    "ALLEVIATES_SYMPTOM": "缓解症状",
    "HAS_EFFECT": "具有功效",
    "HAS_INGREDIENT": "包含药材",
    "HAS_SYMPTOM": "疾病表现症状",
    "FROM_SOURCE": "出自文献",
    "BELONGS_TO": "归属于分类",
    "HAS_MERIDIAN": "归入经络",
}

KG_ENTITY_TYPE_NAMES = frozenset(KG_ENTITY_TYPES.keys())
KG_RELATION_TYPE_NAMES = frozenset(KG_RELATION_TYPES.keys())


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GraphState(TypedDict, total=False):
    user_question: str
    search_question: str
    messages: list[ChatMessage]
    intent: Literal["tcm", "general"]
    intent_reason: str
    is_zhongyi_intent: bool
    final_answer: str
    # 六类用户输入实体
    user_input_symptoms: list[str]
    user_input_diseases: list[str]
    user_input_formulas: list[str]
    user_input_herbs: list[str]
    user_input_effects: list[str]
    user_input_sources: list[str]
    # 六类匹配到的标准实体
    matched_symptoms: list[MatchedEntity]
    matched_diseases: list[MatchedEntity]
    matched_formulas: list[MatchedEntity]
    matched_herbs: list[MatchedEntity]
    matched_effects: list[MatchedEntity]
    matched_sources: list[MatchedEntity]
    # 生成的 Cypher 查询语句
    cypher_queries: list[str]
    # Cypher 校验重试次数（>0 表示生成过程中有过重试）
    cypher_retry_count: int
    # Neo4j 执行结果（去重去噪后的精简上下文）
    neo4j_answer: str
    # W3 文档 RAG（混合检索 + CRAG）
    doc_chunks: list[dict]
    doc_context: str
    refused: bool
    crag_grade: str
    crag_action: str
    crag_confidence: str
    citation_ok: bool
    viewer_dept: str
    viewer_role: str


def question_for_retrieval(state: GraphState) -> str:
    """检索/抽取/Cypher 用消解后的独立问句，没有则用原问题。"""
    return str(state.get("search_question") or state.get("user_question") or "").strip()


def history_as_text(state: GraphState) -> str:
    msgs = list(state.get("messages") or [])
    if not msgs:
        return ""
    lines: list[str] = []
    for item in msgs:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{item.get('content', '')}")
    return "\n".join(lines)

