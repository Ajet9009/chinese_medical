"""LangGraph 问答工作流：意图识别 → 条件路由 → 回答/实体抽取。"""

from langgraph.graph import END, START, StateGraph

try:
    from .answer_generation import answer_generation_node
    from .cypher_executor import cypher_executor_node
    from .cypher_generation import cypher_generation_node
    from .doc_retrieval import doc_retrieval_node
    from .entity_extraction import entity_extraction_node
    from .entity_normalization import entity_normalization_node
    from .general_response import general_response_node
    from .intent_recognition import intent_recognition_node
    from .standalone_query import standalone_query_node
    from .state import GraphState
except ImportError:
    from answer_generation import answer_generation_node
    from cypher_executor import cypher_executor_node
    from cypher_generation import cypher_generation_node
    from doc_retrieval import doc_retrieval_node
    from entity_extraction import entity_extraction_node
    from entity_normalization import entity_normalization_node
    from general_response import general_response_node
    from intent_recognition import intent_recognition_node
    from standalone_query import standalone_query_node
    from state import GraphState


def _route_by_intent(state: GraphState) -> str:
    """中医走图谱；普通先检索文献，有摘录再生成，否则通用回答。"""
    if state.get("is_zhongyi_intent", False):
        return "entity_extraction"
    return "doc_retrieval"


def _route_after_docs(state: GraphState) -> str:
    """图谱路径始终进最终回答；普通路径仅在向量库命中时才用文献作答。"""
    if state.get("is_zhongyi_intent", False):
        return "answer_generation"
    if (state.get("doc_chunks") or []) and str(state.get("doc_context") or "").strip():
        return "answer_generation"
    return "general_response"


def build_graph():
    workflow = StateGraph(GraphState)

    # 节点注册
    workflow.add_node("standalone_query", standalone_query_node)
    workflow.add_node("intent_recognition", intent_recognition_node)
    workflow.add_node("general_response", general_response_node)
    workflow.add_node("entity_extraction", entity_extraction_node)
    workflow.add_node("entity_normalization", entity_normalization_node)
    workflow.add_node("cypher_generation", cypher_generation_node)
    workflow.add_node("cypher_executor", cypher_executor_node)
    workflow.add_node("doc_retrieval", doc_retrieval_node)
    workflow.add_node("answer_generation", answer_generation_node)

    # 边
    workflow.add_edge(START, "standalone_query")
    workflow.add_edge("standalone_query", "intent_recognition")
    workflow.add_conditional_edges(
        "intent_recognition",
        _route_by_intent,
        {
            "entity_extraction": "entity_extraction",
            "doc_retrieval": "doc_retrieval",
        },
    )
    workflow.add_edge("entity_extraction", "entity_normalization")
    workflow.add_edge("entity_normalization", "cypher_generation")
    workflow.add_edge("cypher_generation", "cypher_executor")
    workflow.add_edge("cypher_executor", "doc_retrieval")
    workflow.add_conditional_edges(
        "doc_retrieval",
        _route_after_docs,
        {
            "answer_generation": "answer_generation",
            "general_response": "general_response",
        },
    )
    workflow.add_edge("answer_generation", END)
    workflow.add_edge("general_response", END)

    return workflow.compile()


# ============================================================
# main() — 直接运行本文件验证完整图流程
# ============================================================

_STATE_KEYS = [
    ("user_question",       "用户问题"),
    ("search_question",     "检索问句"),
    ("intent",              "意图"),
    ("is_zhongyi_intent",   "是否中医"),
    ("user_input_symptoms", "输入·症状"),
    ("user_input_diseases", "输入·疾病"),
    ("user_input_formulas", "输入·方剂"),
    ("user_input_herbs",    "输入·药材"),
    ("user_input_effects",  "输入·功效"),
    ("user_input_sources",  "输入·出处"),
    ("matched_symptoms",    "匹配·症状"),
    ("matched_diseases",    "匹配·疾病"),
    ("matched_formulas",    "匹配·方剂"),
    ("matched_herbs",       "匹配·药材"),
    ("matched_effects",     "匹配·功效"),
    ("matched_sources",     "匹配·出处"),
    ("cypher_queries",      "Cypher查询"),
    ("neo4j_answer",        "KG上下文"),
    ("doc_context",         "文献摘录"),
    ("final_answer",        "最终回答"),
]


def _print_state(title: str, state: dict):
    """打印 state 快照。"""
    print(f"  ┌─ {title} " + "─" * 48)
    for key, label in _STATE_KEYS:
        val = state.get(key)
        if val is None or val == "":
            display = "(空)"
        elif isinstance(val, bool):
            display = "True ✓ 中医" if val else "False ✗ 普通"
        elif isinstance(val, list):
            if not val:
                display = "[] (无)"
                print(f"  │  {label + ' (' + key + ')':<30s} = {display}")
            elif isinstance(val[0], dict):
                display = "[" + ", ".join(
                    f"{x.get('name','?')}({x.get('score',0):.2f})" for x in val
                ) + "]"
                print(f"  │  {label + ' (' + key + ')':<30s} = {display}")
            elif key == "cypher_queries":
                print(f"  │  {label + ' (' + key + ')':<30s} = [{len(val)} 条]")
                for qi, q in enumerate(val):
                    print(f"  │  {'':30s}   [{qi+1}] {str(q)[:100]}")
            else:
                display = "[" + ", ".join(str(x) for x in val) + "]"
                print(f"  │  {label + ' (' + key + ')':<30s} = {display}")
            continue
        elif key == "final_answer":
            display = str(val)[:100]
        else:
            display = str(val)[:60]
        print(f"  │  {label + ' (' + key + ')':<30s} = {display}")
        # 长回答换行续显
        if key == "final_answer" and len(str(val)) > 100:
            for chunk_start in range(100, min(len(str(val)), 400), 100):
                print(f"  │  {'':30s}   {str(val)[chunk_start:chunk_start+100]}")
    print("  └" + "─" * 60)


def main():
    """直接运行 python graph.py 验证完整图流程。"""
    import os
    import sys

    # 确保模块所在目录在 path 中（直接运行 .py 时需要）
    HERE = os.path.dirname(os.path.abspath(__file__))
    if HERE not in sys.path:
        sys.path.insert(0, HERE)

    test_cases = [
        "四君子汤有什么功效？",
        "今天有什么电影好看？",
        "肚子疼吃什么药",
    ]

    print("=" * 64)
    print("  LangGraph 完整图流程 · 验证")
    print("=" * 64)
    print("\n编译图...", end=" ", flush=True)
    app = build_graph()
    print("就绪。")

    for i, question in enumerate(test_cases, 1):
        print(f"\n{'─' * 64}")
        print(f"  [{i}/{len(test_cases)}] 输入: {question}")

        initial = {"user_question": question}
        _print_state("初始 State", initial)

        print("\n  ▼ 图开始执行...\n")

        # stream_mode="updates" 每个节点返回其输出增量
        step = 0
        for chunk in app.stream(initial, stream_mode="updates"):
            step += 1
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]

            print(f"  ┌ 节点 [{step}] {node_name}")
            if node_output is None:
                node_output = {}
            for k, v in node_output.items():
                if k == "final_answer":
                    print(f"  │  {k} = {str(v)[:100]}")
                    if len(str(v)) > 100:
                        for cs in range(100, min(len(str(v)), 400), 100):
                            print(f"  │  {'':>10s}{str(v)[cs:cs+100]}")
                elif isinstance(v, list):
                    if not v:
                        display = "[] (无)"
                        print(f"  │  {k} = {display}")
                    elif isinstance(v[0], dict):
                        display = "[" + ", ".join(
                            f"{x.get('name','?')}({x.get('score',0):.2f})" for x in v
                        ) + "]"
                        print(f"  │  {k} = {display}")
                    elif k == "cypher_queries":
                        print(f"  │  {k} = [{len(v)} 条]")
                        for qi, q in enumerate(v):
                            print(f"  │    [{qi+1}] {q[:100]}")
                    else:
                        display = "[" + ", ".join(str(x) for x in v) + "]"
                        print(f"  │  {k} = {display}")
                    continue
                else:
                    print(f"  │  {k} = {v}")
            print("  └")

        # 最终 state
        final_state = app.invoke(initial)
        _print_state("最终 State", final_state)

    print(f"\n{'═' * 64}")
    print("  图流程验证完成。")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
