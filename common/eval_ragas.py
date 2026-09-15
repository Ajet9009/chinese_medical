"""用嵌入生成 RAGAS 风格评测集，并对文献 RAG 打分。

默认对齐 RAGAS Non-LLM 指标（BGE 余弦），不调 LLM、不强制安装 ragas 包。
`scripts/eval_ragas.py official` 才 `pip install ragas` 后调用 `evaluate()`。
CI 与 pytest 注入 encode_fn 即可；本机有 EMBEDDING_MODEL_PATH 时用同一套 BGE。
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from common.doc_rag import REFUSE_ANSWER
from common.env_loader import load_app_env
from common.eval_golden import keyword_hit_ratio, load_golden

EncodeFn = Callable[[list[str]], np.ndarray]
logger = logging.getLogger("eval_ragas")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HASH_DIM = 64
# 与 DOC_MIN_SCORE 同量级；字符哈希嵌入略低，故用 0.35
_SIM_TAU = 0.35

# 嵌入合成问句：用块向量与模板问句的余弦选最贴切的一种
_SYNTH_TEMPLATES = (
    "{name}出自哪部书？",
    "{name}的用法注意事项有哪些？",
    "{name}与其他方剂如何鉴别？",
)

# ragas.evaluate / EvaluationDataset.from_list 需要的 SingleTurnSample 列
RAGAS_DATASET_COLUMNS = (
    "user_input",
    "retrieved_contexts",
    "response",
    "reference",
    "reference_contexts",
)


def list_metrics_step_01() -> list[dict[str, str]]:
    """步骤 01：返回本仓库选用的 RAGAS 指标（英文（中文））。

    面向中医文献 RAG：忠实度防胡编，精确度/召回度看检索，相关性和正确性看答案。
    跳过 Noise Sensitivity 等强依赖 LLM-as-judge 的项。
    """
    return [
        {
            "id": "faithfulness",
            "name": "Faithfulness（忠实度）",
            "why": "答案能否由检索上下文支持，抑制脱离文献的幻觉。",
        },
        {
            "id": "answer_relevancy",
            "name": "Answer Relevancy（答案相关性）",
            "why": "答案是否针对用户问题，而不是答非所问。",
        },
        {
            "id": "context_precision",
            "name": "Context Precision（上下文精确度）",
            "why": "检索块里有多少真正相关，衡量排序噪声。",
        },
        {
            "id": "context_recall",
            "name": "Context Recall（上下文召回度）",
            "why": "参考上下文是否被召回，衡量漏检。",
        },
        {
            "id": "answer_correctness",
            "name": "Answer Correctness（答案正确性）",
            "why": "相对参考答案/期望关键词的事实正确程度。",
        },
        {
            "id": "answer_semantic_similarity",
            "name": "Answer Semantic Similarity（答案语义相似度）",
            "why": "答案与参考的嵌入余弦，RAGAS 原生嵌入指标。",
        },
    ]


def testset_path_step_02() -> Path:
    """步骤 02：解析 RAGAS 评测集路径（eval/ 纳入版本库，不放 data/）。"""
    load_app_env()
    raw = (os.getenv("RAGAS_TESTSET_PATH") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / raw
        return path
    return _REPO_ROOT / "eval" / "ragas_testset.json"


def report_path_step_03() -> Path:
    """步骤 03：解析最近一次评测报告路径（运行产物，默认 gitignore）。"""
    load_app_env()
    raw = (os.getenv("RAGAS_REPORT_PATH") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / raw
        return path
    return _REPO_ROOT / "eval" / "ragas_report.json"


def hash_encode_step_04(texts: list[str]) -> np.ndarray:
    """步骤 04：字符频次哈希嵌入，供 CI / 无 BGE 时与测试注入。

    与 tests/test_doc_rag._encode 同构，保证离线评测可复现。
    """
    mat = np.zeros((len(texts), _HASH_DIM), dtype="float32")
    for i, text in enumerate(texts):
        for ch in text or "":
            mat[i, ord(ch) % _HASH_DIM] += 1.0
    return l2_normalize_step_06(mat)


def resolve_encode_fn_step_05(encode_fn: EncodeFn | None = None) -> EncodeFn:
    """步骤 05：优先注入函数，其次本机 BGE，否则回退哈希嵌入。"""
    if encode_fn is not None:
        return encode_fn
    load_app_env()
    model_path = (os.getenv("EMBEDDING_MODEL_PATH") or "").strip()
    if model_path and Path(model_path).exists():
        from common.doc_store import FaissDocStore

        store = FaissDocStore(
            index_path=_REPO_ROOT / "data" / "faiss" / "_ragas_unused.index",
            metadata_path=_REPO_ROOT / "data" / "faiss" / "_ragas_unused.json",
            model_path=model_path,
        )
        return store._encode
    logger.warning("未配置可用 EMBEDDING_MODEL_PATH，RAGAS 评测回退字符哈希嵌入")
    return hash_encode_step_04


def l2_normalize_step_06(mat: np.ndarray) -> np.ndarray:
    """步骤 06：按行 L2 归一化，使点积等于余弦。"""
    arr = np.asarray(mat, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


def is_refuse_item_step_07(item: dict[str, Any]) -> bool:
    """步骤 07：是否拒答样本（无文献依据时应拒答）。"""
    if str(item.get("category") or "").strip() == "拒答":
        return True
    expect = item.get("expect") or []
    return any(str(x).strip() == "未查到" for x in expect)


def text_covers_step_08(needle: str, haystack: list[str], jaccard: float = 0.7) -> bool:
    """步骤 08：子串或字集合 Jaccard 判定两段文本是否同指一块文献。"""
    a = (needle or "").strip()
    if not a:
        return False
    sa = set(a)
    for raw in haystack:
        b = (raw or "").strip()
        if not b:
            continue
        if a in b or b in a:
            return True
        sb = set(b)
        union = sa | sb
        if union and len(sa & sb) / len(union) >= jaccard:
            return True
    return False


def doc_title_step_09(body: str, fallback: str) -> str:
    """步骤 09：从 Markdown 首个标题取方剂名，去掉括号说明。"""
    for line in (body or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            title = re.split(r"[（(]", title, maxsplit=1)[0].strip()
            if title:
                return title
    return fallback


def load_corpus_chunks_step_10(source_dir: Path | None = None) -> list[dict[str, Any]]:
    """步骤 10：按文献入库规则切块，供嵌入生成参考上下文。"""
    from common.doc_chunking import build_document_chunks_step_04
    from common.doc_store import default_source_dir, read_document

    folder = Path(source_dir) if source_dir is not None else default_source_dir()
    if not folder.is_dir():
        return []
    files = sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() in {".txt", ".md", ".pdf"} and p.is_file()
    )
    size = int(os.getenv("DOC_CHUNK_SIZE", "400"))
    overlap = int(os.getenv("DOC_CHUNK_OVERLAP", "80"))
    chunks: list[dict[str, Any]] = []
    for path in files:
        body = read_document(path)
        doc_type = path.parent.name if path.parent != folder else ""
        built = build_document_chunks_step_04(
            doc_id=path.stem,
            doc_name=path.name,
            body=body,
            doc_type=doc_type,
            size=size,
            overlap=overlap,
        )
        fallback_title = doc_title_step_09(body, path.stem)
        for item in built:
            item["title"] = item.get("title") or fallback_title
        chunks.extend(built)
    return chunks


def pick_reference_contexts_step_11(
    query: str,
    chunks: list[dict[str, Any]],
    chunk_vecs: np.ndarray,
    encode_fn: EncodeFn,
    relevant_docs: list[str] | None,
    top_k: int = 3,
    expect: list[str] | None = None,
    strict_expect: bool = False,
) -> list[str]:
    """步骤 11：只从指定文献块里选参考上下文；严格模式要求覆盖全部 expect。

    有 relevant_docs 时不跨文档降级。strict_expect 时切块必须能拼出全部期望子串，否则报错。
    """
    if not chunks or chunk_vecs.size == 0:
        if strict_expect and expect:
            raise ValueError(f"无文献切块，无法支撑问句: {query}")
        return []
    q = (query or "").strip()
    if not q:
        return []
    needles = [str(x).strip() for x in (expect or []) if str(x).strip()]
    docs = {str(x).strip() for x in (relevant_docs or []) if str(x).strip()}
    idxs = list(range(len(chunks)))
    if docs:
        filtered = [
            i
            for i, item in enumerate(chunks)
            if str(item.get("doc_name") or "") in docs
            or str(item.get("doc_id") or "") in docs
        ]
        if not filtered:
            if strict_expect:
                raise ValueError(f"relevant_docs 在语料中不存在: {sorted(docs)} / {query}")
            return []
        idxs = filtered
    elif needles:
        keyed = [
            i
            for i in idxs
            if any(n in str(chunks[i].get("text") or "") for n in needles)
        ]
        if keyed:
            idxs = keyed
    if needles:
        covering = [
            i
            for i in idxs
            if all(n in str(chunks[i].get("text") or "") for n in needles)
        ]
        if covering:
            idxs = covering
        else:
            partial = [
                i
                for i in idxs
                if any(n in str(chunks[i].get("text") or "") for n in needles)
            ]
            blob = "\n".join(str(chunks[i].get("text") or "") for i in partial)
            if not partial or not all(n in blob for n in needles):
                if strict_expect:
                    raise ValueError(f"切块未覆盖全部 expect: {query} / {needles}")
            elif partial:
                idxs = partial
    qv = l2_normalize_step_06(encode_fn([q]))[0]
    sims = chunk_vecs[idxs] @ qv
    order = np.argsort(-np.asarray(sims).reshape(-1))
    picked: list[str] = []
    k = max(1, int(top_k))
    for rank in order[:k]:
        text = str(chunks[idxs[int(rank)]].get("text") or "").strip()
        if text:
            picked.append(text)
    if strict_expect and needles:
        blob = "\n".join(picked)
        if not all(n in blob for n in needles):
            extra = [
                str(chunks[i].get("text") or "").strip()
                for i in idxs
                if str(chunks[i].get("text") or "").strip() not in picked
                and any(n in str(chunks[i].get("text") or "") for n in needles)
            ]
            picked.extend(extra)
            blob = "\n".join(picked)
        if not all(n in blob for n in needles):
            raise ValueError(f"选出的参考块仍未覆盖 expect: {query}")
    return picked


def synthesize_from_chunks_step_12(
    chunks: list[dict[str, Any]],
    chunk_vecs: np.ndarray,
    encode_fn: EncodeFn,
    seen: set[str],
    extra_per_doc: int = 1,
) -> list[dict[str, Any]]:
    """步骤 12：按文档中心块与模板问句的余弦合成额外样本（RAGAS 嵌入出题）。"""
    if extra_per_doc <= 0 or not chunks or chunk_vecs.size == 0:
        return []
    by_doc: dict[str, list[int]] = {}
    for i, item in enumerate(chunks):
        key = str(item.get("doc_id") or item.get("doc_name") or i)
        by_doc.setdefault(key, []).append(i)
    extras: list[dict[str, Any]] = []
    for doc_id, idxs in by_doc.items():
        sub = chunk_vecs[idxs]
        mean = l2_normalize_step_06(sub.mean(axis=0, keepdims=True))[0]
        centrality = sub @ mean
        best_local = int(np.argmax(centrality))
        best_i = idxs[best_local]
        name = str(chunks[best_i].get("title") or doc_id).strip() or doc_id
        candidates = [t.format(name=name) for t in _SYNTH_TEMPLATES if t.format(name=name) not in seen]
        if not candidates:
            continue
        q_vecs = l2_normalize_step_06(encode_fn(candidates))
        sims = q_vecs @ chunk_vecs[best_i]
        order = np.argsort(-np.asarray(sims).reshape(-1))
        added = 0
        for rank in order:
            if added >= extra_per_doc:
                break
            chosen = candidates[int(rank)]
            if chosen in seen:
                continue
            seen.add(chosen)
            refs = pick_reference_contexts_step_11(
                chosen,
                chunks,
                chunk_vecs,
                encode_fn,
                [str(chunks[best_i].get("doc_name") or "")],
                top_k=2,
                expect=[name] if name else None,
                strict_expect=False,
            )
            extras.append(
                {
                    "user_input": chosen,
                    "query": chosen,
                    "reference": (refs[0][:160] if refs else name),
                    "reference_contexts": refs,
                    "expect": [name] if name else [],
                    "category": "文献",
                    "source": "embedding-synth",
                    "relevant_docs": [str(chunks[best_i].get("doc_name") or "")],
                }
            )
            added += 1
    return extras


def generate_testset_step_13(
    golden_items: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    encode_fn: EncodeFn | None = None,
    top_k: int = 3,
    extra_per_doc: int = 0,
) -> dict[str, Any]:
    """步骤 13：用黄金集问句 + 语料块嵌入生成 RAGAS 评测集。

    默认不合成额外问句。反馈待标注（空 expect）跳过。拒答条目不挂文献上下文。
    非拒答条目的参考块必须覆盖全部 expect，否则报错。
    """
    encode = resolve_encode_fn_step_05(encode_fn)
    gold = list(golden_items if golden_items is not None else load_golden())
    corpus = list(chunks if chunks is not None else load_corpus_chunks_step_10())
    texts = [str(c.get("text") or "") for c in corpus]
    chunk_vecs = (
        l2_normalize_step_06(encode(texts)) if texts else np.zeros((0, _HASH_DIM), dtype="float32")
    )
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in gold:
        query = str(raw.get("query") or "").strip()
        if not query or query in seen:
            continue
        expect = [str(x).strip() for x in (raw.get("expect") or []) if str(x).strip()]
        if str(raw.get("source") or "seed") == "feedback" and not expect:
            continue
        seen.add(query)
        refuse = is_refuse_item_step_07(raw)
        docs = raw.get("relevant_docs") if isinstance(raw.get("relevant_docs"), list) else []
        refs: list[str] = []
        if not refuse:
            refs = pick_reference_contexts_step_11(
                query,
                corpus,
                chunk_vecs,
                encode,
                [str(x) for x in docs],
                top_k=top_k,
                expect=expect,
                strict_expect=True,
            )
        reference = reference_span_step_30(refs, expect, refuse)
        items.append(
            {
                "user_input": query,
                "query": query,
                "reference": reference,
                "reference_contexts": refs,
                "expect": expect,
                "category": str(raw.get("category") or ""),
                "source": str(raw.get("source") or "seed"),
                "relevant_docs": [str(x) for x in docs],
            }
        )
    items.extend(
        synthesize_from_chunks_step_12(
            corpus, chunk_vecs, encode, seen, extra_per_doc=extra_per_doc
        )
    )
    return {
        "version": 1,
        "generator": "ragas-embedding",
        "metrics": list_metrics_step_01(),
        "items": items,
    }


def save_testset_step_14(payload: dict[str, Any], path: Path | None = None) -> Path:
    """步骤 14：把评测集写入 JSON。"""
    p = path or testset_path_step_02()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def load_testset_step_15(path: Path | None = None) -> dict[str, Any]:
    """步骤 15：读取评测集；缺文件返回空结构。"""
    p = path or testset_path_step_02()
    if not p.is_file():
        return {"version": 1, "generator": "ragas-embedding", "metrics": list_metrics_step_01(), "items": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ragas_testset.json 必须是对象")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("ragas_testset.json.items 必须是数组")
    data.setdefault("metrics", list_metrics_step_01())
    return data


def retrieve_for_eval_step_16(
    store: Any,
    question: str,
    top_k: int = 4,
    use_crag: bool = True,
) -> list[dict[str, Any]]:
    """步骤 16：用文献混合检索（可选 CRAG）取块，模拟线上 RAG 检索。"""
    q = (question or "").strip()
    if not q:
        return []

    def _search(query: str) -> list[dict[str, Any]]:
        return store.mixed_search(query, top_k=top_k, min_score=0.0)

    if use_crag:
        from common.rag.pipeline import retrieve_with_crag

        out = retrieve_with_crag(q, _search)
        return list(out.get("hits") or [])
    return _search(q)


def extractive_answer_step_17(hits: list[dict[str, Any]], item: dict[str, Any] | None = None) -> str:
    """步骤 17：离线答案=拼接检索块；无块则拒答（不调 LLM）。

    item 保留与评分入口签名一致；抽取式不读参考答案。
    """
    _ = item
    texts = [str(h.get("text") or "").strip() for h in hits if str(h.get("text") or "").strip()]
    if not texts:
        return REFUSE_ANSWER
    return "\n".join(texts)


def score_sample_step_18(
    item: dict[str, Any],
    retrieved: list[str],
    response: str,
    encode_fn: EncodeFn,
    tau: float = _SIM_TAU,
) -> dict[str, float | None]:
    """步骤 18：按 RAGAS Non-LLM 定义用嵌入给单条样本打分。"""
    query = str(item.get("user_input") or item.get("query") or "").strip()
    refs = [str(x).strip() for x in (item.get("reference_contexts") or []) if str(x).strip()]
    expect = [str(x).strip() for x in (item.get("expect") or []) if str(x).strip()]
    reference = str(item.get("reference") or "").strip()
    answer = response or ""
    refuse = is_refuse_item_step_07(item)
    ret = [str(x).strip() for x in retrieved if str(x).strip()]

    blob = [query, answer, reference, *refs, *ret]
    vecs = l2_normalize_step_06(encode_fn(blob))
    qv, av, rv = vecs[0], vecs[1], vecs[2]
    ref_vecs = vecs[3 : 3 + len(refs)]
    ret_vecs = vecs[3 + len(refs) :]

    if refuse and not ret:
        context_precision: float | None = 1.0
        context_recall: float | None = 1.0
    elif not ret:
        context_precision = 0.0
        context_recall = 0.0 if refs else (1.0 if refuse else 0.0)
    elif not refs:
        sims_q = ret_vecs @ qv
        context_precision = float(np.clip(np.mean(sims_q), 0.0, 1.0))
        context_recall = None if not refuse else 0.0
    else:
        n_rel = 0
        for i, text in enumerate(ret):
            sim = float(np.max(ref_vecs @ ret_vecs[i]))
            if sim >= tau or text_covers_step_08(text, refs):
                n_rel += 1
        context_precision = n_rel / len(ret)
        n_hit = 0
        for i, text in enumerate(refs):
            sim = float(np.max(ret_vecs @ ref_vecs[i]))
            if sim >= tau or text_covers_step_08(text, ret):
                n_hit += 1
        context_recall = n_hit / len(refs)

    if refuse:
        grounded = ("未查到" in answer) or (not ret)
        faithfulness = 1.0 if grounded else 0.0
        answer_relevancy = 1.0 if "未查到" in answer else float(np.clip(float(av @ qv), 0.0, 1.0))
        semantic = 1.0 if "未查到" in answer else float(np.clip(float(av @ rv), 0.0, 1.0))
    elif not ret or not answer.strip():
        faithfulness = 0.0
        answer_relevancy = 0.0 if not answer.strip() else float(np.clip(float(av @ qv), 0.0, 1.0))
        semantic = float(np.clip(float(av @ rv), 0.0, 1.0)) if reference else 0.0
    else:
        concat_vec = l2_normalize_step_06(encode_fn(["\n".join(ret)]))[0]
        faithfulness = float(np.clip(float(av @ concat_vec), 0.0, 1.0))
        answer_relevancy = float(np.clip(float(av @ qv), 0.0, 1.0))
        semantic = float(np.clip(float(av @ rv), 0.0, 1.0)) if reference else 0.0

    kw = keyword_hit_ratio(expect, answer) if expect else semantic
    correctness = (0.5 * kw + 0.5 * semantic) if expect else semantic
    raw = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": correctness,
        "answer_semantic_similarity": semantic,
    }
    return {
        key: (None if value is None else round(float(value), 4))
        for key, value in raw.items()
    }


def aggregate_report_step_19(
    rows: list[dict[str, Any]],
    answer_mode: str = "extractive",
) -> dict[str, Any]:
    """步骤 19：按指标对样本分求均值，带上英文（中文）名称。"""
    catalog = {m["id"]: m for m in list_metrics_step_01()}
    metrics_out: list[dict[str, Any]] = []
    for mid, meta in catalog.items():
        vals = [
            float(r["scores"][mid])
            for r in rows
            if isinstance(r.get("scores"), dict) and r["scores"].get(mid) is not None
        ]
        score = round(sum(vals) / len(vals), 4) if vals else None
        metrics_out.append(
            {
                "id": mid,
                "name": meta["name"],
                "why": meta["why"],
                "score": score,
                "n": len(vals),
            }
        )
    return {
        "n_samples": len(rows),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "ragas-embedding",
        "answer_mode": answer_mode,
        "metrics": metrics_out,
        "items": rows,
    }


def run_rag_eval_step_20(
    testset: dict[str, Any] | list[dict[str, Any]],
    encode_fn: EncodeFn | None = None,
    store: Any | None = None,
    search_fn: Callable[[str], list[dict[str, Any]]] | None = None,
    answer_fn: Callable[[str, list[str]], str] | None = None,
    live_fn: Callable[[str], dict[str, Any]] | None = None,
    top_k: int = 4,
    use_crag: bool = True,
) -> dict[str, Any]:
    """步骤 20：对评测集跑文献 RAG 检索并打分。

    live_fn 有则用线上 /ask 的答案与 doc_chunks；否则抽取式拼接检索块。
    """
    encode = resolve_encode_fn_step_05(encode_fn)
    payload = testset if isinstance(testset, dict) else {"items": testset}
    items = [it for it in (payload.get("items") or []) if isinstance(it, dict)]
    rows: list[dict[str, Any]] = []
    for item in items:
        query = str(item.get("user_input") or item.get("query") or "").strip()
        if not query:
            continue
        if live_fn is not None:
            live = live_fn(query) or {}
            retrieved = [str(x) for x in (live.get("retrieved_contexts") or []) if str(x).strip()]
            response = str(live.get("response") or "")
        elif is_refuse_item_step_07(item):
            # 拒答样本用于校验“无文献依据时不硬答”，离线抽取式评测不应从语料里强行拼答案。
            retrieved = []
            response = REFUSE_ANSWER
        else:
            if search_fn is not None:
                hits = search_fn(query)
            elif store is not None:
                hits = retrieve_for_eval_step_16(store, query, top_k=top_k, use_crag=use_crag)
            else:
                hits = []
            retrieved = [str(h.get("text") or "") for h in hits if str(h.get("text") or "").strip()]
            response = (
                answer_fn(query, retrieved)
                if answer_fn is not None
                else extractive_answer_step_17(hits, item)
            )
        scores = score_sample_step_18(item, retrieved, response, encode)
        sample = to_ragas_sample_step_25(item, retrieved=retrieved, response=response)
        rows.append(
            {
                **sample,
                "category": item.get("category") or "",
                "source": item.get("source") or "",
                "retrieved_n": len(retrieved),
                "scores": scores,
            }
        )
    mode = "live" if live_fn is not None else "extractive"
    return aggregate_report_step_19(rows, answer_mode=mode)


def save_report_step_21(report: dict[str, Any], path: Path | None = None) -> Path:
    """步骤 21：把评测报告写入 JSON。"""
    p = path or report_path_step_03()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def load_report_step_22(path: Path | None = None) -> dict[str, Any] | None:
    """步骤 22：读取最近一次报告；没有则 None。"""
    p = path or report_path_step_03()
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ragas_report.json 必须是对象")
    return data


def summary_for_admin_step_23() -> dict[str, Any]:
    """步骤 23：管理页用的指标目录 + 评测集概况 + 最近报告。"""
    ts_path = testset_path_step_02()
    n_items = 0
    if ts_path.is_file():
        try:
            n_items = len(load_testset_step_15(ts_path).get("items") or [])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("读取 RAGAS 评测集失败: %s", exc)
            n_items = 0
    report = None
    try:
        report = load_report_step_22()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("读取 RAGAS 报告失败: %s", exc)
    ds_path = dataset_path_step_24()
    n_dataset = 0
    if ds_path.is_file():
        try:
            n_dataset = len(load_ragas_dataset_step_28(ds_path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("读取 RAGAS EvaluationDataset 失败: %s", exc)
            n_dataset = 0
    return {
        "metrics": list_metrics_step_01(),
        "testset": {
            "path": str(ts_path),
            "exists": ts_path.is_file(),
            "n_items": n_items,
        },
        "dataset": {
            "path": str(ds_path),
            "exists": ds_path.is_file(),
            "n_items": n_dataset,
            "columns": list(RAGAS_DATASET_COLUMNS),
        },
        "report": report,
    }


def dataset_path_step_24() -> Path:
    """步骤 24：RAGAS EvaluationDataset JSON 路径（from_list 可用的样本数组）。"""
    load_app_env()
    raw = (os.getenv("RAGAS_DATASET_PATH") or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = _REPO_ROOT / raw
        return path
    return _REPO_ROOT / "eval" / "ragas_dataset.json"


def to_ragas_sample_step_25(
    item: dict[str, Any],
    retrieved: list[str] | None = None,
    response: str | None = None,
) -> dict[str, Any]:
    """步骤 25：转成 RAGAS SingleTurnSample 五列（英文列名，供 evaluate()）。"""
    query = str(item.get("user_input") or item.get("query") or "").strip()
    refs = [str(x).strip() for x in (item.get("reference_contexts") or []) if str(x).strip()]
    if retrieved is None:
        retrieved = [str(x).strip() for x in (item.get("retrieved_contexts") or []) if str(x).strip()]
    else:
        retrieved = [str(x).strip() for x in retrieved if str(x).strip()]
    if response is None:
        answer = str(item.get("response") or "")
    else:
        answer = response or ""
    return {
        "user_input": query,
        "retrieved_contexts": retrieved,
        "response": answer,
        "reference": str(item.get("reference") or "").strip(),
        "reference_contexts": refs,
    }


def to_ragas_dataset_step_26(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """步骤 26：把内部 testset / 评测行转成 EvaluationDataset.from_list 数组。"""
    if isinstance(payload, list):
        items = payload
    else:
        items = list(payload.get("items") or [])
        if not items and isinstance(payload.get("samples"), list):
            items = list(payload.get("samples") or [])
    samples: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sample = to_ragas_sample_step_25(item)
        if sample["user_input"]:
            samples.append(sample)
    return samples


def save_ragas_dataset_step_27(
    samples: list[dict[str, Any]],
    path: Path | None = None,
) -> Path:
    """步骤 27：写出 RAGAS dataset JSON 数组，可 EvaluationDataset.from_list 加载。"""
    p = path or dataset_path_step_24()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = [to_ragas_sample_step_25(s) for s in samples if isinstance(s, dict)]
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)
    return p


def load_ragas_dataset_step_28(path: Path | None = None) -> list[dict[str, Any]]:
    """步骤 28：读取官方列 dataset；缺文件返回空列表。"""
    p = path or dataset_path_step_24()
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("samples") or data.get("items") or []
    if not isinstance(data, list):
        raise ValueError("ragas_dataset.json 必须是样本数组")
    return to_ragas_dataset_step_26(data)


def try_ragas_evaluation_dataset_step_29(samples: list[dict[str, Any]]) -> Any:
    """步骤 29：若已安装 ragas，返回 EvaluationDataset；否则返回样本列表。"""
    cleaned = to_ragas_dataset_step_26(samples)
    _patch_ragas_langchain_community_step_31()
    try:
        from ragas import EvaluationDataset
    except ImportError:
        return cleaned
    return EvaluationDataset.from_list(cleaned)


def reference_span_step_30(
    refs: list[str],
    expect: list[str],
    refuse: bool,
) -> str:
    """步骤 30：从参考切块截一段原文作 reference，必须是切块子串。"""
    if refuse:
        return REFUSE_ANSWER
    blob = ""
    needle = expect[0] if expect else ""
    for text in refs:
        raw = str(text or "")
        if needle and needle in raw:
            blob = raw
            break
        if raw and not blob:
            blob = raw
    if not blob:
        return "、".join(expect) if expect else ""
    if needle:
        idx = blob.find(needle)
        if idx >= 0:
            start = max(0, idx - 24)
            end = min(len(blob), idx + 160)
            span = blob[start:end].strip()
            if span:
                return span
    return blob[:160].strip()


def _patch_ragas_langchain_community_step_31() -> None:
    """步骤 31：langchain-community 0.4 去掉 chat_models.vertexai，给 ragas 垫一层。

    不降级 langgraph / langchain-core。由 ragas_official_prereq_step_31 等调用。
    """
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    stub = types.ModuleType(name)

    class ChatVertexAI:
        """占位：仅供 ragas.llms.base 做 isinstance 列表，本仓库不用 Vertex。"""

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[name] = stub


_OFFICIAL_METRIC_NAMES = {
    "faithfulness": ("Faithfulness", "faithfulness"),
    "answer_relevancy": (
        "ResponseRelevancy",
        "AnswerRelevancy",
        "answer_relevancy",
        "response_relevancy",
    ),
    "context_precision": (
        "LLMContextPrecisionWithoutReference",
        "ContextPrecision",
        "context_precision",
    ),
    "context_recall": ("LLMContextRecall", "ContextRecall", "context_recall"),
    "answer_correctness": (
        "FactualCorrectness",
        "AnswerCorrectness",
        "factual_correctness",
        "answer_correctness",
    ),
    "answer_semantic_similarity": (
        "SemanticSimilarity",
        "AnswerSimilarity",
        "semantic_similarity",
        "answer_similarity",
    ),
}

_OFFICIAL_SCORE_ALIASES = {
    "faithfulness": "faithfulness",
    "nv_faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "answer_relevance": "answer_relevancy",
    "response_relevancy": "answer_relevancy",
    "nv_response_relevancy": "answer_relevancy",
    "context_precision": "context_precision",
    "llm_context_precision_without_reference": "context_precision",
    "non_llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
    "llm_context_recall": "context_recall",
    "non_llm_context_recall": "context_recall",
    "answer_correctness": "answer_correctness",
    "factual_correctness": "answer_correctness",
    "semantic_similarity": "answer_semantic_similarity",
    "answer_similarity": "answer_semantic_similarity",
    "answer_semantic_similarity": "answer_semantic_similarity",
}


def ragas_official_prereq_step_31() -> str:
    """步骤 31：检查官方 evaluate 前置条件。空串表示可跑，否则是跳过原因。"""
    _patch_ragas_langchain_community_step_31()
    try:
        import ragas  # noqa: F401
    except ImportError:
        return "未安装 ragas（可选依赖，见 requirements-eval.txt）"
    load_app_env()
    try:
        from common.llm import resolve_llm

        resolve_llm(None)
    except ValueError as exc:
        return str(exc)
    return ""


def wrap_official_models_step_32(
    encode_fn: EncodeFn | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> tuple[Any, Any]:
    """步骤 32：复用 MODEL_* 的 ChatOpenAI 与本机 BGE，包成 ragas wrapper。

    嵌入走本仓库 encode_fn（BGE 或测试注入/哈希），不另起一套向量模型。
    """
    wrapped_llm = llm
    if wrapped_llm is None:
        from common.llm import get_chat_model

        chat = get_chat_model(None, streaming=False, temperature=0)
        wrapped_llm = _wrap_langchain_llm_step_32(chat)

    wrapped_emb = embeddings
    if wrapped_emb is None:
        wrapped_emb = _wrap_encode_fn_embeddings_step_32(encode_fn)
    return wrapped_llm, wrapped_emb


def _wrap_langchain_llm_step_32(chat: Any) -> Any:
    """步骤 32：把 ChatOpenAI 包成 LangchainLLMWrapper。由 wrap_official_models_step_32 调用。"""
    _patch_ragas_langchain_community_step_31()
    for path in ("ragas.llms.base", "ragas.llms"):
        try:
            mod = __import__(path, fromlist=["LangchainLLMWrapper"])
            wrapper = getattr(mod, "LangchainLLMWrapper", None)
            if wrapper is not None:
                try:
                    return wrapper(
                        chat, bypass_n=True, bypass_temperature=True
                    )
                except TypeError:
                    return wrapper(chat)
        except ImportError:
            continue
    return chat


def _wrap_encode_fn_embeddings_step_32(encode_fn: EncodeFn | None = None) -> Any:
    """步骤 32：把 encode_fn 包成 LangChain Embeddings，再交给 ragas。由 wrap_official_models_step_32 调用。"""
    from langchain_core.embeddings import Embeddings

    class EncodeFnLangchainEmbeddings(Embeddings):
        """encode_fn 适配器。步骤：重写 Embeddings.embed_documents / embed_query。"""

        def __init__(self, fn: EncodeFn) -> None:
            super().__init__()
            self._fn = fn

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """步骤：重写 Embeddings.embed_documents。"""
            if not texts:
                return []
            vecs = np.asarray(self._fn(list(texts)), dtype="float32")
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)
            return vecs.tolist()

        def embed_query(self, text: str) -> list[float]:
            """步骤：重写 Embeddings.embed_query。"""
            rows = self.embed_documents([text or ""])
            return rows[0] if rows else []

    encode = resolve_encode_fn_step_05(encode_fn)
    lc_emb = EncodeFnLangchainEmbeddings(encode)
    _patch_ragas_langchain_community_step_31()
    for path in ("ragas.embeddings.base", "ragas.embeddings"):
        try:
            mod = __import__(path, fromlist=["LangchainEmbeddingsWrapper"])
            wrapper = getattr(mod, "LangchainEmbeddingsWrapper", None)
            if wrapper is not None:
                return wrapper(lc_emb)
        except ImportError:
            continue
    return lc_emb


def collect_official_metrics_step_33() -> list[Any]:
    """步骤 33：按现有六项英文（中文）对齐官方 ragas 指标类。

    走 ragas.metrics 的 v1 类（配合 evaluate + LangchainLLMWrapper）。
    collections 版需要 Instructor LLM，不能直接塞进 evaluate()。
    """
    _patch_ragas_langchain_community_step_31()
    try:
        import ragas.metrics as metrics_mod
    except ImportError as exc:
        raise RuntimeError("未安装 ragas，无法构造官方指标") from exc

    built: list[Any] = []
    missing: list[str] = []
    for mid, names in _OFFICIAL_METRIC_NAMES.items():
        obj = None
        for name in names:
            raw = getattr(metrics_mod, name, None)
            if raw is None:
                continue
            obj = raw() if isinstance(raw, type) else raw
            break
        if obj is None:
            missing.append(mid)
        else:
            built.append(obj)
    if not built:
        raise RuntimeError(f"ragas.metrics 无法构造任何指标: {missing}")
    if missing:
        logger.warning("官方 ragas 缺少指标: %s", missing)
    return built


def _canon_official_metric_id_step_34(raw: str) -> str | None:
    """步骤 34：把 ragas 返回的列名映射到本仓库六项 id。由 official_result_to_report_step_34 调用。"""
    key = str(raw or "").strip().lower().replace(" ", "_")
    key = key.replace("(", "_").replace(")", "").replace("=", "_")
    if key in _OFFICIAL_SCORE_ALIASES:
        return _OFFICIAL_SCORE_ALIASES[key]
    for alias, mid in _OFFICIAL_SCORE_ALIASES.items():
        if alias in key or key.startswith(alias):
            return mid
    return None


def _as_metric_float_step_34(val: Any) -> float | None:
    """步骤 34：把 ragas 分数转成可平均的 float。由 official_result_to_report_step_34 调用。"""
    if val is None:
        return None
    if isinstance(val, dict):
        for nested in ("f1", "score", "value", "precision", "recall"):
            if nested in val:
                return _as_metric_float_step_34(val[nested])
        return None
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    if num != num:
        return None
    return num


def official_result_to_report_step_34(
    result: Any,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """步骤 34：把 ragas.evaluate 返回值收成与嵌入评测同结构的报告。"""
    catalog = {m["id"]: m for m in list_metrics_step_01()}
    per_id: dict[str, list[float]] = {mid: [] for mid in catalog}
    item_scores: list[dict[str, Any]] = [{} for _ in samples]

    rows: list[dict[str, Any]] = []
    if hasattr(result, "to_pandas"):
        try:
            frame = result.to_pandas()
            rows = frame.to_dict(orient="records") if frame is not None else []
        except Exception as exc:
            logger.warning("ragas 结果转 pandas 失败: %s", exc)
    if not rows and isinstance(result, dict):
        rows = [result]
    if not rows and hasattr(result, "scores"):
        raw_scores = getattr(result, "scores")
        if isinstance(raw_scores, list):
            rows = [x for x in raw_scores if isinstance(x, dict)]
        elif isinstance(raw_scores, dict):
            rows = [raw_scores]

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        mapped: dict[str, float] = {}
        for key, val in row.items():
            mid = _canon_official_metric_id_step_34(str(key))
            num = _as_metric_float_step_34(val)
            if mid is None or num is None:
                continue
            mapped[mid] = num
            per_id.setdefault(mid, []).append(num)
        if idx < len(item_scores):
            item_scores[idx] = mapped

    metrics_out: list[dict[str, Any]] = []
    for mid, meta in catalog.items():
        vals = per_id.get(mid) or []
        score = round(sum(vals) / len(vals), 4) if vals else None
        metrics_out.append(
            {
                "id": mid,
                "name": meta["name"],
                "why": meta["why"],
                "score": score,
                "n": len(vals),
            }
        )

    items: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        items.append(
            {
                **to_ragas_sample_step_25(sample),
                "retrieved_n": len(sample.get("retrieved_contexts") or []),
                "scores": item_scores[idx] if idx < len(item_scores) else {},
            }
        )
    return {
        "n_samples": len(samples),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "ragas-official",
        "answer_mode": "official",
        "metrics": metrics_out,
        "items": items,
    }


def run_official_evaluate_step_35(
    samples: list[dict[str, Any]] | None = None,
    *,
    encode_fn: EncodeFn | None = None,
    llm: Any | None = None,
    embeddings: Any | None = None,
    evaluate_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """步骤 35：EvaluationDataset.from_list → ragas.evaluate()。

    非拒答行必须已有 retrieved_contexts 与 response（先跑 all/run）。
    抽取式答案上 Faithfulness 会偏高，这是接好流水线的预期。
    """
    reason = ragas_official_prereq_step_31()
    if reason and evaluate_fn is None:
        raise RuntimeError(reason)

    cleaned = to_ragas_dataset_step_26(samples or load_ragas_dataset_step_28())
    if not cleaned:
        raise ValueError("dataset 为空，请先 python scripts/eval_ragas.py all")

    for row in cleaned:
        refs = row.get("reference_contexts") or []
        retrieved = row.get("retrieved_contexts") or []
        response = str(row.get("response") or "").strip()
        refuse = (not refs) and (response == REFUSE_ANSWER or not retrieved)
        if refuse:
            continue
        if not retrieved or not response:
            raise ValueError(
                "非拒答行缺少 retrieved_contexts/response，请先 python scripts/eval_ragas.py all"
            )

    _patch_ragas_langchain_community_step_31()
    from ragas import EvaluationDataset, evaluate as ragas_evaluate

    dataset = EvaluationDataset.from_list(cleaned)
    metrics = collect_official_metrics_step_33()
    wrapped_llm, wrapped_emb = wrap_official_models_step_32(
        encode_fn=encode_fn, llm=llm, embeddings=embeddings
    )
    runner = evaluate_fn or ragas_evaluate
    extra: dict[str, Any] = {"raise_exceptions": False}
    if evaluate_fn is None:
        try:
            from ragas.run_config import RunConfig

            extra["run_config"] = RunConfig(timeout=180)
        except Exception:
            pass
    try:
        result = runner(
            dataset=dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_emb,
            **extra,
        )
    except TypeError:
        extra.pop("run_config", None)
        extra.pop("raise_exceptions", None)
        result = runner(
            dataset,
            metrics=metrics,
            llm=wrapped_llm,
            embeddings=wrapped_emb,
        )
    return official_result_to_report_step_34(result, cleaned)

