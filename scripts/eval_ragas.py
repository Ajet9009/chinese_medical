#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用嵌入生成 RAGAS 评测集并对文献 RAG 打分。CI 禁止 --live。official 才调 ragas 包。"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.eval_golden import load_golden  # noqa: E402
from common.eval_ragas import (  # noqa: E402
    dataset_path_step_24,
    generate_testset_step_13,
    load_corpus_chunks_step_10,
    load_ragas_dataset_step_28,
    load_testset_step_15,
    report_path_step_03,
    resolve_encode_fn_step_05,
    run_official_evaluate_step_35,
    run_rag_eval_step_20,
    save_ragas_dataset_step_27,
    save_report_step_21,
    save_testset_step_14,
    testset_path_step_02,
    to_ragas_dataset_step_26,
)


def resolve_cli_path_step_01(raw: str, default: Path | None = None) -> Path | None:
    """步骤 01：相对路径相对仓库根，与环境变量路径规则一致。"""
    if not (raw or "").strip():
        return default
    path = Path(raw.strip())
    if not path.is_absolute():
        path = ROOT / path
    return path


def build_parser_step_02() -> argparse.ArgumentParser:
    """步骤 02：组装 CLI。默认 generate+run，不打线上 LLM。"""
    parser = argparse.ArgumentParser(description="RAGAS 嵌入评测集生成与文献 RAG 测试")
    parser.add_argument(
        "cmd",
        nargs="?",
        default="all",
        choices=["generate", "run", "all", "official"],
        help="generate 只出题；run 只打分；all 先生成再测 RAG；official 调 ragas.evaluate",
    )
    parser.add_argument("--testset", default="", help="覆盖 RAGAS_TESTSET_PATH")
    parser.add_argument("--dataset", default="", help="覆盖 RAGAS_DATASET_PATH（官方五列 JSON 数组）")
    parser.add_argument("--report", default="", help="覆盖 RAGAS_REPORT_PATH")
    parser.add_argument("--corpus", default="", help="文献目录，默认 DOC_SOURCE_DIR/corpus")
    parser.add_argument("--top-k", type=int, default=4, help="检索条数")
    parser.add_argument("--no-crag", action="store_true", help="评测检索时关闭 CRAG")
    parser.add_argument("--live", action="store_true", help="打本机 /ask（不进 CI）")
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="", help="登录 JWT；--live 时需要")
    return parser


def generate_cli_step_03(corpus: str, testset_out: Path, dataset_out: Path, encode_fn) -> dict:
    """步骤 03：嵌入生成评测集，并写出 RAGAS EvaluationDataset 五列 JSON。"""
    source = resolve_cli_path_step_01(corpus, None)
    chunks = load_corpus_chunks_step_10(source)
    payload = generate_testset_step_13(
        golden_items=load_golden(),
        chunks=chunks,
        encode_fn=encode_fn,
    )
    save_testset_step_14(payload, testset_out)
    samples = to_ragas_dataset_step_26(payload)
    save_ragas_dataset_step_27(samples, dataset_out)
    print(f"testset: {testset_out}  ({len(payload.get('items') or [])} 条)")
    print(f"dataset: {dataset_out}  columns={['user_input','retrieved_contexts','response','reference','reference_contexts']}")
    for item in payload.get("items") or []:
        nref = len(item.get("reference_contexts") or [])
        print(f"  [{item.get('source')}] {item.get('user_input')}  refs={nref}")
    return payload


def live_ask_answers_step_04(base: str, token: str, question: str) -> dict:
    """步骤 04：请求本机 /ask，取答案与文献块（仅 --live）。"""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        base.rstrip("/") + "/ask",
        data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"  [skip] {question}: {exc}", file=sys.stderr)
        return {"response": "", "retrieved_contexts": []}
    chunks = body.get("doc_chunks") or []
    texts = []
    for row in chunks:
        if isinstance(row, dict) and str(row.get("text") or "").strip():
            texts.append(str(row.get("text") or ""))
    return {
        "response": str(body.get("answer") or ""),
        "retrieved_contexts": texts,
    }


def print_report_step_05(report: dict) -> None:
    """步骤 05：把英文（中文）指标均值打到终端。"""
    print(
        f"ragas: n={report.get('n_samples')}  mode={report.get('answer_mode')}  "
        f"at={report.get('generated_at')}"
    )
    if report.get("answer_mode") == "extractive":
        print("  注：抽取式答案的 Faithfulness（忠实度）接近 1 是预期；生成质量请加 --live")
    if report.get("answer_mode") == "official":
        print("  注：official 调 ragas.evaluate；抽取式 dataset 上 Faithfulness 仍会偏高")
    for row in report.get("metrics") or []:
        score = row.get("score")
        shown = "—" if score is None else f"{float(score):.4f}"
        print(f"  {row.get('name')}: {shown}  (n={row.get('n')})")
    for item in report.get("items") or []:
        scores = item.get("scores") or {}
        faith = scores.get("faithfulness")
        rec = scores.get("context_recall")
        print(
            f"  - {item.get('user_input')}  "
            f"faith={faith} recall={rec} retrieved={item.get('retrieved_n')}"
        )


def run_cli_step_06(
    payload: dict,
    encode_fn,
    report_out: Path,
    dataset_out: Path,
    top_k: int,
    use_crag: bool,
    live: bool,
    base: str,
    token: str,
    corpus: str,
) -> dict:
    """步骤 06：对评测集跑文献 RAG 并写报告。"""
    live_fn = None
    if live:
        if not token:
            raise SystemExit("--live 需要 --token")
        live_fn = partial(live_ask_answers_step_04, base, token)
        report = run_rag_eval_step_20(
            payload,
            encode_fn=encode_fn,
            live_fn=live_fn,
            top_k=top_k,
            use_crag=use_crag,
        )
    else:
        from common.doc_store import FaissDocStore

        source = resolve_cli_path_step_01(corpus, None)
        chunks = load_corpus_chunks_step_10(source)
        if not chunks:
            raise SystemExit("语料为空，无法构建评测用文献索引")
        with tempfile.TemporaryDirectory(prefix="ragas_eval_") as tmp:
            store = FaissDocStore(
                index_path=Path(tmp) / "docs.index",
                metadata_path=Path(tmp) / "docs.json",
                encode_fn=encode_fn,
            )
            store.build_chunks(chunks)
            report = run_rag_eval_step_20(
                payload,
                encode_fn=encode_fn,
                store=store,
                top_k=top_k,
                use_crag=use_crag,
            )
    save_report_step_21(report, report_out)
    samples = to_ragas_dataset_step_26(report)
    save_ragas_dataset_step_27(samples, dataset_out)
    print(f"report: {report_out}")
    print(f"dataset: {dataset_out}  ({len(samples)} 条，含 retrieved_contexts/response)")
    print_report_step_05(report)
    return report


def official_cli_step_08(dataset_out: Path, report_out: Path, encode_fn) -> dict:
    """步骤 08：读已填满的五列 dataset，调用 ragas.evaluate 并写报告。"""
    samples = load_ragas_dataset_step_28(dataset_out)
    report = run_official_evaluate_step_35(samples, encode_fn=encode_fn)
    save_report_step_21(report, report_out)
    print(f"report: {report_out}")
    print_report_step_05(report)
    return report


def main() -> int:
    """步骤：07 入口。默认嵌入生成 + 抽取式 RAG 评测，不调 LLM。"""
    args = build_parser_step_02().parse_args()
    ts_path = resolve_cli_path_step_01(args.testset, testset_path_step_02())
    ds_path = resolve_cli_path_step_01(args.dataset, dataset_path_step_24())
    rp_path = resolve_cli_path_step_01(args.report, report_path_step_03())
    encode_fn = resolve_encode_fn_step_05(None)
    if args.cmd == "official":
        official_cli_step_08(ds_path, rp_path, encode_fn)
        return 0
    payload = None
    if args.cmd in ("generate", "all"):
        payload = generate_cli_step_03(args.corpus, ts_path, ds_path, encode_fn)
    if args.cmd in ("run", "all"):
        if payload is None:
            payload = load_testset_step_15(ts_path)
        if not (payload.get("items") or []):
            print("评测集为空，请先 generate", file=sys.stderr)
            return 1
        run_cli_step_06(
            payload,
            encode_fn,
            rp_path,
            ds_path,
            top_k=args.top_k,
            use_crag=not args.no_crag,
            live=args.live,
            base=args.base,
            token=args.token,
            corpus=args.corpus,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
