"""Compare the release page-markdown BM25 index with PDF-inspector BM25."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_hierarchical_retrieval import _derived_metrics, _stratified_folds, _write_run  # noqa: E402
from physics_adaptive_evidence_fusion import _load_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_release_markdown_bm25_baseline"
BM25_PATH = DEFAULT_ROOT / "results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"


def _run_from_index(index: BM25Index, pages: list[str], queries: list[str], qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for query, qid in zip(queries, qids):
        hits = index.search(query, 100)
        output[qid] = [{"chunk_id": pages[position], "doc_id": pages[position], "score": float(score), "rank": rank} for rank, (position, score) in enumerate(hits, 1)]
    return output


def _cached_scores(run: Mapping[str, list[dict[str, Any]]], pages: list[str], qids: list[str]) -> np.ndarray:
    positions = {page: index for index, page in enumerate(pages)}
    matrix = np.zeros((len(qids), len(pages)), dtype=np.float32)
    for query_index, qid in enumerate(qids):
        for item in run[qid][:100]:
            page = str(item["chunk_id"])
            if page in positions:
                matrix[query_index, positions[page]] = max(0.0, float(item.get("score", 0.0)))
    return matrix


def _norm(row: np.ndarray) -> np.ndarray:
    maximum = float(row.max()) if row.size else 0.0
    return row / maximum if maximum else np.zeros_like(row)


def _run_from_scores(pages: list[str], scores: np.ndarray, qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for query_index, qid in enumerate(qids):
        order = np.argsort(-scores[query_index], kind="stable")[:100]
        output[qid] = [{"chunk_id": pages[int(position)], "doc_id": pages[int(position)], "score": float(scores[query_index, int(position)]), "rank": rank} for rank, position in enumerate(order, 1)]
    return output


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    queries = [q.query for q in questions]
    qrels = benchmark.qrels()
    corpus = list(benchmark.corpus())
    pages = [doc.doc_id for doc in corpus]
    texts = [doc.text for doc in corpus]
    release_index = BM25Index(analyzer_name="plain").build([{"chunk_id": page, "doc_id": page, "text": text} for page, text in zip(pages, texts)])
    release_run = _run_from_index(release_index, pages, queries, qids)
    cached = _load_run(BM25_PATH)
    cached_scores = _cached_scores(cached, pages, qids)
    release_scores = np.zeros((len(qids), len(pages)), dtype=np.float32)
    page_pos = {page: index for index, page in enumerate(pages)}
    for query_index, qid in enumerate(qids):
        for item in release_run[qid]:
            release_scores[query_index, page_pos[item["chunk_id"]]] = float(item["score"])
    fused_scores = np.asarray([0.70 * _norm(row) for row in cached_scores]) + np.asarray([0.30 * _norm(row) for row in release_scores])
    fused_run = _run_from_scores(pages, fused_scores, qids)
    runs = {"pdf_inspector_bm25_control": cached, "release_markdown_bm25": release_run, "pdf_plus_release_a070": fused_run}
    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    candidates = list(runs)
    oof = {}
    selections = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append({"fold": fold, "selected": winner, "train_page_recall@10": _metric_for_qids(metrics[winner], train), "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["Release markdown is indexed with the same BM25 implementation and French queries.", "Fusion uses fixed PDF-inspector weight 0.70; no qrel-driven tuning is used.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics release markdown BM25 baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
