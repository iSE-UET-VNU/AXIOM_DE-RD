"""Fixed field-index baseline using Markdown headings as a page field."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import re
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_adaptive_evidence_fusion import _load_page_texts, _load_run  # noqa: E402
from physics_hierarchical_retrieval import _derived_metrics, _paired_comparison, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_structured_text_index_baseline"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
BM25_PATH = DEFAULT_ROOT / "results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"


def _load_pages() -> tuple[list[str], list[str]]:
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    by_page = _load_page_texts(PARSED_RUN)
    texts = [by_page.get(page, "") for page in pages]
    if len(pages) != 1674 or any(not text.strip() for text in texts):
        raise RuntimeError("Expected non-empty text for all Physics pages")
    return pages, texts


def _heading_text(text: str) -> str:
    headings = [line.lstrip("#").strip() for line in str(text).splitlines() if line.lstrip().startswith("#")]
    return "\n".join(headings)


def _score_matrix(index: BM25Index, queries: list[str], page_count: int) -> np.ndarray:
    matrix = np.zeros((len(queries), page_count), dtype=np.float32)
    for query_index, query in enumerate(queries):
        for position, score in index.search(query, page_count):
            matrix[query_index, position] = float(score)
    return matrix


def _norm(row: np.ndarray) -> np.ndarray:
    maximum = float(np.max(row)) if row.size else 0.0
    return row / maximum if maximum > 0.0 else np.zeros_like(row)


def _cached_scores(run: Mapping[str, list[dict[str, Any]]], pages: list[str], qids: list[str]) -> np.ndarray:
    positions = {page: index for index, page in enumerate(pages)}
    matrix = np.zeros((len(qids), len(pages)), dtype=np.float32)
    for query_index, qid in enumerate(qids):
        for item in run[qid][:100]:
            page = str(item["chunk_id"])
            if page in positions:
                matrix[query_index, positions[page]] = max(0.0, float(item.get("score", 0.0)))
    return matrix


def _run_from_scores(pages: list[str], matrix: np.ndarray, qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        order = np.argsort(-matrix[query_index], kind="stable")[:100]
        output[qid] = [{"chunk_id": pages[int(position)], "doc_id": pages[int(position)], "score": float(matrix[query_index, int(position)]), "rank": rank} for rank, position in enumerate(order, 1)]
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
    pages, texts = _load_pages()
    bm25_run = _load_run(BM25_PATH)
    bm25_scores = _cached_scores(bm25_run, pages, qids)
    heading_index = BM25Index(analyzer_name="plain").build([{"chunk_id": page, "doc_id": page, "text": _heading_text(text)} for page, text in zip(pages, texts)])
    heading_scores = _score_matrix(heading_index, queries, len(pages))
    runs = {
        "bm25_control": bm25_run,
        "heading_only": _run_from_scores(pages, heading_scores, qids),
        "bm25_plus_heading_a080": _run_from_scores(pages, np.asarray([0.80 * _norm(row) for row in bm25_scores]) + np.asarray([0.20 * _norm(row) for row in heading_scores]), qids),
    }
    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections = []
    candidates = list(runs)
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append({"fold": fold, "selected": winner, "train_page_recall@10": _metric_for_qids(metrics[winner], train), "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["Heading is an explicit secondary text field extracted from Markdown headings.", "Fusion keeps BM25 as the primary field with a fixed 0.80 weight; no qrel-driven field-weight search is used.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics structured text index baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
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
