"""Test page-specific BM25 length normalization as one fixed baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_adaptive_evidence_fusion import _load_page_texts, _load_run  # noqa: E402
from physics_hierarchical_retrieval import _derived_metrics, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_bm25_page_length_baseline"
RESULT_ROOT = DEFAULT_ROOT / "results"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
BASELINE_PATH = RESULT_ROOT / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"


def _load_pages() -> tuple[list[str], list[str]]:
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    by_page = _load_page_texts(PARSED_RUN)
    texts = [by_page.get(page, "") for page in pages]
    if len(pages) != 1674 or any(not text.strip() for text in texts):
        raise RuntimeError("Expected all 1,674 page texts")
    return pages, texts


def _run(index: BM25Index, pages: list[str], queries: list[str], qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for query, qid in zip(queries, qids):
        hits = index.search(query, 100)
        output[qid] = [{"chunk_id": pages[position], "doc_id": pages[position], "score": float(score), "rank": rank} for rank, (position, score) in enumerate(hits, 1)]
    return output


def _rrf(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], constant: int = 20) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for stream in (left, right):
        for rank, item in enumerate(stream[:100], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


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
    page_rows = [{"chunk_id": page, "doc_id": page, "text": text} for page, text in zip(pages, texts)]
    no_length_index = BM25Index(analyzer_name="plain", k1=1.2, b=0.0).build(page_rows)
    no_length_run = _run(no_length_index, pages, queries, qids)
    baseline = _load_run(BASELINE_PATH)
    c014 = _load_run(RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl")
    cross_kf = _load_run(RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl")
    runs = {
        "plain_bm25_control": baseline,
        "bm25_b0_no_length_norm": no_length_run,
        "rrf_bm25_b0_c014": {qid: _rrf(no_length_run[qid], c014[qid]) for qid in qids},
        "rrf_bm25_b0_cross_kf": {qid: _rrf(no_length_run[qid], cross_kf[qid]) for qid in qids},
    }
    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
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
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "bm25": {"k1": 1.2, "b": 0.0}, "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["b=0 is a single predeclared page-specific hypothesis, not a parameter sweep.", "RRF uses fixed constant 20 and only tests complementarity with existing structural controls.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics BM25 page-length baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
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
