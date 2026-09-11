"""E5 page verification over the union of the two strongest hierarchy runs."""

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

from physics_adaptive_evidence_fusion import _load_run  # noqa: E402
from physics_hierarchical_retrieval import _derived_metrics, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402

DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
RESULT_ROOT = DEFAULT_ROOT / "results"
E5_ROOT = RESULT_ROOT / "physics_multilingual_e5_baseline"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_e5_structural_union"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min()) if len(values) else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(shifted.max()) if len(shifted) else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _rrf_score(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {str(item["chunk_id"]): 1.0 / (20 + rank) for rank, item in enumerate(items[:100], 1)}


def _union_run(dense: np.ndarray, left: Mapping[str, list[dict[str, Any]]], right: Mapping[str, list[dict[str, Any]]], pages: list[str], qids: list[str], dense_weight: float, prior_kind: str) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output = {}
    for query_index, qid in enumerate(qids):
        left_items = left[qid][:100]
        right_items = right[qid][:100]
        candidate_pages = {str(item["chunk_id"]) for item in left_items} | {str(item["chunk_id"]) for item in right_items}
        rrf_left = _rrf_score(left_items)
        rrf_right = _rrf_score(right_items)
        parent_values = np.asarray([
            (max(rrf_left.get(page, 0.0), rrf_right.get(page, 0.0)) if prior_kind == "max_rrf" else rrf_left.get(page, 0.0) + rrf_right.get(page, 0.0))
            for page in candidate_pages
        ], dtype=np.float32)
        dense_values = np.asarray([dense[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        candidate_list = list(candidate_pages)
        combined = dense_weight * _normalise(dense_values) + (1.0 - dense_weight) * _normalise(parent_values)
        order = np.argsort(-combined, kind="stable")[:100]
        output[qid] = [{"chunk_id": candidate_list[int(index)], "doc_id": candidate_list[int(index)], "score": float(combined[int(index)]), "rank": rank} for rank, index in enumerate(order, 1)]
    return output


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--e5-dir", type=Path, default=E5_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    dense = np.load(args.e5_dir / "page_dense_scores.npy", mmap_mode="r")
    c014 = _load_run(RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl")
    cross_kf = _load_run(RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl")
    runs = {"c014_control": c014, "cross_kf_control": cross_kf}
    for parent_kind in ("max_rrf", "sum_rrf"):
        for weight in (0.50, 0.70, 0.85):
            name = f"e5_union_{parent_kind}_w{weight:.2f}"
            runs[name] = _union_run(dense, c014, cross_kf, pages, qids, weight, parent_kind)
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
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "e5_scores": str(args.e5_dir / "page_dense_scores.npy"), "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["Candidate set is the per-query union of c014 and cross-Kf top-100 pages.", "Structural prior is fixed RRF-20 max or sum; E5 is page verifier with predeclared dense weights.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics E5 structural union", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
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
