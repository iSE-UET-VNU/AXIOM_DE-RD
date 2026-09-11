"""Fuse the cached Kf=3 and Kf=4 cascade rankings for Physics.

The two input runs were produced by independent offline experiments.  This
script only tests whether their retained-page sets are complementary; it
does not rerun a parser, encoder or reranker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _paired_comparison,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_KF3_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_legacy_search/runs/best_full_set.jsonl"
DEFAULT_KF4_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_kf_legacy_search/runs/best_full_set.jsonl"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_cross_kf_fusion"


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _normalise(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    values = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in items}
    positive = [value for value in values.values() if value > 0]
    maximum = max(positive, default=0.0)
    return {key: (max(0.0, value) / maximum if maximum else 0.0) for key, value in values.items()}


def _fuse_score(run_a: Sequence[dict[str, Any]], run_b: Sequence[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    left = _normalise(run_a)
    right = _normalise(run_b)
    pages = set(left) | set(right)
    scores = {page: weight * left.get(page, 0.0) + (1.0 - weight) * right.get(page, 0.0) for page in pages}
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [
        {"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank}
        for rank, page in enumerate(ordered, 1)
    ]


def _fuse_rrf(run_a: Sequence[dict[str, Any]], run_b: Sequence[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for run in (run_a, run_b):
        for rank, item in enumerate(run[:100], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--kf3-run", type=Path, default=DEFAULT_KF3_RUN)
    parser.add_argument("--kf4-run", type=Path, default=DEFAULT_KF4_RUN)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    kf3 = _load_run(args.kf3_run)
    kf4 = _load_run(args.kf4_run)
    baseline = _load_run(args.baseline_run)
    if set(kf3) != set(qids) or set(kf4) != set(qids):
        raise RuntimeError("Kf runs do not contain the same 302 qids")
    if any(not (0 < len(kf3[qid]) <= 100 and 0 < len(kf4[qid]) <= 100) for qid in qids):
        raise RuntimeError("Expected 1..100 candidates per input qid")

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {"Kf3 input": kf3, "Kf4 input": kf4}
    methods: dict[str, dict[str, Any]] = {}
    for weight in np.round(np.linspace(0.0, 1.0, 21), 2):
        name = f"score_fusion_kf3_weight_{weight:.2f}"
        runs[name] = {qid: _fuse_score(kf3[qid], kf4[qid], float(weight)) for qid in qids}
    for constant in (5, 10, 20, 40, 60, 100):
        name = f"rrf_k_{constant}"
        runs[name] = {qid: _fuse_rrf(kf3[qid], kf4[qid], constant) for qid in qids}
    baseline_metrics = _derived_metrics(baseline, qids, qrels)
    methods["weighted fusion alpha=0.70 (reference)"] = {"retrieval_metrics": baseline_metrics}
    for name, run in runs.items():
        metrics = _derived_metrics(run, qids, qrels)
        methods[name] = {"retrieval_metrics": metrics, "comparison_to_weighted": _paired_comparison(baseline_metrics, metrics)}

    candidate_names = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    selected: list[dict[str, Any]] = []
    oof_run: dict[str, list[dict[str, Any]]] = {}
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidate_names, key=lambda name: (_metric_for_qids(methods[name]["retrieval_metrics"], train), name))
        for qid in heldout:
            oof_run[qid] = runs[winner][qid]
        selected.append({
            "fold": fold,
            "selected_method": winner,
            "train_page_recall@10": _metric_for_qids(methods[winner]["retrieval_metrics"], train),
            "test_page_recall@10": _metric_for_qids(methods[winner]["retrieval_metrics"], set(heldout)),
        })
    oof_metrics = _derived_metrics(oof_run, qids, qrels)
    oof_comparison = _paired_comparison(baseline_metrics, oof_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(candidate_names, key=lambda name: (methods[name]["retrieval_metrics"]["page_recall@10"], methods[name]["retrieval_metrics"]["ndcg@10"]))
    _write_run(runs_dir / "best_full_set.jsonl", runs[best_name], qids, queries={q.qid: q.query for q in questions})
    _write_run(runs_dir / "oof_selected.jsonl", oof_run, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics", "queries": len(qids),
        "inputs": {"kf3": str(args.kf3_run), "kf4": str(args.kf4_run), "baseline": str(args.baseline_run)},
        "methods": methods,
        "best_full_set": best_name,
        "oof": {"folds": selected, "metrics": oof_metrics, "comparison_to_weighted": oof_comparison},
        "notes": ["Input runs are cached hard-cascade rankings; no new model or parser is run.", "Full-set tuning is exploratory; OOF selection uses training folds only."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics cross-Kf fusion", "", "| Method | Page recall@10 | nDCG@10 | Delta vs weighted pp | File recall@3 |", "|---|---:|---:|---:|---:|"]
    ordered = sorted(candidate_names, key=lambda name: methods[name]["retrieval_metrics"]["page_recall@10"], reverse=True)
    for name in ordered:
        metric = methods[name]["retrieval_metrics"]
        lines.append(f"| {name} | {metric['page_recall@10']:.2%} | {metric['ndcg@10']:.2f} | {methods[name]['comparison_to_weighted']['page_recall_delta_pp']:+.2f} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"OOF page recall@10: **{oof_metrics['page_recall@10']:.2%}** ({oof_comparison['page_recall_delta_pp']:+.2f} pp vs weighted).", "", "| Fold | Selected method | Test recall |", "|---:|---|---:|"]
    for row in selected:
        lines.append(f"| {row['fold']} | `{row['selected_method']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_name, "best_full_page_recall@10": methods[best_name]["retrieval_metrics"]["page_recall@10"], "oof_page_recall@10": oof_metrics["page_recall@10"], "oof_delta_pp": oof_comparison["page_recall_delta_pp"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
