"""Test a three-stream rank fusion for the cached Physics retrieval runs."""

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

from research.experiments.physics_hierarchical_retrieval import _derived_metrics, _paired_comparison, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_WEIGHTED = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl"
DEFAULT_KF3 = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_legacy_search/runs/best_full_set.jsonl"
DEFAULT_KF4 = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_kf_legacy_search/runs/best_full_set.jsonl"
DEFAULT_OUTPUT = ROOT / "data/benchmark/vidore_v3/results/physics_threeway_rrf"


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _rrf(runs: Sequence[Sequence[dict[str, Any]]], weights: Sequence[float], constant: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for stream, weight in zip(runs, weights):
        for rank, item in enumerate(stream[:100], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + float(weight) / (constant + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


def _mean_recall(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--weighted-run", type=Path, default=DEFAULT_WEIGHTED)
    parser.add_argument("--kf3-run", type=Path, default=DEFAULT_KF3)
    parser.add_argument("--kf4-run", type=Path, default=DEFAULT_KF4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    weighted, kf3, kf4 = (_load(path) for path in (args.weighted_run, args.kf3_run, args.kf4_run))
    if set(weighted) != set(qids) or set(kf3) != set(qids) or set(kf4) != set(qids):
        raise RuntimeError("The three input runs must contain the same 302 qids")

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    # Fixed, predeclared rank weights.  Equal RRF is the main hypothesis;
    # nearby variants test whether the third stream is useful without tuning
    # on the complete test qrels.
    weight_grid = [(1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 1.0, 2.0), (1.0, 2.0, 2.0), (2.0, 1.0, 2.0)]
    for constant in (10, 20, 40, 60):
        for weights in weight_grid:
            name = f"rrf3_k{constant}_w{weights[0]:g}{weights[1]:g}{weights[2]:g}"
            runs[name] = {qid: _rrf((weighted[qid], kf3[qid], kf4[qid]), weights, constant) for qid in qids}

    baseline_metrics = _derived_metrics(weighted, qids, qrels)
    methods: dict[str, dict[str, Any]] = {"weighted fusion alpha=0.70 (reference)": {"retrieval_metrics": baseline_metrics}}
    for name, run in runs.items():
        metrics = _derived_metrics(run, qids, qrels)
        methods[name] = {"retrieval_metrics": metrics, "comparison_to_weighted": _paired_comparison(baseline_metrics, metrics)}

    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    selected: list[dict[str, Any]] = []
    oof: dict[str, list[dict[str, Any]]] = {}
    candidate_names = list(runs)
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidate_names, key=lambda name: (_mean_recall(methods[name]["retrieval_metrics"], train), name))
        for qid in heldout:
            oof[qid] = runs[winner][qid]
        selected.append({"fold": fold, "selected_method": winner, "train_page_recall@10": _mean_recall(methods[winner]["retrieval_metrics"], train), "test_page_recall@10": _mean_recall(methods[winner]["retrieval_metrics"], set(heldout))})
    oof_metrics = _derived_metrics(oof, qids, qrels)
    oof_comparison = _paired_comparison(baseline_metrics, oof_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(candidate_names, key=lambda name: (methods[name]["retrieval_metrics"]["page_recall@10"], methods[name]["retrieval_metrics"]["ndcg@10"]))
    _write_run(runs_dir / "best_full_set.jsonl", runs[best_name], qids, queries={q.qid: q.query for q in questions})
    _write_run(runs_dir / "oof_selected.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {"dataset": "vidore_v3/physics", "inputs": {"weighted": str(args.weighted_run), "kf3": str(args.kf3_run), "kf4": str(args.kf4_run)}, "methods": methods, "best_full_set": best_name, "oof": {"folds": selected, "metrics": oof_metrics, "comparison_to_weighted": oof_comparison}}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics three-way RRF", "", "| Method | Page recall@10 | nDCG@10 | Delta vs weighted pp | File recall@3 |", "|---|---:|---:|---:|---:|"]
    for name in sorted(candidate_names, key=lambda n: methods[n]["retrieval_metrics"]["page_recall@10"], reverse=True):
        metric = methods[name]["retrieval_metrics"]
        lines.append(f"| {name} | {metric['page_recall@10']:.2%} | {metric['ndcg@10']:.2f} | {methods[name]['comparison_to_weighted']['page_recall_delta_pp']:+.2f} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"OOF page recall@10: **{oof_metrics['page_recall@10']:.2%}** ({oof_comparison['page_recall_delta_pp']:+.2f} pp vs weighted).", "", "| Fold | Selected method | Test recall |", "|---:|---|---:|"]
    for row in selected:
        lines.append(f"| {row['fold']} | `{row['selected_method']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_name, "best_full_page_recall@10": methods[best_name]["retrieval_metrics"]["page_recall@10"], "oof_page_recall@10": oof_metrics["page_recall@10"], "oof_delta_pp": oof_comparison["page_recall_delta_pp"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
