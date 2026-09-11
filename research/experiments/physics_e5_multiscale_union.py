"""Combine two fixed E5 segment scales inside a structural candidate union."""

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
LONG_E5 = RESULT_ROOT / "physics_multilingual_e5_baseline/page_dense_scores.npy"
SHORT_E5 = RESULT_ROOT / "physics_multilingual_e5_short_segments/page_dense_scores.npy"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_e5_multiscale_union"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min()) if len(values) else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(shifted.max()) if len(shifted) else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _rrf_score(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {str(item["chunk_id"]): 1.0 / (20 + rank) for rank, item in enumerate(items[:100], 1)}


def _run(
    long_dense: np.ndarray,
    short_dense: np.ndarray,
    structural: Sequence[Mapping[str, list[dict[str, Any]]]],
    pages: list[str],
    qids: list[str],
    long_weight: float,
    short_weight: float,
) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output: dict[str, list[dict[str, Any]]] = {}
    prior_weight = 1.0 - long_weight - short_weight
    for query_index, qid in enumerate(qids):
        streams = [parent[qid][:100] for parent in structural]
        candidate_pages = sorted({str(item["chunk_id"]) for items in streams for item in items})
        rrf_maps = [_rrf_score(items) for items in streams]
        prior = np.asarray([max(rrf.get(page, 0.0) for rrf in rrf_maps) for page in candidate_pages], dtype=np.float32)
        long_values = np.asarray([long_dense[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        short_values = np.asarray([short_dense[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        combined = (
            long_weight * _normalise(long_values)
            + short_weight * _normalise(short_values)
            + prior_weight * _normalise(prior)
        )
        order = np.argsort(-combined, kind="stable")[:100]
        output[qid] = [
            {"chunk_id": candidate_pages[int(index)], "doc_id": candidate_pages[int(index)], "score": float(combined[int(index)]), "rank": rank}
            for rank, index in enumerate(order, 1)
        ]
    return output


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--long-scores", type=Path, default=LONG_E5)
    parser.add_argument("--short-scores", type=Path, default=SHORT_E5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    long_dense = np.load(args.long_scores, mmap_mode="r")
    short_dense = np.load(args.short_scores, mmap_mode="r")
    if long_dense.shape != (302, 1674) or short_dense.shape != (302, 1674):
        raise RuntimeError(f"Unexpected score shapes: {long_dense.shape}, {short_dense.shape}")
    structural_paths = [
        RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
        RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
        RESULT_ROOT / "physics_cascade_kf_legacy_search/runs/oof_selected.jsonl",
        RESULT_ROOT / "physics_cascade_grid_search/runs/oof_selected.jsonl",
        RESULT_ROOT / "physics_hierarchical_retrieval/runs/cascade-all_text-max-kf3-fusion.jsonl",
    ]
    structural = [_load_run(path) for path in structural_paths]
    runs = {
        "long_e5_control": _run(long_dense, long_dense, structural, pages, qids, 1.0, 0.0),
        "multiscale_l070_s015": _run(long_dense, short_dense, structural, pages, qids, 0.70, 0.15),
        "multiscale_l080_s010": _run(long_dense, short_dense, structural, pages, qids, 0.80, 0.10),
        "multiscale_l080_s015": _run(long_dense, short_dense, structural, pages, qids, 0.80, 0.15),
    }
    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
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
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "long_scores": str(args.long_scores),
        "short_scores": str(args.short_scores),
        "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()},
        "best_full_set": best_full,
        "oof": {"metrics": oof_metric, "folds": selections},
        "notes": ["Both E5 scales are max-pooled to page scores before fusion.", "The structural candidate union and max-RRF prior are fixed; scale weights are predeclared.", "All results use full page IDs and canonical derived metrics."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics E5 multiscale union", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
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
