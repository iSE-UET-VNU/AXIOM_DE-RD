"""Fold-fitted lightweight reranking over the multi-index candidate union.

This follows the candidate-ceiling diagnosis: the existing streams contain
more relevant pages than any one final rank exposes.  The model is a tiny
pairwise linear ranker over rank/presence features only.  It is deliberately
not a neural reranker and uses no page text, qrel modality or benchmark answer.
Gold page labels are used only for training folds; the headline number is OOF.
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

from research.experiments.physics_hierarchical_retrieval import _derived_metrics, _load_run, _paired_comparison, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_multistream_ltr"


STREAMS = ("bm25", "splade", "weighted", "hierarchical", "c014", "cross_kf", "legacy_second")
FEATURE_NAMES = tuple(
    [f"{name}_rank" for name in STREAMS]
    + [f"{name}_top10" for name in STREAMS]
    + ["stream_count", "best_rank", "rank_agreement"]
)


def _rank_map(stream: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {str(row["chunk_id"]): rank for rank, row in enumerate(stream[:100], 1)}


def _features(streams: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, list[float]]:
    ranks = {name: _rank_map(streams[name]) for name in STREAMS}
    pages = set().union(*(mapping for mapping in ranks.values()))
    output = {}
    for page in pages:
        raw = [ranks[name].get(page, 0) for name in STREAMS]
        rank_values = [1.0 / (rank + 1) if rank else 0.0 for rank in raw]
        top_values = [float(1 <= rank <= 10) for rank in raw]
        present = [rank for rank in raw if rank]
        best = min(present) if present else 101
        # Agreement is a fixed pair count: how many stream pairs both contain
        # this page in their top-10. It rewards corroboration without using a
        # relevance label.
        top_names = [name for name, rank in zip(STREAMS, raw) if 1 <= rank <= 10]
        agreement = len(top_names) * (len(top_names) - 1) / 2.0
        output[page] = rank_values + top_values + [float(len(present)), 1.0 / (best + 1), agreement]
    return output


def _fit_pairwise(
    train_qids: Sequence[str],
    rows_by_qid: Mapping[str, list[dict[str, Any]]],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    train_rows = [row for qid in train_qids for row in rows_by_qid[qid]]
    matrix = np.asarray([row["features"] for row in train_rows], dtype=np.float64)
    mean = matrix.mean(axis=0) if len(matrix) else np.zeros(len(FEATURE_NAMES))
    std = matrix.std(axis=0) if len(matrix) else np.ones(len(FEATURE_NAMES))
    std[std < 1e-8] = 1.0
    rng = np.random.default_rng(seed)
    differences = []
    pair_count = 0
    for qid in train_qids:
        positives = [row for row in rows_by_qid[qid] if row["label"]]
        negatives = [row for row in rows_by_qid[qid] if not row["label"]]
        if not positives or not negatives:
            continue
        for _ in range(min(128, len(positives) * len(negatives))):
            positive = positives[int(rng.integers(len(positives)))]
            negative = negatives[int(rng.integers(len(negatives)))]
            differences.append(
                (np.asarray(positive["features"], dtype=np.float64) - mean) / std
                - (np.asarray(negative["features"], dtype=np.float64) - mean) / std
            )
            pair_count += 1
    pair_matrix = np.asarray(differences, dtype=np.float64) if differences else np.zeros((0, len(FEATURE_NAMES)))
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(100):
        if len(pair_matrix):
            margins = np.clip(pair_matrix @ weights, -40.0, 40.0)
            gradient = -((pair_matrix.T @ (1.0 / (1.0 + np.exp(margins)))) / len(pair_matrix))
        else:
            gradient = np.zeros_like(weights)
        weights -= 0.05 * (gradient + 0.001 * weights)
    return weights, mean, std, pair_count


def _rank(rows_by_qid: Mapping[str, list[dict[str, Any]]], weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for qid, rows in rows_by_qid.items():
        ranked = []
        for row in rows:
            score = float(((np.asarray(row["features"], dtype=np.float64) - mean) / std) @ weights)
            ranked.append({"chunk_id": row["page_id"], "doc_id": row["page_id"], "score": score})
        ordered = sorted(ranked, key=lambda row: (-row["score"], row["chunk_id"]))[:100]
        output[qid] = [{**row, "rank": rank} for rank, row in enumerate(ordered, 1)]
    return output


def _mean(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return sum(float(row["page_recall@10"]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    root = args.benchmark_root / "results"
    paths = {
        "bm25": root / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
        "splade": root / "physics_vsplade_bm25_fusion/vsplade_french_bm25-french_vs-english.jsonl",
        "weighted": root / "physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl",
        "hierarchical": root / "physics_hierarchical_retrieval/oof_run.jsonl",
        "c014": root / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
        "cross_kf": root / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
        "legacy_second": root / "physics_legacy_second_retrieval/oof_run.jsonl",
    }
    loaded = {name: _load_run(path) for name, path in paths.items()}
    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        stream_qid = {name: loaded[name][qid] for name in STREAMS}
        feature_map = _features(stream_qid)
        gold = set(qrels.get(qid, {}))
        rows_by_qid[qid] = [{"page_id": page_id, "features": values, "label": page_id in gold} for page_id, values in feature_map.items()]

    folds = [qids[i::5] for i in range(5)]
    oof: dict[str, list[dict[str, Any]]] = {}
    fold_rows = []
    weight_rows = []
    for fold, heldout in enumerate(folds):
        train = [qid for qid in qids if qid not in set(heldout)]
        weights, mean, std, pair_count = _fit_pairwise(train, rows_by_qid, seed=42 + fold)
        test_rows = {qid: rows_by_qid[qid] for qid in heldout}
        ranked = _rank(test_rows, weights, mean, std)
        oof.update(ranked)
        fold_metric = _derived_metrics(ranked, heldout, qrels)
        fold_rows.append({"fold": fold, "test_page_recall@10": fold_metric["page_recall@10"], "test_ndcg@10": fold_metric["ndcg@10"], "pair_count": pair_count})
        weight_rows.append({"fold": fold, "weights": {name: float(value) for name, value in zip(FEATURE_NAMES, weights)}})
    metric = _derived_metrics(oof, qids, qrels)
    refs = {
        "c014": _derived_metrics(loaded["c014"], qids, qrels),
        "cross_kf": _derived_metrics(loaded["cross_kf"], qids, qrels),
        "weighted": _derived_metrics(loaded["weighted"], qids, qrels),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "candidate_streams": {name: str(path) for name, path in paths.items()},
        "feature_names": FEATURE_NAMES,
        "cv": {"folds": fold_rows, "retrieval_metrics": metric, "weight_rows": weight_rows},
        "references": refs,
        "comparisons": {name: _paired_comparison(ref, metric) for name, ref in refs.items()},
        "notes": [
            "Candidate set is the union of top-100 pages from seven cache-only streams.",
            "The pairwise linear ranker is fit independently on each training fold; held-out qrels are not used in fitting.",
            "Features are ranks/presence/agreement only; no text, modality labels or answer content is used.",
            "V-SPLADE streams use cached English query vectors against French qrels.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics multi-stream LTR", "", "| Method | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs c014 pp |", "|---|---:|---:|---:|---:|---:|"]
    for name, ref in refs.items():
        lines.append(f"| {name} reference | {ref['ndcg@10']:.2f} | {ref['page_recall@10']:.2%} | {ref['page_hit@10']:.2%} | {ref['file_metrics_by_k']['3']['file_recall']:.2%} | +0.00 |")
    lines.append(f"| **multistream-ltr OOF** | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} | {report['comparisons']['c014']['page_recall_delta_pp']:+.2f} |")
    lines += ["", f"OOF fold rows: `{fold_rows}`.", "Feature learning uses only rank metadata; qrel labels appear only as training targets within non-held-out folds."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "oof_page_recall@10": round(metric["page_recall@10"] * 100, 2), "oof_file_recall@3": round(metric["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "delta_vs_c014_pp": report["comparisons"]["c014"]["page_recall_delta_pp"], "folds": fold_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
