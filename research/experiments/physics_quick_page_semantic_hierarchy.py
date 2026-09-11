"""Quick page-stage semantic fusion over the current Physics hierarchy.

This is deliberately narrow: keep the current hierarchical OOF page run as
one page-ranking stream, add the cached multilingual-E5 page stream, and
measure fixed score/RRF fusions.  It does not change file selection and does
not encode new images or documents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _file_id,
    _load_run,
    _page_vector_units,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_HIERARCHICAL_RUN = DEFAULT_ROOT / "results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_E5_SCORES = DEFAULT_ROOT / "results/physics_multilingual_e5_short_segments/page_dense_scores.npy"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_quick_page_semantic_hierarchy"
DEPTH = 100


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(np.min(values)) if values.size else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(np.max(shifted)) if values.size else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _run_from_scores(
    pages: list[str],
    qids: list[str],
    scores: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        order = np.argsort(-scores[query_index], kind="stable")[:DEPTH]
        output[qid] = [
            {
                "chunk_id": pages[int(position)],
                "doc_id": pages[int(position)],
                "score": float(scores[query_index, int(position)]),
                "rank": rank,
            }
            for rank, position in enumerate(order, 1)
        ]
    return output


def _fuse(
    current: Mapping[str, list[dict[str, Any]]],
    dense: np.ndarray,
    pages: list[str],
    qids: list[str],
    *,
    current_weight: float,
) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        current_items = current[qid][:DEPTH]
        current_scores = {str(item["chunk_id"]): max(0.0, float(item.get("score", 0.0))) for item in current_items}
        dense_order = np.argsort(-dense[query_index], kind="stable")[:DEPTH]
        dense_pages = {pages[int(index)] for index in dense_order}
        candidate_pages = sorted(set(current_scores) | dense_pages)
        current_values = np.asarray([current_scores.get(page, 0.0) for page in candidate_pages], dtype=np.float32)
        dense_values = np.asarray([dense[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        combined = current_weight * _normalise(current_values) + (1.0 - current_weight) * _normalise(dense_values)
        order = np.argsort(-combined, kind="stable")[:DEPTH]
        output[qid] = [
            {"chunk_id": candidate_pages[int(index)], "doc_id": candidate_pages[int(index)], "score": float(combined[int(index)]), "rank": rank}
            for rank, index in enumerate(order, 1)
        ]
    return output


def _rrf(
    current: Mapping[str, list[dict[str, Any]]],
    dense: np.ndarray,
    pages: list[str],
    qids: list[str],
    *,
    constant: int = 20,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    position = {page: index for index, page in enumerate(pages)}
    for query_index, qid in enumerate(qids):
        scores: dict[str, float] = {}
        for rank, item in enumerate(current[qid][:DEPTH], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
        for rank, page_index in enumerate(np.argsort(-dense[query_index], kind="stable")[:DEPTH], 1):
            page = pages[int(page_index)]
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:DEPTH]
        output[qid] = [
            {"chunk_id": page, "doc_id": page, "score": float(score), "rank": rank}
            for rank, (page, score) in enumerate(ranked, 1)
        ]
    return output


def _metric_for_qids(metric: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metric["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def _candidate_recall(
    run_left: Mapping[str, list[dict[str, Any]]],
    run_right: Mapping[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> float:
    values = []
    for qid in qids:
        candidates = {str(item["chunk_id"]) for item in run_left[qid][:DEPTH]} | {
            str(item["chunk_id"]) for item in run_right[qid][:DEPTH]
        }
        gold = set(qrels.get(qid, {}))
        values.append(len(candidates & gold) / len(gold) if gold else 0.0)
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--e5-scores", type=Path, default=DEFAULT_E5_SCORES)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    pages = _page_vector_units(args.page_vector_dir)
    current = _load_run(args.hierarchical_run)
    dense = np.load(args.e5_scores, mmap_mode="r")
    if dense.shape != (len(qids), len(pages)):
        raise RuntimeError(f"Unexpected E5 score shape: {dense.shape}")

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "current_hierarchy": current,
        "e5_dense": _run_from_scores(pages, qids, dense),
    }
    for weight in (0.70, 0.50, 0.30):
        runs[f"current_plus_e5_score_w{weight:.2f}"] = _fuse(
            current, dense, pages, qids, current_weight=weight
        )
    runs["current_plus_e5_rrf20"] = _rrf(current, dense, pages, qids)

    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append({
            "fold": fold,
            "selected": winner,
            "train_page_recall@10": _metric_for_qids(metrics[winner], train),
            "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout)),
        })
    oof_metric = _derived_metrics(oof, qids, qrels)
    union_recall = _candidate_recall(current, runs["e5_dense"], qids, qrels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(runs_dir / f"{name}.jsonl", run, qids, queries={q.qid: q.query for q in questions})
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(pages),
        "e5_scores": str(args.e5_scores),
        "hierarchical_run": str(args.hierarchical_run),
        "candidate_union_recall": union_recall,
        "methods": {
            name: {
                "page_recall@10": value["page_recall@10"],
                "ndcg@10": value["ndcg@10"],
                "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"],
            }
            for name, value in metrics.items()
        },
        "oof": {"metrics": oof_metric, "folds": selections},
        "notes": [
            "This is page-stage fusion only; file selection is inherited from the current hierarchy run.",
            "Multilingual-E5 page scores and query vectors are cached; no new encoding is performed.",
            "Weights and RRF constant are fixed before OOF selection.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics quick page semantic hierarchy",
        "",
        "| Method | Page recall@10 | nDCG@10 | File recall@3 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(metrics, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += [
        "",
        f"Candidate union page recall before reranking: **{union_recall:.2%}**.",
        f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.",
        "",
        "| Fold | Selected | Test page recall@10 |",
        "|---:|---|---:|",
    ]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "candidate_union_recall": round(union_recall * 100, 2),
        "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2),
        "best_full_set": max(candidates, key=lambda name: metrics[name]["page_recall@10"]),
        "selections": selections,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
