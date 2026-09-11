"""E5 page verification over a fixed union of independent structural runs."""

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
DEFAULT_OUTPUT = RESULT_ROOT / "physics_e5_broad_structural_union"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min()) if len(values) else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(shifted.max()) if len(shifted) else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _rrf_score(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {str(item["chunk_id"]): 1.0 / (20 + rank) for rank, item in enumerate(items[:100], 1)}


def _union_run(
    dense: np.ndarray,
    parents: Sequence[Mapping[str, list[dict[str, Any]]]],
    pages: list[str],
    qids: list[str],
    dense_weight: float,
    prior_kind: str,
) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        per_parent = [parent[qid][:100] for parent in parents]
        candidate_pages = sorted({str(item["chunk_id"]) for items in per_parent for item in items})
        rrf_maps = [_rrf_score(items) for items in per_parent]
        parent_values = np.asarray(
            [
                (
                    max(rrf.get(page, 0.0) for rrf in rrf_maps)
                    if prior_kind == "max_rrf"
                    else sum(rrf.get(page, 0.0) for rrf in rrf_maps)
                )
                for page in candidate_pages
            ],
            dtype=np.float32,
        )
        dense_values = np.asarray([dense[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        combined = dense_weight * _normalise(dense_values) + (1.0 - dense_weight) * _normalise(parent_values)
        order = np.argsort(-combined, kind="stable")[:100]
        output[qid] = [
            {
                "chunk_id": candidate_pages[int(index)],
                "doc_id": candidate_pages[int(index)],
                "score": float(combined[int(index)]),
                "rank": rank,
            }
            for rank, index in enumerate(order, 1)
        ]
    return output


def _union_run_with_bm25(
    dense: np.ndarray,
    candidate_parents: Sequence[Mapping[str, list[dict[str, Any]]]],
    prior_parents: Sequence[Mapping[str, list[dict[str, Any]]]],
    bm25_parent: Mapping[str, list[dict[str, Any]]],
    pages: list[str],
    qids: list[str],
    dense_weight: float,
    bm25_weight: float,
    prior_kind: str,
) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        candidate_items = [parent[qid][:100] for parent in candidate_parents]
        candidate_pages = sorted({str(item["chunk_id"]) for items in candidate_items for item in items})
        prior_maps = [_rrf_score(parent[qid][:100]) for parent in prior_parents]
        prior_values = np.asarray(
            [
                (
                    max(rrf.get(page, 0.0) for rrf in prior_maps)
                    if prior_kind == "max_rrf"
                    else sum(rrf.get(page, 0.0) for rrf in prior_maps)
                )
                for page in candidate_pages
            ],
            dtype=np.float32,
        )
        dense_values = np.asarray([dense[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        bm25_map = {str(item["chunk_id"]): max(0.0, float(item.get("score", 0.0))) for item in bm25_parent[qid][:100]}
        bm25_values = np.asarray([bm25_map.get(page, 0.0) for page in candidate_pages], dtype=np.float32)
        prior_weight = 1.0 - dense_weight - bm25_weight
        combined = (
            dense_weight * _normalise(dense_values)
            + bm25_weight * _normalise(bm25_values)
            + prior_weight * _normalise(prior_values)
        )
        order = np.argsort(-combined, kind="stable")[:100]
        output[qid] = [
            {
                "chunk_id": candidate_pages[int(index)],
                "doc_id": candidate_pages[int(index)],
                "score": float(combined[int(index)]),
                "rank": rank,
            }
            for rank, index in enumerate(order, 1)
        ]
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
    if dense.shape != (302, 1674):
        raise RuntimeError(f"Unexpected E5 dense score shape: {dense.shape}")

    parent_paths = {
        "bm25": RESULT_ROOT / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
        "weighted": RESULT_ROOT / "physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl",
        "splade": RESULT_ROOT / "physics_vsplade_bm25_fusion/vsplade_french_bm25-french_vs-english.jsonl",
        "c014": RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
        "cross_kf": RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
        "cascade_legacy": RESULT_ROOT / "physics_cascade_kf_legacy_search/runs/oof_selected.jsonl",
        "grid": RESULT_ROOT / "physics_cascade_grid_search/runs/oof_selected.jsonl",
        "hierarchical": RESULT_ROOT / "physics_hierarchical_retrieval/runs/cascade-all_text-max-kf3-fusion.jsonl",
        "adaptive_ltr": RESULT_ROOT / "physics_adaptive_evidence_fusion/runs/linear_ltr_oof.jsonl",
    }
    parents = {name: _load_run(path) for name, path in parent_paths.items()}
    e5_parent: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        order = np.argsort(-dense[query_index], kind="stable")[:100]
        e5_parent[qid] = [
            {"chunk_id": pages[int(index)], "doc_id": pages[int(index)], "score": float(dense[query_index, int(index)]), "rank": rank}
            for rank, index in enumerate(order, 1)
        ]
    parents["e5_dense"] = e5_parent
    structural_names = ("c014", "cross_kf", "cascade_legacy", "grid", "hierarchical")
    light_names = ("bm25", "weighted", "splade", "adaptive_ltr") + structural_names
    parent_sets = {
        "two": ("c014", "cross_kf"),
        "three": ("c014", "cross_kf", "cascade_legacy"),
        "all_structural": structural_names,
        "all_light": light_names + ("e5_dense",),
    }

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for parent_name, parent_names in parent_sets.items():
        chosen = [parents[name] for name in parent_names]
        for prior_kind in ("max_rrf", "sum_rrf"):
            for weight in (0.30, 0.50, 0.70, 0.85):
                name = f"e5_{parent_name}_{prior_kind}_w{weight:.2f}"
                runs[name] = _union_run(dense, chosen, pages, qids, weight, prior_kind)

    structural_parents = [parents[name] for name in structural_names]
    bm25_candidate_parents = structural_parents + [parents["bm25"]]
    for candidate_name, candidate_parents in (
        ("structural", structural_parents),
        ("structural_plus_bm25", bm25_candidate_parents),
    ):
        for prior_kind in ("max_rrf", "sum_rrf"):
            for dense_weight, bm25_weight in ((0.70, 0.15), (0.80, 0.10), (0.85, 0.10)):
                name = f"mix_{candidate_name}_{prior_kind}_e{dense_weight:.2f}_b{bm25_weight:.2f}"
                runs[name] = _union_run_with_bm25(
                    dense,
                    candidate_parents,
                    structural_parents,
                    parents["bm25"],
                    pages,
                    qids,
                    dense_weight,
                    bm25_weight,
                    prior_kind,
                )

    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append(
            {
                "fold": fold,
                "selected": winner,
                "train_page_recall@10": _metric_for_qids(metrics[winner], train),
                "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout)),
            }
        )
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "e5_scores": str(args.e5_dir / "page_dense_scores.npy"),
        "parents": {name: str(path) for name, path in parent_paths.items()},
        "methods": {
            name: {
                "page_recall@10": value["page_recall@10"],
                "ndcg@10": value["ndcg@10"],
                "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"],
            }
            for name, value in metrics.items()
        },
        "best_full_set": best_full,
        "oof": {"metrics": oof_metric, "folds": selections},
        "notes": [
            "Candidate set is a per-query union of fixed top-100 pages from pre-existing structural runs.",
            "Structural prior is fixed RRF-20 max or sum; E5 is the page verifier with predeclared dense weights.",
            "All results use full page IDs and canonical derived metrics.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics E5 broad structural union",
        "",
        "| Method | Page recall@10 | nDCG@10 | File recall@3 |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(
            f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |"
        )
    lines += [
        "",
        f"Best full-set method: `{best_full}`.",
        f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.",
        "",
        "| Fold | Selected | Test page recall@10 |",
        "|---:|---|---:|",
    ]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "best_full_set": best_full,
                "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2),
                "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2),
                "selections": selections,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
