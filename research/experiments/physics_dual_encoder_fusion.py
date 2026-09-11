"""Fuse two fixed multilingual page encoders inside structural candidates."""

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
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
E5_SCORES = RESULT_ROOT / "physics_multilingual_e5_baseline/page_dense_scores.npy"
MINILM_SCORES = RESULT_ROOT / "physics_paraphrase_multilingual_minilm/page_dense_scores.npy"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_dual_encoder_fusion"


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min()) if len(values) else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(shifted.max()) if len(shifted) else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _rrf_score(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {str(item["chunk_id"]): 1.0 / (20 + rank) for rank, item in enumerate(items[:100], 1)}


def _run(
    e5: np.ndarray,
    minilm: np.ndarray,
    candidate_parents: Sequence[Mapping[str, list[dict[str, Any]]]],
    prior_parents: Sequence[Mapping[str, list[dict[str, Any]]]],
    pages: list[str],
    qids: list[str],
    minilm_weight: float,
    e5_weight: float,
    prior_kind: str,
) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output: dict[str, list[dict[str, Any]]] = {}
    prior_weight = 1.0 - minilm_weight - e5_weight
    for query_index, qid in enumerate(qids):
        candidates = [parent[qid][:100] for parent in candidate_parents]
        candidate_pages = sorted({str(item["chunk_id"]) for items in candidates for item in items})
        prior_maps = [_rrf_score(parent[qid][:100]) for parent in prior_parents]
        prior = np.asarray([
            max(rrf.get(page, 0.0) for rrf in prior_maps)
            if prior_kind == "max_rrf"
            else sum(rrf.get(page, 0.0) for rrf in prior_maps)
            for page in candidate_pages
        ], dtype=np.float32)
        e5_values = np.asarray([e5[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        minilm_values = np.asarray([minilm[query_index, position[page]] for page in candidate_pages], dtype=np.float32)
        combined = (
            minilm_weight * _normalise(minilm_values)
            + e5_weight * _normalise(e5_values)
            + prior_weight * _normalise(prior)
        )
        order = np.argsort(-combined, kind="stable")[:100]
        output[qid] = [{"chunk_id": candidate_pages[int(index)], "doc_id": candidate_pages[int(index)], "score": float(combined[int(index)]), "rank": rank} for rank, index in enumerate(order, 1)]
    return output


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--e5-scores", type=Path, default=E5_SCORES)
    parser.add_argument("--minilm-scores", type=Path, default=MINILM_SCORES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    e5 = np.load(args.e5_scores, mmap_mode="r")
    minilm = np.load(args.minilm_scores, mmap_mode="r")
    c014 = _load_run(RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl")
    cross_kf = _load_run(RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl")
    runs = {
        "minilm_c014": _run(e5, minilm, [c014], [c014], pages, qids, 0.30, 0.0, "max_rrf"),
        "dual_c014_m030_e015": _run(e5, minilm, [c014], [c014], pages, qids, 0.30, 0.15, "max_rrf"),
        "dual_union_m030_e015": _run(e5, minilm, [c014, cross_kf], [c014, cross_kf], pages, qids, 0.30, 0.15, "sum_rrf"),
        "dual_union_m025_e020": _run(e5, minilm, [c014, cross_kf], [c014, cross_kf], pages, qids, 0.25, 0.20, "sum_rrf"),
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
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "e5_scores": str(args.e5_scores), "minilm_scores": str(args.minilm_scores), "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["Both encoders are queried in French and page-pooled from fixed 240-word segments.", "Candidate sets and RRF-20 structural priors are fixed; encoder weights are predeclared.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics dual encoder fusion", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
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
