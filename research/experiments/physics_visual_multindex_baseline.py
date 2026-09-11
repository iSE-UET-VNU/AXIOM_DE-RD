"""Controlled fusion of cached French visual indexes with hierarchy runs."""

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

from physics_hierarchical_retrieval import _derived_metrics, _paired_comparison, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402

RESULT_ROOT = ROOT / "data/benchmark/vidore_v3/results"
VIS_ROOT = ROOT / "data/output/visual_retrieval"
DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_visual_multindex_baseline"

PATHS = {
    "jina_clip": VIS_ROOT / "vidore_v3_physics_jina_clip_v2/retrieval_french.jsonl",
    "colsmol": VIS_ROOT / "vidore_v3_physics_colsmol/retrieval_french.jsonl",
    "siglip2": VIS_ROOT / "vidore_v3_physics_siglip2/retrieval_french.jsonl",
    "c014": RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
    "cross_kf": RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
}


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _rrf(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], constant: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for items in (left, right):
        for rank, item in enumerate(items[:100], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


def _score_fusion(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    left_scores = {str(item["chunk_id"]): max(0.0, float(item.get("score", 0.0))) for item in left[:100]}
    right_scores = {str(item["chunk_id"]): max(0.0, float(item.get("score", 0.0))) for item in right[:100]}
    left_max = max(left_scores.values(), default=0.0)
    right_max = max(right_scores.values(), default=0.0)
    pages = set(left_scores) | set(right_scores)
    scores = {
        page: weight * (left_scores.get(page, 0.0) / left_max if left_max else 0.0)
        + (1.0 - weight) * (right_scores.get(page, 0.0) / right_max if right_max else 0.0)
        for page in pages
    }
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
    qrels = benchmark.qrels()
    loaded = {name: _load(path) for name, path in PATHS.items()}
    if any(set(run) != set(qids) for run in loaded.values()):
        raise RuntimeError("Visual and hierarchy runs must contain the same 302 qids")

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {name: {qid: loaded[name][qid] for qid in qids} for name in PATHS}
    for visual in ("jina_clip", "colsmol", "siglip2"):
        for structural in ("c014", "cross_kf"):
            for constant in (20, 60):
                name = f"rrf_{visual}_{structural}_k{constant}"
                runs[name] = {qid: _rrf(loaded[visual][qid], loaded[structural][qid], constant) for qid in qids}
            name = f"score_{visual}_{structural}_w0.70"
            runs[name] = {qid: _score_fusion(loaded[visual][qid], loaded[structural][qid], 0.70) for qid in qids}

    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections: list[dict[str, Any]] = []
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
        "dataset": "vidore_v3/physics", "queries": len(qids), "inputs": {name: str(path) for name, path in PATHS.items()},
        "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()},
        "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections},
        "notes": ["Jina-CLIP-V2, ColSmol and SigLIP2 are cached French-query visual retrieval runs.", "Only a fixed small set of RRF/score-fusion controls is screened; OOF selection uses training folds.", "All results use full page IDs and canonical derived metrics."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics visual multi-index baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        item = metrics[name]
        lines.append(f"| {name} | {item['page_recall@10']:.2%} | {item['ndcg@10']:.2f} | {item['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set exploratory method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
