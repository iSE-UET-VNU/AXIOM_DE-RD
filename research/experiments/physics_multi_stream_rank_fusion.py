"""Multi-stream reciprocal-rank fusion over existing Physics runs.

The candidate-ceiling audit shows that several already cached retrievers carry
complementary gold pages.  This experiment is the ranking counterpart of a
multi-index lakehouse planner: combine existing candidate streams by rank,
then keep a fixed top-100 page output.  It does not inspect qrels while
scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import _derived_metrics, _load_run, _paired_comparison, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_multi_stream_rank_fusion"


def _rrf(streams: Sequence[Sequence[dict[str, Any]]], *, constant: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for stream in streams:
        for rank, row in enumerate(stream[:100], 1):
            page = str(row["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": round(scores[page], 8), "rank": rank} for rank, page in enumerate(ordered, 1)]


def _mean_recall(metrics: Mapping[str, Any], qids: set[str]) -> float:
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
    loaded = {name: _load_run(path) for name, path in paths.items() if path.is_file()}
    if any(set(run) != set(qids) for run in loaded.values()):
        raise RuntimeError("All input runs must contain the same 302 Physics qids")
    specs = {
        "rrf-lexical-visual-c20": (20, ("bm25", "splade", "weighted")),
        "rrf-structural-c20": (20, ("weighted", "hierarchical", "c014", "cross_kf", "legacy_second")),
        "rrf-all-c20": (20, tuple(loaded)),
        "rrf-all-c60": (60, tuple(loaded)),
    }
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    methods: dict[str, dict[str, Any]] = {}
    for name, (constant, stream_names) in specs.items():
        runs[name] = {qid: _rrf([loaded[stream][qid] for stream in stream_names], constant=constant) for qid in qids}
        metric = _derived_metrics(runs[name], qids, qrels)
        methods[name] = {"constant": constant, "streams": stream_names, "retrieval_metrics": metric}
    baseline = _derived_metrics(loaded["weighted"], qids, qrels)
    hierarchical = _derived_metrics(loaded["hierarchical"], qids, qrels)
    for item in methods.values():
        item["comparison_to_weighted"] = _paired_comparison(baseline, item["retrieval_metrics"])
        item["comparison_to_hierarchical"] = _paired_comparison(hierarchical, item["retrieval_metrics"])

    # Predeclared stream sets are selected OOF by training-fold page recall.
    folds = [qids[index::5] for index in range(5)]
    oof: dict[str, list[dict[str, Any]]] = {}
    fold_rows = []
    all_qids = set(qids)
    for fold, heldout in enumerate(folds):
        train = all_qids - set(heldout)
        winner = max(methods, key=lambda name: (_mean_recall(methods[name]["retrieval_metrics"], train), name))
        for qid in heldout:
            oof[qid] = runs[winner][qid]
        fold_rows.append({"fold": fold, "selected_method": winner, "test_page_recall@10": _mean_recall(methods[winner]["retrieval_metrics"], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "methods": methods,
        "cv": {"folds": fold_rows, "retrieval_metrics": oof_metric, "selected_arm_counts": {name: sum(row["selected_method"] == name for row in fold_rows) for name in methods}},
        "inputs": {name: str(path) for name, path in paths.items() if path.is_file()},
        "notes": [
            "All RRF streams are cache-only top-100 runs; no qrels are used in scoring.",
            "Stream sets and constants are predeclared; OOF selection is reported separately from full-set screening.",
            "V-SPLADE streams use cached English query vectors against French qrels.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics multi-stream rank fusion", "", "| Method | Streams | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs hierarchical pp |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, item in methods.items():
        metric = item["retrieval_metrics"]
        lines.append(f"| {name} | {', '.join(item['streams'])} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} | {item['comparison_to_hierarchical']['page_recall_delta_pp']:+.2f} |")
    lines += ["", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", f"OOF file recall@3: **{oof_metric['file_metrics_by_k']['3']['file_recall']:.2%}**.", f"OOF selected methods: `{report['cv']['selected_arm_counts']}`."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()}, "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "oof_file_recall@3": round(oof_metric["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "selected_arm_counts": report["cv"]["selected_arm_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
