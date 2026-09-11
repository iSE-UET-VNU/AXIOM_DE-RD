"""Fixed stream-budget diversity over cached Physics page runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import _derived_metrics, _load_run, _paired_comparison, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_stream_budget_diversity"


def _quota(streams: Mapping[str, list[dict[str, Any]]], schedule: list[tuple[str, int]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursors = {name: 0 for name, _ in schedule}
    slots = {name: budget for name, budget in schedule}
    while slots and len(output) < 100:
        progressed = False
        for name, _ in schedule:
            if slots.get(name, 0) <= 0:
                continue
            stream = streams[name]
            while cursors[name] < len(stream) and str(stream[cursors[name]]["chunk_id"]) in seen:
                cursors[name] += 1
            if cursors[name] >= len(stream):
                slots[name] = 0
                continue
            page = str(stream[cursors[name]]["chunk_id"])
            cursors[name] += 1
            slots[name] -= 1
            if page not in seen:
                seen.add(page)
                output.append({"chunk_id": page, "doc_id": page, "score": 1.0, "rank": len(output) + 1})
            progressed = True
        if not progressed:
            break
    # Fill after the protected first-slot budget using the strongest structural
    # stream; this keeps file discovery defined through 100 pages.
    for name in ("c014", "cross_kf", "weighted", "bm25", "legacy_second", "splade"):
        for row in streams[name][:100]:
            page = str(row["chunk_id"])
            if page not in seen:
                seen.add(page)
                output.append({"chunk_id": page, "doc_id": page, "score": 1.0, "rank": len(output) + 1})
            if len(output) >= 100:
                return output
    return output[:100]


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
        "c014": root / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
        "cross_kf": root / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
        "legacy_second": root / "physics_legacy_second_retrieval/oof_run.jsonl",
    }
    loaded = {name: _load_run(path) for name, path in paths.items() if path.is_file()}
    schedules = {
        "c014-control": [("c014", 10)],
        "budget-c0146-cross2-bm251-legacy1": [("c014", 6), ("cross_kf", 2), ("bm25", 1), ("legacy_second", 1)],
        "budget-c0145-cross2-weighted2-bm251": [("c014", 5), ("cross_kf", 2), ("weighted", 2), ("bm25", 1)],
        "budget-c0144-cross3-legacy2-bm251": [("c014", 4), ("cross_kf", 3), ("legacy_second", 2), ("bm25", 1)],
    }
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in schedules}
    for qid in qids:
        streams = {name: loaded[name][qid] for name in loaded}
        for name, schedule in schedules.items():
            # A one-pass schedule is a fixed page-slot allocation.  The
            # round-robin quota helper handles duplicate pages correctly.
            runs[name][qid] = _quota(streams, schedule)
    baseline = _derived_metrics(loaded["weighted"], qids, qrels)
    hierarchical = _derived_metrics(loaded["c014"], qids, qrels)
    methods = {}
    for name, run in runs.items():
        metric = _derived_metrics(run, qids, qrels)
        methods[name] = {
            "schedule": schedules[name],
            "retrieval_metrics": metric,
            "comparison_to_weighted": _paired_comparison(baseline, metric),
            "comparison_to_c014": _paired_comparison(hierarchical, metric),
        }
    folds = [qids[i::5] for i in range(5)]
    oof: dict[str, list[dict[str, Any]]] = {}
    fold_rows = []
    all_qids = set(qids)
    for fold, heldout in enumerate(folds):
        train = all_qids - set(heldout)
        winner = max(methods, key=lambda name: (_mean(methods[name]["retrieval_metrics"], train), name))
        for qid in heldout:
            oof[qid] = runs[winner][qid]
        fold_rows.append({"fold": fold, "selected_method": winner, "test_page_recall@10": _mean(methods[winner]["retrieval_metrics"], set(heldout))})
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
            "Only rank positions from pre-existing runs are used; no qrels or modality labels enter the slot allocation.",
            "The protected first 10 slots are fixed schedules; remaining slots are filled in structural-stream order.",
            "V-SPLADE streams use cached English query vectors against French qrels.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics stream-budget diversity", "", "| Method | Schedule | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs c014 pp |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, item in methods.items():
        metric = item["retrieval_metrics"]
        lines.append(f"| {name} | {item['schedule']} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} | {item['comparison_to_c014']['page_recall_delta_pp']:+.2f} |")
    lines += ["", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", f"OOF file recall@3: **{oof_metric['file_metrics_by_k']['3']['file_recall']:.2%}**.", f"OOF selected methods: `{report['cv']['selected_arm_counts']}`."]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()}, "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "oof_file_recall@3": round(oof_metric["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "selected_arm_counts": report["cv"]["selected_arm_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
