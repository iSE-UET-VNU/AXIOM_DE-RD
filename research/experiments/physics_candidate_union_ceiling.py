"""Measure the page-level ceiling of existing cache-only candidate sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import _derived_metrics, _load_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = ROOT / "data/benchmark/vidore_v3/results/physics_candidate_union_ceiling"


def _union_pages(runs: list[dict[str, list[dict[str, Any]]]], qids: list[str], depth: int) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for qid in qids:
        pages: set[str] = set()
        for run in runs:
            for row in run[qid][:depth]:
                pages.add(str(row["chunk_id"]))
        output[qid] = pages
    return output


def _coverage(union: dict[str, set[str]], qids: list[str], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    recalls = []
    hits = []
    for qid in qids:
        gold = set(qrels.get(qid, {}))
        found = union.get(qid, set()) & gold
        recalls.append(len(found) / len(gold) if gold else 0.0)
        hits.append(float(bool(found)))
    return {
        "candidate_page_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "candidate_page_hit": sum(hits) / len(hits) if hits else 0.0,
    }


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
    specs = {
        "bm25": root / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
        "weighted": root / "physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl",
        "splade": root / "physics_vsplade_bm25_fusion/vsplade_french_bm25-french_vs-english.jsonl",
        "hierarchical": root / "physics_hierarchical_retrieval/oof_run.jsonl",
        "c014": root / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
        "cross_kf": root / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
        "legacy_second": root / "physics_legacy_second_retrieval/legacy-second-page100-max-gamma0_25.jsonl",
        "adaptive_ltr": root / "physics_adaptive_evidence_fusion/linear_ltr_oof.jsonl",
    }
    loaded = {name: _load_run(path) for name, path in specs.items() if path.is_file()}
    methods = {}
    for name, run in loaded.items():
        methods[name] = _derived_metrics(run, qids, qrels)
    unions = {}
    for name, names in {
        "all": list(loaded),
        "lexical_visual": [n for n in ("bm25", "splade", "weighted") if n in loaded],
        "hierarchical_family": [n for n in ("hierarchical", "c014", "cross_kf", "legacy_second") if n in loaded],
        "strong_structural": [n for n in ("c014", "cross_kf", "legacy_second", "adaptive_ltr") if n in loaded],
    }.items():
        for depth in (10, 100):
            union_pages = _union_pages([loaded[n] for n in names], qids, depth)
            unions[f"{name}@{depth}"] = _coverage(union_pages, qids, qrels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"methods": methods, "unions": unions, "inputs": {name: str(path) for name, path in specs.items() if path.is_file()}}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics candidate union ceiling", "", "| Set | Depth | Candidate page recall | Candidate page hit |", "|---|---:|---:|---:|"]
    for name, metric in methods.items():
        lines.append(f"| {name} | 10 | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} |")
    for name, metric in unions.items():
        base, depth = name.rsplit("@", 1)
        lines.append(f"| union:{base} | {depth} | {metric['candidate_page_recall']:.2%} | {metric['candidate_page_hit']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"methods": {name: round(metric["page_recall@10"] * 100, 2) for name, metric in methods.items()}, "unions": {name: round(metric["candidate_page_recall"] * 100, 2) for name, metric in unions.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
