"""Ablate the two mechanisms in the Physics hierarchical retriever.

The experiment holds page fusion and file scoring fixed and separates:

1. hard file restriction (Kf=3 versus all files); and
2. parent file score added to the page score (parent_weight=0 versus 0.25).

It uses cached KDL/PDF-inspector text and V-SPLADE vectors only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _load_visual_scores,
    _paired_comparison,
    _write_run,
)
from research.data_discovery.hierarchical import CascadeConfig  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_ablation"


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["qid"])] = list(row["chunks"])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in metadata]
    if len(qids) != 302 or len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected 302 queries and 1,674 unique pages, got {len(qids)} and {len(page_ids)}")

    corpus, page_index, file_indexes, _fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)
    retriever = HierarchicalRetriever(corpus, page_index=page_index, file_indexes=file_indexes, fine_indexes={}, visual_scores=visual_scores)

    # The four cells use the c014 page/filer score settings.  Kf=42 means all
    # Physics files are eligible; it removes the hard restriction while
    # retaining the same file score calculation for the parent-only cell.
    cells = [
        ("neither_restriction_nor_parent", 42, 0.00),
        ("restriction_only_kf3", 3, 0.00),
        ("parent_only_all_files", 42, 0.25),
        ("both_kf3_and_parent", 3, 0.25),
    ]
    methods: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, k_files, parent_weight in cells:
        config = CascadeConfig(
            name=name,
            file_pool="sum_top2",
            k_files=k_files,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.60,
            parent_weight=parent_weight,
            file_direct_weight=0.50,
            file_pool_source="page_base",
        )
        run: dict[str, list[dict[str, Any]]] = {}
        for question in questions:
            page_run, _trace = retriever.retrieve(question.qid, question.query, config)
            run[question.qid] = page_run
        runs[name] = run
        methods[name] = {"config": config.__dict__, "retrieval_metrics": _derived_metrics(run, qids, qrels)}

    baseline_run = _load_run(args.baseline_run)
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)
    methods["weighted fusion alpha=0.70 (reference)"] = {"retrieval_metrics": baseline_metrics}
    for method in methods.values():
        method["comparison_to_weighted"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(runs_dir / f"{name}.jsonl", run, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "fixed_controls": {
            "page_bm25_weight": 0.60,
            "file_pool": "sum_top2",
            "file_direct_weight": 0.50,
            "file_pool_source": "page_base",
        },
        "methods": methods,
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta, "index_counts": index_counts},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "The parent-only cell still computes the same file scores but selects all 42 files, so it measures parent-score ranking without hard file filtering.",
            "The restriction-only cell selects Kf=3 files but does not add parent file score to page ranking.",
            "The neither cell is page_base at BM25 weight 0.60 and is the control for this ablation; weighted alpha=0.70 remains the external reference.",
            "No qrel or modality label is used as a scoring feature.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics hierarchical ablation",
        "",
        "| Cell | Kf | Parent weight | nDCG@10 | Page recall@10 | Delta vs weighted pp | File recall@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _k_files, _parent in cells:
        method = methods[name]
        metric = method["retrieval_metrics"]
        lines.append(
            f"| {name} | {method['config']['k_files']} | {method['config']['parent_weight']:.2f} | "
            f"{metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | "
            f"{method['comparison_to_weighted']['page_recall_delta_pp']:+.2f} | "
            f"{metric['file_metrics_by_k']['3']['file_recall']:.2%} |"
        )
    lines += [
        "",
        "Interpretation: restriction contribution = restriction-only minus neither; parent contribution = parent-only minus neither; interaction = both minus the sum of the two single-factor gains.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "cells": {
            name: {
                "page_recall@10": methods[name]["retrieval_metrics"]["page_recall@10"],
                "delta_vs_weighted_pp": methods[name]["comparison_to_weighted"]["page_recall_delta_pp"],
                "file_recall@3": methods[name]["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"],
            }
            for name, _k, _p in cells
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
