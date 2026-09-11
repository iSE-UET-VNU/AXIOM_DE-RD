"""Section-locality baseline for Physics light retrieval.

This is a small companion to ``physics_data_skipping_baselines.py``.  It
replaces fixed-size row groups with contiguous section zones derived from the
already parsed heading metadata.  The query still verifies pages with the
existing page-level BM25 + V-SPLADE score.  No qrels, modality labels or new
model inference are used for scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from collections import defaultdict
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import HierarchyCorpus, build_bm25  # noqa: E402
from research.experiments.physics_data_skipping_baselines import (  # noqa: E402
    DataSkippingRetriever,
    Region,
    SkipConfig,
    _load_qrels_and_questions,
    _metric_for_qids,
    _oof_select,
    _stage_metrics,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _build_corpus_and_indexes,
    _derived_metrics,
    _load_run,
    _load_visual_scores,
    _paired_comparison,
    _write_run,
)


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
DEFAULT_HIERARCHICAL_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_section_zone_baseline"


def _page_section_labels(corpus: HierarchyCorpus) -> dict[str, str]:
    """Use the last active heading on each page as its section label."""
    labels: dict[str, str] = {}
    for page_id in corpus.page_order:
        blocks = sorted(
            (node for node in corpus.blocks.values() if node.page_id == page_id),
            key=lambda node: (node.block_index if node.block_index is not None else -1, node.node_id),
        )
        label = ""
        for block in blocks:
            candidate = str(block.metadata.get("section") or "").strip()
            if candidate:
                label = candidate
        labels[page_id] = label or "__preamble__"
    return labels


def _section_regions(corpus: HierarchyCorpus) -> list[Region]:
    """Make contiguous zones; repeated headings start a new zone."""
    labels = _page_section_labels(corpus)
    by_file: dict[str, list[str]] = defaultdict(list)
    for page_id in corpus.page_order:
        by_file[page_id.split("#page=", 1)[0]].append(page_id)

    output: list[Region] = []
    for file_id in corpus.file_order:
        pages = sorted(by_file.get(file_id, []), key=lambda page: int(page.rsplit("#page=", 1)[-1]))
        current_label = None
        current_pages: list[str] = []
        zone_index = 0
        for page_id in pages + [""]:
            label = labels.get(page_id) if page_id else None
            if current_pages and label != current_label:
                text = "\n".join(corpus.pages[page].text for page in current_pages).strip()
                output.append(Region(f"{file_id}#section={zone_index}", file_id, tuple(current_pages), text))
                zone_index += 1
                current_pages = []
            if not page_id:
                break
            current_label = label
            current_pages.append(page_id)
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics section-zone data-skipping baseline",
        "",
        "Offline section-locality experiment over the cached Physics corpus.",
        "",
        "| Arm | Zones | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Page candidate recall | Mean page skip | Δ vs current hierarchical pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["methods"].items():
        metric = item["retrieval_metrics"]
        stage = item["stage_metrics"]
        delta = item["comparison_to_hierarchical"]["page_recall_delta_pp"]
        lines.append(
            f"| {name} | {report['zones']} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | "
            f"{metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} | "
            f"{stage['page_candidate_recall']:.2%} | {stage['mean_page_skip_fraction']:.2%} | {delta:+.2f} |"
        )
    cv = report["cv"]
    lines += [
        "",
        f"- OOF page recall@10: **{cv['retrieval_metrics']['page_recall@10']:.2%}**.",
        f"- OOF file recall@3: **{cv['retrieval_metrics']['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"- OOF selected arms: `{cv['selected_arm_counts']}`.",
        "",
        "Section zones are structural metadata priors. Hard section pruning is invalidated when page candidate recall falls, even if it reduces work.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()
    questions, qids, qrels = _load_qrels_and_questions(args.benchmark_root)
    page_metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in page_metadata]
    corpus, page_index, file_indexes, _fine, index_counts = _build_corpus_and_indexes(args.parsed_run, page_ids, fine_units=[])
    zones = _section_regions(corpus)
    zone_index = build_bm25((zone.region_id, zone.text) for zone in zones)
    visual_scores, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)
    retriever = DataSkippingRetriever(
        corpus,
        page_index=page_index,
        file_index=file_indexes["all_text"],
        region_indexes={0: zone_index},
        regions_by_size={0: zones},
        visual_scores=visual_scores,
    )
    configs = [
        SkipConfig("section-soft-kf3", 0, 3, None),
        SkipConfig("section-hard16-kf3", 0, 3, 16),
        SkipConfig("section-soft-all-files", 0, 42, None),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    traces_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for config in configs:
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        trace_by_qid: dict[str, dict[str, Any]] = {}
        for question in questions:
            run, trace = retriever.retrieve(question.qid, question.query, config)
            run_by_qid[question.qid] = run
            trace_by_qid[question.qid] = trace
        runs[config.name] = run_by_qid
        traces_by_name[config.name] = trace_by_qid
        methods[config.name] = {
            "config": asdict(config),
            "retrieval_metrics": _derived_metrics(run_by_qid, qids, qrels),
            "stage_metrics": _stage_metrics(trace_by_qid, qids, qrels),
        }
    baseline_metrics = _derived_metrics(_load_run(args.baseline_run), qids, qrels)
    hierarchical_metrics = _derived_metrics(_load_run(args.hierarchical_run), qids, qrels)
    for item in methods.values():
        item["comparison_to_baseline"] = _paired_comparison(baseline_metrics, item["retrieval_metrics"])
        item["comparison_to_hierarchical"] = _paired_comparison(hierarchical_metrics, item["retrieval_metrics"])
    oof_run, cv_meta = _oof_select(qids, {name: item["retrieval_metrics"] for name, item in methods.items()}, runs, qrels)
    cv_metrics = _derived_metrics(oof_run, qids, qrels)
    cv_traces = {}
    for fold in cv_meta["folds"]:
        for qid in fold["heldout_qids"]:
            cv_traces[qid] = traces_by_name[fold["selected_arm"]][qid]
    cv = {
        **cv_meta,
        "retrieval_metrics": cv_metrics,
        "stage_metrics": _stage_metrics(cv_traces, qids, qrels),
        "comparison_to_baseline": _paired_comparison(baseline_metrics, cv_metrics),
        "comparison_to_hierarchical": _paired_comparison(hierarchical_metrics, cv_metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof_run, qids, queries={question.qid: question.query for question in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "zones": len(zones),
        "baseline_cached_page_bm25": {"path": str(args.baseline_run), "retrieval_metrics": baseline_metrics},
        "current_hierarchical": {"path": str(args.hierarchical_run), "retrieval_metrics": hierarchical_metrics},
        "methods": methods,
        "cv": cv,
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta, "index_counts": index_counts},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "Section zones use only parsed heading context and contiguous page order.",
            "V-SPLADE query vectors are English translations evaluated against French qrels.",
            "No full stage traces are written.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "zones": len(zones),
        "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()},
        "oof_page_recall@10": round(cv_metrics["page_recall@10"] * 100, 2),
        "oof_file_recall@3": round(cv_metrics["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
        "selected_arm_counts": cv["selected_arm_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
