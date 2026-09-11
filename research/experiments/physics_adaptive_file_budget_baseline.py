"""Confidence-aware file pruning for Physics hierarchical retrieval.

Lakehouse planners prune partitions/files when metadata is selective and keep
more candidates when the metadata boundary is ambiguous.  This experiment
applies that idea to the fixed c014 page/file score:

* keep 3 files when the normalized gap between file 3 and file 4 is at least
  5% of the top file score;
* widen to 4 files otherwise.

The 5% rule is a predeclared engineering threshold, not learned from Physics
qrels.  The fixed Kf=3 and Kf=4 controls are reported beside the adaptive arm.
All page scores are exact cache-derived page BM25 + V-SPLADE scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import HierarchyCorpus, aggregate_file_scores, build_bm25, normalise_scores, sort_scores  # noqa: E402
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _load_visual_scores,
    _paired_comparison,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
DEFAULT_HIERARCHICAL_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_adaptive_file_budget_baseline"


def _retrieve(
    corpus: HierarchyCorpus,
    page_index: Any,
    file_index: Any,
    visual_scores: Mapping[str, float],
    query: str,
    *,
    file_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page_ids = corpus.page_order
    page_bm25 = normalise_scores(_index_scores(page_index, query, top_k=len(page_ids)))
    visual = normalise_scores(visual_scores)
    page_base = {page_id: 0.60 * page_bm25.get(page_id, 0.0) + 0.40 * visual.get(page_id, 0.0) for page_id in page_ids}
    page_to_file = {page_id: _file_id(page_id) for page_id in page_ids}
    file_pool = normalise_scores(aggregate_file_scores(page_base, page_to_file, corpus.file_order, "sum_top2"))
    file_direct = normalise_scores(_index_scores(file_index, query, top_k=len(corpus.file_order)))
    file_scores = {file_id: 0.50 * file_direct.get(file_id, 0.0) + 0.50 * file_pool.get(file_id, 0.0) for file_id in corpus.file_order}
    file_ranked = sort_scores(file_scores)
    selected_files = {file_id for file_id, _ in file_ranked[:file_budget]}
    parent = normalise_scores(file_scores)
    final_scores = {
        page_id: 0.75 * page_base[page_id] + 0.25 * parent.get(page_to_file[page_id], 0.0)
        for page_id in page_ids
        if page_to_file[page_id] in selected_files
    }
    ranked = sort_scores(final_scores)[:100]
    return [
        {"chunk_id": page_id, "doc_id": page_id, "text": corpus.pages[page_id].text, "score": round(float(score), 8), "rank": rank}
        for rank, (page_id, score) in enumerate(ranked, 1)
    ], {
        "file_ranked": [file_id for file_id, _ in file_ranked],
        "selected_files": sorted(selected_files),
        "selected_file_scores": {file_id: round(float(score), 8) for file_id, score in file_ranked[:file_budget]},
        "file_gap_3_4": round(float((file_ranked[2][1] - file_ranked[3][1]) if len(file_ranked) > 3 else 0.0), 8),
        "top_file_score": round(float(file_ranked[0][1] if file_ranked else 0.0), 8),
    }


def _metrics_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return sum(float(row["page_recall@10"]) for row in rows) / len(rows) if rows else 0.0


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

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    page_metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in page_metadata]
    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="physics", page_ids=page_ids)
    page_index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in page_ids)
    file_index = build_bm25(corpus.file_texts("all_text").items())
    visual_by_qid, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {"fixed-kf3": {}, "fixed-kf4": {}, "adaptive-gap05": {}}
    traces: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in runs}
    for question in questions:
        page3, trace3 = _retrieve(corpus, page_index, file_index, visual_by_qid[question.qid], question.query, file_budget=3)
        page4, trace4 = _retrieve(corpus, page_index, file_index, visual_by_qid[question.qid], question.query, file_budget=4)
        # A 5% normalized score gap is the fixed planner confidence rule.
        top = trace3["top_file_score"]
        gap = trace3["file_gap_3_4"]
        budget = 3 if top <= 0.0 or gap / top >= 0.05 else 4
        adaptive_page, adaptive_trace = (page3, trace3) if budget == 3 else (page4, trace4)
        runs["fixed-kf3"][question.qid], traces["fixed-kf3"][question.qid] = page3, trace3
        runs["fixed-kf4"][question.qid], traces["fixed-kf4"][question.qid] = page4, trace4
        runs["adaptive-gap05"][question.qid], traces["adaptive-gap05"][question.qid] = adaptive_page, {**adaptive_trace, "adaptive_budget": budget}

    baseline = _derived_metrics(_load_run(args.baseline_run), qids, qrels)
    hierarchical = _derived_metrics(_load_run(args.hierarchical_run), qids, qrels)
    methods: dict[str, dict[str, Any]] = {}
    for name, run in runs.items():
        methods[name] = {
            "retrieval_metrics": _derived_metrics(run, qids, qrels),
            "comparison_to_baseline": _paired_comparison(baseline, _derived_metrics(run, qids, qrels)),
            "comparison_to_hierarchical": _paired_comparison(hierarchical, _derived_metrics(run, qids, qrels)),
        }
    adaptive_budgets = [trace["adaptive_budget"] for trace in traces["adaptive-gap05"].values()]
    adaptive_metrics = methods["adaptive-gap05"]["retrieval_metrics"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "adaptive-gap05.jsonl", runs["adaptive-gap05"], qids, queries={question.qid: question.query for question in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.file_order),
        "threshold": 0.05,
        "methods": methods,
        "adaptive_budget_distribution": {str(budget): adaptive_budgets.count(budget) for budget in sorted(set(adaptive_budgets))},
        "references": {"cached_page_bm25": baseline, "current_hierarchical": hierarchical},
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "Adaptive selection uses only the file-score gap and does not use qrels.",
            "fixed-kf3 and fixed-kf4 are controls; the adaptive arm widens only when the file boundary is ambiguous.",
            "V-SPLADE query vectors are cached English translations against French qrels.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics adaptive file-budget baseline",
        "",
        "| Method | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs hierarchical pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        metric = item["retrieval_metrics"]
        lines.append(
            f"| {name} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | "
            f"{metric['file_metrics_by_k']['3']['file_recall']:.2%} | {item['comparison_to_hierarchical']['page_recall_delta_pp']:+.2f} |"
        )
    lines += [
        "",
        f"Adaptive file-budget distribution: `{report['adaptive_budget_distribution']}`.",
        "Threshold 0.05 was fixed before evaluation; no qrel or modality label enters the decision.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()},
        "adaptive_budget_distribution": report["adaptive_budget_distribution"],
        "adaptive_page_recall@10": round(adaptive_metrics["page_recall@10"] * 100, 2),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
