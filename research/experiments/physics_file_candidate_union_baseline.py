"""Multi-index file-candidate union for Physics light retrieval.

Lakehouse planners may have several metadata indexes (partition stats, column
stats, bloom filters).  A conservative document analogue is to let a lexical
file index and a visual page-stat index each propose a small file set, take
their union, and then verify/rank pages with the exact page score.

The main arm is a fixed top-2 lexical-file plus top-2 visual-file union.  A
top-3 lexical variant is a sensitivity control, not a parameter sweep.
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
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_file_candidate_union_baseline"


def _retrieve(
    corpus: HierarchyCorpus,
    page_index: Any,
    file_index: Any,
    visual_scores: Mapping[str, float],
    query: str,
    *,
    lexical_k: int,
    visual_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page_ids = corpus.page_order
    page_bm25 = normalise_scores(_index_scores(page_index, query, top_k=len(page_ids)))
    visual = normalise_scores(visual_scores)
    page_base = {page_id: 0.60 * page_bm25.get(page_id, 0.0) + 0.40 * visual.get(page_id, 0.0) for page_id in page_ids}
    page_to_file = {page_id: _file_id(page_id) for page_id in page_ids}
    hybrid_file = normalise_scores(aggregate_file_scores(page_base, page_to_file, corpus.file_order, "sum_top2"))
    file_direct = normalise_scores(_index_scores(file_index, query, top_k=len(corpus.file_order)))
    hybrid_scores = {file_id: 0.50 * file_direct.get(file_id, 0.0) + 0.50 * hybrid_file.get(file_id, 0.0) for file_id in corpus.file_order}
    lexical_ranked = sort_scores({file_id: file_direct.get(file_id, 0.0) for file_id in corpus.file_order})
    visual_file_raw = aggregate_file_scores(visual, page_to_file, corpus.file_order, "max")
    visual_ranked = sort_scores(visual_file_raw)
    hybrid_ranked = sort_scores(hybrid_scores)
    if lexical_k == 3 and visual_k == 0:
        selected_files = {file_id for file_id, _ in hybrid_ranked[:3]}
    else:
        selected_files = {file_id for file_id, _ in lexical_ranked[:lexical_k]}
        selected_files.update(file_id for file_id, _ in visual_ranked[:visual_k])
    parent = normalise_scores(hybrid_scores)
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
        "selected_files": sorted(selected_files),
        "lexical_files": [file_id for file_id, _ in lexical_ranked[:lexical_k]],
        "visual_files": [file_id for file_id, _ in visual_ranked[:visual_k]],
        "hybrid_files": [file_id for file_id, _ in hybrid_ranked[:3]],
    }


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return sum(float(row["page_recall@10"]) for row in rows) / len(rows) if rows else 0.0


def _oof(qids: list[str], metrics: Mapping[str, Mapping[str, Any]], runs: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    folds: list[dict[str, Any]] = []
    all_qids = set(qids)
    for fold in range(5):
        heldout = qids[fold::5]
        train = all_qids - set(heldout)
        winner = max(metrics, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        for qid in heldout:
            output[qid] = runs[winner][qid]
        folds.append({"fold": fold, "selected_arm": winner, "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    return output, folds


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
    metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in metadata]
    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="physics", page_ids=page_ids)
    page_index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in page_ids)
    file_index = build_bm25(corpus.file_texts("all_text").items())
    visual_by_qid, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)
    specs = {
        "hybrid-kf3": (3, 0),
        "file-union-lex2-visual2": (2, 2),
        "file-union-lex3-visual2": (3, 2),
    }
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {name: {} for name in specs}
    traces: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in specs}
    for question in questions:
        for name, (lexical_k, visual_k) in specs.items():
            runs[name][question.qid], traces[name][question.qid] = _retrieve(
                corpus, page_index, file_index, visual_by_qid[question.qid], question.query,
                lexical_k=lexical_k, visual_k=visual_k,
            )
    baseline = _derived_metrics(_load_run(args.baseline_run), qids, qrels)
    hierarchical = _derived_metrics(_load_run(args.hierarchical_run), qids, qrels)
    methods = {}
    for name, run in runs.items():
        metric = _derived_metrics(run, qids, qrels)
        methods[name] = {
            "spec": specs[name],
            "retrieval_metrics": metric,
            "comparison_to_baseline": _paired_comparison(baseline, metric),
            "comparison_to_hierarchical": _paired_comparison(hierarchical, metric),
            "mean_selected_files": sum(len(trace["selected_files"]) for trace in traces[name].values()) / len(qids),
        }
    oof_run, folds = _oof(qids, {name: item["retrieval_metrics"] for name, item in methods.items()}, runs)
    oof_metric = _derived_metrics(oof_run, qids, qrels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof_run, qids, queries={question.qid: question.query for question in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.file_order),
        "methods": methods,
        "references": {"cached_page_bm25": baseline, "current_hierarchical": hierarchical},
        "cv": {"folds": folds, "retrieval_metrics": oof_metric, "selected_arm_counts": {name: sum(row["selected_arm"] == name for row in folds) for name in methods}},
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "The main union arm uses top-2 lexical and top-2 visual file proposals; exact page ranking happens after the union.",
            "The top-3 lexical row is a small sensitivity control, not a broad parameter sweep.",
            "V-SPLADE query vectors are cached English translations evaluated against French qrels.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics file-candidate union baseline",
        "",
        "| Method | Mean files | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs hierarchical pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        metric = item["retrieval_metrics"]
        lines.append(
            f"| {name} | {item['mean_selected_files']:.2f} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | "
            f"{metric['file_metrics_by_k']['3']['file_recall']:.2%} | {item['comparison_to_hierarchical']['page_recall_delta_pp']:+.2f} |"
        )
    lines += [
        "",
        f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.",
        f"OOF file recall@3: **{oof_metric['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"OOF selected arms: `{report['cv']['selected_arm_counts']}`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()},
        "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2),
        "oof_file_recall@3": round(oof_metric["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
        "selected_arm_counts": report["cv"]["selected_arm_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
