"""Locality-aware page prior for the Physics hierarchy.

Lakehouse clustering makes nearby rows co-located so a coarse zone score can
guide a precise scan.  This cache-only document analogue gives a page a small
prior when an adjacent page in the same file has a strong exact page score.
The file stage remains the fixed c014 file/region-style hierarchy (top 3
files); only pages inside those files are eligible.

The radius values are predeclared locality tests, not a learned router.  The
main result is the five-fold OOF selection between no locality, one-page and
two-page support.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_neighbor_locality_baseline"


def _run_one(
    corpus: HierarchyCorpus,
    page_index: Any,
    file_index: Any,
    visual_scores: Mapping[str, float],
    query: str,
    *,
    radius: int,
) -> list[dict[str, Any]]:
    page_ids = corpus.page_order
    page_bm25 = normalise_scores(_index_scores(page_index, query, top_k=len(page_ids)))
    visual = normalise_scores(visual_scores)
    page_base = {
        page_id: 0.60 * page_bm25.get(page_id, 0.0) + 0.40 * visual.get(page_id, 0.0)
        for page_id in page_ids
    }
    page_to_file = {page_id: _file_id(page_id) for page_id in page_ids}
    file_pool = aggregate_file_scores(page_base, page_to_file, corpus.file_order, "sum_top2")
    file_direct = normalise_scores(_index_scores(file_index, query, top_k=len(corpus.file_order)))
    file_score = {
        file_id: 0.50 * file_direct.get(file_id, 0.0) + 0.50 * normalise_scores(file_pool).get(file_id, 0.0)
        for file_id in corpus.file_order
    }
    selected_files = {file_id for file_id, _ in sort_scores(file_score)[:3]}
    file_norm = normalise_scores(file_score)

    pages_by_file: dict[str, list[str]] = {}
    for file_id in selected_files:
        pages_by_file[file_id] = sorted(
            [page_id for page_id in page_ids if page_to_file[page_id] == file_id],
            key=lambda page_id: int(page_id.rsplit("#page=", 1)[-1]),
        )
    position: dict[str, tuple[str, int]] = {
        page_id: (file_id, index)
        for file_id, pages in pages_by_file.items()
        for index, page_id in enumerate(pages)
    }
    scores: dict[str, float] = {}
    for page_id, (file_id, index) in position.items():
        neighbor_values = []
        pages = pages_by_file[file_id]
        for offset in range(1, radius + 1):
            for candidate_index in (index - offset, index + offset):
                if 0 <= candidate_index < len(pages):
                    neighbor_values.append(page_base[pages[candidate_index]])
        neighbor = max(neighbor_values, default=0.0)
        # c014 control uses 0.75 page_base + 0.25 file parent.  The locality
        # arms reserve 0.15 for neighbor support and keep the parent signal.
        scores[page_id] = (
            (0.60 if radius > 0 else 0.75) * page_base[page_id]
            + 0.25 * file_norm.get(file_id, 0.0)
            + (0.15 * neighbor if radius > 0 else 0.0)
        )
    ranked = sort_scores(scores)[:100]
    return [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
        }
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return sum(float(row["page_recall@10"]) for row in rows) / len(rows) if rows else 0.0


def _oof(qids: list[str], metrics: Mapping[str, Mapping[str, Any]], runs: Mapping[str, Mapping[str, list[dict[str, Any]]]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    all_qids = set(qids)
    for fold in range(5):
        heldout = qids[fold::5]
        train = all_qids - set(heldout)
        winner = max(metrics, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        for qid in heldout:
            output[qid] = runs[winner][qid]
        rows.append({"fold": fold, "selected_arm": winner, "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    return output, rows


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

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for radius in (0, 1, 2):
        name = "c014-control" if radius == 0 else f"neighbor-radius{radius}"
        runs[name] = {}
        for question in questions:
            runs[name][question.qid] = _run_one(
                corpus, page_index, file_index, visual_by_qid[question.qid], question.query, radius=radius
            )
        metrics[name] = _derived_metrics(runs[name], qids, qrels)

    baseline = _derived_metrics(_load_run(args.baseline_run), qids, qrels)
    hierarchical = _derived_metrics(_load_run(args.hierarchical_run), qids, qrels)
    methods = {
        name: {
            "radius": 0 if name == "c014-control" else int(name.rsplit("radius", 1)[-1]),
            "retrieval_metrics": metric,
            "comparison_to_baseline": _paired_comparison(baseline, metric),
            "comparison_to_hierarchical": _paired_comparison(hierarchical, metric),
        }
        for name, metric in metrics.items()
    }
    oof_run, folds = _oof(qids, metrics, runs)
    oof_metrics = _derived_metrics(oof_run, qids, qrels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof_run, qids, queries={question.qid: question.query for question in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "methods": methods,
        "references": {"cached_page_bm25": baseline, "current_hierarchical": hierarchical},
        "cv": {"folds": folds, "retrieval_metrics": oof_metrics, "selected_arm_counts": {name: sum(row["selected_arm"] == name for row in folds) for name in metrics}},
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "Radius 0 is the c014 file->page control; radius 1 and 2 add only same-file locality support.",
            "V-SPLADE query vectors are cached English translations evaluated against French qrels.",
            "No qrel or modality label is used during scoring.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics neighbor-locality baseline",
        "",
        "| Arm | Radius | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Δ vs c014 pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in methods.items():
        metric = item["retrieval_metrics"]
        lines.append(
            f"| {name} | {item['radius']} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | "
            f"{metric['file_metrics_by_k']['3']['file_recall']:.2%} | {item['comparison_to_hierarchical']['page_recall_delta_pp']:+.2f} |"
        )
    lines += [
        "",
        f"OOF page recall@10: **{oof_metrics['page_recall@10']:.2%}**.",
        f"OOF file recall@3: **{oof_metrics['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"OOF selected arms: `{report['cv']['selected_arm_counts']}`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "methods": {name: round(item["retrieval_metrics"]["page_recall@10"] * 100, 2) for name, item in methods.items()},
        "oof_page_recall@10": round(oof_metrics["page_recall@10"] * 100, 2),
        "oof_file_recall@3": round(oof_metrics["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
        "selected_arm_counts": report["cv"]["selected_arm_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
