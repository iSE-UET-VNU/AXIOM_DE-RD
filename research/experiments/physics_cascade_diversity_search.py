"""Screen page diversity quotas for the best cached Physics cascade.

The experiment tests whether a per-file page quota improves evidence recall
after file selection.  It is intentionally separate from all existing
scripts and uses only cached KDL/PDF-inspector text and V-SPLADE scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import CascadeConfig  # noqa: E402
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _load_visual_scores,
    _paired_comparison,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_diversity_search"


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def _oof(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    metrics: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    output: dict[str, list[dict[str, Any]]] = {}
    fold_rows: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        train = all_qids - set(heldout)
        winner = max(metrics, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        for qid in heldout:
            output[qid] = runs[winner][qid]
        train_value = _metric_for_qids(metrics[winner], train)
        test_value = _metric_for_qids(metrics[winner], set(heldout))
        fold_rows.append({
            "fold": fold,
            "selected_method": winner,
            "train_page_recall@10": train_value,
            "test_page_recall@10": test_value,
        })
    return {"run": output, "folds": fold_rows}


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
    questions_list = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    page_metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in page_metadata]
    if len(qids) != 302 or len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected 302 queries and 1,674 unique pages, got {len(qids)} and {len(page_ids)}")

    corpus, page_index, file_indexes, _fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores=visual_scores,
    )
    quotas = (0, 2, 3, 4, 5, 8, 10, 15)
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    methods: dict[str, dict[str, Any]] = {}
    for quota in quotas:
        name = f"cascade-c014-file-quota-{quota}"
        config = CascadeConfig(
            name=name,
            file_pool="sum_top2",
            k_files=3,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.60,
            parent_weight=0.25,
            file_direct_weight=0.50,
            file_pool_source="page_base",
            file_quota=quota,
        )
        run: dict[str, list[dict[str, Any]]] = {}
        for qid, question in questions.items():
            page_run, _trace = retriever.retrieve(qid, question.query, config)
            run[qid] = page_run
        runs[name] = run
        methods[name] = {
            "config": config.__dict__,
            "retrieval_metrics": _derived_metrics(run, qids, qrels),
        }

    baseline_run = _load_run(args.baseline_run)
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)
    methods["weighted fusion alpha=0.70 (reference)"] = {"retrieval_metrics": baseline_metrics}
    for method in methods.values():
        method["comparison_to_weighted"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    candidates = {name: method["retrieval_metrics"] for name, method in methods.items() if name in runs}
    cv = _oof(qids, questions, qrels, candidates, runs)
    oof_metrics = _derived_metrics(cv["run"], qids, qrels)
    oof_comparison = _paired_comparison(baseline_metrics, oof_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_name = max(candidates, key=lambda name: (candidates[name]["page_recall@10"], candidates[name]["ndcg@10"]))
    _write_run(runs_dir / "best_full_set.jsonl", runs[best_name], qids, queries={qid: questions[qid].query for qid in qids})
    _write_run(runs_dir / "oof_selected.jsonl", cv["run"], qids, queries={qid: questions[qid].query for qid in qids})

    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "baseline": {"path": str(args.baseline_run), "metrics": baseline_metrics},
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta, "index_counts": index_counts},
        "methods": methods,
        "best_full_set": best_name,
        "oof": {
            "folds": cv["folds"],
            "metrics": oof_metrics,
            "comparison_to_weighted": oof_comparison,
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "Only page ordering changes; all methods use the same c014 file/page scores and Kf=3.",
            "Quota 0 is the no-diversification c014 reference.",
            "V-SPLADE uses cached English query vectors against French qrels.",
            "Full-set values are exploratory; OOF selection uses training folds only.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics cascade page-diversity search",
        "",
        "| Method | nDCG@10 | Page recall@10 | Delta vs weighted pp | File recall@3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, method in methods.items():
        metrics = method["retrieval_metrics"]
        delta = method["comparison_to_weighted"]["page_recall_delta_pp"]
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_recall@10']:.2%} | {delta:+.2f} | "
            f"{metrics['file_metrics_by_k']['3']['file_recall']:.2%} |"
        )
    lines += [
        "",
        f"Best full-set quota: **{best_name}**.",
        f"OOF page recall@10: **{oof_metrics['page_recall@10']:.2%}** ({oof_comparison['page_recall_delta_pp']:+.2f} pp vs weighted).",
        "",
        "| Fold | Selected method | Test page recall@10 |",
        "|---:|---|---:|",
    ]
    for fold in cv["folds"]:
        lines.append(f"| {fold['fold']} | `{fold['selected_method']}` | {fold['test_page_recall@10']:.2%} |")
    lines += [
        "",
        "The quota is a structural diversity control, not a modality/evidence label.",
        "Per-query results and runs are under `report.json` and `runs/`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "best_full_set": best_name,
        "best_full_page_recall@10": methods[best_name]["retrieval_metrics"]["page_recall@10"],
        "oof_page_recall@10": oof_metrics["page_recall@10"],
        "oof_delta_pp": oof_comparison["page_recall_delta_pp"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
