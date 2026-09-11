"""Controlled French-vs-English V-SPLADE query test for Physics hierarchy.

The repository handoff identifies English V-SPLADE queries against French
qrels as a confound.  This runner reuses the cached French 302-query vectors
and evaluates a small predeclared set of existing hierarchical configurations
without writing the very large stage-trace artifact.
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

from research.data_discovery.hierarchical import CascadeConfig  # noqa: E402
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _load_run,
    _load_visual_scores,
    _paired_comparison,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_ENGLISH_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_FRENCH_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_french_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_language_control"


def _configs() -> list[CascadeConfig]:
    """Existing structural choices, fixed before this language comparison."""
    return [
        CascadeConfig(
            name="hier-current-max-a070-kf3",
            file_pool="max",
            k_files=3,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.70,
            parent_weight=0.15,
            file_direct_weight=0.50,
            file_pool_source="page_base",
        ),
        CascadeConfig(
            name="hier-c014-sum2-a060-kf3",
            file_pool="sum_top2",
            k_files=3,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.60,
            parent_weight=0.25,
            file_direct_weight=0.50,
            file_pool_source="page_base",
        ),
        CascadeConfig(
            name="hier-c065-sum2-a070-kf3",
            file_pool="sum_top2",
            k_files=3,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.70,
            parent_weight=0.25,
            file_direct_weight=0.25,
            file_pool_source="page_base",
        ),
    ]


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return sum(float(row["page_recall@10"]) for row in rows) / len(rows) if rows else 0.0


def _oof(
    qids: list[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    method_metrics: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    folds = _stratified_folds(qids, questions, qrels)
    qid_set = set(qids)
    output: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        train = qid_set - set(heldout)
        winner = max(method_metrics, key=lambda name: (_metric_for_qids(method_metrics[name], train), name))
        for qid in heldout:
            output[qid] = runs[winner][qid]
        rows.append({
            "fold": fold,
            "selected_method": winner,
            "train_page_recall@10": _metric_for_qids(method_metrics[winner], train),
            "test_page_recall@10": _metric_for_qids(method_metrics[winner], set(heldout)),
        })
    return output, rows


def _summary_table(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics hierarchical language control",
        "",
        "Offline comparison of cached English and French V-SPLADE query vectors on the same French Physics qrels.",
        "",
        "| Query vectors | Arm | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for language, payload in report["languages"].items():
        for name, item in payload["methods"].items():
            metric = item["retrieval_metrics"]
            lines.append(
                f"| {language} | {name} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | "
                f"{metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} |"
            )
        cv = payload["cv"]
        metric = cv["retrieval_metrics"]
        lines.append(
            f"| {language} | **OOF selected** | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | "
            f"{metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- French V-SPLADE is evaluated against French qrels; English V-SPLADE remains the historical reference.",
        "- Configurations are existing structural arms, not a new weight sweep.",
        "- No query qrel or content-modality label is used by the scorer.",
        "- This runner omits `stage_traces.jsonl` to avoid multi-gigabyte output; only the OOF run is persisted.",
    ]
    return "\n".join(lines) + "\n"


def _run_language(
    *,
    language: str,
    query_dir: Path,
    corpus: Any,
    page_index: Any,
    file_indexes: Mapping[str, Any],
    page_ids: list[str],
    qids: list[str],
    questions: list[Any],
    qrels: Mapping[str, Mapping[str, int]],
    output_dir: Path,
) -> dict[str, Any]:
    visual_scores, visual_meta = _load_visual_scores(query_dir.parent / "vidore_v3_physics_48q", query_dir, page_ids, qids)
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores=visual_scores,
    )
    methods: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for config in _configs():
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        for question in questions:
            run, _trace = retriever.retrieve(question.qid, question.query, config)
            run_by_qid[question.qid] = run
        runs[config.name] = run_by_qid
        methods[config.name] = {
            "config": asdict(config),
            "retrieval_metrics": _derived_metrics(run_by_qid, qids, qrels),
        }
    metric_map = {name: item["retrieval_metrics"] for name, item in methods.items()}
    oof_run, fold_rows = _oof(qids, {question.qid: question for question in questions}, qrels, metric_map, runs)
    oof_metrics = _derived_metrics(oof_run, qids, qrels)
    language_output = output_dir / language
    language_output.mkdir(parents=True, exist_ok=True)
    _write_run(language_output / "oof_run.jsonl", oof_run, qids, queries={question.qid: question.query for question in questions})
    return {
        "query_dir": str(query_dir),
        "visual": visual_meta,
        "methods": methods,
        "cv": {"folds": fold_rows, "retrieval_metrics": oof_metrics, "selected_arm_counts": {name: sum(row["selected_method"] == name for row in fold_rows) for name in methods}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--english-dir", type=Path, default=DEFAULT_ENGLISH_DIR)
    parser.add_argument("--french-dir", type=Path, default=DEFAULT_FRENCH_DIR)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    page_metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in page_metadata]
    if len(questions) != 302 or len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError("Expected 302 questions and 1,674 unique pages")
    corpus, page_index, file_indexes, _fine, index_counts = _build_corpus_and_indexes(args.parsed_run, page_ids, fine_units=[])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    languages = {}
    for language, query_dir in (("english", args.english_dir), ("french", args.french_dir)):
        languages[language] = _run_language(
            language=language,
            query_dir=query_dir,
            corpus=corpus,
            page_index=page_index,
            file_indexes=file_indexes,
            page_ids=page_ids,
            qids=qids,
            questions=questions,
            qrels=qrels,
            output_dir=args.output_dir,
        )
    baseline = _derived_metrics(_load_run(args.baseline_run), qids, qrels)
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "baseline_cached_page_bm25": {"path": str(args.baseline_run), "retrieval_metrics": baseline},
        "languages": languages,
        "sources": {"parsed_run": str(args.parsed_run), "index_counts": index_counts},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "The same French qrels are used for both query-vector languages.",
            "The V-SPLADE page vectors are shared; only the cached query vectors change.",
            "No stage trace is written because the generic hierarchy trace exceeds available local disk space.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_summary_table(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "languages": {
            language: {
                "oof_page_recall@10": round(payload["cv"]["retrieval_metrics"]["page_recall@10"] * 100, 2),
                "oof_file_recall@3": round(payload["cv"]["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
                "selected_arm_counts": payload["cv"]["selected_arm_counts"],
            }
            for language, payload in languages.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
