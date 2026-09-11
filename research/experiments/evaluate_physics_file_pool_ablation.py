"""Disentangle page fusion, file pooling and direct file BM25 on Physics.

This is a small controlled ablation for the hierarchical retrieval experiment.
Every BM25 arm operates on text extracted by PDF-inspector; no arm indexes raw
PDF bytes.  The important distinction is whether the file pool consumes
page-BM25 scores only or the already fused BM25 + V-SPLADE page score.

Run::

    python research/experiments/evaluate_physics_file_pool_ablation.py
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_PAIR_DIR,
    DEFAULT_PAGE_VECTOR_DIR,
    DEFAULT_PARSED_RUN,
    DEFAULT_QUERY_VECTOR_DIR,
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _file_id,
    _load_run,
    _load_visual_scores,
    _paired_comparison,
    _safe_name,
    _stage_metrics,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from research.data_discovery.hierarchical import CascadeConfig  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_file_pool_ablation"
)


def _configurations() -> list[CascadeConfig]:
    """Return fixed arms that isolate the hierarchy components."""
    common = dict(
        file_representation="all_text",
        file_pool="max",
        page_depth=100,
        bm25_weight=0.70,
        fine_unit="none",
        fine_weight=0.0,
    )
    return [
        CascadeConfig(
            name="A-page-fusion-no-file-stage",
            k_files=42,
            parent_weight=0.0,
            file_direct_weight=0.0,
            file_pool_source="page_bm25",
            **common,
        ),
        CascadeConfig(
            name="F-soft-bm25-page-pool",
            k_files=42,
            parent_weight=0.15,
            file_direct_weight=0.0,
            file_pool_source="page_bm25",
            **common,
        ),
        CascadeConfig(
            name="G-soft-fused-page-pool",
            k_files=42,
            parent_weight=0.15,
            file_direct_weight=0.0,
            file_pool_source="page_base",
            **common,
        ),
        CascadeConfig(
            name="B-bm25-page-pool-kf3",
            k_files=3,
            parent_weight=0.15,
            file_direct_weight=0.0,
            file_pool_source="page_bm25",
            **common,
        ),
        CascadeConfig(
            name="C-fused-page-pool-kf3",
            k_files=3,
            parent_weight=0.15,
            file_direct_weight=0.0,
            file_pool_source="page_base",
            **common,
        ),
        CascadeConfig(
            name="D1-current-full-cascade-kf1",
            k_files=1,
            parent_weight=0.15,
            file_direct_weight=0.50,
            file_pool_source="page_base",
            **common,
        ),
        CascadeConfig(
            name="D2-current-full-cascade-kf2",
            k_files=2,
            parent_weight=0.15,
            file_direct_weight=0.50,
            file_pool_source="page_base",
            **common,
        ),
        CascadeConfig(
            name="D-current-full-cascade",
            k_files=3,
            parent_weight=0.15,
            file_direct_weight=0.50,
            file_pool_source="page_base",
            **common,
        ),
    ]


def _summary_row(name: str, method: dict[str, Any]) -> dict[str, Any]:
    metrics = method["retrieval_metrics"]
    comparison = method.get("comparison_to_baseline") or {}
    stage = method.get("stage_metrics") or {}
    return {
        "method": name,
        "page_recall@10": round(100.0 * metrics["page_recall@10"], 4),
        "page_hit@10": round(100.0 * metrics["page_hit@10"], 4),
        "file_recall@3": round(
            100.0 * metrics["file_metrics_by_k"]["3"]["file_recall"], 4
        ),
        "file_hit@3": round(
            100.0 * metrics["file_metrics_by_k"]["3"]["file_hit"], 4
        ),
        "file_candidate_recall@3": (
            round(100.0 * stage["file_candidate_recall"]["@3"], 4)
            if stage.get("file_candidate_recall")
            else None
        ),
        "page_delta_pp": round(comparison.get("page_recall_delta_pp", 0.0), 4),
        "file_delta_pp": round(comparison.get("file_recall_delta_pp", 0.0), 4),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics file-pooling ablation",
        "",
        "All BM25 scores use text extracted by PDF-inspector; raw PDF bytes are never indexed.",
        "",
        "## Controlled arms",
        "",
        "| Arm | File pool input | Direct file BM25 | Hard Kf | Page recall@10 | File candidate recall@3 | File recall@3 | Δ page pp | Δ file pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        if name == "PDF-inspector + BM25 baseline":
            config = "baseline"
            pool_source = "page BM25 (cached baseline)"
            direct = "—"
            k_files = "—"
        elif not isinstance(method["config"], dict) or "file_pool_source" not in method["config"]:
            config = "cached page-fusion run"
            pool_source = "cached BM25 + V-SPLADE"
            direct = "—"
            k_files = "—"
        else:
            config = method["config"]
            pool_source = config["file_pool_source"]
            direct = f"{config['file_direct_weight']:.2f}"
            k_files = str(config["k_files"])
        metrics = method["retrieval_metrics"]
        comparison = method.get("comparison_to_baseline") or {}
        stage = method.get("stage_metrics") or {}
        file_candidate = (
            f"{stage['file_candidate_recall']['@3']:.2%}"
            if stage.get("file_candidate_recall")
            else "—"
        )
        lines.append(
            f"| {name} | {pool_source} | {direct} | {k_files} | "
            f"{metrics['page_recall@10']:.2%} | "
            f"{file_candidate} | "
            f"{metrics['file_metrics_by_k']['3']['file_recall']:.2%} | "
            f"{comparison.get('page_recall_delta_pp', 0.0):+.2f} | "
            f"{comparison.get('file_recall_delta_pp', 0.0):+.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Arm A measures BM25 + V-SPLADE page fusion without a restrictive file stage.",
        "- Arms F/G compute a file prior but keep all files eligible; they isolate the soft parent prior from hard file filtering.",
        "- Arm B is the intended variant: file score is pooled from PDF-inspector/BM25 page scores only, then used as a page parent prior with Kf=3.",
        "- Arm C changes only the file pool input from BM25-only to fused BM25 + V-SPLADE page scores.",
        "- Arm D additionally enables the direct file-BM25 branch used by the current full cascade.",
        "- `file recall@3` follows the repository protocol: first 3 unique files from the first 100 page results. `file candidate recall@3` is the direct file-stage metric.",
        "",
        f"Query language: **{report['sources']['query_language']}** V-SPLADE vectors against French Physics qrels.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    started = time.perf_counter()
    benchmark = ViDoreV3(
        root=DEFAULT_BENCHMARK_ROOT, subset="physics", language="french"
    )
    questions_list = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    page_ids = json.loads(
        (DEFAULT_PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8")
    )
    page_ids = [str(row["unit_id"]) for row in page_ids]
    configurations = _configurations()
    corpus, page_index, file_indexes, fine_indexes, index_counts = (
        _build_corpus_and_indexes(DEFAULT_PARSED_RUN, page_ids, fine_units=[])
    )
    visual_scores, visual_meta = _load_visual_scores(
        DEFAULT_PAGE_VECTOR_DIR, DEFAULT_QUERY_VECTOR_DIR, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores=visual_scores,
    )

    baseline_path = DEFAULT_PAIR_DIR / "bm25_french_bm25-french_vs-english.jsonl"
    baseline_run = _load_run(baseline_path)
    methods: dict[str, dict[str, Any]] = {
        "PDF-inspector + BM25 baseline": {
            "config": None,
            "retrieval_metrics": _derived_metrics(baseline_run, qids, qrels),
            "stage_metrics": {},
        }
    }
    cached_weighted_path = (
        DEFAULT_PAIR_DIR / "weighted_french_bm25-french_vs-english.jsonl"
    )
    if cached_weighted_path.is_file():
        cached_weighted_run = _load_run(cached_weighted_path)
        if set(cached_weighted_run) == set(qids):
            methods["E-cached-weighted-page-fusion"] = {
                "config": {"source": str(cached_weighted_path)},
                "retrieval_metrics": _derived_metrics(cached_weighted_run, qids, qrels),
                "stage_metrics": {},
            }
    output_dir = DEFAULT_OUTPUT_DIR
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for config in configurations:
        arm_started = time.perf_counter()
        runs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for question in questions_list:
            run, trace = retriever.retrieve(question.qid, question.query, config)
            runs[question.qid] = run
            traces[question.qid] = trace
        methods[config.name] = {
            "config": asdict(config),
            "retrieval_metrics": _derived_metrics(runs, qids, qrels),
            "stage_metrics": _stage_metrics(traces, qids, qrels),
            "timing_seconds": round(time.perf_counter() - arm_started, 6),
        }
        _write_run(
            runs_dir / f"{_safe_name(config.name)}.jsonl",
            runs,
            qids,
            queries={question.qid: question.query for question in questions_list},
        )

    baseline_metrics = methods["PDF-inspector + BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "PDF-inspector + BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(
                baseline_metrics, method["retrieval_metrics"]
            )

    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "sources": {"baseline_run": str(baseline_path), **visual_meta},
        "index_counts": index_counts,
        "methods": methods,
        "summary": [_summary_row(name, method) for name, method in methods.items()],
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "BM25 indexes only PDF-inspector extracted text; raw PDFs are not used as BM25 input.",
            "This ablation separates page fusion, pooled file prior and direct file BM25.",
            "File recall@3 is still the repository's page-derived discovery metric.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
