"""Search the hard file budget (Kf) with the cached legacy second stage.

This independent experiment checks whether the previous Physics result is
limited by selecting too few files before page ranking.  It reuses the
cached KDL/PDF-inspector, V-SPLADE and legacy chunk artifacts only.
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
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from physics_cascade_legacy_search import (  # noqa: E402
    _load_run,
    _make_run,
    _metric_for_qids,
    _oof_select,
    _paired_comparison,
    _safe_name,
)
from research.data_discovery.hierarchical import CascadeConfig, normalise_scores  # noqa: E402
from research.experiments.evaluate_physics_legacy_second_retrieval import (  # noqa: E402
    _aggregate_hits_to_pages,
    _load_cached_query_vectors,
    _load_legacy_chunks,
    _retrieve_legacy_chunks_for_query,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _load_visual_scores,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_kf_legacy_search"


def _base_configs() -> list[CascadeConfig]:
    return [
        CascadeConfig(
            name=f"cascade-c014-kf{kf}",
            file_pool="sum_top2",
            k_files=kf,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.60,
            parent_weight=0.25,
            file_direct_weight=0.50,
            file_pool_source="page_base",
        )
        for kf in (1, 2, 3, 4, 5, 7, 10)
    ]


def _legacy_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for alpha in (0.50, 0.70, 0.90):
        for gamma in (0.10, 0.20, 0.25):
            specs.append({"pool": "max", "alpha": alpha, "gamma": gamma, "depth": 100})
    for gamma in (0.15, 0.25):
        specs.append({"pool": "sum_top2", "alpha": 0.70, "gamma": gamma, "depth": 100})
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--legacy-cache", type=Path, default=DEFAULT_LEGACY_CACHE)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions_list = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in metadata]
    if len(qids) != 302 or len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected 302 queries and 1,674 unique pages, got {len(qids)} and {len(page_ids)}")

    corpus, page_index, file_indexes, _fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(args.page_vector_dir, args.query_vector_dir, page_ids, qids)
    chunks, legacy_meta = _load_legacy_chunks(args.parsed_run, corpus)
    query_vectors, query_meta = _load_cached_query_vectors(questions_list, args.legacy_cache)
    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    chunk_matrix /= np.clip(np.linalg.norm(chunk_matrix, axis=1, keepdims=True), 1e-12, None)
    chunk_index = BM25Index(analyzer_name="auto").build(
        [{"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text} for chunk in chunks]
    )

    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores=visual_scores,
    )
    base_traces: dict[str, dict[str, dict[str, Any]]] = {}
    all_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    methods: dict[str, dict[str, Any]] = {}
    for config in _base_configs():
        run: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for qid, question in questions.items():
            page_run, trace = retriever.retrieve(qid, question.query, config)
            run[qid] = page_run
            traces[qid] = trace
        base_traces[config.name] = traces
        all_runs[config.name] = run
        methods[config.name] = {"kind": "cascade", "config": config.__dict__, "retrieval_metrics": _derived_metrics(run, qids, qrels)}

    # Materialise the legacy page evidence once for every (Kf, alpha, pool).
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    specs = _legacy_specs()
    cache: dict[tuple[str, float, str], dict[str, dict[str, float]]] = {}
    for base_name, traces in base_traces.items():
        keys = sorted({(float(item["alpha"]), str(item["pool"]), int(item["depth"])) for item in specs})
        for alpha, pool, depth in keys:
            scores: dict[str, dict[str, float]] = {}
            for index, qid in enumerate(qids, 1):
                base_ranked = [str(row["node_id"]) for row in traces[qid].get("page_candidates", [])][:100]
                allowed_pages = set(base_ranked)
                hits = _retrieve_legacy_chunks_for_query(
                    chunks, chunk_index, chunk_matrix, questions[qid].query,
                    query_vectors[qid], allowed_pages, alpha=alpha, depth=depth,
                )
                scores[qid] = _aggregate_hits_to_pages(hits, by_id, allowed_pages, pool)
                if index == len(qids) and base_name == list(base_traces)[-1]:
                    print(f"legacy features {base_name} alpha={alpha:g} pool={pool}: {index}/{len(qids)}")
            cache[(base_name, alpha, pool)] = scores

    for base_name in base_traces:
        for item in specs:
            name = f"{base_name}+legacy-{item['pool']}-a{item['alpha']:g}-g{item['gamma']:g}"
            scores = cache[(base_name, float(item["alpha"]), str(item["pool"]))]
            run, _trace = _make_run(
                base_traces[base_name], corpus, scores,
                gamma=float(item["gamma"]), spec=item,
            )
            all_runs[name] = run
            methods[name] = {
                "kind": "cascade_plus_legacy",
                "base": base_name,
                "legacy": item,
                "retrieval_metrics": _derived_metrics(run, qids, qrels),
            }

    baseline_run = _load_run(args.baseline_run)
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)
    methods["weighted fusion alpha=0.70 (reference)"] = {"kind": "reference", "retrieval_metrics": baseline_metrics}
    for method in methods.values():
        method["comparison_to_weighted"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    candidate_names = [name for name in methods if name != "weighted fusion alpha=0.70 (reference)"]
    cv = _oof_select(
        qids,
        questions,
        qrels,
        {name: methods[name]["retrieval_metrics"] for name in candidate_names},
        {name: all_runs[name] for name in candidate_names},
    )
    oof_metrics = _derived_metrics(cv["run"], qids, qrels)
    oof_comparison = _paired_comparison(baseline_metrics, oof_metrics)
    folds = cv["folds"]
    fold_rows: list[dict[str, Any]] = []
    stratified = __import__("physics_hierarchical_retrieval")._stratified_folds(qids, questions, qrels)
    for fold, heldout in enumerate(stratified):
        selected = folds[fold]["selected_method"]
        rows = {row["qid"]: row for row in methods[selected]["retrieval_metrics"]["per_query"]}
        base_rows = {row["qid"]: row for row in baseline_metrics["per_query"]}
        delta = float(np.mean([rows[qid]["page_recall@10"] - base_rows[qid]["page_recall@10"] for qid in heldout]))
        fold_rows.append({**folds[fold], "test_page_recall@10": _metric_for_qids(methods[selected]["retrieval_metrics"], set(heldout))[0], "test_delta_pp": 100.0 * delta})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        (
            {"name": name, "page_recall@10": methods[name]["retrieval_metrics"]["page_recall@10"], "ndcg@10": methods[name]["retrieval_metrics"]["ndcg@10"], "file_recall@3": methods[name]["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"]}
            for name in candidate_names
        ),
        key=lambda row: (row["page_recall@10"], row["ndcg@10"]), reverse=True,
    )
    _write_run(runs_dir / "best_full_set.jsonl", all_runs[candidates[0]["name"]], qids, queries={qid: questions[qid].query for qid in qids})
    _write_run(runs_dir / "oof_selected.jsonl", cv["run"], qids, queries={qid: questions[qid].query for qid in qids})
    report = {
        "dataset": "vidore_v3/physics", "queries": len(qids), "pages": len(page_ids), "files": len(corpus.files),
        "baseline": {"path": str(args.baseline_run), "metrics": baseline_metrics},
        "sources": {"parsed_run": str(args.parsed_run), "visual": visual_meta, "legacy_cache": legacy_meta, "query_cache": query_meta, "index_counts": index_counts},
        "methods": methods, "best_full_set": candidates[:15],
        "oof": {"folds": fold_rows, "selected_method_counts": cv["selected_method_counts"], "metrics": oof_metrics, "comparison_to_weighted": oof_comparison},
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "config": {"k_files": [1, 2, 3, 4, 5, 7, 10], "legacy_specs": specs, "scope": "page100", "seed": 20260729},
        "notes": ["Hard cascade: legacy stage only reorders the first 100 pages retained by the selected Kf files.", "Full-set values are exploratory; OOF selection uses four training folds only.", "V-SPLADE uses cached English query vectors against French qrels.", "No qrel/modality/evidence label is used as a scoring feature."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics Kf + legacy second-stage search", "", "| Method | Page recall@10 | nDCG@10 | Delta vs weighted pp | File recall@3 |", "|---|---:|---:|---:|---:|"]
    for row in candidates[:30]:
        name = row["name"]
        lines.append(f"| {name} | {row['page_recall@10']:.2%} | {row['ndcg@10']:.2f} | {methods[name]['comparison_to_weighted']['page_recall_delta_pp']:+.2f} | {row['file_recall@3']:.2%} |")
    lines += ["", f"OOF page recall@10: **{oof_metrics['page_recall@10']:.2%}** ({oof_comparison['page_recall_delta_pp']:+.2f} pp vs weighted).", "", "| Fold | Selected method | Test page recall@10 | Test delta pp |", "|---:|---|---:|---:|"]
    for row in fold_rows:
        lines.append(f"| {row['fold']} | `{row['selected_method']}` | {row['test_page_recall@10']:.2%} | {row['test_delta_pp']:+.2f} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": candidates[:5], "oof_page_recall@10": oof_metrics["page_recall@10"], "oof_delta_pp": oof_comparison["page_recall_delta_pp"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
