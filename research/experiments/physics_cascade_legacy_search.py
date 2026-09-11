"""Search a cached Physics cascade followed by a legacy chunk reranker.

This is an independent, retrieval-only research experiment.  It reuses the
existing KDL + pdf-inspector page text, cached V-SPLADE scores and cached
legacy text-chunk embeddings.  It does not render, encode, call an API or
modify an existing experiment.

The hypothesis is deliberately narrow: a better file -> page cascade can
provide a cleaner candidate set, while the old chunk-level BM25 + dense
retriever can recover lexical details inside those pages.  The second stage
is restricted to the first 100 pages of each cascade, so the hard-cascade
result cannot gain pages that the first stage discarded.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import CascadeConfig, HierarchyCorpus, normalise_scores, sort_scores  # noqa: E402
from research.experiments.evaluate_physics_legacy_second_retrieval import (  # noqa: E402
    LegacyChunk,
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
    _paired_comparison,
    _stratified_folds,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_LEGACY_CACHE = ROOT / "data/work/vidore_physics_emb"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_legacy_search"


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["qid"])] = list(row["chunks"])
    return output


def _base_configs() -> list[CascadeConfig]:
    # These were fixed before inspecting the new second-stage results.  They
    # cover the best full-set cascade, the best OOF cascade and the current
    # published cascade used by the legacy experiment.
    return [
        CascadeConfig(
            name="cascade-c014-a060-sum2-d050-p025-kf3",
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
            name="cascade-c065-a070-sum2-d025-p025-kf3",
            file_pool="sum_top2",
            k_files=3,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.70,
            parent_weight=0.25,
            file_direct_weight=0.25,
            file_pool_source="page_base",
        ),
        CascadeConfig(
            name="cascade-current-a070-max-d050-p015-kf3",
            file_pool="max",
            k_files=3,
            page_depth=100,
            final_depth=100,
            bm25_weight=0.70,
            parent_weight=0.15,
            file_direct_weight=0.50,
            file_pool_source="page_base",
        ),
    ]


def _second_specs() -> list[dict[str, Any]]:
    # A small predeclared grid.  The dense/sparse chunk scores are computed
    # once per (base, alpha, depth), then these gamma/pooling variants are
    # cheap to materialise.
    specs: list[dict[str, Any]] = []
    for alpha in (0.50, 0.70, 0.90):
        for gamma in (0.10, 0.20, 0.25, 0.30):
            specs.append({"pool": "max", "gamma": gamma, "alpha": alpha, "depth": 100})
    for depth in (50, 200):
        specs.append({"pool": "max", "gamma": 0.25, "alpha": 0.70, "depth": depth})
    for gamma in (0.15, 0.25, 0.35):
        specs.append({"pool": "sum_top2", "gamma": gamma, "alpha": 0.70, "depth": 100})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for spec in specs:
        key = tuple(spec[name] for name in ("pool", "gamma", "alpha", "depth"))
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def _make_run(
    base_traces: Mapping[str, Mapping[str, Any]],
    corpus: HierarchyCorpus,
    fine_scores: Mapping[str, Mapping[str, float]],
    *,
    gamma: float,
    spec: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for qid, base_trace in base_traces.items():
        base_ranked = [str(row["node_id"]) for row in base_trace.get("page_candidates", [])][:100]
        base_scores = {
            str(row["node_id"]): float(row["score"])
            for row in base_trace.get("page_candidates", [])
            if str(row["node_id"]) in base_ranked
        }
        base_norm = normalise_scores(base_scores)
        fine_norm = normalise_scores(fine_scores.get(qid, {}))
        final_scores = {
            page_id: (1.0 - gamma) * base_norm.get(page_id, 0.0)
            + gamma * fine_norm.get(page_id, 0.0)
            for page_id in base_ranked
        }
        ranked = sort_scores(final_scores)[:100]
        runs[qid] = [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "text": corpus.pages[page_id].text,
                "score": round(float(score), 8),
                "rank": rank,
                "scores": {
                    "page_base": round(float(base_norm.get(page_id, 0.0)), 8),
                    "legacy_chunk_score": round(float(fine_norm.get(page_id, 0.0)), 8),
                    "final_score": round(float(score), 8),
                },
            }
            for rank, (page_id, score) in enumerate(ranked, 1)
        ]
        traces[qid] = {
            "base_page_pool": base_ranked,
            "legacy_pages": sorted(fine_scores.get(qid, {}), key=fine_scores.get(qid, {}).get, reverse=True)[:20],
            "gamma": gamma,
            "legacy_alpha_dense": float(spec["alpha"]),
            "legacy_depth": int(spec["depth"]),
            "pool": str(spec["pool"]),
            "page_candidates": len(base_ranked),
            "legacy_candidate_pages": len(fine_scores.get(qid, {})),
        }
    return runs, traces


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> tuple[float, float]:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    if not rows:
        return 0.0, 0.0
    page = float(np.mean([float(row["page_recall@10"]) for row in rows]))
    file_values: list[float] = []
    for row in rows:
        candidates = set(row["top10_files_from_top100_pages"][:3])
        gold = set(row["gold_files"])
        file_values.append(len(candidates & gold) / len(gold) if gold else 0.0)
    return page, float(np.mean(file_values))


def _oof_select(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    metrics: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    selected: list[dict[str, Any]] = []
    output_run: dict[str, list[dict[str, Any]]] = {}
    for fold, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = all_qids - heldout_set
        ranked: list[tuple[float, float, str]] = []
        for name, method_metrics in metrics.items():
            page, file = _metric_for_qids(method_metrics, train)
            ranked.append((page, file, name))
        train_page, train_file, winner = max(ranked, key=lambda row: (row[0], row[1], row[2]))
        for qid in heldout:
            output_run[qid] = runs[winner][qid]
        selected.append(
            {
                "fold": fold,
                "heldout_qids": heldout,
                "selected_method": winner,
                "train_page_recall@10": train_page,
                "train_file_recall@3": train_file,
            }
        )
    return {
        "folds": selected,
        "selected_method_counts": {
            name: sum(row["selected_method"] == name for row in selected)
            for name in metrics
        },
        "run": output_run,
    }


def _per_query_report(
    path: Path,
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    methods: Mapping[str, Mapping[str, Any]],
    qids: Sequence[str],
) -> None:
    folds = {qid: fold for fold, group in enumerate(_stratified_folds(qids, questions, qrels)) for qid in group}
    with path.open("w", encoding="utf-8") as handle:
        for qid in qids:
            row: dict[str, Any] = {
                "qid": qid,
                "fold": folds[qid],
                "query": questions[qid].query,
                "gold_pages": sorted(qrels.get(qid, {})),
                "gold_files": sorted({page.split("#page=", 1)[0] for page in qrels.get(qid, {})}),
                "methods": {},
            }
            for name, method in methods.items():
                metric_row = next(item for item in method["retrieval_metrics"]["per_query"] if item["qid"] == qid)
                row["methods"][name] = {
                    "top10_pages": metric_row["top10_pages"],
                    "top3_files": metric_row["top10_files_from_top100_pages"][:3],
                    "page_hit@10": metric_row["page_hit@10"],
                    "page_recall@10": metric_row["page_recall@10"],
                    "page_precision@10": metric_row["page_precision@10"],
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics cascade + legacy chunk search",
        "",
        "Offline hard-cascade experiment using cached PDF-inspector text, cached V-SPLADE scores and cached legacy chunk embeddings.",
        "The second stage can only reorder the first 100 pages retained by each cascade.",
        "V-SPLADE uses English query vectors against French Physics qrels.",
        "",
        "## Main results",
        "",
        "| Method | nDCG@10 | Page recall@10 | Delta vs weighted pp | File recall@3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metrics = method["retrieval_metrics"]
        comparison = method.get("comparison_to_weighted") or {}
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_recall@10']:.2%} | "
            f"{comparison.get('page_recall_delta_pp', 0.0):+.2f} | "
            f"{metrics['file_metrics_by_k']['3']['file_recall']:.2%} |"
        )
    lines += [
        "",
        "## Best full-set methods",
        "",
        "| Method | Page recall@10 | nDCG@10 | File recall@3 |",
        "|---|---:|---:|---:|",
    ]
    for item in report["best_full_set"]:
        lines.append(
            f"| {item['name']} | {item['page_recall@10']:.2%} | {item['ndcg@10']:.2f} | {item['file_recall@3']:.2%} |"
        )
    cv = report["oof"]
    lines += [
        "",
        "## 5-fold OOF method selection",
        "",
        f"- OOF page recall@10: **{cv['metrics']['page_recall@10']:.2%}** ({cv['comparison_to_weighted']['page_recall_delta_pp']:+.2f} pp vs weighted alpha=0.7).",
        f"- OOF nDCG@10: **{cv['metrics']['ndcg@10']:.2f}**; file recall@3: **{cv['metrics']['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"- Selected methods by fold: `{cv['selected_method_counts']}`.",
        "",
        "| Fold | Selected method | Test page recall@10 | Test delta pp |",
        "|---:|---|---:|---:|",
    ]
    for fold in cv["folds"]:
        lines.append(
            f"| {fold['fold']} | `{fold['selected_method']}` | {fold['test_page_recall@10']:.2%} | {fold['test_delta_pp']:+.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Full-set numbers are exploratory because all fixed variants are scored on the complete qrels.",
        "- The OOF result is the held-out comparison; each fold selects a method using only the other four folds.",
        "- Since the legacy stage is restricted to the cascade page pool, it cannot fix a page omitted by the first stage.",
        "- Cross-page legacy chunks are mapped to every touched page, matching the existing legacy experiment's provenance convention.",
        "- This experiment does not make an end-to-end QA claim.",
        "",
        "Timing and per-query details are in `report.json`, `per_query.jsonl` and `runs/`.",
    ]
    return "\n".join(lines) + "\n"


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
    if len(qids) != 302:
        raise RuntimeError(f"Expected 302 Physics queries, got {len(qids)}")

    page_ids = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in page_ids]
    if len(page_ids) != 1674 or len(set(page_ids)) != len(page_ids):
        raise RuntimeError(f"Expected 1,674 unique page vector units, got {len(page_ids)}")

    load_started = time.perf_counter()
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
    load_seconds = time.perf_counter() - load_started

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
    feature_started = time.perf_counter()
    base_specs = _base_configs()
    for config in base_specs:
        runs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for qid, question in questions.items():
            run, trace = retriever.retrieve(qid, question.query, config)
            runs[qid] = run
            traces[qid] = trace
        base_traces[config.name] = traces
        all_runs[config.name] = runs
        methods[config.name] = {
            "kind": "cascade",
            "config": asdict(config),
            "retrieval_metrics": _derived_metrics(runs, qids, qrels),
        }
    feature_seconds = time.perf_counter() - feature_started

    fit_started = time.perf_counter()
    variant_traces: dict[str, dict[str, dict[str, Any]]] = {}
    second_specs = _second_specs()
    cache: dict[tuple[str, float, int, str], dict[str, dict[str, float]]] = {}
    for base_name, traces in base_traces.items():
        for spec in sorted({
            (float(item["alpha"]), int(item["depth"]), str(item["pool"]))
            for item in second_specs
        }):
            alpha, depth, pool = spec
            score_by_qid: dict[str, dict[str, float]] = {}
            for index, qid in enumerate(qids, 1):
                base_ranked = [str(row["node_id"]) for row in traces[qid].get("page_candidates", [])][:100]
                allowed_pages = set(base_ranked)
                hits = _retrieve_legacy_chunks_for_query(
                    chunks,
                    chunk_index,
                    chunk_matrix,
                    questions[qid].query,
                    query_vectors[qid],
                    allowed_pages,
                    alpha=alpha,
                    depth=depth,
                )
                by_id = {chunk.chunk_id: chunk for chunk in chunks}
                score_by_qid[qid] = _aggregate_hits_to_pages(hits, by_id, allowed_pages, pool)
                if (index == len(qids) and base_name == list(base_traces)[-1]) or (index % 100 == 0 and alpha == 0.5 and depth == 100 and pool == "max"):
                    print(f"legacy features {base_name} alpha={alpha:g} depth={depth} pool={pool}: {index}/{len(qids)}")
            cache[(base_name, alpha, depth, pool)] = score_by_qid

    materialise_started = time.perf_counter()
    for base_name in base_traces:
        for item in second_specs:
            name = (
                f"{base_name}+legacy-{item['pool']}-a{item['alpha']:g}-"
                f"g{item['gamma']:g}-d{item['depth']}"
            )
            key = (base_name, float(item["alpha"]), int(item["depth"]), str(item["pool"]))
            runs, traces = _make_run(
                base_traces[base_name],
                corpus,
                cache[key],
                gamma=float(item["gamma"]),
                spec=item,
            )
            all_runs[name] = runs
            variant_traces[name] = traces
            methods[name] = {
                "kind": "cascade_plus_legacy",
                "base": base_name,
                "legacy": dict(item),
                "retrieval_metrics": _derived_metrics(runs, qids, qrels),
            }
    materialise_seconds = time.perf_counter() - materialise_started
    fit_seconds = time.perf_counter() - fit_started

    baseline_run = _load_run(args.baseline_run)
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)
    methods = {
        "weighted fusion alpha=0.70 (reference)": {
            "kind": "reference",
            "retrieval_metrics": baseline_metrics,
        },
        **methods,
    }
    for method in methods.values():
        method["comparison_to_weighted"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    oof_started = time.perf_counter()
    candidate_names = [name for name in methods if name != "weighted fusion alpha=0.70 (reference)"]
    cv = _oof_select(
        qids,
        questions,
        qrels,
        {name: methods[name]["retrieval_metrics"] for name in candidate_names},
        {name: all_runs[name] for name in candidate_names},
    )
    oof_metrics = _derived_metrics(cv["run"], qids, qrels)
    folds = _stratified_folds(qids, questions, qrels)
    base_by_qid = {row["qid"]: row for row in baseline_metrics["per_query"]}
    oof_fold_rows: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        selected_name = cv["folds"][fold]["selected_method"]
        selected_rows = {row["qid"]: row for row in methods[selected_name]["retrieval_metrics"]["per_query"]}
        deltas = [
            float(selected_rows[qid]["page_recall@10"]) - float(base_by_qid[qid]["page_recall@10"])
            for qid in heldout
        ]
        oof_fold_rows.append({
            **cv["folds"][fold],
            "test_page_recall@10": float(np.mean([selected_rows[qid]["page_recall@10"] for qid in heldout])),
            "test_delta_pp": 100.0 * float(np.mean(deltas)),
        })
    oof_seconds = time.perf_counter() - oof_started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    best_full = sorted(
        (
            {
                "name": name,
                "page_recall@10": method["retrieval_metrics"]["page_recall@10"],
                "ndcg@10": method["retrieval_metrics"]["ndcg@10"],
                "file_recall@3": method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"],
            }
            for name, method in methods.items()
            if name != "weighted fusion alpha=0.70 (reference)"
        ),
        key=lambda row: (row["page_recall@10"], row["ndcg@10"]),
        reverse=True,
    )[:15]
    full_best_name = best_full[0]["name"]
    _write_run(runs_dir / "best_full_set.jsonl", all_runs[full_best_name], qids, queries={qid: questions[qid].query for qid in qids})
    _write_run(runs_dir / "oof_selected.jsonl", cv["run"], qids, queries={qid: questions[qid].query for qid in qids})

    per_query_path = args.output_dir / "per_query.jsonl"
    _per_query_report(per_query_path, questions, qrels, methods, qids)
    comparison_oof = _paired_comparison(baseline_metrics, oof_metrics)
    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "baseline": {
            "name": "weighted fusion alpha=0.70",
            "page_recall@10": baseline_metrics["page_recall@10"],
            "source": str(args.baseline_run),
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "page_vector_artifact": visual_meta,
            "legacy_cache": legacy_meta,
            "query_cache": query_meta,
        },
        "index_counts": index_counts,
        "method_count": len(methods),
        "methods": methods,
        "best_full_set": best_full,
        "oof": {
            "folds": oof_fold_rows,
            "selected_method_counts": cv["selected_method_counts"],
            "metrics": oof_metrics,
            "comparison_to_weighted": comparison_oof,
        },
        "timing_seconds": {
            "load_cached_artifacts": round(load_seconds, 6),
            "base_feature_extraction": round(feature_seconds, 6),
            "legacy_feature_fit_and_materialisation": round(fit_seconds, 6),
            "variant_materialisation": round(materialise_seconds, 6),
            "oof_selection": round(oof_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "config": {
            "base_configs": [asdict(config) for config in base_specs],
            "second_stage_specs": second_specs,
            "scope": "page100",
            "seed": 20260729,
            "selection": "5-fold stratified query split; choose by training page recall, then file recall@3",
        },
        "notes": [
            "This is a hard cascade: legacy reranking can only reorder the first 100 pages of the selected cascade.",
            "Full-set rankings are exploratory; OOF selection is the held-out comparison.",
            "V-SPLADE page scores use cached English query vectors and French Physics qrels, a language confound.",
            "Legacy text chunks are mapped to every touched page for cross-page provenance, matching the existing legacy experiment.",
            "No qrel, modality label or evidence label is used as a scoring feature.",
            "This experiment is retrieval-only and makes no E2E QA claim.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(report["config"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "best_full_set": best_full[:5],
        "oof_page_recall@10": oof_metrics["page_recall@10"],
        "oof_delta_pp": comparison_oof["page_recall_delta_pp"],
        "oof_selected_method_counts": cv["selected_method_counts"],
        "timing_seconds": report["timing_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
