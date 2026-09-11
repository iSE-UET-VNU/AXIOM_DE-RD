"""Semantic title/heading file retrieval combined with hierarchical page search.

This experiment uses the cached French multilingual-E5 query vectors and
encodes only file-level structural fields from the KDL/PDF-inspector cache.
It tests whether semantic matching of ``title``, ``heading`` and the flat
nearest-heading ``section_context`` helps the existing file -> page cascade.

No qrels are used to build vectors.  The only semantic choices are fixed in
advance: field-wise max pooling, a 3/3/1 weighted field mean, and the existing
hierarchy's fixed page fusion.  Configuration selection remains query-level
five-fold out-of-fold.
"""

from __future__ import annotations

from dataclasses import replace
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

from research.data_discovery.hierarchical import (  # noqa: E402
    HierarchyCorpus,
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_fielded_hierarchical_retrieval import (  # noqa: E402
    EXPECTED_PAGES,
    EXPECTED_QUERIES,
    FIELD_WEIGHTS,
    FieldedBM25,
    QuerySignals,
    _build_field_records,
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _load_visual_scores,
    _oof_select,
    _page_vector_units,
    _paired_comparison,
    _path_size_bytes,
    _retrieve_hierarchical,
    _safe_name,
    _stage_metrics,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_EMBEDDINGS = (
    ROOT / "data/benchmark/vidore_v3/results/physics_multilingual_e5_baseline/query_embeddings.npy"
)
DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_HIERARCHICAL_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_semantic_field_hierarchical"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"

SEMANTIC_FIELDS = ("title", "heading", "section_context")
SEMANTIC_FIELD_WEIGHTS = {"title": 3.0, "heading": 3.0, "section_context": 1.0}
PAGE_DEPTH = 100


def _normalise_dense(values: np.ndarray) -> np.ndarray:
    """Shift cosine scores if needed and scale a row to [0, 1]."""
    minimum = float(np.min(values)) if values.size else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(np.max(shifted)) if shifted.size else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _encode_file_fields(
    records: Mapping[str, Mapping[str, str]],
    file_order: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode one passage per file/field and return [field, file, dim]."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    passages: list[str] = []
    for field_name in SEMANTIC_FIELDS:
        for file_id in file_order:
            text = str(records[file_id].get(field_name) or "").strip()
            passages.append("passage: " + text)
    vectors = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    if vectors.shape[0] != len(SEMANTIC_FIELDS) * len(file_order):
        raise RuntimeError(f"Unexpected semantic field vector shape: {vectors.shape}")
    vectors = vectors.reshape(len(SEMANTIC_FIELDS), len(file_order), vectors.shape[1])
    return vectors, {
        "model": model_name,
        "prefix": "passage: ",
        "fields": list(SEMANTIC_FIELDS),
        "field_weights": SEMANTIC_FIELD_WEIGHTS,
        "field_count": len(SEMANTIC_FIELDS),
        "file_count": len(file_order),
        "dimension": int(vectors.shape[2]),
    }


def _semantic_file_scores(
    query_vectors: np.ndarray,
    field_vectors: np.ndarray,
    file_order: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Return fixed max and weighted-mean semantic file score maps."""
    if query_vectors.ndim != 2 or field_vectors.ndim != 3:
        raise RuntimeError("Semantic vectors must be [query, dim] and [field, file, dim]")
    if query_vectors.shape[1] != field_vectors.shape[2]:
        raise RuntimeError("Query and field embedding dimensions do not match")
    per_field = np.asarray(
        [query_vectors @ field_vectors[index].T for index in range(field_vectors.shape[0])],
        dtype=np.float32,
    )
    max_scores = np.max(per_field, axis=0)
    weighted_scores = np.zeros_like(max_scores)
    total_weight = sum(SEMANTIC_FIELD_WEIGHTS.values())
    for field_index, field_name in enumerate(SEMANTIC_FIELDS):
        weighted_scores += (
            SEMANTIC_FIELD_WEIGHTS[field_name]
            * np.asarray([_normalise_dense(row) for row in per_field[field_index]], dtype=np.float32)
        )
    weighted_scores /= total_weight
    output: dict[str, dict[str, float]] = {"semantic_field_max": {}, "semantic_field_weighted": {}}
    for query_index in range(query_vectors.shape[0]):
        output["semantic_field_max"][str(query_index)] = {
            file_id: float(max_scores[query_index, file_index])
            for file_index, file_id in enumerate(file_order)
        }
        output["semantic_field_weighted"][str(query_index)] = {
            file_id: float(weighted_scores[query_index, file_index])
            for file_index, file_id in enumerate(file_order)
        }
    return output


def _semantic_ranks(
    scores_by_qid: Mapping[str, Mapping[str, float]],
    qids: Sequence[str],
) -> dict[str, list[str]]:
    return {
        qid: [file_id for file_id, _ in sort_scores(scores_by_qid[qid])]
        for qid in qids
    }


def _direct_file_metrics(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    scores_by_qid: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    values = {1: [], 3: [], 5: [], 10: []}
    ranks = _semantic_ranks(scores_by_qid, qids)
    for qid in qids:
        gold = {_file_id(page_id) for page_id in qrels.get(qid, {})}
        for k in values:
            values[k].append(len(set(ranks[qid][:k]) & gold) / len(gold) if gold else 0.0)
    return {
        "file_recall": {
            f"@{k}": round(sum(rows) / len(rows), 6) if rows else 0.0
            for k, rows in values.items()
        }
    }


def _write_per_query(
    path: Path,
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    bundles: Mapping[str, tuple[Mapping[str, list[dict[str, Any]]], Mapping[str, dict[str, Any]]]],
    *,
    selected: Mapping[str, str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for qid in qids:
            gold_pages = set(qrels.get(qid, {}))
            gold_files = {_file_id(page_id) for page_id in gold_pages}
            row: dict[str, Any] = {
                "qid": qid,
                "gold_files": sorted(gold_files),
                "selected_method": selected.get(qid),
                "methods": {},
            }
            for method, (runs, traces) in bundles.items():
                run = runs[qid]
                top10 = [str(item["chunk_id"]) for item in run[:10]]
                top100_files: list[str] = []
                for item in run[:PAGE_DEPTH]:
                    file_id = _file_id(str(item["chunk_id"]))
                    if file_id not in top100_files:
                        top100_files.append(file_id)
                trace = traces[qid]
                row["methods"][method] = {
                    "page_recall@10": len(set(top10) & gold_pages) / len(gold_pages) if gold_pages else 0.0,
                    "file_recall@3": len(set(top100_files[:3]) & gold_files) / len(gold_files) if gold_files else 0.0,
                    "proposal_files": trace.get("counts", {}).get("proposal_files"),
                    "selected_files": trace.get("selected_files", []),
                    "pages_scanned": trace.get("counts", {}).get("pages_scanned"),
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics semantic structural-field hierarchy",
        "",
        "Multilingual-E5 semantic search over file title/heading/section-context fields combined with the existing hierarchical page verifier.",
        "",
        "## Protocol",
        "",
        f"- Corpus: **{report['queries']}** French queries, **{report['pages']}** pages, **{report['files']}** files.",
        f"- Model: `{report['semantic']['model']}`; query vectors are the cached French vectors from the E5 baseline.",
        "- Semantic fields: title, heading and flat nearest-heading section context; no body/page text is embedded in this experiment.",
        "- Semantic variants: field-wise max and fixed 3/3/1 weighted mean.",
        "- Hierarchy: fixed top-30 lexical/flat/visual proposal union, fixed page fusion, and five-fold OOF selection.",
        "",
        "## Direct semantic file retrieval",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in report["direct_file_metrics"].items():
        recall = payload["file_recall"]
        lines.append(
            f"| {name} | {recall['@1']:.2%} | {recall['@3']:.2%} | "
            f"{recall['@5']:.2%} | {recall['@10']:.2%} |"
        )
    lines += ["", "## Hierarchical screening", "", "| Method | Kf | nDCG@10 | Page recall@10 | File recall@3 | Proposal coverage | Page candidate recall | Mean pages |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, method in report["methods"].items():
        metric = method["retrieval_metrics"]
        stage = method["stage_metrics"]
        lines.append(
            f"| {name} | {method['k_files']} | {metric['ndcg@10']:.2f} | "
            f"{metric['page_recall@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} | "
            f"{stage['proposal_file_coverage']:.2%} | {stage['page_candidate_recall']:.2%} | "
            f"{stage['mean_pages_scanned']:.2f} |"
        )
    lines += ["", "## Out-of-fold results", ""]
    for key, payload in report["cv"].items():
        metric = payload["retrieval_metrics"]
        stage = payload["stage_metrics"]
        lines += [
            f"### {key}",
            "",
            f"- Selected methods: `{payload['selected_method_counts']}`.",
            f"- Page recall@10: **{metric['page_recall@10']:.2%}**; nDCG@10: **{metric['ndcg@10']:.2f}**.",
            f"- File recall@3: **{metric['file_metrics_by_k']['3']['file_recall']:.2%}**.",
            f"- Proposal coverage: **{stage['proposal_file_coverage']:.2%}**; selected-file recall@3: **{stage['selected_file_recall']['@3']:.2%}**.",
            f"- Page candidate recall: **{stage['page_candidate_recall']:.2%}**; mean pages scanned: **{stage['mean_pages_scanned']:.2f}**.",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "Semantic vectors are used as file-level structural evidence and are not a replacement for the page verifier.",
        "The E5 query cache is French and the document fields use the multilingual-E5 passage prefix.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-embeddings", type=Path, default=DEFAULT_QUERY_EMBEDDINGS)
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    started = time.perf_counter()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions_list = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [question.qid for question in questions_list]
    questions = {question.qid: question for question in questions_list}
    qrels = benchmark.qrels()
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(page_ids) != EXPECTED_PAGES or len(qids) != EXPECTED_QUERIES:
        raise RuntimeError(f"Inventory mismatch: queries={len(qids)}, pages={len(page_ids)}")
    query_vectors = np.load(args.query_embeddings, mmap_mode="r").astype(np.float32, copy=False)
    if query_vectors.shape != (EXPECTED_QUERIES, 384):
        raise RuntimeError(f"Unexpected cached query embedding shape: {query_vectors.shape}")

    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="physics", page_ids=page_ids)
    if len(corpus.file_order) != 42 or len(corpus.page_order) != EXPECTED_PAGES:
        raise RuntimeError(f"Parsed inventory mismatch: files={len(corpus.file_order)}, pages={len(corpus.page_order)}")
    field_records = _build_field_records(corpus, args.parsed_run)

    encode_started = time.perf_counter()
    field_vectors, semantic_meta = _encode_file_fields(
        field_records.file_records,
        corpus.file_order,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    semantic_meta["encoding_seconds"] = round(time.perf_counter() - encode_started, 6)
    semantic_by_index = _semantic_file_scores(query_vectors, field_vectors, corpus.file_order)
    semantic_by_qid: dict[str, dict[str, dict[str, float]]] = {
        representation: {
            qid: semantic_by_index[representation][str(index)]
            for index, qid in enumerate(qids)
        }
        for representation in semantic_by_index
    }

    page_fields = ("title", "heading", "body", "caption", "table", "formula", "figure", "boilerplate")
    page_fields_with_context = (*page_fields, "section_context")
    page_field_index = FieldedBM25.build(field_records.page_records, fields=page_fields, weights=FIELD_WEIGHTS)
    page_field_context_index = FieldedBM25.build(field_records.page_records, fields=page_fields_with_context, weights=FIELD_WEIGHTS)
    flat_index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in corpus.page_order)
    visual_by_qid, visual_meta = _load_visual_scores(args.page_vector_dir, ROOT / "data/output/vsplade/vidore_v3_physics_english_302q", page_ids, qids)

    signal_started = time.perf_counter()
    base_states: dict[str, QuerySignals] = {}
    for question in questions_list:
        field_scores, field_components = page_field_index.score(question.query)
        field_section_scores, field_section_components = page_field_context_index.score(question.query)
        flat_scores = _index_scores(flat_index, question.query, top_k=len(page_ids))
        base_states[question.qid] = QuerySignals(
            flat_scores=flat_scores,
            field_scores=field_scores,
            field_components=field_components,
            field_section_scores=field_section_scores,
            field_section_components=field_section_components,
            file_scores={},
            file_components={},
            visual_scores=visual_by_qid[question.qid],
        )

    direct_file_metrics = {
        representation: _direct_file_metrics(qids, qrels, scores)
        for representation, scores in semantic_by_qid.items()
    }
    methods: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    traces: dict[str, dict[str, dict[str, Any]]] = {}
    for representation, score_by_qid in semantic_by_qid.items():
        for strategy in ("synopsis", "sum_top2", "mean_synopsis_sum_top2"):
            for k_files in (3, 5, 10):
                name = f"hier-{representation}-{strategy}-kf{k_files}"
                run_by_qid: dict[str, list[dict[str, Any]]] = {}
                trace_by_qid: dict[str, dict[str, Any]] = {}
                for question in questions_list:
                    state = replace(base_states[question.qid], file_scores=score_by_qid[question.qid])
                    run_by_qid[question.qid], trace_by_qid[question.qid] = _retrieve_hierarchical(
                        corpus,
                        state,
                        strategy=strategy,
                        k_files=k_files,
                    )
                runs[name] = run_by_qid
                traces[name] = trace_by_qid
                metric = _derived_metrics(run_by_qid, qids, qrels)
                methods[name] = {
                    "k_files": k_files,
                    "representation": representation,
                    "strategy": strategy,
                    "retrieval_metrics": metric,
                    "stage_metrics": _stage_metrics(trace_by_qid, qids, qrels),
                    "query_latency_seconds": round(
                        sum(float(t.get("timing_seconds", {}).get("total", 0.0)) for t in trace_by_qid.values()) / len(qids),
                        6,
                    ),
                }

    baseline_path = args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl"
    baseline_metric = _derived_metrics(_load_run(baseline_path), qids, qrels)
    hierarchical_metric = _derived_metrics(_load_run(args.hierarchical_run), qids, qrels)
    for method in methods.values():
        method["comparison_to_cached_bm25"] = _paired_comparison(baseline_metric, method["retrieval_metrics"])
        method["comparison_to_current_hierarchical"] = _paired_comparison(hierarchical_metric, method["retrieval_metrics"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for name, run in runs.items():
        _write_run(runs_dir / f"{_safe_name(name)}.jsonl", run, qids, queries={q.qid: q.query for q in questions_list})

    cv: dict[str, Any] = {}
    oof_bundles: dict[str, tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]] = {}
    for k_files in (3, 5, 10):
        candidates = {
            name: methods[name]["retrieval_metrics"]
            for name in methods
            if methods[name]["k_files"] == k_files
        }
        meta, bundle, result = _oof_select(
            qids,
            questions,
            qrels,
            candidates,
            runs,
            traces,
            file_k=k_files,
        )
        metric = result["metrics"]
        stage = _stage_metrics(bundle["traces"], qids, qrels)
        cv[f"hierarchical_kf{k_files}"] = {
            **meta,
            "retrieval_metrics": metric,
            "stage_metrics": stage,
            "comparison_to_cached_bm25": _paired_comparison(baseline_metric, metric),
            "comparison_to_current_hierarchical": _paired_comparison(hierarchical_metric, metric),
        }
        oof_bundles[f"hierarchical_kf{k_files}"] = (bundle["run"], bundle["traces"])
        _write_run(args.output_dir / f"oof_kf{k_files}.jsonl", bundle["run"], qids, queries={q.qid: q.query for q in questions_list})

    selected_by_qid: dict[str, str] = {}
    primary_meta = cv["hierarchical_kf3"]
    for fold in primary_meta["folds"]:
        for qid in fold["heldout_qids"]:
            selected_by_qid[qid] = fold["selected_method"]
    _write_per_query(
        args.output_dir / "per_query.jsonl",
        qids,
        qrels,
        {name: (runs[name], traces[name]) for name in runs},
        selected=selected_by_qid,
    )

    np.save(args.output_dir / "file_field_embeddings.npy", field_vectors)
    np.save(
        args.output_dir / "semantic_file_scores.npy",
        np.asarray(
            [
                [semantic_by_qid["semantic_field_max"][qid][file_id] for file_id in corpus.file_order]
                for qid in qids
            ],
            dtype=np.float32,
        ),
    )
    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.file_order),
        "references": {
            "cached_pdf_inspector_bm25": baseline_metric,
            "current_hierarchical_oof": hierarchical_metric,
        },
        "semantic": {
            **semantic_meta,
            "query_embeddings": str(args.query_embeddings),
            "query_embedding_shape": list(query_vectors.shape),
            "device": "cpu" if not _cuda_available() else "cuda",
        },
        "field_inventory": field_records.stats,
        "direct_file_metrics": direct_file_metrics,
        "methods": methods,
        "cv": cv,
        "fixed_policy": {
            "semantic_fields": list(SEMANTIC_FIELDS),
            "semantic_field_weights": SEMANTIC_FIELD_WEIGHTS,
            "page_proposal_depth": 30,
            "file_synopsis_depth": 10,
            "visual_weight": 0.30,
            "no_qrels_used_for_embedding": True,
            "query_language": "French",
        },
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(baseline_path),
            "hierarchical_reference_run": str(args.hierarchical_run),
            **visual_meta,
        },
        "timing_seconds": {
            "signals": round(time.perf_counter() - signal_started, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report) + "\n", encoding="utf-8")
    primary = cv["hierarchical_kf3"]["retrieval_metrics"]
    print(json.dumps({
        "output": str(args.output_dir),
        "direct_file_recall@3": {
            name: round(payload["file_recall"]["@3"] * 100, 2)
            for name, payload in direct_file_metrics.items()
        },
        "oof_kf3_page_recall@10": round(primary["page_recall@10"] * 100, 2),
        "oof_kf3_file_recall@3": round(primary["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
        "selected_methods": cv["hierarchical_kf3"]["selected_method_counts"],
    }, ensure_ascii=False, indent=2))


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
