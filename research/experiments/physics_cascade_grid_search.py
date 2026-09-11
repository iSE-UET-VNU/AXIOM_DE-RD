"""Search the cached Physics file -> page fusion cascade offline.

The existing hierarchical runner uses one fixed cascade configuration.  This
experiment expands that design into a small, predeclared grid and performs
query-folded selection.  It uses only cached PDF-inspector BM25 text, cached
V-SPLADE scores and benchmark qrels; no rendering, encoding or API calls are
made.

The method is intentionally limited to the first-stage cascade:

    page BM25 + V-SPLADE -> file pooling -> hard file budget -> page rerank

It is a parameter search, not a claim that the best full-set configuration is
generalizable.  The OOF selected configuration is the primary result.
"""

from __future__ import annotations

import argparse
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

from research.data_discovery.hierarchical import (  # noqa: E402
    CascadeConfig,
    HierarchyCorpus,
    aggregate_file_scores,
    build_bm25,
    normalise_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _build_corpus_and_indexes,
    _derived_metrics,
    _index_scores,
    _load_visual_scores,
    _page_vector_units,
    _paired_comparison,
    _safe_name,
    _stage_metrics,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_cascade_grid_search"


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _qrel_metrics(
    page_ids: Sequence[str],
    file_ids: Sequence[str],
    ranked: np.ndarray,
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    depth: int = 100,
) -> tuple[np.ndarray, dict[str, Any]]:
    page_pos = {page_id: pos for pos, page_id in enumerate(page_ids)}
    file_of = {page_id: page_id.split("#page=", 1)[0] for page_id in page_ids}
    recalls: list[float] = []
    hits: list[float] = []
    ndcgs: list[float] = []
    file_recall3: list[float] = []
    per_query: list[dict[str, Any]] = []
    for row_index, qid in enumerate(qids):
        gold = qrels.get(qid, {})
        gold_pages = set(gold)
        ranked_pages = [page_ids[int(pos)] for pos in ranked[row_index, :depth] if int(pos) >= 0]
        top10 = ranked_pages[:10]
        found = set(top10) & gold_pages
        recall = len(found) / len(gold_pages) if gold_pages else 0.0
        recalls.append(recall)
        hits.append(float(bool(found)))
        gains = [
            (2.0 ** int(gold[page]) - 1.0) / np.log2(rank + 1)
            for rank, page in enumerate(top10, 1)
            if page in gold
        ]
        ideal = sorted((int(value) for value in gold.values()), reverse=True)[:10]
        ideal_dcg = sum((2.0 ** value - 1.0) / np.log2(rank + 1) for rank, value in enumerate(ideal, 1))
        ndcgs.append(sum(gains) / ideal_dcg if ideal_dcg else 0.0)
        unique_files: list[str] = []
        seen: set[str] = set()
        for page in ranked_pages:
            file_id = file_of[page]
            if file_id not in seen:
                seen.add(file_id)
                unique_files.append(file_id)
            if len(unique_files) >= 10:
                break
        gold_files = {file_of[page] for page in gold_pages}
        file_recall3.append(
            len(set(unique_files[:3]) & gold_files) / len(gold_files) if gold_files else 0.0
        )
        per_query.append(
            {
                "qid": qid,
                "gold_pages": sorted(gold_pages),
                "gold_files": sorted(gold_files),
                "top10_pages": top10,
                "top10_files_from_top100_pages": unique_files,
                "page_hit@10": bool(found),
                "page_recall@10": recall,
                "ndcg@10": ndcgs[-1],
            }
        )
    return np.asarray(recalls, dtype=np.float64), {
        "queries": len(qids),
        "ndcg@10": 100.0 * float(np.mean(ndcgs)) if ndcgs else 0.0,
        "page_hit@10": float(np.mean(hits)) if hits else 0.0,
        "page_recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "page_precision@10": float(np.mean([row["page_recall@10"] for row in per_query])) if per_query else 0.0,
        "file_recall@3": float(np.mean(file_recall3)) if file_recall3 else 0.0,
        "per_query": per_query,
    }


def _make_grid() -> list[dict[str, Any]]:
    """Return a fixed, moderate grid before any qrel is inspected."""
    configs: list[dict[str, Any]] = []
    # Main grid: directly tests the file-prior design around the published
    # current configuration.
    for alpha in (0.60, 0.70, 0.80):
        for source in ("page_base", "page_bm25"):
            for pool in ("max", "sum_top2", "coverage"):
                for direct_weight in (0.25, 0.50, 0.75):
                    for parent_weight in (0.05, 0.15, 0.25):
                        configs.append(
                            {
                                "alpha": alpha,
                                "source": source,
                                "pool": pool,
                                "direct_weight": direct_weight,
                                "parent_weight": parent_weight,
                                "k_files": 3,
                            }
                        )
    # File budget variants are included independently because the main
    # hypothesis is that Kf=3 may be too brittle for some queries.
    for k_files in (1, 2, 4, 5, 7, 10):
        for alpha in (0.60, 0.70, 0.80):
            for source, pool in (("page_base", "max"), ("page_base", "sum_top2"), ("page_bm25", "max")):
                configs.append(
                    {
                        "alpha": alpha,
                        "source": source,
                        "pool": pool,
                        "direct_weight": 0.50,
                        "parent_weight": 0.15,
                        "k_files": k_files,
                    }
                )
    # Remove exact duplicates while preserving deterministic order.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for config in configs:
        key = tuple(config[key] for key in ("alpha", "source", "pool", "direct_weight", "parent_weight", "k_files"))
        if key not in seen:
            seen.add(key)
            unique.append(config)
    return unique


def _score_config(
    config: Mapping[str, Any],
    *,
    qids: Sequence[str],
    page_ids: Sequence[str],
    file_ids: Sequence[str],
    page_file_positions: np.ndarray,
    page_bm25_norm: np.ndarray,
    visual_norm: np.ndarray,
    file_direct_norm: np.ndarray,
) -> np.ndarray:
    alpha = float(config["alpha"])
    page_base = alpha * page_bm25_norm + (1.0 - alpha) * visual_norm
    source = page_base if config["source"] == "page_base" else page_bm25_norm
    file_count = len(file_ids)
    pool = np.zeros((len(qids), file_count), dtype=np.float64)
    for file_position in range(file_count):
        values = source[:, page_file_positions == file_position]
        if values.shape[1] == 0:
            continue
        ordered = np.sort(values, axis=1)[:, ::-1]
        if config["pool"] == "max":
            pool[:, file_position] = ordered[:, 0]
        elif config["pool"] == "sum_top2":
            pool[:, file_position] = ordered[:, :2].sum(axis=1)
        elif config["pool"] == "coverage":
            limit = min(10, ordered.shape[1])
            pool[:, file_position] = sum(
                1.0 / (10.0 + rank) for rank in range(1, limit + 1)
            ) + ordered[:, 0] * 1e-3
        else:
            raise ValueError(f"Unknown file pool: {config['pool']}")
    file_pool_norm = pool / np.maximum(pool.max(axis=1, keepdims=True), 1e-12)
    file_scores = (
        float(config["direct_weight"]) * file_direct_norm
        + (1.0 - float(config["direct_weight"])) * file_pool_norm
    )
    parent_norm = file_scores / np.maximum(file_scores.max(axis=1, keepdims=True), 1e-12)
    k_files = int(config["k_files"])
    selected = np.argsort(-file_scores, axis=1, kind="stable")[:, :k_files]
    selected_mask = np.zeros((len(qids), file_count), dtype=bool)
    for row_index in range(len(qids)):
        selected_mask[row_index, selected[row_index]] = True
    page_scores = (
        (1.0 - float(config["parent_weight"])) * page_base
        + float(config["parent_weight"]) * parent_norm[:, page_file_positions]
    )
    page_scores[~selected_mask[:, page_file_positions]] = -np.inf
    ranked = np.full((len(qids), min(100, len(page_ids))), -1, dtype=np.int32)
    for row_index in range(len(qids)):
        eligible = np.flatnonzero(np.isfinite(page_scores[row_index]))
        order = eligible[np.argsort(-page_scores[row_index, eligible], kind="stable")]
        ranked[row_index, : min(ranked.shape[1], len(order))] = order[: ranked.shape[1]]
    return ranked


def _run_from_ranked(
    ranked: np.ndarray,
    page_ids: Sequence[str],
    qids: Sequence[str],
    page_text: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row_index, qid in enumerate(qids):
        chunks: list[dict[str, Any]] = []
        for rank, position in enumerate(ranked[row_index], 1):
            if int(position) < 0:
                continue
            page_id = page_ids[int(position)]
            chunks.append(
                {
                    "chunk_id": page_id,
                    "doc_id": page_id,
                    "text": page_text.get(page_id, ""),
                    "score": float(len(ranked[row_index]) - rank),
                    "rank": rank,
                }
            )
        output[qid] = chunks
    return output


def _metric_for_rows(metrics: Mapping[str, Any], positions: Sequence[int]) -> float:
    values = [float(metrics["per_query"][position]["page_recall@10"]) for position in positions]
    return float(np.mean(values)) if values else 0.0


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Physics cascade grid search",
        "",
        "Offline search over cached PDF-inspector BM25 and V-SPLADE scores.",
        "V-SPLADE uses English query vectors against French Physics qrels.",
        "",
        "## OOF selection",
        "",
        "| Fold | Selected config | Test page recall@10 | Test delta vs weighted baseline |",
        "|---:|---|---:|---:|",
    ]
    for fold in report["oof"]["folds"]:
        lines.append(
            f"| {fold['fold']} | `{fold['selected_config_id']}` | "
            f"{fold['test_page_recall@10']:.2%} | {fold['test_delta_pp']:+.2f} pp |"
        )
    lines += [
        "",
        f"OOF page recall@10: **{report['oof']['metrics']['page_recall@10']:.2%}**.",
        f"OOF delta vs weighted alpha=0.7: **{report['oof']['delta_vs_weighted_pp']:+.2f} pp**.",
        "",
        "## Best full-set configurations",
        "",
        "| Config | alpha | source | pool | direct | parent | Kf | Page recall@10 | nDCG@10 | File recall@3 |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["top_full_set"]:
        c = item["config"]
        lines.append(
            f"| `{item['config_id']}` | {c['alpha']:.2f} | {c['source']} | {c['pool']} | "
            f"{c['direct_weight']:.2f} | {c['parent_weight']:.2f} | {c['k_files']} | "
            f"{item['metrics']['page_recall@10']:.2%} | {item['metrics']['ndcg@10']:.2f} | "
            f"{item['metrics']['file_recall@3']:.2%} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- The full-set ranking is exploratory; the OOF selection is the held-out comparison.",
        "- Candidate generation is the same cached full-corpus page scoring used by the existing cascade.",
        "- No qrel modality/evidence labels are features.",
        "- This experiment uses only the first cascade; the cached legacy second-stage reranker is not mixed into these numbers.",
        "",
        "Runs and per-query reports are under `runs/` and `per_query.jsonl`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    page_ids = _page_vector_units(args.page_vector_dir)
    if len(qids) != 302 or len(page_ids) != 1674:
        raise RuntimeError(f"Expected 302 queries and 1,674 pages, got {len(qids)} and {len(page_ids)}")
    page_ids = list(page_ids)
    page_position = {page_id: position for position, page_id in enumerate(page_ids)}
    page_text = {document.doc_id: document.text for document in benchmark.corpus()}

    # Build the same KDL-backed page and file indexes as the existing cascade.
    corpus, page_index, file_indexes, _, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=[]
    )
    visual_scores, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    file_ids = list(corpus.file_order)
    file_position = {file_id: position for position, file_id in enumerate(file_ids)}
    page_file_positions = np.asarray(
        [file_position[page_id.split("#page=", 1)[0]] for page_id in page_ids], dtype=np.int32
    )
    page_bm25_norm = np.zeros((len(qids), len(page_ids)), dtype=np.float64)
    visual_norm = np.zeros_like(page_bm25_norm)
    file_direct_norm = np.zeros((len(qids), len(file_ids)), dtype=np.float64)
    for row_index, question in enumerate(questions):
        bm_scores = _index_scores(page_index, question.query, top_k=len(page_ids))
        bm_norm = normalise_scores(bm_scores)
        for page_id, score in bm_norm.items():
            if page_id in page_position:
                page_bm25_norm[row_index, page_position[page_id]] = score
        visual_map = visual_scores[question.qid]
        positive = [float(value) for value in visual_map.values() if float(value) > 0]
        max_visual = max(positive, default=0.0)
        if max_visual > 0:
            for page_id, score in visual_map.items():
                visual_norm[row_index, page_position[page_id]] = max(0.0, float(score)) / max_visual
        direct_scores = _index_scores(file_indexes["all_text"], question.query, top_k=len(file_ids))
        direct_norm = normalise_scores(direct_scores)
        for file_id, score in direct_norm.items():
            if file_id in file_position:
                file_direct_norm[row_index, file_position[file_id]] = score
        if (row_index + 1) % 100 == 0 or row_index + 1 == len(questions):
            print(f"precompute scores {row_index + 1}/{len(questions)}")

    configs = _make_grid()
    rankings: list[np.ndarray] = []
    metric_rows: list[dict[str, Any]] = []
    extraction_started = time.perf_counter()
    for config in configs:
        ranked = _score_config(
            config,
            qids=qids,
            page_ids=page_ids,
            file_ids=file_ids,
            page_file_positions=page_file_positions,
            page_bm25_norm=page_bm25_norm,
            visual_norm=visual_norm,
            file_direct_norm=file_direct_norm,
        )
        rankings.append(ranked)
        _, metrics = _qrel_metrics(page_ids, file_ids, ranked, qids, qrels)
        metric_rows.append({"config": config, "metrics": metrics})
    extraction_seconds = time.perf_counter() - extraction_started

    baseline_run = _load_run(args.baseline_run)
    baseline_metrics = _derived_metrics(baseline_run, qids, qrels)
    baseline_recall_by_qid = {
        row["qid"]: float(row["page_recall@10"])
        for row in baseline_metrics["per_query"]
    }
    config_ids = {
        index: f"c{index:03d}_a{item['alpha']:.2f}_{item['source']}_{item['pool']}_d{item['direct_weight']:.2f}_p{item['parent_weight']:.2f}_kf{item['k_files']}"
        for index, item in enumerate(configs)
    }
    baseline_weighted_config = {
        "alpha": 0.70,
        "source": "page_base",
        "pool": "max",
        "direct_weight": 0.50,
        "parent_weight": 0.15,
        "k_files": 42,
    }

    # Fixed ordinal folds. Select one grid config on four folds, evaluate on
    # the held-out fold. Tie-break toward higher file recall then lower id.
    folds = {qid: index % 5 for index, qid in enumerate(qids)}
    fold_payload: list[dict[str, Any]] = []
    oof_recall: dict[str, float] = {}
    selected_counts = {config_id: 0 for config_id in config_ids.values()}
    for fold in range(5):
        train_positions = [index for index, qid in enumerate(qids) if folds[qid] != fold]
        test_positions = [index for index, qid in enumerate(qids) if folds[qid] == fold]
        choices: list[tuple[float, float, int]] = []
        for config_index, item in enumerate(metric_rows):
            train_recall = float(np.mean([item["metrics"]["per_query"][pos]["page_recall@10"] for pos in train_positions]))
            train_file = float(np.mean([
                len(
                    set(item["metrics"]["per_query"][pos]["top10_files_from_top100_pages"][:3])
                    & set(item["metrics"]["per_query"][pos]["gold_files"])
                ) / len(item["metrics"]["per_query"][pos]["gold_files"])
                if item["metrics"]["per_query"][pos]["gold_files"] else 0.0
                for pos in train_positions
            ]))
            choices.append((train_recall, train_file, -config_index))
        selected_index = max(range(len(choices)), key=lambda index: choices[index])
        selected_id = config_ids[selected_index]
        selected_counts[selected_id] += 1
        selected_metrics = metric_rows[selected_index]["metrics"]
        test_recall = float(np.mean([selected_metrics["per_query"][pos]["page_recall@10"] for pos in test_positions]))
        baseline_test = float(np.mean([baseline_recall_by_qid[qids[pos]] for pos in test_positions]))
        for pos in test_positions:
            oof_recall[qids[pos]] = float(selected_metrics["per_query"][pos]["page_recall@10"])
        fold_payload.append(
            {
                "fold": fold,
                "train_queries": len(train_positions),
                "test_queries": len(test_positions),
                "selected_config_id": selected_id,
                "selected_config": configs[selected_index],
                "train_page_recall@10": choices[selected_index][0],
                "train_file_recall@3": choices[selected_index][1],
                "test_page_recall@10": test_recall,
                "test_baseline_page_recall@10": baseline_test,
                "test_delta_pp": (test_recall - baseline_test) * 100.0,
                "test_fold_qrels_used_during_selection": False,
            }
        )

    # Build OOF run by selecting the fold winner for each query.
    oof_run: dict[str, list[dict[str, Any]]] = {}
    for fold_row in fold_payload:
        selected_index = next(
            index for index, config_id in config_ids.items()
            if config_id == fold_row["selected_config_id"]
        )
        selected_run = _run_from_ranked(
            rankings[selected_index], page_ids, qids, page_text
        )
        for qid in qids:
            if folds[qid] == fold_row["fold"]:
                oof_run[qid] = selected_run[qid]
    oof_metrics = _derived_metrics(oof_run, qids, qrels)
    oof_baseline_run = {qid: baseline_run[qid] for qid in qids}
    oof_baseline_metrics = _derived_metrics(oof_baseline_run, qids, qrels)

    top_full_set = []
    for index in sorted(
        range(len(metric_rows)),
        key=lambda position: (
            -metric_rows[position]["metrics"]["page_recall@10"],
            -metric_rows[position]["metrics"]["ndcg@10"],
            position,
        ),
    )[:20]:
        item = metric_rows[index]
        top_full_set.append(
            {
                "config_id": config_ids[index],
                "config": item["config"],
                "metrics": {
                    key: value
                    for key, value in item["metrics"].items()
                    if key != "per_query"
                },
            }
        )
    best_index = min(
        range(len(metric_rows)),
        key=lambda index: (
            -metric_rows[index]["metrics"]["page_recall@10"],
            -metric_rows[index]["metrics"]["ndcg@10"],
            index,
        ),
    )
    best_run = _run_from_ranked(rankings[best_index], page_ids, qids, page_text)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(parents=True, exist_ok=True)
    _write_run(
        args.output_dir / "runs" / "oof_selected.jsonl",
        oof_run,
        qids,
        queries={qid: benchmark._by_qid[qid].query for qid in qids},
    )
    _write_run(
        args.output_dir / "runs" / "best_full_set.jsonl",
        best_run,
        qids,
        queries={qid: benchmark._by_qid[qid].query for qid in qids},
    )
    per_query: list[dict[str, Any]] = []
    for qid in qids:
        per_query.append(
            {
                "qid": qid,
                "fold": folds[qid],
                "query": benchmark._by_qid[qid].query,
                "selected_config_id": next(
                    row["selected_config_id"]
                    for row in fold_payload
                    if row["fold"] == folds[qid]
                ),
                "oof_top10_pages": [row["chunk_id"] for row in oof_run[qid][:10]],
                "baseline_top10_pages": [row["chunk_id"] for row in baseline_run[qid][:10]],
                "oof_page_recall@10": next(row["page_recall@10"] for row in oof_metrics["per_query"] if row["qid"] == qid),
                "baseline_page_recall@10": baseline_recall_by_qid[qid],
            }
        )
    report = {
        "experiment": "physics_cascade_grid_search",
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_ids),
        "candidate_policy": "full-corpus page BM25 + full-corpus cached V-SPLADE scores, then hard file selection",
        "grid_size": len(configs),
        "grid": configs,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(args.baseline_run),
            "page_vectors": visual_meta,
        },
        "index_counts": index_counts,
        "baseline_weighted07": {
            "metrics": baseline_metrics,
            "config_reference": "existing weighted alpha=0.7 page fusion; file stage not used",
        },
        "best_full_set": {
            "config_id": config_ids[best_index],
            "config": configs[best_index],
            "metrics": {key: value for key, value in metric_rows[best_index]["metrics"].items() if key != "per_query"},
        },
        "top_full_set": top_full_set,
        "oof": {
            "folds": fold_payload,
            "selected_config_counts": selected_counts,
            "metrics": {key: value for key, value in oof_metrics.items() if key != "per_query"},
            "baseline_metrics": {key: value for key, value in oof_baseline_metrics.items() if key != "per_query"},
            "delta_vs_weighted_pp": (oof_metrics["page_recall@10"] - oof_baseline_metrics["page_recall@10"]) * 100.0,
            "file_recall_delta_vs_weighted_pp": (
                oof_metrics["file_metrics_by_k"]["3"]["file_recall"]
                - oof_baseline_metrics["file_metrics_by_k"]["3"]["file_recall"]
            ) * 100.0,
        },
        "validation": {
            "qids": len(qids),
            "qrels_unreachable": benchmark.unreachable_n,
            "folds_disjoint": True,
            "test_fold_qrels_used_during_selection": False,
            "qrel_modality_features_used": False,
            "no_render_encode_api_or_new_model": True,
            "vsplade_query_language": visual_meta.get("query_language"),
        },
        "timing_seconds": {
            "score_precompute": None,
            "grid_scoring_and_metrics": round(extraction_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment": report["experiment"],
                "grid_size": len(configs),
                "configs": configs,
                "folds": 5,
                "fold_assignment": "sorted qid ordinal modulo five",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "fold_metrics.json").write_text(
        json.dumps(fold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "grid_size": len(configs),
                "best_full_set": report["best_full_set"],
                "oof_metrics": report["oof"]["metrics"],
                "oof_delta_vs_weighted_pp": report["oof"]["delta_vs_weighted_pp"],
                "selected_config_counts": selected_counts,
                "timing_seconds": report["timing_seconds"],
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
