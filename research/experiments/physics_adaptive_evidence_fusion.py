"""Offline adaptive evidence-aware fusion for ViDoRe V3 Physics.

This is a self-contained research experiment.  It reuses the cached French
BM25/PDF-inspector run, the cached English-query V-SPLADE run, and the cached
page/query sparse matrices.  It does not render pages, run models, call an
API, or modify an existing experiment.

The experiment compares the existing alpha=0.7 weighted fusion with:

* an interpretable, fold-fitted evidence-aware rule scorer;
* a small NumPy pairwise linear ranker.

All reported primary comparisons are out-of-fold over five fixed query folds.
Gold labels are used only for fitting on the training folds.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
for _path in (ROOT, EXPERIMENT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.chunking_embedding.lexical import analyze
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3, unit_id
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks
from src.retrieval.sparse import BM25Index
from evaluate_physics_file_level import _evaluate, _load_physics_questions_and_qrels


PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_adaptive_evidence_fusion"

BM25_RUN = PAIR_DIR / "bm25_french_bm25-french_vs-english.jsonl"
VSPLADE_RUN = PAIR_DIR / "vsplade_french_bm25-french_vs-english.jsonl"
FEATURE_NAMES = (
    "bm25_score_raw",
    "bm25_score_norm",
    "bm25_rank_norm",
    "bm25_present",
    "bm25_top10",
    "vsplade_score_raw",
    "vsplade_score_norm",
    "vsplade_rank_norm",
    "vsplade_present",
    "vsplade_top10",
    "lexical_term_coverage",
    "lexical_matched_term_count",
    "lexical_matched_term_fraction",
    "lexical_idf_mass_norm",
    "anchor_coverage",
    "anchor_matched_fraction",
    "exact_anchor_match",
    "page_length_log",
    "page_token_count_log",
    "has_formula_marker",
    "has_table_marker",
    "has_image_marker",
    "has_chart_marker",
    "visual_percentile",
    "visual_shared_dims",
    "visual_shared_dims_log",
    "visual_shared_contribution_norm",
    "visual_query_nnz_log",
    "visual_query_nnz",
    "visual_page_nnz_log",
    "visual_page_nnz",
    "both_top10",
    "both_top100",
    "rank_gap_norm",
    "semantic_rescue_signal",
    "lexical_guard_signal",
    "conflict_signal",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
FORMULA_MARKER = re.compile(
    r"\d|[=+\-*/^_{}[\]()]|commut|équation|theorem|théorème|formule|calcul",
    re.IGNORECASE,
)
TABLE_MARKER = re.compile(r"tableau|table|colonnes?|lignes?|tabular", re.IGNORECASE)
IMAGE_MARKER = re.compile(r"image|figure|illustration|schéma|schema|photo", re.IGNORECASE)
CHART_MARKER = re.compile(r"graphique|graphe|chart|diagramme|courbe|histogramme", re.IGNORECASE)


def _load_csr(path: Path) -> sparse.csr_matrix:
    payload = np.load(path, allow_pickle=True)
    shape = tuple(int(value) for value in payload["shape"])
    return sparse.csr_matrix(
        (
            payload["data"].astype(np.float32, copy=False),
            payload["indices"].astype(np.int32, copy=False),
            payload["indptr"].astype(np.int32, copy=False),
        ),
        shape=shape,
    )


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached run: {path}")
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _load_page_texts(parsed_run: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for document in documents(parsed_run):
        file_name = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            result[unit_id("physics", file_name, page)] = "\n".join(
                str(block.get("text") or "")
                for block in blocks
                if str(block.get("text") or "").strip()
            )
    return result


def _normalise(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {position: 0.0 for position in scores}
    return {position: value / maximum for position, value in scores.items()}


def _positions_from_run(
    run: dict[str, list[dict[str, Any]]], page_position: dict[str, int]
) -> dict[str, list[tuple[int, float]]]:
    result: dict[str, list[tuple[int, float]]] = {}
    for qid, chunks in run.items():
        result[qid] = [
            (page_position[str(chunk["chunk_id"])], float(chunk.get("score", 0.0)))
            for chunk in chunks
            if str(chunk["chunk_id"]) in page_position
        ]
    return result


def _rank_map(hits: list[tuple[int, float]]) -> dict[int, int]:
    return {position: rank for rank, (position, _) in enumerate(hits, 1)}


def _weighted_rank(
    bm25_hits: list[tuple[int, float]],
    visual_hits: list[tuple[int, float]],
    alpha: float = 0.7,
) -> list[tuple[int, float]]:
    bm25 = dict(bm25_hits)
    visual = dict(visual_hits)
    bm25_norm = _normalise(bm25)
    visual_norm = _normalise(visual)
    positions = set(bm25) | set(visual)
    scores = {
        position: alpha * bm25_norm.get(position, 0.0)
        + (1.0 - alpha) * visual_norm.get(position, 0.0)
        for position in positions
    }
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _unique_files(ranking: list[dict[str, Any]], depth: int = 100, limit: int = 10) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for item in ranking[:depth]:
        file_id = _file_id(str(item["chunk_id"]))
        if file_id not in seen:
            seen.add(file_id)
            files.append(file_id)
        if len(files) >= limit:
            break
    return files


def _run_from_rankings(
    rankings: dict[str, list[dict[str, Any]]],
    page_units: list[str],
    page_texts: list[str],
    depth: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, items in rankings.items():
        result[qid] = [
            {
                "chunk_id": page_units[int(item["position"])],
                "doc_id": page_units[int(item["position"])],
                "score": float(item["score"]),
                "rank": rank,
                "text": page_texts[int(item["position"])],
                **({"reason": item["reason"]} if item.get("reason") else {}),
            }
            for rank, item in enumerate(items[:depth], 1)
        ]
    return result


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_percentile(values: np.ndarray, value: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values <= value))


def _query_metadata(benchmark: ViDoreV3, qids: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for qid in qids:
        question = benchmark._by_qid[qid]
        result[qid] = {
            "query": question.query,
            "query_types": list(question.query_types),
            "query_format": question.query_format or "unknown",
            "modalities_posthoc": list(question.modalities),
        }
    return result


def _build_idf(index: BM25Index, page_count: int) -> dict[str, float]:
    return {
        term: math.log(1.0 + (page_count - len(posting) + 0.5) / (len(posting) + 0.5))
        for term, posting in index.postings.items()
    }


def _sparse_row_dict(matrix: sparse.csr_matrix, row: int) -> dict[int, float]:
    vector = matrix.getrow(row)
    return {int(index): float(value) for index, value in zip(vector.indices, vector.data)}


def _build_feature_rows(
    qids: list[str],
    metadata: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    page_units: list[str],
    page_texts: list[str],
    bm25_hits: dict[str, list[tuple[int, float]]],
    visual_hits: dict[str, list[tuple[int, float]]],
    bm25_index: BM25Index,
    page_vectors: sparse.csr_matrix,
    query_vectors: sparse.csr_matrix,
) -> list[dict[str, Any]]:
    page_count = len(page_units)
    idf = _build_idf(bm25_index, page_count)
    idf_values = np.asarray(list(idf.values()), dtype=np.float32)
    rare_cutoff = float(np.percentile(idf_values, 75)) if idf_values.size else 0.0
    visual_matrix = (query_vectors @ page_vectors.T).toarray()
    rows: list[dict[str, Any]] = []
    for query_index, qid in enumerate(qids):
        query_terms = analyze(metadata[qid]["query"])
        unique_terms = list(dict.fromkeys(query_terms))
        term_weights = {term: idf.get(term, max(idf_values, default=1.0)) for term in unique_terms}
        anchor_terms = [
            term for term in unique_terms
            if any(character.isdigit() for character in term)
            or idf.get(term, 0.0) >= rare_cutoff
        ]
        query_vector = _sparse_row_dict(query_vectors, query_index)
        query_nnz = len(query_vector)
        visual_all = visual_matrix[query_index]
        visual_hits_q = visual_hits[qid]
        visual_score_map = dict(visual_hits_q)
        visual_values = np.asarray(list(visual_score_map.values()), dtype=np.float32)
        visual_ranks = _rank_map(visual_hits_q)
        bm25_ranks = _rank_map(bm25_hits[qid])
        bm25_score_map = dict(bm25_hits[qid])
        bm25_norm = _normalise(bm25_score_map)
        visual_norm = _normalise(visual_score_map)
        candidate_positions = sorted(set(bm25_score_map) | set(visual_score_map))
        total_query_idf = sum(term_weights.values()) or 1.0
        for position in candidate_positions:
            tokens = analyze(page_texts[position])
            token_set = set(tokens)
            matched = [term for term in unique_terms if term in token_set]
            matched_idf = sum(term_weights[term] for term in matched)
            matched_anchors = [term for term in anchor_terms if term in token_set]
            page_vector = _sparse_row_dict(page_vectors, position)
            shared = set(query_vector) & set(page_vector)
            shared_contribution = sum(
                query_vector[index] * page_vector[index] for index in shared
            )
            bm_rank = bm25_ranks.get(position, 101)
            vis_rank = visual_ranks.get(position, 101)
            lexical_coverage = matched_idf / total_query_idf
            anchor_coverage = (
                sum(term_weights[term] for term in matched_anchors)
                / (sum(term_weights[term] for term in anchor_terms) or 1.0)
            )
            visual_raw = float(visual_all[position])
            visual_percentile = _safe_percentile(visual_all, visual_raw)
            bm_score = float(bm25_score_map.get(position, 0.0))
            vis_score = float(visual_score_map.get(position, visual_raw))
            both_top10 = float(bm_rank <= 10 and vis_rank <= 10)
            both_top100 = float(bm_rank <= 100 and vis_rank <= 100)
            semantic_rescue = float(visual_percentile >= 0.75 and lexical_coverage <= 0.10)
            lexical_guard = float(
                bool(matched_anchors) and (anchor_coverage >= 0.25 or len(matched_anchors) >= 2)
            )
            conflict = float(
                (bm_rank <= 10 and vis_rank > 50)
                or (vis_rank <= 10 and bm_rank > 50)
            )
            values = [
                bm_score,
                bm25_norm.get(position, 0.0),
                1.0 - min(bm_rank, 100) / 100.0 if bm_rank <= 100 else 0.0,
                float(position in bm25_score_map),
                float(bm_rank <= 10),
                vis_score,
                visual_norm.get(position, 0.0),
                1.0 - min(vis_rank, 100) / 100.0 if vis_rank <= 100 else 0.0,
                float(position in visual_score_map),
                float(vis_rank <= 10),
                lexical_coverage,
                float(len(matched)),
                len(matched) / (len(unique_terms) or 1),
                matched_idf / total_query_idf,
                anchor_coverage,
                len(matched_anchors) / (len(anchor_terms) or 1),
                float(bool(matched_anchors)),
                math.log1p(len(page_texts[position])),
                math.log1p(len(tokens)),
                float(bool(FORMULA_MARKER.search(page_texts[position]))),
                float(bool(TABLE_MARKER.search(page_texts[position]))),
                float(bool(IMAGE_MARKER.search(page_texts[position]))),
                float(bool(CHART_MARKER.search(page_texts[position]))),
                visual_percentile,
                float(len(shared)),
                math.log1p(len(shared)),
                shared_contribution / (float(np.sum(query_vectors.getrow(query_index).data ** 2)) ** 0.5 + 1e-6),
                math.log1p(query_nnz),
                float(query_nnz),
                math.log1p(len(page_vector)),
                float(len(page_vector)),
                both_top10,
                float(both_top100),
                abs(bm_rank - vis_rank) / 100.0 if bm_rank <= 100 and vis_rank <= 100 else 1.0,
                semantic_rescue,
                lexical_guard,
                conflict,
            ]
            rows.append(
                {
                    "qid": qid,
                    "position": position,
                    "page_id": page_units[position],
                    "relevance": int(qrels.get(qid, {}).get(page_units[position], 0)),
                    "query_length": len(unique_terms),
                    "anchor_count": len(anchor_terms),
                    "features": [float(value) for value in values],
                    "signals": {
                        "semantic_rescue": bool(semantic_rescue),
                        "lexical_guard": bool(lexical_guard),
                        "agreement": bool(both_top10),
                        "conflict": bool(conflict),
                        "matched_terms": matched[:30],
                        "matched_anchors": matched_anchors[:30],
                        "bm25_rank": bm_rank if bm_rank <= 100 else None,
                        "vsplade_rank": vis_rank if vis_rank <= 100 else None,
                    },
                }
            )
    return rows


def _rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row["qid"])].append(row)
    return dict(result)


def _rule_score(row: dict[str, Any], params: dict[str, float]) -> tuple[float, str]:
    feature = dict(zip(FEATURE_NAMES, row["features"]))
    lexical_guard = (
        feature["exact_anchor_match"] > 0
        and feature["anchor_coverage"] >= params["guard_threshold"]
    )
    semantic_rescue = (
        feature["visual_percentile"] >= params["visual_threshold"]
        and feature["lexical_term_coverage"] <= params["low_lexical_threshold"]
    )
    if lexical_guard:
        alpha = params["lexical_alpha"]
        reason = "lexical_guard"
    elif semantic_rescue:
        alpha = params["semantic_alpha"]
        reason = "semantic_rescue"
    else:
        alpha = 0.7
        reason = "balanced_default"
    score = (
        alpha * feature["bm25_score_norm"]
        + (1.0 - alpha) * feature["vsplade_score_norm"]
    )
    if feature["both_top10"] > 0:
        score += params["agreement_bonus"]
        reason += "+agreement"
    if semantic_rescue:
        score += params["rescue_bonus"]
    if feature["conflict_signal"] > 0:
        score -= params["conflict_penalty"]
    return float(score), reason


def _rule_rank(
    rows_by_qid: dict[str, list[dict[str, Any]]],
    params: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in rows_by_qid.items():
        ranked: list[dict[str, Any]] = []
        for row in rows:
            score, reason = _rule_score(row, params)
            ranked.append({**row, "score": score, "reason": reason})
        result[qid] = sorted(ranked, key=lambda item: (-item["score"], item["page_id"]))
    return result


def _baseline_rank(
    rows_by_qid: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in rows_by_qid.items():
        ranked = []
        for row in rows:
            feature = dict(zip(FEATURE_NAMES, row["features"]))
            score = (
                0.7 * feature["bm25_score_norm"]
                + 0.3 * feature["vsplade_score_norm"]
            )
            ranked.append({**row, "score": float(score), "reason": "weighted_alpha_0.7"})
        result[qid] = sorted(ranked, key=lambda item: (-item["score"], item["page_id"]))
    return result


def _fit_rule(
    train_qids: list[str],
    rows_by_qid: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
) -> tuple[dict[str, float], dict[str, Any]]:
    grid: list[dict[str, float]] = []
    for guard_threshold in (0.25, 0.50):
        for visual_threshold in (0.75, 0.85):
            for low_lexical_threshold in (0.05, 0.10):
                for lexical_alpha in (0.80, 0.90):
                    for semantic_alpha in (0.30, 0.40):
                        for agreement_bonus in (0.00, 0.03):
                            grid.append(
                                {
                                    "guard_threshold": guard_threshold,
                                    "visual_threshold": visual_threshold,
                                    "low_lexical_threshold": low_lexical_threshold,
                                    "lexical_alpha": lexical_alpha,
                                    "semantic_alpha": semantic_alpha,
                                    "agreement_bonus": agreement_bonus,
                                    "rescue_bonus": 0.02,
                                    "conflict_penalty": 0.0,
                                }
                            )
    best_params = grid[0]
    best_key = (-1.0, -1.0, -1.0)
    for params in grid:
        ranked = _rule_rank(
            {qid: rows_by_qid[qid] for qid in train_qids},
            params,
        )
        run = _run_from_rankings(ranked, PAGE_UNITS, PAGE_TEXTS, depth=100)
        metrics = _evaluate(run, train_qids, qrels)
        key = (
            float(metrics["page_recall@10"]),
            float(metrics["page_hit@10"]),
            float(metrics["ndcg@10"]),
        )
        if key > best_key:
            best_key = key
            best_params = params
    return dict(best_params), {
        "grid_size": len(grid),
        "train_page_recall@10": best_key[0],
        "train_page_hit@10": best_key[1],
        "train_ndcg@10": best_key[2],
    }


def _fit_standardizer(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([row["features"] for row in rows], dtype=np.float64)
    mean = matrix.mean(axis=0) if len(matrix) else np.zeros(len(FEATURE_NAMES))
    std = matrix.std(axis=0) if len(matrix) else np.ones(len(FEATURE_NAMES))
    std[std < 1e-8] = 1.0
    return mean, std


def _fit_linear_ranker(
    train_qids: list[str],
    rows_by_qid: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    max_pairs_per_query: int = 128,
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_rows = [row for qid in train_qids for row in rows_by_qid[qid]]
    mean, std = _fit_standardizer(train_rows)
    rng = np.random.default_rng(seed)
    differences: list[np.ndarray] = []
    pair_counts: dict[str, int] = {}
    for qid in train_qids:
        positives = [row for row in rows_by_qid[qid] if int(row["relevance"]) > 0]
        negatives = [row for row in rows_by_qid[qid] if int(row["relevance"]) <= 0]
        possible = len(positives) * len(negatives)
        if not positives or not negatives:
            pair_counts[qid] = 0
            continue
        pair_count = min(max_pairs_per_query, possible)
        pair_counts[qid] = pair_count
        for _ in range(pair_count):
            positive = positives[int(rng.integers(len(positives)))]
            negative = negatives[int(rng.integers(len(negatives)))]
            left = (np.asarray(positive["features"], dtype=np.float64) - mean) / std
            right = (np.asarray(negative["features"], dtype=np.float64) - mean) / std
            differences.append(left - right)
    if differences:
        pair_matrix = np.asarray(differences, dtype=np.float64)
    else:
        pair_matrix = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(epochs):
        if len(pair_matrix):
            margins = np.clip(pair_matrix @ weights, -40.0, 40.0)
            gradient = -((pair_matrix.T @ (1.0 / (1.0 + np.exp(margins)))) / len(pair_matrix))
        else:
            gradient = np.zeros_like(weights)
        weights -= learning_rate * (gradient + l2 * weights)
    return (
        {
            "weights": weights.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "pair_count": int(len(pair_matrix)),
            "pair_counts_nonzero": int(sum(value > 0 for value in pair_counts.values())),
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "max_pairs_per_query": max_pairs_per_query,
        },
        {
            "weight_by_feature": {
                name: float(value) for name, value in zip(FEATURE_NAMES, weights)
            },
            "pair_counts": pair_counts,
        },
    )


def _linear_rank(
    rows_by_qid: dict[str, list[dict[str, Any]]],
    model: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    weights = np.asarray(model["weights"], dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    std = np.asarray(model["std"], dtype=np.float64)
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in rows_by_qid.items():
        ranked: list[dict[str, Any]] = []
        for row in rows:
            values = (np.asarray(row["features"], dtype=np.float64) - mean) / std
            score = float(values @ weights)
            ranked.append({**row, "score": score, "reason": "linear_evidence_ranker"})
        result[qid] = sorted(ranked, key=lambda item: (-item["score"], item["page_id"]))
    return result


def _metric_summary(
    run: dict[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    metrics = _evaluate(run, qids, qrels)
    rows = metrics.pop("per_query")
    single = [row for row in rows if len(row["gold_pages"]) == 1]
    multi = [row for row in rows if len(row["gold_pages"]) > 1]
    for label, subset in (("single_evidence", single), ("multi_evidence", multi)):
        metrics[f"{label}_queries"] = len(subset)
        metrics[f"{label}_page_recall@10"] = (
            sum(row["page_recall@10"] for row in subset) / len(subset)
            if subset else 0.0
        )
        metrics[f"{label}_page_hit@10"] = (
            sum(float(row["page_hit@10"]) for row in subset) / len(subset)
            if subset else 0.0
        )
    return metrics


def _bootstrap_delta(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    samples: int = 10000,
) -> dict[str, float]:
    candidate = np.asarray([row["page_recall@10"] for row in candidate_rows], dtype=np.float64)
    baseline = np.asarray([row["page_recall@10"] for row in baseline_rows], dtype=np.float64)
    delta = candidate - baseline
    if not len(delta):
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    sample_means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        indexes = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        sample_means[start:stop] = delta[indexes].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(np.quantile(sample_means, 0.025)),
        "ci95_high": float(np.quantile(sample_means, 0.975)),
    }


def _validate_inputs(
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    bm25_run: dict[str, list[dict[str, Any]]],
    visual_run: dict[str, list[dict[str, Any]]],
    page_units: list[str],
    page_vectors: sparse.csr_matrix,
    query_vectors: sparse.csr_matrix,
    page_metadata: list[dict[str, Any]],
) -> None:
    if len(qids) != 302:
        raise RuntimeError(f"Expected 302 French queries, found {len(qids)}")
    expected = set(qids)
    for name, run in (("BM25", bm25_run), ("V-SPLADE", visual_run)):
        if set(run) != expected:
            raise RuntimeError(
                f"{name} qid mismatch: missing={sorted(expected - set(run))[:3]}, "
                f"extra={sorted(set(run) - expected)[:3]}"
            )
        for qid in qids:
            chunks = run[qid]
            ids = [str(chunk["chunk_id"]) for chunk in chunks]
            if len(chunks) != 100 or len(set(ids)) != 100:
                raise RuntimeError(f"{name} does not have 100 unique candidates for {qid}")
    if page_vectors.shape != (1674, 50368):
        raise RuntimeError(f"Expected page vector shape (1674, 50368), got {page_vectors.shape}")
    if query_vectors.shape != (302, 50368):
        raise RuntimeError(f"Expected query vector shape (302, 50368), got {query_vectors.shape}")
    if len(page_units) != 1674 or len(page_metadata) != 1674:
        raise RuntimeError("Page vector metadata does not describe all 1,674 Physics pages")
    if len(set(page_units)) != len(page_units):
        raise RuntimeError("Page vector to page ID mapping contains duplicates")
    reachable = set(page_units)
    unreachable = sorted(
        page for qid in qids for page in qrels.get(qid, {}) if page not in reachable
    )
    if unreachable:
        raise RuntimeError(f"Found unreachable qrel pages, examples: {unreachable[:5]}")


def _folds(qids: list[str], fold_count: int = 5) -> dict[str, int]:
    ordered = sorted(qids, key=lambda qid: int(qid.rsplit("::", 1)[1]))
    result = {qid: index % fold_count for index, qid in enumerate(ordered)}
    if len(set(result)) != len(qids):
        raise RuntimeError("Fold assignment duplicated a qid")
    for left in range(fold_count):
        for right in range(left + 1, fold_count):
            if set(qid for qid, fold in result.items() if fold == left) & set(
                qid for qid, fold in result.items() if fold == right
            ):
                raise RuntimeError("Fold assignment overlaps query IDs")
    return result


def _run_query_metrics(
    run: dict[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, dict[str, Any]]:
    payload = _evaluate(run, qids, qrels)
    return {row["qid"]: row for row in payload["per_query"]}


def _top_pages(
    run: dict[str, list[dict[str, Any]]],
    qid: str,
    depth: int = 10,
) -> list[str]:
    return [str(item["chunk_id"]) for item in run[qid][:depth]]


def _posthoc_groups(
    qids: list[str],
    metadata: dict[str, dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    runs: dict[str, dict[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    for qid in qids:
        groups["query_format=" + metadata[qid]["query_format"]].append(qid)
        for query_type in metadata[qid]["query_types"] or ["unknown"]:
            groups["query_type=" + query_type].append(qid)
        groups["gold_pages=" + ("single" if len(qrels.get(qid, {})) == 1 else "multi")].append(qid)
    output: dict[str, Any] = {}
    for group, group_qids in sorted(groups.items()):
        output[group] = {"queries": len(group_qids), "methods": {}}
        for method, run in runs.items():
            metrics = _metric_summary(run, group_qids, qrels)
            output[group]["methods"][method] = {
                "page_recall@10": metrics["page_recall@10"],
                "page_hit@10": metrics["page_hit@10"],
                "ndcg@10": metrics["ndcg@10"],
            }
    return output


def _comparison_rows(
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    metadata: dict[str, dict[str, Any]],
    rows_by_qid: dict[str, list[dict[str, Any]]],
    bm25_run: dict[str, list[dict[str, Any]]],
    visual_run: dict[str, list[dict[str, Any]]],
    baseline_run: dict[str, list[dict[str, Any]]],
    rule_run: dict[str, list[dict[str, Any]]],
    linear_run: dict[str, list[dict[str, Any]]],
    folds: dict[str, int],
) -> list[dict[str, Any]]:
    methods = {
        "baseline_weighted07": baseline_run,
        "adaptive_rule": rule_run,
        "linear_ltr": linear_run,
    }
    method_metrics = {
        name: _run_query_metrics(run, qids, qrels) for name, run in methods.items()
    }
    result: list[dict[str, Any]] = []
    for qid in qids:
        gold_pages = sorted(qrels.get(qid, {}))
        gold_files = sorted({_file_id(page) for page in gold_pages})
        feature_by_page = {row["page_id"]: row for row in rows_by_qid[qid]}
        method_payload: dict[str, Any] = {}
        for name, run in methods.items():
            top10 = _top_pages(run, qid, 10)
            metric = method_metrics[name][qid]
            method_payload[name] = {
                "top10_pages": top10,
                "top3_files": _unique_files(run[qid], 100, 3),
                "page_hit@10": metric["page_hit@10"],
                "page_recall@10": metric["page_recall@10"],
                "page_precision@10": metric["page_precision@10"],
                "file_recall@3": (
                    len(set(_unique_files(run[qid], 100, 3)) & set(gold_files))
                    / len(gold_files)
                    if gold_files else 0.0
                ),
                "top10_feature_summary": [
                    {
                        "page_id": str(item["chunk_id"]),
                        "score": float(item["score"]),
                        "reason": item.get("reason", ""),
                        "features": dict(zip(
                            FEATURE_NAMES,
                            feature_by_page[str(item["chunk_id"])]["features"],
                        )),
                        "signals": feature_by_page[str(item["chunk_id"])]["signals"],
                    }
                    for item in run[qid][:10]
                ],
                "promoted_reasons": [
                    item.get("reason", "")
                    for item in run[qid][:10]
                    if item.get("reason")
                ],
            }
        bm25_top = set(_top_pages(bm25_run, qid))
        visual_top = set(_top_pages(visual_run, qid))
        gold = set(gold_pages)
        class_name = (
            "both-hit" if bm25_top & gold and visual_top & gold
            else "BM25-only" if bm25_top & gold
            else "V-SPLADE-only" if visual_top & gold
            else "both-miss"
        )
        baseline_top = set(method_payload["baseline_weighted07"]["top10_pages"])
        rule_top = set(method_payload["adaptive_rule"]["top10_pages"])
        rescue_pages = [
            row["page_id"] for row in rows_by_qid[qid]
            if row["signals"]["semantic_rescue"]
            and row["page_id"] in rule_top
        ]
        successful_rescue = sorted(set(rescue_pages) & gold - baseline_top)
        wrong_rescue = sorted(set(rescue_pages) - gold)
        pushed_guard = [
            row["page_id"] for row in rows_by_qid[qid]
            if row["signals"]["lexical_guard"]
            and row["page_id"] in baseline_top
            and row["page_id"] not in rule_top
            and row["page_id"] in gold
        ]
        result.append(
            {
                "qid": qid,
                "fold": folds[qid],
                "query": metadata[qid]["query"],
                "query_types": metadata[qid]["query_types"],
                "query_format": metadata[qid]["query_format"],
                "modalities_posthoc": metadata[qid]["modalities_posthoc"],
                "gold_pages": gold_pages,
                "gold_files": gold_files,
                "retriever_page_hit_class": class_name,
                "methods": method_payload,
                "semantic_rescue_pages_in_rule_top10": sorted(set(rescue_pages)),
                "semantic_rescue_success_pages": successful_rescue,
                "semantic_rescue_wrong_pages": wrong_rescue,
                "lexical_guard_gold_pages_pushed_down": pushed_guard,
                "candidate_count": len(rows_by_qid[qid]),
                "candidate_union_ceiling_page_hit@10": bool(
                    set(_top_pages(bm25_run, qid)) | set(_top_pages(visual_run, qid))
                ) and bool((set(_top_pages(bm25_run, qid)) | set(_top_pages(visual_run, qid))) & gold),
            }
        )
    return result


def _examples(rows: list[dict[str, Any]], method: str, limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["methods"][method]["page_recall@10"]
            - row["methods"]["baseline_weighted07"]["page_recall@10"],
            row["qid"],
        ),
    )
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "qid": row["qid"],
            "query": row["query"],
            "delta_page_recall@10": row["methods"][method]["page_recall@10"]
            - row["methods"]["baseline_weighted07"]["page_recall@10"],
            "gold_pages": row["gold_pages"],
            "top10_pages": row["methods"][method]["top10_pages"],
            "semantic_rescue_success_pages": row["semantic_rescue_success_pages"],
            "semantic_rescue_wrong_pages": row["semantic_rescue_wrong_pages"],
        }
    return {
        "largest_losses": [compact(row) for row in ordered[:limit]],
        "largest_gains": [compact(row) for row in ordered[-limit:][::-1]],
    }


def _format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report_markdown(
    path: Path,
    report: dict[str, Any],
    comparisons: list[dict[str, Any]],
) -> None:
    target_status = report["success_criterion"]["status"]
    lines = [
        "# Physics adaptive evidence-aware fusion",
        "",
        "Offline five-fold query cross-validation over 302 French Physics queries.",
        "The V-SPLADE leg uses cached English query vectors; this language mismatch "
        "is retained as an explicit experimental confound.",
        "",
        "## Main OOF metrics",
        "",
        "| Method | nDCG@10 | page hit@10 | page recall@10 | page precision@10 | file recall@3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, metrics in report["oof_metrics"].items():
        lines.append(
            f"| {method} | {metrics['ndcg@10']:.2f} | "
            f"{_format_pct(metrics['page_hit@10'])} | "
            f"{_format_pct(metrics['page_recall@10'])} | "
            f"{_format_pct(metrics['page_precision@10'])} | "
            f"{_format_pct(metrics['file_metrics_by_k']['3']['file_recall'])} |"
        )
    lines += [
        "",
        "## Delta versus weighted alpha=0.7",
        "",
        "| Method | delta page recall@10 | bootstrap 95% CI | improved queries | degraded queries |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, delta in report["deltas_vs_baseline"].items():
        lines.append(
            f"| {method} | {100.0 * delta['mean']:.2f} pp | "
            f"[{100.0 * delta['ci95_low']:.2f}, {100.0 * delta['ci95_high']:.2f}] pp | "
            f"{delta['improved_queries']} | {delta['degraded_queries']} |"
        )
    lines += [
        "",
        f"## Success criterion: {target_status}",
        "",
        f"Target: page recall@10 at least {report['success_criterion']['target_percent']:.2f}%. "
        f"Baseline: {report['success_criterion']['baseline_percent']:.2f}%. "
        f"Required delta: +{report['success_criterion']['required_delta_pp']:.2f} pp.",
        "",
        f"Adaptive rule delta: {report['success_criterion']['adaptive_rule_delta_pp']:.2f} pp. "
        f"Linear ranker delta: {report['success_criterion']['linear_ltr_delta_pp']:.2f} pp.",
        "",
        "## Fold metrics",
        "",
        "| Fold | Method | nDCG@10 | page recall@10 | file recall@3 |",
        "|---:|---|---:|---:|---:|",
    ]
    for fold in report["fold_metrics"]:
        for method, metrics in fold["methods"].items():
            lines.append(
                f"| {fold['fold']} | {method} | {metrics['ndcg@10']:.2f} | "
                f"{_format_pct(metrics['page_recall@10'])} | "
                f"{_format_pct(metrics['file_metrics_by_k']['3']['file_recall'])} |"
            )
    lines += [
        "",
        "## Query groups",
        "",
        "| Group | Queries | Baseline recall | Adaptive recall | Linear recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, payload in report["posthoc_groups"].items():
        methods = payload["methods"]
        lines.append(
            f"| {group} | {payload['queries']} | "
            f"{_format_pct(methods['baseline_weighted07']['page_recall@10'])} | "
            f"{_format_pct(methods['adaptive_rule']['page_recall@10'])} | "
            f"{_format_pct(methods['linear_ltr']['page_recall@10'])} |"
        )
    lines += [
        "",
        "## Evidence diagnostics",
        "",
        f"- Semantic rescue pages in adaptive top-10: {report['diagnostics']['semantic_rescue_pages_in_top10']}.",
        f"- Queries with successful semantic rescue: {report['diagnostics']['queries_with_successful_semantic_rescue']}.",
        f"- Queries with wrong semantic rescue: {report['diagnostics']['queries_with_wrong_semantic_rescue']}.",
        f"- Queries where a gold lexical-guard page was pushed down: {report['diagnostics']['queries_with_guard_pushdown']}.",
        f"- Queries where both original retrievers missed the gold page in top-10: {report['diagnostics']['both_miss_queries']}.",
        f"- Candidate-union page-hit ceiling: {_format_pct(report['diagnostics']['candidate_union_page_hit@10'])}.",
        "",
        "## Interpretation examples",
        "",
        "The full per-query comparison is in per_query.jsonl. The JSON report also "
        "contains the largest gains and losses for both learned methods.",
        "",
        "## Reproducibility",
        "",
        "Folds use sorted ordinal qids modulo five. The linear ranker uses NumPy "
        "pairwise logistic ranking with seed 42 plus the fold number. No test-fold "
        "qrels are used to fit rule thresholds or linear weights.",
        "",
        "Timing is separated into artifact loading, feature extraction, rule fitting, "
        "linear fitting, out-of-fold ranking, and report writing.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parsed-run", type=Path, default=PARSED_RUN)
    parser.add_argument("--page-vector-dir", type=Path, default=PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=QUERY_VECTOR_DIR)
    parser.add_argument("--pair-dir", type=Path, default=PAIR_DIR)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "data/benchmark/vidore_v3")
    args = parser.parse_args()
    global PAGE_UNITS, PAGE_TEXTS

    output_dir = args.output_dir
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    load_started = time.perf_counter()
    qids, qrels = _load_physics_questions_and_qrels()
    qids = sorted(qids, key=lambda qid: int(qid.rsplit("::", 1)[1]))
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    metadata = _query_metadata(benchmark, qids)
    bm25_path = args.pair_dir / BM25_RUN.name
    visual_path = args.pair_dir / VSPLADE_RUN.name
    bm25_run = _load_run(bm25_path)
    visual_run = _load_run(visual_path)
    page_vectors = _load_csr(args.page_vector_dir / "page_vectors.npz")
    query_vectors = _load_csr(args.query_vector_dir / "query_vectors.npz")
    page_metadata = json.loads(
        (args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8")
    )
    PAGE_UNITS = [str(row["unit_id"]) for row in page_metadata]
    PAGE_TEXTS_BY_UNIT = _load_page_texts(args.parsed_run)
    PAGE_TEXTS = [PAGE_TEXTS_BY_UNIT.get(unit, "") for unit in PAGE_UNITS]
    page_position = {unit: index for index, unit in enumerate(PAGE_UNITS)}
    bm25_positions = _positions_from_run(bm25_run, page_position)
    visual_positions = _positions_from_run(visual_run, page_position)
    timings["load_cached_artifacts"] = time.perf_counter() - load_started

    _validate_inputs(
        qids,
        qrels,
        bm25_run,
        visual_run,
        PAGE_UNITS,
        page_vectors,
        query_vectors,
        page_metadata,
    )
    if any(not text.strip() for text in PAGE_TEXTS):
        missing = sum(not text.strip() for text in PAGE_TEXTS)
        raise RuntimeError(f"Missing PDF-inspector text for {missing} pages")

    index_started = time.perf_counter()
    bm25_index = BM25Index(analyzer_name="plain").build(
        [
            {"chunk_id": page, "doc_id": page, "text": text}
            for page, text in zip(PAGE_UNITS, PAGE_TEXTS)
        ]
    )
    timings["build_bm25_index"] = time.perf_counter() - index_started

    feature_started = time.perf_counter()
    rows = _build_feature_rows(
        qids,
        metadata,
        qrels,
        PAGE_UNITS,
        PAGE_TEXTS,
        bm25_positions,
        visual_positions,
        bm25_index,
        page_vectors,
        query_vectors,
    )
    rows_by_qid = _rows_by_query(rows)
    timings["feature_extraction"] = time.perf_counter() - feature_started
    if set(rows_by_qid) != set(qids) or any(not rows_by_qid[qid] for qid in qids):
        raise RuntimeError("Feature extraction did not produce candidates for every qid")

    feature_export = [
        {
            "qid": row["qid"],
            "page_id": row["page_id"],
            "features": row["features"],
            "signals": row["signals"],
        }
        for row in rows
    ]
    _write_jsonl(output_dir / "features.jsonl", feature_export)

    fold_by_qid = _folds(qids)
    fold_metrics: list[dict[str, Any]] = []
    baseline_oof: dict[str, list[dict[str, Any]]] = {}
    rule_oof: dict[str, list[dict[str, Any]]] = {}
    linear_oof: dict[str, list[dict[str, Any]]] = {}
    rule_models: dict[str, Any] = {}
    linear_models: dict[str, Any] = {}
    baseline_rankings = _baseline_rank(rows_by_qid)

    rule_fit_seconds = 0.0
    linear_fit_seconds = 0.0
    oof_rank_seconds = 0.0
    for fold in range(5):
        test_qids = [qid for qid in qids if fold_by_qid[qid] == fold]
        train_qids = [qid for qid in qids if fold_by_qid[qid] != fold]
        rule_started = time.perf_counter()
        rule_params, rule_fit_info = _fit_rule(train_qids, rows_by_qid, qrels)
        rule_fit_seconds += time.perf_counter() - rule_started
        rule_models[str(fold)] = {"params": rule_params, "fit": rule_fit_info}
        linear_started = time.perf_counter()
        linear_model, linear_info = _fit_linear_ranker(
            train_qids,
            rows_by_qid,
            seed=42 + fold,
        )
        linear_fit_seconds += time.perf_counter() - linear_started
        linear_models[str(fold)] = {**linear_model, "fit_info": linear_info}
        rank_started = time.perf_counter()
        baseline_rank_test = {qid: baseline_rankings[qid] for qid in test_qids}
        rule_rank_test = _rule_rank(
            {qid: rows_by_qid[qid] for qid in test_qids},
            rule_params,
        )
        linear_rank_test = _linear_rank(
            {qid: rows_by_qid[qid] for qid in test_qids},
            linear_model,
        )
        baseline_oof.update(baseline_rank_test)
        rule_oof.update(rule_rank_test)
        linear_oof.update(linear_rank_test)
        oof_rank_seconds += time.perf_counter() - rank_started
        baseline_test_run = _run_from_rankings(baseline_rank_test, PAGE_UNITS, PAGE_TEXTS)
        rule_test_run = _run_from_rankings(rule_rank_test, PAGE_UNITS, PAGE_TEXTS)
        linear_test_run = _run_from_rankings(linear_rank_test, PAGE_UNITS, PAGE_TEXTS)
        fold_method_metrics = {
            "baseline_weighted07": _metric_summary(baseline_test_run, test_qids, qrels),
            "adaptive_rule": _metric_summary(rule_test_run, test_qids, qrels),
            "linear_ltr": _metric_summary(linear_test_run, test_qids, qrels),
        }
        fold_deltas = {
            method: {
                "page_recall@10": (
                    fold_method_metrics[method]["page_recall@10"]
                    - fold_method_metrics["baseline_weighted07"]["page_recall@10"]
                ),
                "ndcg@10": (
                    fold_method_metrics[method]["ndcg@10"]
                    - fold_method_metrics["baseline_weighted07"]["ndcg@10"]
                ),
                "file_recall@3": (
                    fold_method_metrics[method]["file_metrics_by_k"]["3"]["file_recall"]
                    - fold_method_metrics["baseline_weighted07"]["file_metrics_by_k"]["3"]["file_recall"]
                ),
            }
            for method in ("adaptive_rule", "linear_ltr")
        }
        fold_metrics.append(
            {
                "fold": fold,
                "train_queries": len(train_qids),
                "test_queries": len(test_qids),
                "test_qids": test_qids,
                "methods": fold_method_metrics,
                "delta_vs_baseline": fold_deltas,
                "rule_fit": rule_models[str(fold)],
                "linear_fit": {
                    "pair_count": linear_model["pair_count"],
                    "weight_by_feature": linear_info["weight_by_feature"],
                },
                "test_fold_qrels_used_during_fit": False,
            }
        )
    timings["fit_rule_parameters"] = rule_fit_seconds
    timings["fit_linear_rankers"] = linear_fit_seconds
    timings["out_of_fold_ranking"] = oof_rank_seconds

    baseline_oof_run = _run_from_rankings(baseline_oof, PAGE_UNITS, PAGE_TEXTS)
    rule_oof_run = _run_from_rankings(rule_oof, PAGE_UNITS, PAGE_TEXTS)
    linear_oof_run = _run_from_rankings(linear_oof, PAGE_UNITS, PAGE_TEXTS)
    _write_jsonl(runs_dir / "baseline_weighted07_oof.jsonl", [
        {"qid": qid, "chunks": chunks} for qid, chunks in baseline_oof_run.items()
    ])
    _write_jsonl(runs_dir / "adaptive_rule_oof.jsonl", [
        {"qid": qid, "chunks": chunks} for qid, chunks in rule_oof_run.items()
    ])
    _write_jsonl(runs_dir / "linear_ltr_oof.jsonl", [
        {"qid": qid, "chunks": chunks} for qid, chunks in linear_oof_run.items()
    ])

    full_rule_started = time.perf_counter()
    full_rule_params, full_rule_fit_info = _fit_rule(qids, rows_by_qid, qrels)
    full_rule_rank = _rule_rank(rows_by_qid, full_rule_params)
    full_rule_run = _run_from_rankings(full_rule_rank, PAGE_UNITS, PAGE_TEXTS)
    _write_jsonl(runs_dir / "adaptive_full_fit.jsonl", [
        {"qid": qid, "chunks": chunks} for qid, chunks in full_rule_run.items()
    ])
    timings["full_fit_for_qualitative_reading"] = time.perf_counter() - full_rule_started

    eval_started = time.perf_counter()
    oof_runs = {
        "baseline_weighted07": baseline_oof_run,
        "adaptive_rule": rule_oof_run,
        "linear_ltr": linear_oof_run,
    }
    oof_metrics = {
        method: _metric_summary(run, qids, qrels)
        for method, run in oof_runs.items()
    }
    comparison_rows = _comparison_rows(
        qids,
        qrels,
        metadata,
        rows_by_qid,
        bm25_run,
        visual_run,
        baseline_oof_run,
        rule_oof_run,
        linear_oof_run,
        fold_by_qid,
    )
    comparison_by_qid = {row["qid"]: row for row in comparison_rows}
    baseline_query_metrics = _run_query_metrics(baseline_oof_run, qids, qrels)
    deltas: dict[str, Any] = {}
    for method in ("adaptive_rule", "linear_ltr"):
        candidate_metrics = _run_query_metrics(oof_runs[method], qids, qrels)
        candidate_rows = [candidate_metrics[qid] for qid in qids]
        base_rows = [baseline_query_metrics[qid] for qid in qids]
        delta = _bootstrap_delta(candidate_rows, base_rows)
        delta["improved_queries"] = int(sum(
            candidate_metrics[qid]["page_recall@10"] > baseline_query_metrics[qid]["page_recall@10"]
            for qid in qids
        ))
        delta["degraded_queries"] = int(sum(
            candidate_metrics[qid]["page_recall@10"] < baseline_query_metrics[qid]["page_recall@10"]
            for qid in qids
        ))
        delta["unchanged_queries"] = len(qids) - delta["improved_queries"] - delta["degraded_queries"]
        deltas[method] = delta
    timings["evaluation"] = time.perf_counter() - eval_started

    posthoc_groups = _posthoc_groups(qids, metadata, qrels, oof_runs)
    semantic_rescue_pages = sum(
        len(row["semantic_rescue_pages_in_rule_top10"]) for row in comparison_rows
    )
    successful_rescue_queries = sum(bool(row["semantic_rescue_success_pages"]) for row in comparison_rows)
    wrong_rescue_queries = sum(bool(row["semantic_rescue_wrong_pages"]) for row in comparison_rows)
    guard_pushdown_queries = sum(bool(row["lexical_guard_gold_pages_pushed_down"]) for row in comparison_rows)
    both_miss_queries = sum(row["retriever_page_hit_class"] == "both-miss" for row in comparison_rows)
    union_hit = sum(row["candidate_union_ceiling_page_hit@10"] for row in comparison_rows) / len(qids)
    report = {
        "experiment": "physics_adaptive_evidence_fusion",
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(PAGE_UNITS),
        "candidate_depth_per_retriever": 100,
        "candidate_pool": "union(top-100 BM25, top-100 V-SPLADE)",
        "primary_metric": "page_recall@10",
        "baseline": {
            "method": "weighted fusion",
            "alpha": 0.7,
            "oof_page_recall@10": oof_metrics["baseline_weighted07"]["page_recall@10"],
            "expected_reference_percent": 41.28,
        },
        "oof_metrics": oof_metrics,
        "deltas_vs_baseline": deltas,
        "success_criterion": {
            "definition": "absolute percentage-point improvement in OOF page_recall@10",
            "baseline_percent": 100.0 * oof_metrics["baseline_weighted07"]["page_recall@10"],
            "target_percent": 100.0 * oof_metrics["baseline_weighted07"]["page_recall@10"] + 5.0,
            "required_delta_pp": 5.0,
            "adaptive_rule_delta_pp": 100.0 * deltas["adaptive_rule"]["mean"],
            "linear_ltr_delta_pp": 100.0 * deltas["linear_ltr"]["mean"],
            "status": (
                "PASS"
                if max(deltas["adaptive_rule"]["mean"], deltas["linear_ltr"]["mean"]) >= 0.05
                else "FAIL"
            ),
        },
        "fold_metrics": fold_metrics,
        "posthoc_groups": posthoc_groups,
        "diagnostics": {
            "semantic_rescue_pages_in_top10": semantic_rescue_pages,
            "queries_with_successful_semantic_rescue": successful_rescue_queries,
            "queries_with_wrong_semantic_rescue": wrong_rescue_queries,
            "queries_with_guard_pushdown": guard_pushdown_queries,
            "both_miss_queries": both_miss_queries,
            "candidate_union_page_hit@10": union_hit,
        },
        "examples": {
            "adaptive_rule": _examples(comparison_rows, "adaptive_rule"),
            "linear_ltr": _examples(comparison_rows, "linear_ltr"),
        },
        "full_fit_qualitative": {
            "adaptive_rule_metrics": _metric_summary(full_rule_run, qids, qrels),
            "adaptive_rule_params": full_rule_params,
            "adaptive_rule_fit": full_rule_fit_info,
            "linear_ltr_fit_by_fold": linear_models,
        },
        "timing_seconds": {
            **timings,
            "total": time.perf_counter() - total_started,
        },
        "validation": {
            "qids": len(qids),
            "each_input_run_has_100_candidates": True,
            "page_vector_shape": list(page_vectors.shape),
            "query_vector_shape": list(query_vectors.shape),
            "unique_page_vector_ids": len(set(PAGE_UNITS)) == len(PAGE_UNITS),
            "all_qrel_pages_reachable": True,
            "folds_disjoint": True,
            "test_fold_qrels_used_during_fit": False,
            "features_use_gold_labels": False,
            "vsplade_query_language": "english",
            "vsplade_language_confounded_with_french_evaluation": True,
        },
        "artifacts": {
            "bm25_run": str(bm25_path),
            "vsplade_run": str(visual_path),
            "page_vectors": str(args.page_vector_dir / "page_vectors.npz"),
            "query_vectors": str(args.query_vector_dir / "query_vectors.npz"),
            "parsed_page_text": str(args.parsed_run),
        },
    }
    report_started = time.perf_counter()
    _write_json(output_dir / "config.json", {
        "experiment": report["experiment"],
        "seed": 42,
        "folds": 5,
        "alpha": 0.7,
        "candidate_depth": 100,
        "primary_metric": "page_recall@10",
        "feature_names": list(FEATURE_NAMES),
        "artifacts": report["artifacts"],
        "no_render_encode_api_or_new_model": True,
    })
    _write_json(output_dir / "fold_metrics.json", fold_metrics)
    _write_jsonl(output_dir / "per_query.jsonl", comparison_rows)
    _write_json(output_dir / "report.json", report)
    _write_report_markdown(output_dir / "report.md", report, comparison_rows)
    timings["report_writing"] = time.perf_counter() - report_started
    report["timing_seconds"] = {**timings, "total": time.perf_counter() - total_started}
    _write_json(output_dir / "report.json", report)
    print(json.dumps({
        "status": report["success_criterion"]["status"],
        "baseline_page_recall@10": oof_metrics["baseline_weighted07"]["page_recall@10"],
        "adaptive_rule_page_recall@10": oof_metrics["adaptive_rule"]["page_recall@10"],
        "linear_ltr_page_recall@10": oof_metrics["linear_ltr"]["page_recall@10"],
        "adaptive_rule_delta_pp": report["success_criterion"]["adaptive_rule_delta_pp"],
        "linear_ltr_delta_pp": report["success_criterion"]["linear_ltr_delta_pp"],
        "total_seconds": report["timing_seconds"]["total"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
