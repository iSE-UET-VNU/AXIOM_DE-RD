"""Fold-safe page ranking over a union of lightweight index candidates.

This experiment treats the cached retrieval runs as document-level/zone-level
skip indexes.  Their top-100 union is the candidate set, and a small linear
pairwise ranker uses only cheap page signals to order that set:

* reciprocal rank, score and presence from the lexical, visual and hierarchy
  streams;
* query-term coverage and IDF mass from the existing PDF-inspector text;
* formula/table/image markers and page length.

The candidate union is fixed before fitting.  Gold labels are used only on the
four training folds; the reported result is out-of-fold.  This is deliberately
one predeclared model, not a benchmark-wide parameter sweep.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (ROOT, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from physics_adaptive_evidence_fusion import (  # noqa: E402
    _build_idf,
    _load_csr,
    _load_page_texts,
    _load_run,
    _positions_from_run,
    _sparse_row_dict,
)
from physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _paired_comparison,
    _write_run,
)
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_union_feature_ranker"
PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
RESULT_ROOT = ROOT / "data/benchmark/vidore_v3/results"

STREAM_PATHS = {
    "bm25": PAIR_DIR / "bm25_french_bm25-french_vs-english.jsonl",
    "splade": PAIR_DIR / "vsplade_french_bm25-french_vs-english.jsonl",
    "weighted": PAIR_DIR / "weighted_french_bm25-french_vs-english.jsonl",
    "hierarchical": RESULT_ROOT / "physics_hierarchical_retrieval/oof_run.jsonl",
    "c014": RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
    "cross_kf": RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
    "legacy_second": RESULT_ROOT / "physics_legacy_second_retrieval/oof_run.jsonl",
}
STREAMS = tuple(STREAM_PATHS)
STREAM_FEATURES = tuple(
    feature
    for stream in STREAMS
    for feature in (f"{stream}_score", f"{stream}_rank", f"{stream}_present")
)
FEATURE_NAMES = STREAM_FEATURES + (
    "stream_count",
    "top10_stream_count",
    "rank_agreement",
    "lexical_coverage",
    "lexical_idf_mass",
    "matched_term_count",
    "exact_anchor_match",
    "formula_marker",
    "table_marker",
    "image_marker",
    "page_length_log",
    "page_position_norm",
)

FORMULA_MARKER = re.compile(
    r"\d|[=+\-*/^_{}\[\]()]|équation|theoreme|théorème|formule|calcul",
    re.IGNORECASE,
)
TABLE_MARKER = re.compile(r"tableau|table|colonnes?|lignes?|tabular", re.IGNORECASE)
IMAGE_MARKER = re.compile(r"image|figure|illustration|schéma|schema|photo", re.IGNORECASE)


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    positive = [value for value in values.values() if value > 0.0]
    maximum = max(positive, default=0.0)
    return {
        key: max(0.0, float(value)) / maximum if maximum else 0.0
        for key, value in values.items()
    }


def _rank_and_scores(items: Sequence[dict[str, Any]]) -> tuple[dict[str, int], dict[str, float]]:
    ranks: dict[str, int] = {}
    scores: dict[str, float] = {}
    for rank, item in enumerate(items[:100], 1):
        page = str(item["chunk_id"])
        ranks[page] = rank
        scores[page] = float(item.get("score", 0.0))
    return ranks, scores


def _page_number(page_id: str) -> int:
    try:
        return int(page_id.rsplit("#page=", 1)[1])
    except (IndexError, ValueError):
        return 0


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _candidate_pages(
    qid: str,
    loaded: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> list[str]:
    pages: set[str] = set()
    for stream in STREAMS:
        pages.update(str(item["chunk_id"]) for item in loaded[stream][qid][:100])
    return sorted(pages)


def _build_rows(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    loaded: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    page_units: Sequence[str],
    page_texts: Sequence[str],
    bm25_index: BM25Index,
) -> dict[str, list[dict[str, Any]]]:
    page_position = {page: index for index, page in enumerate(page_units)}
    page_idf = _build_idf(bm25_index, len(page_units))
    idf_values = np.asarray(list(page_idf.values()), dtype=np.float64)
    rare_cutoff = float(np.percentile(idf_values, 75)) if len(idf_values) else 0.0
    page_tokens = [analyze(text) for text in page_texts]
    page_sets = [set(tokens) for tokens in page_tokens]
    page_counts_by_file: dict[str, int] = defaultdict(int)
    page_offsets: dict[str, list[int]] = defaultdict(list)
    for position, page in enumerate(page_units):
        file_id = _file_id(page)
        page_counts_by_file[file_id] += 1
        page_offsets[file_id].append(position)

    stream_data: dict[str, dict[str, dict[str, float | int]]] = {}
    for stream in STREAMS:
        stream_data[stream] = {}
        for qid in qids:
            ranks, scores = _rank_and_scores(loaded[stream][qid])
            normalised = _normalise(scores)
            stream_data[stream][qid] = {
                page: (normalised.get(page, 0.0), ranks.get(page, 0), scores.get(page, 0.0))
                for page in set(ranks) | set(scores)
            }

    output: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        query_terms = list(dict.fromkeys(analyze(str(questions[qid].query))))
        query_set = set(query_terms)
        term_idf = {term: float(page_idf.get(term, 0.0)) for term in query_terms}
        total_idf = sum(term_idf.values()) or 1.0
        anchors = [term for term in query_terms if any(ch.isdigit() for ch in term) or term_idf[term] >= rare_cutoff]
        candidate_pages = _candidate_pages(qid, loaded)
        rows: list[dict[str, Any]] = []
        for page in candidate_pages:
            position = page_position[page]
            tokens = page_tokens[position]
            token_set = page_sets[position]
            matched = [term for term in query_terms if term in token_set]
            matched_idf = sum(term_idf[term] for term in matched)
            matched_anchors = [term for term in anchors if term in token_set]
            values: list[float] = []
            rank_values: list[int] = []
            top10_count = 0
            present_count = 0
            top10_streams: list[str] = []
            for stream in STREAMS:
                data = stream_data[stream][qid].get(page)
                score = float(data[0]) if data else 0.0
                rank = int(data[1]) if data else 0
                values.extend([score, 1.0 / (rank + 1) if rank else 0.0, float(bool(rank))])
                rank_values.append(rank)
                if rank:
                    present_count += 1
                if 1 <= rank <= 10:
                    top10_count += 1
                    top10_streams.append(stream)
            present_ranks = [rank for rank in rank_values if rank]
            best_rank = min(present_ranks, default=101)
            rank_agreement = len(top10_streams) * (len(top10_streams) - 1) / 2.0
            file_id = _file_id(page)
            file_page_numbers = [_page_number(page_units[item]) for item in page_offsets[file_id]]
            max_page_number = max(file_page_numbers, default=1)
            page_position_norm = _page_number(page) / max_page_number if max_page_number else 0.0
            values.extend(
                [
                    float(present_count),
                    float(top10_count),
                    rank_agreement / 21.0,
                    matched_idf / total_idf,
                    matched_idf / total_idf,
                    float(len(matched)),
                    float(bool(matched_anchors)),
                    float(bool(FORMULA_MARKER.search(page_texts[position]))),
                    float(bool(TABLE_MARKER.search(page_texts[position]))),
                    float(bool(IMAGE_MARKER.search(page_texts[position]))),
                    math.log1p(len(tokens)),
                    page_position_norm,
                ]
            )
            # The two IDF-derived fields have different semantics in the
            # contract: lexical coverage is a fractional query match, while
            # lexical IDF mass is intentionally normalized by query mass. Keep
            # them equal here rather than inventing a tuned weighting.
            rows.append(
                {
                    "qid": qid,
                    "page_id": page,
                    "features": values,
                    "label": int(qrels.get(qid, {}).get(page, 0) > 0),
                }
            )
        output[qid] = rows
    return output


def _fit_pairwise(
    train_qids: Sequence[str],
    rows_by_qid: Mapping[str, Sequence[dict[str, Any]]],
    *,
    seed: int,
    max_pairs_per_query: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    train_rows = [row for qid in train_qids for row in rows_by_qid[qid]]
    matrix = np.asarray([row["features"] for row in train_rows], dtype=np.float64)
    mean = matrix.mean(axis=0) if len(matrix) else np.zeros(len(FEATURE_NAMES))
    std = matrix.std(axis=0) if len(matrix) else np.ones(len(FEATURE_NAMES))
    std[std < 1e-8] = 1.0
    rng = np.random.default_rng(seed)
    differences: list[np.ndarray] = []
    for qid in train_qids:
        positives = [row for row in rows_by_qid[qid] if row["label"]]
        negatives = [row for row in rows_by_qid[qid] if not row["label"]]
        if not positives or not negatives:
            continue
        for _ in range(min(max_pairs_per_query, len(positives) * len(negatives))):
            positive = positives[int(rng.integers(len(positives)))]
            negative = negatives[int(rng.integers(len(negatives)))]
            left = (np.asarray(positive["features"], dtype=np.float64) - mean) / std
            right = (np.asarray(negative["features"], dtype=np.float64) - mean) / std
            differences.append(left - right)
    pair_matrix = np.asarray(differences, dtype=np.float64)
    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    for _ in range(160):
        if len(pair_matrix):
            margins = np.clip(pair_matrix @ weights, -40.0, 40.0)
            gradient = -((pair_matrix.T @ (1.0 / (1.0 + np.exp(margins)))) / len(pair_matrix))
        else:
            gradient = np.zeros_like(weights)
        weights -= 0.04 * (gradient + 0.002 * weights)
    return weights, mean, std, int(len(pair_matrix))


def _rank(
    rows_by_qid: Mapping[str, Sequence[dict[str, Any]]],
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in rows_by_qid.items():
        scored = []
        for row in rows:
            values = (np.asarray(row["features"], dtype=np.float64) - mean) / std
            scored.append((float(values @ weights), str(row["page_id"])))
        ordered = sorted(scored, key=lambda item: (-item[0], item[1]))[:100]
        output[qid] = [
            {"chunk_id": page, "doc_id": page, "score": score, "rank": rank}
            for rank, (score, page) in enumerate(ordered, 1)
        ]
    return output


def _rank_baseline(rows_by_qid: Mapping[str, Sequence[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    # The untrained control uses the first (lexical) stream score only. It is
    # included to verify that the candidate-union data path is not itself a
    # hidden relevance improvement.
    output: dict[str, list[dict[str, Any]]] = {}
    lexical_index = FEATURE_NAMES.index("bm25_score")
    for qid, rows in rows_by_qid.items():
        ordered = sorted(rows, key=lambda row: (-float(row["features"][lexical_index]), row["page_id"]))[:100]
        output[qid] = [
            {"chunk_id": row["page_id"], "doc_id": row["page_id"], "score": float(row["features"][lexical_index]), "rank": rank}
            for rank, row in enumerate(ordered, 1)
        ]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parsed-run", type=Path, default=PARSED_RUN)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    question_by_qid = {q.qid: q for q in questions}
    qrels = benchmark.qrels()
    loaded = {name: _load_run(path) for name, path in STREAM_PATHS.items()}
    if any(set(loaded[name]) != set(qids) for name in STREAMS):
        raise RuntimeError("Every stream must contain the same 302 qids")

    page_metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    page_units = [str(row["unit_id"]) for row in page_metadata]
    page_text_by_unit = _load_page_texts(args.parsed_run)
    page_texts = [page_text_by_unit.get(page, "") for page in page_units]
    if len(page_units) != 1674 or any(not text.strip() for text in page_texts):
        raise RuntimeError("Expected non-empty PDF-inspector text for all 1,674 pages")
    page_vectors = _load_csr(PAGE_VECTOR_DIR / "page_vectors.npz")
    query_vectors = _load_csr(QUERY_VECTOR_DIR / "query_vectors.npz")
    if page_vectors.shape != (1674, 50368) or query_vectors.shape != (302, 50368):
        raise RuntimeError("Unexpected cached V-SPLADE matrix shape")

    bm25_index = BM25Index(analyzer_name="plain").build(
        [{"chunk_id": page, "doc_id": page, "text": text} for page, text in zip(page_units, page_texts)]
    )
    rows_by_qid = _build_rows(qids, question_by_qid, qrels, loaded, page_units, page_texts, bm25_index)
    candidate_counts = [len(rows_by_qid[qid]) for qid in qids]
    if min(candidate_counts) < 100:
        raise RuntimeError("Candidate union must contain at least 100 pages per query")

    folds = {qid: index % 5 for index, qid in enumerate(qids)}
    oof: dict[str, list[dict[str, Any]]] = {}
    fold_rows: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []
    for fold in range(5):
        heldout = [qid for qid in qids if folds[qid] == fold]
        train = [qid for qid in qids if folds[qid] != fold]
        weights, mean, std, pair_count = _fit_pairwise(train, rows_by_qid, seed=20260907 + fold)
        ranked = _rank({qid: rows_by_qid[qid] for qid in heldout}, weights, mean, std)
        oof.update(ranked)
        metrics = _derived_metrics(ranked, heldout, qrels)
        fold_rows.append({"fold": fold, "test_qids": len(heldout), "test_page_recall@10": metrics["page_recall@10"], "test_ndcg@10": metrics["ndcg@10"], "pair_count": pair_count})
        models.append({"fold": fold, "weights": {name: float(value) for name, value in zip(FEATURE_NAMES, weights)}, "pair_count": pair_count})

    metric = _derived_metrics(oof, qids, qrels)
    union_bm25 = _rank_baseline(rows_by_qid)
    union_metric = _derived_metrics(union_bm25, qids, qrels)
    c014_path = RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl"
    c014 = _load_run(c014_path)
    c014_metric = _derived_metrics(c014, qids, qrels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "candidate_streams": {name: str(path) for name, path in STREAM_PATHS.items()},
        "candidate_count": {"min": min(candidate_counts), "max": max(candidate_counts), "mean": float(np.mean(candidate_counts))},
        "feature_names": FEATURE_NAMES,
        "oof": {"metrics": metric, "folds": fold_rows, "models": models},
        "controls": {"bm25_over_union": union_metric, "c014": c014_metric},
        "comparisons": {"vs_c014": _paired_comparison(c014_metric, metric)},
        "notes": [
            "The seven top-100 stream union is the fixed lightweight-index candidate set.",
            "The pairwise ranker is fitted independently on four training folds; held-out qrels are never used in fitting.",
            "Features are cheap metadata/text markers and stream scores/ranks. No answer, modality or gold evidence metadata is used.",
            "The visual streams use cached English query vectors against French Physics qrels.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Physics union feature ranker",
        "",
        "Fold-safe ranking over the union of seven cached lightweight-index streams.",
        "",
        "| Method | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 |",
        "|---|---:|---:|---:|---:|",
        f"| BM25 score over union control | {union_metric['ndcg@10']:.2f} | {union_metric['page_recall@10']:.2%} | {union_metric['page_hit@10']:.2%} | {union_metric['file_metrics_by_k']['3']['file_recall']:.2%} |",
        f"| c014 structural reference | {c014_metric['ndcg@10']:.2f} | {c014_metric['page_recall@10']:.2%} | {c014_metric['page_hit@10']:.2%} | {c014_metric['file_metrics_by_k']['3']['file_recall']:.2%} |",
        f"| **union feature ranker OOF** | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | {metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} |",
        "",
        f"Candidate count per query: {min(candidate_counts)}..{max(candidate_counts)}, mean {np.mean(candidate_counts):.1f}.",
        f"OOF delta vs c014: {report['comparisons']['vs_c014']['page_recall_delta_pp']:+.2f} pp.",
        "",
        f"Fold rows: `{fold_rows}`.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "oof_page_recall@10": round(metric["page_recall@10"] * 100, 2), "oof_file_recall@3": round(metric["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "delta_vs_c014_pp": report["comparisons"]["vs_c014"]["page_recall_delta_pp"], "candidate_count": report["candidate_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
