"""Core data, features, metrics, and leak-free helpers for Adaptive-K experiments.

This module deliberately keeps benchmark loading separate from model selection.
Any feature that depends on labels (notably neighbour target-K summaries) must be
constructed by a caller from the fitting split only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = Path(os.environ.get("AXIOM_ADAPTIVE_BASELINE_ROOT", ROOT / "_repro_retrieval_baseline")).resolve()
BENCHMARK_ROOT = BASELINE_ROOT / "0.1. BENCHMARK"
PREP_ROOT = BASELINE_ROOT / "data/work/ondemand_v2/ef7f11811803/light_prep"
VECTOR_ROOT = BASELINE_ROOT / "data/work/ondemand_v2/ef7f11811803/light_v2"
CAP = 200
SEED = 42
POINT_RANKS = (1, 2, 5, 10, 20, 50, 100, 150)
BANDS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 150), (151, 200))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def minmax_topk(scores: np.ndarray, cutoff: int = 100) -> np.ndarray:
    top = np.sort(scores)[-cutoff:]
    return np.clip((scores - top[0]) / (top[-1] - top[0] + 1e-9), 0.0, None)


@dataclass
class FrozenData:
    rows: list[dict[str, Any]]
    query_vectors: np.ndarray
    fused_full: np.ndarray
    lexical_full: np.ndarray
    dense_full: np.ndarray
    page_ids: list[str]
    page_texts: list[str]
    protocol: dict[str, Any]
    lexical_raw_full: np.ndarray | None = None
    dense_raw_full: np.ndarray | None = None


def load_frozen_data() -> FrozenData:
    """Recreate the frozen PP-OCRv5/Qwen ranking and retain its components."""
    sys.path.insert(0, str(BASELINE_ROOT))
    from ondemand.text import PageBM25, enfr, real_text  # type: ignore

    pages = read_jsonl(PREP_ROOT / "pages_ocr_ppocr.jsonl")
    query_rows = read_jsonl(BENCHMARK_ROOT / "queries.jsonl")
    queries = {str(row["query_id"]): row for row in query_rows}
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(PREP_ROOT / "qrels_page_level.jsonl"):
        if int(row.get("relevance", 1)) > 0:
            qrels[str(row["query_id"])].add(str(row["page_id"]))

    page_ids = [str(row["page_id"]) for row in pages]
    page_texts = [real_text(str(row.get("text", ""))) for row in pages]
    page_tokens = [enfr(text) for text in page_texts]
    bm25 = PageBM25(page_tokens)
    page_vectors = np.asarray(np.load(VECTOR_ROOT / "dense_qwen8b_ppocr_pages.npy", mmap_mode="r"), dtype=np.float32)
    query_vectors = np.asarray(np.load(VECTOR_ROOT / "dense_qwen8b_ppocr_queries.npy", mmap_mode="r"), dtype=np.float32)
    page_vectors = page_vectors / (np.linalg.norm(page_vectors, axis=1, keepdims=True) + 1e-9)
    query_vectors = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-9)
    metadata = json.loads((VECTOR_ROOT / "dense_qwen8b_ppocr_meta.json").read_text(encoding="utf-8"))

    if len(page_ids) != 5091:
        raise ValueError(f"Expected 5091 pages, found {len(page_ids)}")
    if len(metadata["qids"]) != 220 or query_vectors.shape[0] != 220:
        raise ValueError("Frozen query cache must contain exactly 220 aligned queries")

    lexical_full = np.empty((220, len(page_ids)), dtype=np.float32)
    dense_full = np.empty_like(lexical_full)
    lexical_raw_full = np.empty_like(lexical_full)
    dense_raw_full = np.empty_like(lexical_full)
    fused_full = np.empty_like(lexical_full)
    rows: list[dict[str, Any]] = []
    for index, qid_value in enumerate(metadata["qids"]):
        qid = str(qid_value)
        query = str(queries[qid]["query"])
        lexical_raw = np.asarray(bm25.scores(enfr(query)), dtype=np.float32)
        dense_raw = np.asarray(page_vectors @ query_vectors[index], dtype=np.float32)
        lexical = minmax_topk(lexical_raw).astype(np.float32)
        dense = minmax_topk(dense_raw).astype(np.float32)
        fused = 0.2 * lexical + 0.8 * dense
        order = np.argsort(-fused, kind="stable")[:CAP]
        lexical_full[index] = lexical
        dense_full[index] = dense
        lexical_raw_full[index] = lexical_raw
        dense_raw_full[index] = dense_raw
        fused_full[index] = fused
        rows.append({
            "qid": qid,
            "source": qid.split("::", 1)[0],
            "query": query,
            "ranked": [page_ids[int(position)] for position in order],
            "ranked_indices": order.astype(np.int32),
            "scores": fused[order].astype(np.float32),
            "lexical_scores": lexical[order].astype(np.float32),
            "dense_scores": dense[order].astype(np.float32),
            "gold": qrels[qid],
        })

    return FrozenData(
        rows=rows,
        query_vectors=query_vectors,
        fused_full=fused_full,
        lexical_full=lexical_full,
        dense_full=dense_full,
        page_ids=page_ids,
        page_texts=page_texts,
        protocol={
            "benchmark": "Physics 80 + Industrial 75 + OHR-Bench 65",
            "queries": 220,
            "pages": 5091,
            "page_text": str(PREP_ROOT / "pages_ocr_ppocr.jsonl"),
            "ocr_engine": "ppocrv5_doc",
            "dense_model": "Qwen3-Embedding-8B (cached vectors)",
            "ranking": "global 0.2 * minmax_top100(BM25) + 0.8 * minmax_top100(dense)",
            "candidate_depth": CAP,
        },
        lexical_raw_full=lexical_raw_full,
        dense_raw_full=dense_raw_full,
    )


def recall_at(row: Mapping[str, Any], cutoff: int) -> float:
    gold = set(row["gold"])
    return len(set(row["ranked"][: int(cutoff)]) & gold) / max(len(gold), 1)


def required_k(row: Mapping[str, Any], target_recall: float) -> tuple[int, bool]:
    gold = set(row["gold"])
    ranks = sorted(rank for rank, page_id in enumerate(row["ranked"][:CAP], 1) if page_id in gold)
    need = int(math.ceil(target_recall * len(gold) - 1e-12))
    reachable = need > 0 and len(ranks) >= need
    return (int(ranks[need - 1]) if reachable else CAP), (not reachable)


def target_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for percent in (80, 90, 100):
        pairs = [required_k(row, percent / 100.0) for row in rows]
        output[f"k{percent}"] = np.asarray([item[0] for item in pairs], dtype=np.float32)
        output[f"censored{percent}"] = np.asarray([item[1] for item in pairs], dtype=bool)
    return output


def _entropy(values: np.ndarray) -> float:
    shifted = values - float(np.min(values)) + 1e-9
    probabilities = shifted / float(np.sum(shifted))
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(len(values))) if len(values) > 1 else 0.0


def _point_features(scores: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    return ([f"{prefix}_at_{rank}" for rank in POINT_RANKS], [float(scores[rank - 1]) for rank in POINT_RANKS])


def _local_features(scores: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    names, values = [], []
    for rank in POINT_RANKS:
        left, right = max(0, rank - 3), min(len(scores), rank + 2)
        names.append(f"{prefix}_local5_at_{rank}")
        values.append(float(np.mean(scores[left:right])))
    return names, values


def _band_features(scores: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    names, values = [], []
    for start, end in BANDS:
        names.append(f"{prefix}_band_{start}_{end}")
        values.append(float(np.mean(scores[start - 1:end])))
    return names, values


def _shape_features(scores: np.ndarray, prefix: str) -> tuple[list[str], list[float]]:
    eps = 1e-9
    means = {k: float(np.mean(scores[:k])) for k in (5, 10, 20)}
    names = [
        f"{prefix}_gap_1_2", f"{prefix}_gap_1_5mean", f"{prefix}_ratio_1_2",
        f"{prefix}_ratio_1_5mean", f"{prefix}_drop_1_20", f"{prefix}_drop_20_100",
        f"{prefix}_curvature_2", f"{prefix}_curvature_5",
        f"{prefix}_entropy_5", f"{prefix}_entropy_10", f"{prefix}_entropy_20",
    ]
    values = [
        float(scores[0] - scores[1]), float(scores[0] - means[5]),
        float(scores[0] / (scores[1] + eps)), float(scores[0] / (means[5] + eps)),
        float(scores[0] - scores[19]), float(scores[19] - scores[99]),
        float(scores[0] - 2 * scores[1] + scores[2]),
        float(scores[3] - 2 * scores[4] + scores[5]),
        _entropy(scores[:5]), _entropy(scores[:10]), _entropy(scores[:20]),
    ]
    return names, values


def _legacy_features(scores: np.ndarray) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    for k in (1, 2, 3, 5, 10, 20, 50, 100, 150, 200):
        names.append(f"legacy_mean_top{k}")
        values.append(float(np.mean(scores[:k])))
    for a, b in ((1, 2), (2, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 150), (150, 200)):
        names.append(f"legacy_gap_{a}_{b}")
        values.append(float(scores[a - 1] - scores[b - 1]))
    names += ["legacy_range", "legacy_std"]
    values += [float(scores[0] - scores[-1]), float(np.std(scores))]
    return names, values


def build_score_feature_families(data: FrozenData) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Return independent families; callers decide forward-add combinations."""
    rows = data.rows
    values: dict[str, list[list[float]]] = defaultdict(list)
    names: dict[str, list[str]] = {}
    for row_index, row in enumerate(rows):
        fused = np.asarray(row["scores"], dtype=float)
        lexical = np.asarray(row["lexical_scores"], dtype=float)
        dense = np.asarray(row["dense_scores"], dtype=float)
        builders = {
            "baseline21": _legacy_features(fused),
            "clean": ([], []),
            "point": _point_features(fused, "fused"),
            "local": _local_features(fused, "fused"),
            "band": _band_features(fused, "fused"),
            "shape": _shape_features(fused, "fused"),
        }
        # Clean baseline removes exact/redundant top-1 mean, range duplication and rank-200 artefacts.
        clean_names = ["clean_top1", "clean_mean_2_5", "clean_mean_6_10", "clean_mean_11_20", "clean_mean_21_50", "clean_mean_51_100", "clean_std_1_100", "clean_drop_1_20", "clean_drop_20_100"]
        clean_values = [float(fused[0]), float(np.mean(fused[1:5])), float(np.mean(fused[5:10])), float(np.mean(fused[10:20])), float(np.mean(fused[20:50])), float(np.mean(fused[50:100])), float(np.std(fused[:100])), float(fused[0] - fused[19]), float(fused[19] - fused[99])]
        builders["clean"] = (clean_names, clean_values)

        modality_names: list[str] = []
        modality_values: list[float] = []
        for component_name, component in (("bm25", lexical), ("dense", dense)):
            for fn in (_point_features, _band_features):
                n, v = fn(component, component_name)
                modality_names.extend(n); modality_values.extend(v)
        lexical_order = np.argsort(-data.lexical_full[row_index], kind="stable")
        dense_order = np.argsort(-data.dense_full[row_index], kind="stable")
        fused_top = int(row["ranked_indices"][0])
        modality_names += ["fused_top_rank_in_bm25", "fused_top_rank_in_dense", "bm25_dense_top1_agree"]
        modality_values += [float(np.flatnonzero(lexical_order == fused_top)[0] + 1), float(np.flatnonzero(dense_order == fused_top)[0] + 1), float(lexical_order[0] == dense_order[0])]
        for k in (5, 10, 20):
            overlap = len(set(lexical_order[:k].tolist()) & set(dense_order[:k].tolist())) / k
            modality_names.append(f"bm25_dense_overlap_{k}"); modality_values.append(float(overlap))
        modality_names += ["bm25_margin_1_2", "dense_margin_1_2"]
        modality_values += [float(data.lexical_full[row_index, lexical_order[0]] - data.lexical_full[row_index, lexical_order[1]]), float(data.dense_full[row_index, dense_order[0]] - data.dense_full[row_index, dense_order[1]])]
        builders["modality"] = (modality_names, modality_values)

        for family, (family_names, family_values) in builders.items():
            if family not in names:
                names[family] = family_names
            values[family].append(family_values)
    return {key: np.asarray(matrix, dtype=np.float32) for key, matrix in values.items()}, names


_NUMBER = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?|[IVXLCDM]+)(?!\w)", re.I)
_UNIT = re.compile(r"\b(?:kg|g|mg|m|cm|mm|km|s|ms|hz|khz|mhz|ghz|v|a|w|kw|kwh|pa|bar|Â°c|%)\b", re.I)
_TABLE_FORM = re.compile(r"\b(?:table|tableau|figure|fig\.?|form|formula|Ã©quation|equation|schÃ©ma|schema|graph|chart|diagram)\b", re.I)


def build_query_features(data: FrozenData) -> tuple[np.ndarray, list[str]]:
    sys.path.insert(0, str(BASELINE_ROOT))
    from ondemand.text import enfr  # type: ignore
    tokenized_pages = [set(enfr(text)) for text in data.page_texts]
    document_frequency = Counter(token for tokens in tokenized_pages for token in tokens)
    n_pages = len(tokenized_pages)
    names = ["query_tokens", "query_chars", "query_number_count", "query_unit_count", "query_table_form", "query_anchor_idf_mean", "query_anchor_idf_max", "query_anchor_rare_fraction"]
    matrix = []
    for row in data.rows:
        query = str(row["query"])
        tokens = enfr(query)
        idfs = [math.log(1.0 + (n_pages - document_frequency[token] + 0.5) / (document_frequency[token] + 0.5)) for token in tokens]
        matrix.append([
            len(tokens), len(query), len(_NUMBER.findall(query)), len(_UNIT.findall(query)), bool(_TABLE_FORM.search(query)),
            float(np.mean(idfs)) if idfs else 0.0, float(np.max(idfs)) if idfs else 0.0,
            float(np.mean([document_frequency[token] <= 5 for token in tokens])) if tokens else 0.0,
        ])
    return np.asarray(matrix, dtype=np.float32), names


def combine_families(families: Mapping[str, np.ndarray], names: Mapping[str, list[str]], keys: Iterable[str]) -> tuple[np.ndarray, list[str]]:
    selected = list(keys)
    return np.hstack([families[key] for key in selected]), [name for key in selected for name in names[key]]


def metric_summary(rows: Sequence[Mapping[str, Any]], choices: np.ndarray, indices: np.ndarray | None = None) -> dict[str, Any]:
    indices = np.arange(len(rows)) if indices is None else np.asarray(indices)
    local_choices = np.asarray(choices)[indices]
    recalls = np.asarray([recall_at(rows[int(index)], int(k)) for index, k in zip(indices, local_choices)], dtype=float)
    return {
        "queries": int(len(indices)), "mean_k": float(np.mean(local_choices)), "median_k": float(np.median(local_choices)),
        "p90_k": float(np.percentile(local_choices, 90)), "macro_recall": float(np.mean(recalls)), "recalls": recalls,
    }


def bootstrap_ci(rows: Sequence[Mapping[str, Any]], choices: np.ndarray, samples: int = 2000, seed: int = SEED) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    recalls = np.asarray([recall_at(row, int(k)) for row, k in zip(rows, choices)])
    ks = np.asarray(choices, dtype=float)
    draw = rng.integers(0, len(rows), size=(samples, len(rows)))
    return {
        "macro_recall": np.percentile(np.mean(recalls[draw], axis=1), [2.5, 97.5]).tolist(),
        "mean_k": np.percentile(np.mean(ks[draw], axis=1), [2.5, 97.5]).tolist(),
    }


def validate_data(data: FrozenData) -> dict[str, Any]:
    qids = [str(row["qid"]) for row in data.rows]
    monotone = [bool(np.all(np.diff(np.asarray(row["scores"], dtype=float)) <= 1e-7)) for row in data.rows]
    finite = all(np.isfinite(np.asarray(row["scores"], dtype=float)).all() for row in data.rows)
    rank200_zero = [bool(abs(float(row["scores"][-1])) < 1e-12) for row in data.rows]
    checks = {
        "queries_220": len(qids) == 220 and len(set(qids)) == 220,
        "pages_5091": len(data.page_ids) == 5091 and len(set(data.page_ids)) == 5091,
        "candidates_200_each": all(len(row["ranked"]) == CAP and len(set(row["ranked"])) == CAP for row in data.rows),
        "score_monotone": all(monotone), "finite_scores": finite,
        "rank200_zero_count": int(sum(rank200_zero)),
    }
    if not all(value for key, value in checks.items() if not key.endswith("_count")):
        raise ValueError(f"Frozen data acceptance check failed: {checks}")
    return checks
