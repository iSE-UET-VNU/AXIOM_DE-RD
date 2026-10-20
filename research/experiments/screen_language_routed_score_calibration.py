"""Screen score calibration for the cached language-routed hybrid retrieval.

No qrels are used to rank.  The arms differ only in how cached BM25 and dense
scores are mapped to a common query-local scale before the existing Kf=3
file-stage and page-stage formula is applied.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    OLD_LANGUAGE, file_id, load_inputs, metric_rows, read_jsonl, write_jsonl,
)

BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/score_calibration_screen"


def max_norm(values: np.ndarray) -> np.ndarray:
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values, dtype=np.float64)


def percentile(values: np.ndarray) -> np.ndarray:
    """Stable query-local empirical percentile; all-zero vectors map to zero."""
    if not values.size or float(values.max()) <= 0:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    out = np.zeros_like(values, dtype=np.float64)
    # Tie-aware average rank, then map the lowest non-zero score close to 0.
    sorted_values = values[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        out[order[start:end]] = (rank - 1.0) / max(1, len(order) - 1)
        start = end
    out[values <= 0] = 0.0
    return out


def weighted_rrf(values: np.ndarray, k: int = 60) -> np.ndarray:
    """Rank-only reciprocal score, scaled so the best rank is one."""
    if not values.size or float(values.max()) <= 0:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    out = np.zeros_like(values, dtype=np.float64)
    ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    out[order] = (k + 1.0) / (k + ranks)
    out[values <= 0] = 0.0
    return out


def language_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for row in read_jsonl(OLD_LANGUAGE):
        fid = str(row.get("file_id") or row.get("doc_id") or "")
        if fid:
            result[fid] = str(row.get("detected_language") or "en")
    return result


def build_arm(
    name: str,
    transform: Callable[[np.ndarray], np.ndarray],
    page_bm_raw: np.ndarray,
    file_bm_raw: np.ndarray,
    dense_raw: np.ndarray,
    page_ids: list[str],
    file_ids: list[str],
    query_ids: list[str],
    langs: dict[str, str],
) -> list[dict]:
    page_file_pos = np.asarray([file_ids.index(file_id(page)) for page in page_ids], dtype=np.int32)
    rows: list[dict] = []
    for qi, qid in enumerate(query_ids):
        page_bm = transform(page_bm_raw[qi])
        dense = transform(dense_raw[qi])
        file_bm = transform(file_bm_raw[qi])
        page_score = np.empty(len(page_ids), dtype=np.float64)
        direct = np.empty(len(file_ids), dtype=np.float64)
        for fi, fid in enumerate(file_ids):
            mask = page_file_pos == fi
            if langs.get(fid, "en") == "en":
                page_score[mask] = 0.70 * page_bm[mask] + 0.30 * dense[mask]
                direct[fi] = 0.70 * file_bm[fi] + 0.30 * float(dense[mask].max(initial=0.0))
            else:
                page_score[mask] = dense[mask]
                direct[fi] = float(dense[mask].max(initial=0.0))
        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_score)
        file_score = 0.50 * direct + 0.50 * transform(page_max)
        selected = np.argsort(-file_score, kind="stable")[:3]
        final = 0.85 * page_score + 0.15 * transform(file_score)[page_file_pos]
        allowed = np.flatnonzero(np.isin(page_file_pos, selected))
        order = allowed[np.argsort(-final[allowed], kind="stable")]
        rows.append({
            "query_id": qid, "arm": name,
            "selected_files": [file_ids[int(i)] for i in selected],
            "candidate_pages": [page_ids[int(i)] for i in order[:200]],
            "returned_pages": [page_ids[int(i)] for i in order[:20]],
            "returned_scores": [float(final[int(i)]) for i in order[:20]],
        })
    return rows


def source_metrics(rows: list[dict], qrels: dict[str, set[str]]) -> dict[str, dict]:
    prefixes = {"physics": "vidore_physics::", "industrial": "vidore_industrial::", "ohr": "ohrbench::", "all": ""}
    result = {}
    for source, prefix in prefixes.items():
        subset = [row for row in rows if not prefix or str(row["query_id"]).startswith(prefix)]
        sqrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
        result[source] = {"@10": metric_rows(subset, sqrels, 10), "@20": metric_rows(subset, sqrels, 20)}
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(x) for x in metadata["page_ids"]]
    file_ids = [str(x) for x in metadata["file_ids"]]
    query_ids = [str(x) for x in metadata["query_ids"]]
    page_bm_raw = np.load(BASE / "page_bm25_raw.npy")
    file_bm_raw = np.load(BASE / "file_bm25_raw.npy")
    dense_raw = np.load(BASE / "query_page_dense.npy")
    queries, qrels, inferred_langs = load_inputs()
    if [str(row["query_id"]) for row in queries] != query_ids:
        raise RuntimeError("Cached score matrix query ordering mismatch")
    langs = {**language_map(), **inferred_langs}
    if page_bm_raw.shape != dense_raw.shape or page_bm_raw.shape != (220, 5091):
        raise RuntimeError(f"Unexpected page matrices: BM25={page_bm_raw.shape}, dense={dense_raw.shape}")
    if file_bm_raw.shape != (220, 103):
        raise RuntimeError(f"Unexpected file BM25 shape: {file_bm_raw.shape}")

    arms = {
        "max_norm_current": max_norm,
        "percentile_rank": percentile,
        "weighted_rrf_k60": weighted_rrf,
    }
    report = {"protocol": {
        "same_language": "0.70 BM25 + 0.30 dense", "cross_language": "dense only",
        "file_k": 3, "candidate_depth": 200, "qrels_used_for_ranking": False,
        "transforms": {
            "max_norm_current": "score / query maximum (current implementation)",
            "percentile_rank": "tie-aware empirical percentile per query and signal",
            "weighted_rrf_k60": "(k+1)/(k+rank), k=60, then fixed 0.70/0.30 weights",
        },
    }, "arms": {}}
    for name, transform in arms.items():
        rows = build_arm(name, transform, page_bm_raw, file_bm_raw, dense_raw, page_ids, file_ids, query_ids, langs)
        write_jsonl(OUT / "runs" / f"{name}.jsonl", rows)
        report["arms"][name] = source_metrics(rows, qrels)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Language-routed score calibration screening", "", "| Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |", "|---|---:|---:|---:|---:|---:|"]
    for name, values in report["arms"].items():
        m10, m20 = values["all"]["@10"], values["all"]["@20"]
        lines.append(f"| {name} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
