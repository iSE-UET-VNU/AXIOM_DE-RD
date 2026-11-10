"""Evaluate the cached canonical PP-OCR/Qwen light retriever at deeper K."""

from __future__ import annotations

import json

import numpy as np

from .bench import BENCH, gold, load_jsonl, queries, work
from .evaluate import as_run, evaluate
from .text import PageBM25, enfr, real_text


DEPTHS = (20, 30, 50, 100, 150)


def minmax_topk(scores: np.ndarray, cutoff: int = 100) -> np.ndarray:
    top = np.sort(scores)[-cutoff:]
    return np.clip((scores - top[0]) / (top[-1] - top[0] + 1e-9), 0, None)


def rank(scores: np.ndarray, page_ids: list[str], depth: int = 200) -> list[str]:
    indices = np.argpartition(-scores, min(depth, len(scores) - 1))[:depth]
    indices = indices[np.argsort(-scores[indices], kind="stable")]
    return [page_ids[index] for index in indices]


def score_group(runs: dict[str, dict[str, float]], qrels, qids: list[str]) -> dict[str, float]:
    group_run = {qid: runs[qid] for qid in qids}
    metrics = {"queries": len(qids)}
    for depth in DEPTHS:
        ndcg, recall, _ = evaluate(group_run, qrels, depth)
        metrics[f"recall@{depth}"] = round(recall, 4)
        metrics[f"ndcg@{depth}"] = round(ndcg, 4)
    return metrics


def main() -> None:
    prep_dir = work("light_prep", bench=BENCH)
    pages = load_jsonl(prep_dir / "pages_ocr_ppocr.jsonl")
    page_ids = [page["page_id"] for page in pages]
    query_rows = queries(BENCH)
    qrels = gold(BENCH)
    qids = sorted(qrels)

    bm25 = PageBM25([enfr(real_text(page["text"])) for page in pages])
    vector_dir = work("light_v2", bench=BENCH)
    page_vectors = np.load(vector_dir / "dense_qwen8b_ppocr_pages.npy")
    query_vectors = np.load(vector_dir / "dense_qwen8b_ppocr_queries.npy")
    page_vectors /= np.linalg.norm(page_vectors, axis=1, keepdims=True) + 1e-9
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-9
    dense_scores = query_vectors @ page_vectors.T

    ranked_runs = {}
    for index, qid in enumerate(qids):
        lexical = bm25.scores(enfr(query_rows[qid]["query"]))
        fused = 0.2 * minmax_topk(lexical) + 0.8 * minmax_topk(dense_scores[index])
        ranked_runs[qid] = rank(fused, page_ids)

    scored_runs = {qid: as_run(pages) for qid, pages in ranked_runs.items()}
    groups = {
        "all": qids,
        "physics": [qid for qid in qids if qid.startswith("vidore_physics::")],
        "industrial": [qid for qid in qids if qid.startswith("vidore_industrial::")],
        "ohr": [qid for qid in qids if qid.startswith("ohrbench::")],
    }
    result = {
        "benchmark": "Physics 80 + Industrial 75 + OHR 65",
        "pages": len(page_ids),
        "pages_source": "pages_ocr_ppocr.jsonl",
        "embedding_model": "qwen/qwen3-embedding-8b",
        "fusion": {"bm25": 0.2, "dense": 0.8, "normalization_top_k": 100},
        "metrics_percent": {
            name: score_group(scored_runs, qrels, members) for name, members in groups.items()
        },
    }
    output = prep_dir / "ppocr_topk_depth_sweep.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
