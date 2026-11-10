"""Offline ablation for the per-component gate cutoff.

This reuses the existing PP-OCRv5 page text, BM25 index inputs and cached
Qwen3-Embedding-8B vectors. It does not call an embedding API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bench import BENCH, gold, load_jsonl, queries, work
from .evaluate import as_run, evaluate, page_recall
from .text import PageBM25, enfr, real_text


def minmax_topk(x: np.ndarray, k: int) -> np.ndarray:
    top = np.sort(x)[-k:]
    return np.clip((x - top[0]) / (top[-1] - top[0] + 1e-9), 0, None)


def ranked(scores: np.ndarray, units: list[str], n: int) -> list[str]:
    idx = np.argpartition(-scores, min(n, len(scores) - 1))[:n]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    return [units[i] for i in idx]


def main() -> None:
    out = work("light_prep", bench=BENCH)
    pages_name = "pages_ocr_ppocr.jsonl"
    rows = load_jsonl(out / pages_name)
    units = [row["page_id"] for row in rows]
    qs = queries(BENCH)
    qrels = gold(BENCH)
    qids = sorted(qrels)
    bm25 = PageBM25([enfr(real_text(row["text"])) for row in rows])

    vector_dir = work("light_v2", bench=BENCH)
    page_vectors = np.load(vector_dir / "dense_qwen8b_ppocr_pages.npy")
    query_vectors = np.load(vector_dir / "dense_qwen8b_ppocr_queries.npy")
    page_vectors = page_vectors / (np.linalg.norm(page_vectors, axis=1, keepdims=True) + 1e-9)
    query_vectors = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-9)
    dense_matrix = query_vectors @ page_vectors.T

    results = {}
    for cutoff in (100, 200):
        runs = {}
        for i, qid in enumerate(qids):
            lexical = bm25.scores(enfr(qs[qid]["query"]))
            fused = 0.2 * minmax_topk(lexical, cutoff) + 0.8 * minmax_topk(dense_matrix[i], cutoff)
            runs[qid] = ranked(fused, units, min(200, len(units)))
        scored = {qid: as_run(run) for qid, run in runs.items()}
        metrics = {}
        for k in (10, 20, 100, 200):
            ndcg, recall, _ = evaluate(scored, qrels, k)
            metrics[f"recall@{k}"] = round(recall, 4)
            metrics[f"ndcg@{k}"] = round(ndcg, 4)
        metrics["candidate_ceiling@200"] = round(
            sum(page_recall(runs[qid][:200], qrels[qid]) for qid in qids) / len(qids), 4
        )
        for name, predicate in {
            "physics": lambda q: q.startswith("vidore_physics::"),
            "industrial": lambda q: q.startswith("vidore_industrial::"),
        }.items():
            members = [qid for qid in qids if predicate(qid)]
            group_run = {qid: scored[qid] for qid in members}
            group_metrics = {}
            for k in (10, 20, 200):
                ndcg, recall, _ = evaluate(group_run, qrels, k)
                group_metrics[f"recall@{k}"] = round(recall, 4)
                group_metrics[f"ndcg@{k}"] = round(ndcg, 4)
            group_metrics["candidate_ceiling@200"] = round(
                sum(page_recall(runs[qid][:200], qrels[qid]) for qid in members) / len(members), 4
            )
            metrics[name] = group_metrics
        results[str(cutoff)] = metrics

    output = out / "gate_cutoff_experiment.json"
    output.write_text(
        json.dumps(
            {
                "pages_source": pages_name,
                "embedding_model": "qwen/qwen3-embedding-8b",
                "alpha_bm25": 0.2,
                "alpha_dense": 0.8,
                "cutoffs": results,
                "note": "Recall values are fractions; no API calls were made.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2))
    print(output)


if __name__ == "__main__":
    main()
