"""Fair dense/hybrid/adaptive comparison on the cached 220-query benchmark.

All ranking policies use the same cached PDF-inspector + Tesseract page text,
the same dense embedding cache, and the same global page candidate depth.
Adaptive policies are calibrated to exactly the fixed total budget (mean K=12)
using ranking scores only; qrels are loaded only for the final evaluation.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
EMBEDDING_DIR = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
OUTPUT = CACHE / "fair_dense_hybrid_adaptive"
FIXED_K = 12
DEPTH = 100


def _normalise(values: Mapping[str, float]) -> dict[str, float]:
    maximum = max(values.values(), default=0.0)
    return {key: float(value) / maximum if maximum > 0 else 0.0 for key, value in values.items()}


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _ndcg(ranked: Sequence[str], gold: set[str], k: int) -> float:
    ranked_gold = [page_id for page_id in ranked[:k] if page_id in gold]
    dcg = sum(1.0 / np.log2(rank + 2) for rank, _page_id in enumerate(ranked_gold))
    ideal_len = min(k, len(gold))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))
    return float(dcg / idcg) if idcg else 0.0


def _metrics(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]]) -> dict[str, float]:
    page_returned, page10, page20 = [], [], []
    file_returned, file10, file20 = [], [], []
    ndcg10, ndcg20 = [], []
    ks = []
    for row in rows:
        qid = str(row["query_id"])
        gold = qrels[qid]
        pages = [str(page_id) for page_id in row["pages"]]
        denominator = max(len(gold), 1)
        page_returned.append(len(set(pages) & gold) / denominator)
        page10.append(len(set(pages[:10]) & gold) / denominator)
        page20.append(len(set(pages[:20]) & gold) / denominator)
        ndcg10.append(_ndcg(pages, gold, 10))
        ndcg20.append(_ndcg(pages, gold, 20))
        gold_files = {_file_id(page_id) for page_id in gold}
        file_returned.append(len({_file_id(page_id) for page_id in pages} & gold_files) / max(len(gold_files), 1))
        file10.append(len({_file_id(page_id) for page_id in pages[:10]} & gold_files) / max(len(gold_files), 1))
        file20.append(len({_file_id(page_id) for page_id in pages[:20]} & gold_files) / max(len(gold_files), 1))
        ks.append(len(pages))
    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {
        "queries": len(rows),
        "page_recall_at_returned_k": mean(page_returned),
        "page_recall@10": mean(page10),
        "page_recall@20": mean(page20),
        "page_ndcg@10": mean(ndcg10),
        "page_ndcg@20": mean(ndcg20),
        "file_recall_at_returned_k": mean(file_returned),
        "file_recall@10": mean(file10),
        "file_recall@20": mean(file20),
        "mean_k": mean(ks),
        "min_k": int(min(ks)) if ks else 0,
        "max_k": int(max(ks)) if ks else 0,
    }


def _rows(
    qids: Sequence[str],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    choices: Mapping[str, int],
    policy: str,
) -> list[dict[str, Any]]:
    rows = []
    for qid in qids:
        ranking = list(rankings[qid])
        k = max(0, min(int(choices[qid]), len(ranking)))
        rows.append({
            "query_id": qid,
            "policy": policy,
            "k": k,
            "pages": [page_id for page_id, _score in ranking[:k]],
            "scores": [float(score) for _page_id, score in ranking[:k]],
        })
    return rows


def _adaptive_choices(
    qids: Sequence[str],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    retrieve_more: int,
) -> dict[str, int]:
    return {
        qid: int(adaptive_k(
            [score for _page_id, score in rankings[qid]],
            ignore_extreme=0,
            ignore_extreme_tail=0.1,
            retrieve_more=retrieve_more,
        )["k"])
        for qid in qids
    }


def _calibrate_choices(
    qids: Sequence[str],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    choices: Mapping[str, int],
    target_total: int,
) -> dict[str, int]:
    calibrated = {qid: max(1, min(int(choices[qid]), len(rankings[qid]))) for qid in qids}
    total = sum(calibrated.values())
    while total > target_total:
        candidates = [qid for qid in qids if calibrated[qid] > 1]
        if not candidates:
            break
        qid = min(candidates, key=lambda item: (rankings[item][calibrated[item] - 1][1], item))
        calibrated[qid] -= 1
        total -= 1
    while total < target_total:
        candidates = [qid for qid in qids if calibrated[qid] < len(rankings[qid])]
        if not candidates:
            break
        qid = max(candidates, key=lambda item: (rankings[item][calibrated[item]][1], item))
        calibrated[qid] += 1
        total += 1
    return calibrated


def main() -> int:
    started = time.perf_counter()
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [page for page in pages_all if page.text_source in {"pdf_inspector", "tesseract"} and page.text.strip()]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(query["query_id"]) for query in queries]

    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    index = BM25Index(analyzer_name="auto").build(
        {"chunk_id": page.page_id, "doc_id": page.doc_id, "text": page.text}
        for page in pages
    )
    timings["bm25_index_build_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_page_ids = [page.page_id for page in pages_all]
    position_by_id = {page_id: position for position, page_id in enumerate(all_page_ids)}
    positions = np.asarray([position_by_id[page.page_id] for page in pages], dtype=np.int64)
    page_vectors_all = np.load(EMBEDDING_DIR / "page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(EMBEDDING_DIR / "query_embeddings.npy", mmap_mode="r")
    page_vectors = page_vectors_all[positions]
    page_norms = np.linalg.norm(page_vectors, axis=1)
    query_norms = np.linalg.norm(query_vectors, axis=1)
    # Touch the mapped arrays so this timing includes loading the reusable cache.
    _ = float(page_vectors[0, 0]) + float(query_vectors[0, 0])
    timings["embedding_cache_load_seconds"] = time.perf_counter() - t0

    rankings: dict[str, dict[str, list[tuple[str, float]]]] = {
        "bm25": {},
        "dense": {},
        "hybrid": {},
        "rrf10": {},
    }
    t0 = time.perf_counter()
    page_ids = [page.page_id for page in pages]
    for query_index, query in enumerate(queries):
        qid = str(query["query_id"])
        text = str(query["query"])
        bm = [(index.chunk_ids[position], float(score)) for position, score in index.search(text, DEPTH)]
        rankings["bm25"][qid] = bm
        bm_by_id = dict(bm)
        bm_rank = {page_id: rank for rank, (page_id, _score) in enumerate(bm, 1)}

        qv = query_vectors[query_index]
        dense_scores = (page_vectors @ qv) / (page_norms * query_norms[query_index] + 1e-9)
        dense = sorted(
            ((page_id, float(score)) for page_id, score in zip(page_ids, dense_scores)),
            key=lambda item: (-item[1], item[0]),
        )[:DEPTH]
        rankings["dense"][qid] = dense
        dense_by_id = dict(dense)
        dense_rank = {page_id: rank for rank, (page_id, _score) in enumerate(dense, 1)}

        bm_norm = _normalise(bm_by_id)
        dense_norm = _normalise(dense_by_id)
        candidate_ids = set(bm_by_id) | set(dense_by_id)
        hybrid_scores = {
            page_id: 0.10 * bm_norm.get(page_id, 0.0) + 0.90 * dense_norm.get(page_id, 0.0)
            for page_id in candidate_ids
        }
        rankings["hybrid"][qid] = sorted(hybrid_scores.items(), key=lambda item: (-item[1], item[0]))[:DEPTH]
        rrf_scores = {
            page_id: 1.0 / (10 + bm_rank.get(page_id, 1000)) + 1.0 / (10 + dense_rank.get(page_id, 1000))
            for page_id in candidate_ids
        }
        rankings["rrf10"][qid] = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))[:DEPTH]
    timings["ranking_seconds"] = time.perf_counter() - t0

    policies: dict[str, dict[str, float]] = {}
    policy_rows: dict[str, list[dict[str, Any]]] = {}
    t0 = time.perf_counter()
    target_total = FIXED_K * len(qids)
    for name, ranking in rankings.items():
        fixed_choices = {qid: FIXED_K for qid in qids}
        fixed_name = f"{name}_fixed_k{FIXED_K}"
        rows = _rows(qids, ranking, fixed_choices, fixed_name)
        policy_rows[fixed_name] = rows
        policies[fixed_name] = _metrics(rows, qrels)

        raw_choices = _adaptive_choices(qids, ranking, retrieve_more=5)
        raw_name = f"{name}_adaptive_gap_plus5_raw"
        raw_rows = _rows(qids, ranking, raw_choices, raw_name)
        policy_rows[raw_name] = raw_rows
        policies[raw_name] = _metrics(raw_rows, qrels)

        exact_choices = _calibrate_choices(qids, ranking, raw_choices, target_total)
        exact_name = f"{name}_adaptive_gap_plus5_exact_mean12"
        exact_rows = _rows(qids, ranking, exact_choices, exact_name)
        policy_rows[exact_name] = exact_rows
        policies[exact_name] = _metrics(exact_rows, qrels)
    timings["adaptive_calibration_seconds"] = time.perf_counter() - t0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, rows in policy_rows.items():
        (OUTPUT / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )
    embedding_report_path = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/report.json"
    embedding_seconds = None
    if embedding_report_path.exists():
        embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
        embedding_seconds = (
            embedding_report.get("timing_seconds", {}).get("embedding")
            or embedding_report.get("embedding_seconds")
        )
    timings["cached_embedding_preparation_seconds_from_previous_run"] = embedding_seconds
    timings["local_reused_run_seconds"] = time.perf_counter() - started
    timings["fresh_run_estimated_seconds"] = (
        timings["local_reused_run_seconds"] + float(embedding_seconds or 0.0)
    )
    report = {
        "benchmark": {"queries": len(qids), "pages_all": len(pages_all), "indexed_pages": len(pages)},
        "protocol": {
            "text_sources": ["pdf_inspector", "tesseract"],
            "scope": "global page retrieval; no hierarchical file stage",
            "fixed_k": FIXED_K,
            "adaptive_rule": "largest descending score gap, tail=0.1, retrieve_more=5",
            "adaptive_budget_calibration": "score-only trimming/adding to total K=12*220",
            "qrels_used_for_ranking_or_calibration": False,
            "hybrid": "normalized BM25 0.10 + dense cosine 0.90 over union of top-100 candidates",
            "rrf": "RRF constant 10 over BM25 and dense top-100 candidates",
        },
        "timings_seconds": timings,
        "policies": policies,
        "notes": [
            "Dense page/query embeddings were reused from the existing cache; this run made no embedding API call.",
            "The previous embedding report records 155.64 seconds for the original embedding preparation.",
            "All fixed and exact adaptive policies have the same total page budget; raw adaptive rows are included to show the uncalibrated choice.",
        ],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fair dense / hybrid / adaptive comparison",
        "",
        f"- Benchmark: {len(qids)} queries; {len(pages)} indexed pages from PDF-inspector + Tesseract.",
        f"- Fixed budget: K={FIXED_K}; exact adaptive rows have the same total budget (mean K=12).",
        "- Dense embeddings are cached and reused; no new API call was made.",
        "",
        "| Policy | Page recall@K | Page recall@10 | Page recall@20 | nDCG@10 | nDCG@20 | File recall@K | File recall@10 | File recall@20 | Mean K |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in policies.items():
        lines.append(
            f"| {name} | {value['page_recall_at_returned_k']:.2%} | {value['page_recall@10']:.2%} | {value['page_recall@20']:.2%} | {value['page_ndcg@10']:.4f} | {value['page_ndcg@20']:.4f} | {value['file_recall_at_returned_k']:.2%} | {value['file_recall@10']:.2%} | {value['file_recall@20']:.2%} | {value['mean_k']:.2f} |"
        )
    lines.extend([
        "",
        "## Timing",
        "",
        "```json",
        json.dumps(timings, ensure_ascii=False, indent=2),
        "```",
    ])
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
