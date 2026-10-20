"""Compare the original bundle light BM25 run with dense and BM25 Adaptive-k.

The bundle is the source of truth for the original Kf=3/Kp=10 BM25 result
(37.16% page recall@10). Dense vectors are reused from the existing benchmark
cache and are evaluated with the same hierarchical Kf=3 page-stage protocol.
"""

from __future__ import annotations

from collections import defaultdict
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

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_BUNDLE = ROOT / "data/work/benchmark_kf3_top20_kdl_ocr64_t300-20260917T181644Z-1-001"
DEFAULT_EMBEDDINGS = ROOT / "data/work/benchmark_hierarchical_local/embedding_adaptive_openrouter_te3s_fulltext/embeddings"
DEFAULT_DATASET = ROOT / "data/raw/BENCHMARK"
DEFAULT_OUTPUT = ROOT / "data/work/benchmark_hierarchical_local/bundle_light_dense_bm25_adaptive"
FILE_K = 3
FIXED_PAGE_K = 10
SAVED_PAGE_K = 20
ADAPTIVE_RETRIEVE_MORE = 5


def _bundle_run_dir(bundle: Path) -> Path:
    candidates = sorted(bundle.glob("*/light_retry32_eng_v2"))
    if not candidates:
        raise FileNotFoundError(f"Could not find light_retry32_eng_v2 under {bundle}")
    return candidates[-1]


def _normalise(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max((float(value) for value in scores.values() if float(value) > 0), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: max(0.0, float(value)) / maximum for key, value in scores.items()}


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _ndcg(ranked: Sequence[str], gold: Mapping[str, int], k: int) -> float:
    gains = {str(page_id): int(score) for page_id, score in gold.items()}
    dcg = sum(
        (2.0 ** gains[page_id] - 1.0) / np.log2(rank + 2)
        for rank, page_id in enumerate(ranked[:k])
        if page_id in gains
    )
    ideal_scores = sorted(gains.values(), reverse=True)[:k]
    ideal = sum((2.0 ** score - 1.0) / np.log2(rank + 2) for rank, score in enumerate(ideal_scores))
    return float(dcg / ideal) if ideal else 0.0


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "median": ordered[len(ordered) // 2],
        "p90": ordered[min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.90)) - 1)],
        "max": max(values),
        "mean": float(np.mean(values)),
        "counts": dict(sorted((str(k), v) for k, v in defaultdict(int, ((x, values.count(x)) for x in set(values))).items())),
    }


def _metrics(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    returned, recall10, recall20, hit10, ndcg_returned, ndcg10, ndcg20 = [], [], [], [], [], [], []
    selected_file_recall, derived_file_recall, returned_union_file_recall = [], [], []
    page_counts, candidate_counts = [], []
    for row in rows:
        qid = str(row["query_id"])
        gold_relevance = qrels[qid]
        gold = set(gold_relevance)
        pages = [str(page_id) for page_id in row["returned_pages"]]
        denominator = max(len(gold), 1)
        returned.append(len(set(pages) & gold) / denominator)
        recall10.append(len(set(pages[:10]) & gold) / denominator)
        recall20.append(len(set(pages[:20]) & gold) / denominator)
        hit10.append(float(bool(set(pages[:10]) & gold)))
        ndcg_returned.append(_ndcg(pages, gold_relevance, len(pages)))
        ndcg10.append(_ndcg(pages, gold_relevance, 10))
        ndcg20.append(_ndcg(pages, gold_relevance, 20))
        gold_files = {_file_id(page_id) for page_id in gold}
        selected = {str(item["file_id"]) for item in row["selected_files"]}
        selected_file_recall.append(len(selected & gold_files) / max(len(gold_files), 1))
        returned_union = {_file_id(page_id) for page_id in pages}
        returned_union_file_recall.append(len(returned_union & gold_files) / max(len(gold_files), 1))
        derived = []
        for page_id in pages[:20]:
            file_id = _file_id(page_id)
            if file_id not in derived:
                derived.append(file_id)
        derived_file_recall.append(len(set(derived[:3]) & gold_files) / max(len(gold_files), 1))
        page_counts.append(len(pages))
        candidate_counts.append(int(row["candidate_page_count"]))
    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {
        "queries": len(rows),
        "page_recall_at_returned_k": mean(returned),
        "page_recall@10": mean(recall10),
        "page_recall@20": mean(recall20),
        "page_hit@10": mean(hit10),
        "ndcg_at_returned_k": 100.0 * mean(ndcg_returned),
        "ndcg@10": 100.0 * mean(ndcg10),
        "ndcg@20": 100.0 * mean(ndcg20),
        "file_candidate_recall@3": mean(selected_file_recall),
        "derived_file_recall@3": mean(derived_file_recall),
        "file_recall_at_returned_k_union": mean(returned_union_file_recall),
        "mean_returned_pages": mean(page_counts),
        "mean_candidate_pages": mean(candidate_counts),
        "returned_page_distribution": _distribution(page_counts),
    }


def _make_rows(
    qids: Sequence[str],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    choices: Mapping[str, int],
    selected_files: Mapping[str, Sequence[str]],
    candidate_counts: Mapping[str, int],
    policy: str,
) -> list[dict[str, Any]]:
    rows = []
    for qid in qids:
        ranking = list(rankings[qid])
        k = max(1, min(int(choices[qid]), len(ranking)))
        rows.append({
            "query_id": qid,
            "policy": policy,
            "kf": len(selected_files[qid]),
            "kp": k,
            "selected_files": [{"file_id": file_id} for file_id in selected_files[qid]],
            "candidate_page_count": candidate_counts[qid],
            "returned_pages": [page_id for page_id, _score in ranking[:k]],
            "chunks": [{"page_id": page_id, "score": float(score), "rank": rank} for rank, (page_id, score) in enumerate(ranking[:k], 1)],
        })
    return rows


def _adaptive_choices(qids: Sequence[str], rankings: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, int]:
    return {
        qid: int(adaptive_k(
            [score for _page_id, score in rankings[qid]],
            ignore_extreme=0,
            ignore_extreme_tail=0.1,
            retrieve_more=ADAPTIVE_RETRIEVE_MORE,
        )["k"])
        for qid in qids
    }


def _calibrate(qids: Sequence[str], rankings: Mapping[str, Sequence[tuple[str, float]]], choices: Mapping[str, int], target_total: int) -> dict[str, int]:
    output = {qid: max(1, min(int(choices[qid]), len(rankings[qid]))) for qid in qids}
    total = sum(output.values())
    while total > target_total:
        candidates = [qid for qid in qids if output[qid] > 1]
        qid = min(candidates, key=lambda item: (rankings[item][output[item] - 1][1], item))
        output[qid] -= 1
        total -= 1
    while total < target_total:
        candidates = [qid for qid in qids if output[qid] < len(rankings[qid])]
        qid = max(candidates, key=lambda item: (rankings[item][output[item]][1], item))
        output[qid] += 1
        total += 1
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()

    run_dir = _bundle_run_dir(args.bundle.resolve())
    pages = [RawPage(**row) for row in read_jsonl(run_dir / "parsing/pages.jsonl")]
    pages = [page for page in pages if page.text.strip()]
    queries = read_jsonl(args.dataset_root.resolve() / "queries.jsonl")
    qids = [str(row["query_id"]) for row in queries]
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in read_jsonl(run_dir / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))
    page_ids = [page.page_id for page in pages]
    page_by_id = {page.page_id: page for page in pages}
    file_ids = sorted({page.doc_id for page in pages})
    page_to_file = {page.page_id: page.doc_id for page in pages}
    page_counts_by_file: dict[str, int] = defaultdict(int)
    for page in pages:
        page_counts_by_file[page.doc_id] += 1

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    page_index = BM25Index.load(run_dir / "indexes/page_bm25.json")
    file_index = BM25Index.load(run_dir / "indexes/file_bm25.json")
    timings["load_bundle_bm25_indexes_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    all_cache_pages = [RawPage(**row) for row in read_jsonl(args.embedding_dir.resolve().parents[1] / "parsing/pages.jsonl")]
    all_cache_ids = [page.page_id for page in all_cache_pages]
    cache_position = {page_id: index for index, page_id in enumerate(all_cache_ids)}
    positions = np.asarray([cache_position[page_id] for page_id in page_ids], dtype=np.int64)
    page_vectors_all = np.load(args.embedding_dir / "page_embeddings.npy", mmap_mode="r")
    query_vectors = np.load(args.embedding_dir / "query_embeddings.npy", mmap_mode="r")
    page_vectors = page_vectors_all[positions]
    page_norms = np.linalg.norm(page_vectors, axis=1)
    query_norms = np.linalg.norm(query_vectors, axis=1)
    _ = float(page_vectors[0, 0]) + float(query_vectors[0, 0])
    timings["load_dense_cache_seconds"] = time.perf_counter() - t0

    rankings: dict[str, dict[str, list[tuple[str, float]]]] = {
        "bm25": {}, "dense": {}, "bm25_global": {}, "dense_global": {}
    }
    selected_files: dict[str, dict[str, list[str]]] = {
        "bm25": {}, "dense": {}, "bm25_global": {}, "dense_global": {}
    }
    candidate_counts: dict[str, dict[str, int]] = {
        "bm25": {}, "dense": {}, "bm25_global": {}, "dense_global": {}
    }
    t0 = time.perf_counter()
    for query_index, query in enumerate(queries):
        qid = str(query["query_id"])
        text = str(query["query"])

        page_hits = page_index.search(text, len(pages))
        page_bm25 = {page_index.chunk_ids[position]: float(score) for position, score in page_hits}
        page_norm = _normalise(page_bm25)
        page_base = {page_id: 0.70 * page_norm.get(page_id, 0.0) for page_id in page_ids}
        direct_hits = file_index.search(text, len(file_ids))
        direct = {file_index.chunk_ids[position]: float(score) for position, score in direct_hits}
        file_pool = {file_id: max((page_base[page_id] for page_id in page_ids if page_to_file[page_id] == file_id), default=0.0) for file_id in file_ids}
        file_scores = {
            file_id: 0.50 * _normalise(direct).get(file_id, 0.0) + 0.50 * _normalise(file_pool).get(file_id, 0.0)
            for file_id in file_ids
        }
        bm25_files = [file_id for file_id, _score in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))[:FILE_K]]
        bm25_parent = _normalise(file_scores)
        bm25_page_scores = {
            page_id: 0.85 * page_base[page_id] + 0.15 * bm25_parent.get(page_to_file[page_id], 0.0)
            for page_id in page_ids if page_to_file[page_id] in set(bm25_files)
        }
        rankings["bm25"][qid] = sorted(bm25_page_scores.items(), key=lambda item: (-item[1], item[0]))
        selected_files["bm25"][qid] = bm25_files
        candidate_counts["bm25"][qid] = sum(page_counts_by_file[file_id] for file_id in bm25_files)
        rankings["bm25_global"][qid] = sorted(
            ((page_id, page_norm.get(page_id, 0.0)) for page_id in page_ids),
            key=lambda item: (-item[1], item[0]),
        )
        global_files = []
        for page_id, _score in rankings["bm25_global"][qid][:SAVED_PAGE_K]:
            file_id = page_to_file[page_id]
            if file_id not in global_files:
                global_files.append(file_id)
        selected_files["bm25_global"][qid] = global_files[:FILE_K]
        candidate_counts["bm25_global"][qid] = len(page_ids)

        dense_scores = (page_vectors @ query_vectors[query_index]) / (page_norms * query_norms[query_index] + 1e-9)
        dense_raw = {page_id: float(score) for page_id, score in zip(page_ids, dense_scores)}
        dense_base = _normalise(dense_raw)
        dense_file_pool = {file_id: max((dense_base[page_id] for page_id in page_ids if page_to_file[page_id] == file_id), default=0.0) for file_id in file_ids}
        dense_files = [file_id for file_id, _score in sorted(dense_file_pool.items(), key=lambda item: (-item[1], item[0]))[:FILE_K]]
        dense_parent = _normalise(dense_file_pool)
        dense_page_scores = {
            page_id: 0.85 * dense_base[page_id] + 0.15 * dense_parent.get(page_to_file[page_id], 0.0)
            for page_id in page_ids if page_to_file[page_id] in set(dense_files)
        }
        rankings["dense"][qid] = sorted(dense_page_scores.items(), key=lambda item: (-item[1], item[0]))
        selected_files["dense"][qid] = dense_files
        candidate_counts["dense"][qid] = sum(page_counts_by_file[file_id] for file_id in dense_files)
        rankings["dense_global"][qid] = sorted(
            ((page_id, dense_base[page_id]) for page_id in page_ids),
            key=lambda item: (-item[1], item[0]),
        )
        global_files = []
        for page_id, _score in rankings["dense_global"][qid][:SAVED_PAGE_K]:
            file_id = page_to_file[page_id]
            if file_id not in global_files:
                global_files.append(file_id)
        selected_files["dense_global"][qid] = global_files[:FILE_K]
        candidate_counts["dense_global"][qid] = len(page_ids)
    timings["query_ranking_seconds"] = time.perf_counter() - t0

    policies: dict[str, dict[str, Any]] = {}
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    t0 = time.perf_counter()
    for name in ("bm25", "dense", "bm25_global", "dense_global"):
        fixed10 = {qid: FIXED_PAGE_K for qid in qids}
        fixed20 = {qid: SAVED_PAGE_K for qid in qids}
        is_global = name.endswith("_global")
        budget_label = "fixed_kp" if is_global else "fixed_kf3_kp"
        p10 = f"{name}_{budget_label}10"
        p20 = f"{name}_{budget_label}20"
        rows_by_policy[p10] = _make_rows(qids, rankings[name], fixed10, selected_files[name], candidate_counts[name], p10)
        rows_by_policy[p20] = _make_rows(qids, rankings[name], fixed20, selected_files[name], candidate_counts[name], p20)
        policies[p10] = _metrics(rows_by_policy[p10], qrels)
        policies[p20] = _metrics(rows_by_policy[p20], qrels)

        raw_choices = _adaptive_choices(qids, rankings[name])
        raw_name = f"{name}_adaptive_gap_plus{ADAPTIVE_RETRIEVE_MORE}_raw"
        rows_by_policy[raw_name] = _make_rows(qids, rankings[name], raw_choices, selected_files[name], candidate_counts[name], raw_name)
        policies[raw_name] = _metrics(rows_by_policy[raw_name], qrels)

        exact_choices = _calibrate(qids, rankings[name], raw_choices, FIXED_PAGE_K * len(qids))
        exact_name = f"{name}_adaptive_gap_plus{ADAPTIVE_RETRIEVE_MORE}_exact_mean10"
        rows_by_policy[exact_name] = _make_rows(qids, rankings[name], exact_choices, selected_files[name], candidate_counts[name], exact_name)
        policies[exact_name] = _metrics(rows_by_policy[exact_name], qrels)
    timings["adaptive_calibration_and_evaluation_seconds"] = time.perf_counter() - t0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for policy, rows in rows_by_policy.items():
        (args.output_dir / f"{policy}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows) + "\n",
            encoding="utf-8",
        )

    original_report = json.loads((run_dir / "reports/report.json").read_text(encoding="utf-8"))
    bundle_timing = original_report.get("timing", {})
    timings["this_offline_run_seconds"] = time.perf_counter() - started
    report = {
        "bundle_run_dir": str(run_dir),
        "benchmark": {"queries": len(qids), "pages_indexed": len(pages), "files": len(file_ids)},
        "original_bundle_reference": {
            "page_recall@10": original_report.get("metrics", {}).get("page_recall@10"),
            "page_recall@20": original_report.get("metrics", {}).get("page_recall@20"),
            "report": str(run_dir / "reports/report.json"),
            "config": str(run_dir / "config.json"),
            "original_retrieval_seconds": bundle_timing.get("retrieval_seconds"),
        },
        "protocol": {
            "file_k": FILE_K,
            "fixed_page_k": FIXED_PAGE_K,
            "saved_page_k": SAVED_PAGE_K,
            "adaptive": "largest descending score gap, tail=0.1, retrieve_more=5",
            "adaptive_exact_budget": "score-only calibration to total K=10*220",
            "bm25_page_score": "0.85 * normalized page BM25 + 0.15 * normalized selected-file score",
            "dense_page_score": "0.85 * normalized dense cosine + 0.15 * normalized dense max-page file score",
            "no_file_stage": "bm25_global/dense_global rank all indexed pages directly; file recall is derived from top-20 page files",
            "qrels_used_for_ranking_or_calibration": False,
            "dense_cache_note": "Dense cache is aligned by page_id; its preparation used the existing local benchmark text cache.",
        },
        "timings_seconds": timings,
        "policies": policies,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Bundle light BM25 versus dense and Adaptive-k",
        "",
        f"- Bundle reference: `{run_dir}`; {len(qids)} queries, {len(pages)} non-empty indexed pages, {len(file_ids)} files.",
        "- Hierarchical protocol: fixed Kf=3; page metrics at Kp=10 and Kp=20.",
        "- BM25 reference is the original bundle result (page recall@10=37.16%).",
        "- Adaptive rows are shown both raw and score-only calibrated to the same total Kp=10 budget.",
        "",
        "| Policy | Recall@K | nDCG@K | Recall@10 | Recall@20 | nDCG@10 | nDCG@20 | File recall@3 | Derived file recall@3 | File recall@K union | Mean Kp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in policies.items():
        lines.append(f"| {name} | {value['page_recall_at_returned_k']:.2%} | {value['ndcg_at_returned_k']:.2f} | {value['page_recall@10']:.2%} | {value['page_recall@20']:.2%} | {value['ndcg@10']:.2f} | {value['ndcg@20']:.2f} | {value['file_candidate_recall@3']:.2%} | {value['derived_file_recall@3']:.2%} | {value['file_recall_at_returned_k_union']:.2%} | {value['mean_returned_pages']:.2f} |")
    lines.extend(["", "## Timing", "", "```json", json.dumps(timings, ensure_ascii=False, indent=2), "```"])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
