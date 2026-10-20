"""Run the upstream Adaptive-k rule on the AXIOM benchmark cache.

This adapter keeps the retrieval scores and hierarchy from the completed
light-retrieval run, and ports only the Adaptive-k selection rule from
``megagonlabs/adaptive-k-retrieval``:

* sort candidate scores in descending order;
* find the largest drop in the score curve;
* optionally ignore the tail and add a small retrieval buffer; and
* return the prefix ending at that boundary.

The direct comparison is page-level Adaptive-k with fixed ``Kf=3``.  The
script also emits fixed ``Kp=10`` and ``Kp=20`` baselines in the same output
directory; Kp=20 matches the saved depth of the completed light run.
No parsing, OCR, rendering, encoding, or API calls are performed. Qrels are
loaded only after score-based rankings have been produced.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import (  # noqa: E402
    RawPage,
    aggregate_file_scores,
    normalise_scores,
    read_jsonl,
    sort_scores,
)
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_CACHE_DIR = ROOT / "data/work/benchmark_hierarchical_local"
DEFAULT_DATASET_ROOT = ROOT / "data/raw/BENCHMARK"
DEFAULT_OUTPUT_DIR = DEFAULT_CACHE_DIR / "adaptive_k_upstream"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _int_or_float(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        return float(value)


def adaptive_k(
    scores: Sequence[float],
    *,
    ignore_extreme: int | float = 0.0,
    ignore_extreme_tail: int | float = 0.1,
    retrieve_more: int | float = 5,
) -> dict[str, float | int]:
    """Port ``Retriever.find_threshold_largest_gap`` faithfully.

    The upstream implementation returns the zero-based gap index and then
    includes ranks ``0..threshold``.  Therefore the unbuffered prefix length
    is ``gap_index + 1``.  The optional ``retrieve_more`` adjustment is made
    to the threshold before converting it to a prefix length, matching the
    upstream code.
    """
    values = [float(score) for score in scores]
    if not values:
        return {"gap_index": -1, "gap_rank": 0, "gap": 0.0, "k": 0}
    if len(values) == 1:
        return {"gap_index": 0, "gap_rank": 1, "gap": 0.0, "k": 1}

    def cut(value: int | float) -> int:
        if isinstance(value, float):
            return int((len(values) - 1) * value)
        return int(value)

    start = max(0, min(len(values) - 2, cut(ignore_extreme)))
    tail = max(0, cut(ignore_extreme_tail))
    end = len(values) - 1 - tail
    end = max(start + 1, min(len(values) - 1, end))

    # The upstream code uses argmin(diff(scores)) because scores are sorted
    # descending. This is the same as argmax(score[i] - score[i+1]).
    gap_index = min(
        range(start, end),
        key=lambda index: (-(values[index] - values[index + 1]), index),
    )
    gap = max(0.0, values[gap_index] - values[gap_index + 1])
    threshold = gap_index
    if isinstance(retrieve_more, float):
        threshold = int(threshold * retrieve_more)
    elif isinstance(retrieve_more, int):
        threshold = threshold + retrieve_more

    k = max(1, min(len(values), threshold + 1))
    return {
        "gap_index": int(gap_index),
        "gap_rank": int(gap_index + 1),
        "gap": float(gap),
        "k": int(k),
    }


def _ndcg(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    gains = {str(page): int(score) for page, score in qrels.items()}
    actual = sum(
        (2.0**gains[page] - 1.0) / __import__("math").log2(rank + 1)
        for rank, page in enumerate(ranked[:k], 1)
        if page in gains
    )
    ideal_scores = sorted(gains.values(), reverse=True)[:k]
    ideal = sum(
        (2.0**score - 1.0) / __import__("math").log2(rank + 1)
        for rank, score in enumerate(ideal_scores, 1)
    )
    return actual / ideal if ideal else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    counts = Counter(int(value) for value in values)
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "max": 0, "mean": 0.0, "counts": {}}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[(len(ordered) - 1) // 2],
        "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "max": ordered[-1],
        "mean": _mean(ordered),
        "counts": {str(key): counts[key] for key in sorted(counts)},
    }


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    metric_page_k: int,
) -> dict[str, Any]:
    page_recall_returned: list[float] = []
    page_recall10: list[float] = []
    page_recall20: list[float] = []
    page_hit10: list[float] = []
    ndcg10: list[float] = []
    file_recall_selected: list[float] = []
    derived_file_recall3: list[float] = []
    returned_pages: list[int] = []
    returned_files: list[int] = []
    selected_files_count: list[int] = []
    candidate_pages: list[int] = []
    kf_values: list[int] = []
    kp_values: list[int] = []

    for row in rows:
        query_id = str(row["query_id"])
        gold = set(qrels.get(query_id, {}))
        returned = [str(page_id) for page_id in row["returned_pages"]]
        selected = [
            str(item["file_id"]) if isinstance(item, Mapping) else str(item)
            for item in row["selected_files"]
        ]
        top10 = returned[:metric_page_k]
        top20 = returned[:20]
        gold_files = {page_id.split("#page=", 1)[0] for page_id in gold}
        derived: list[str] = []
        for page_id in top20:
            file_id = page_id.split("#page=", 1)[0]
            if file_id not in derived:
                derived.append(file_id)

        denominator = len(gold)
        page_recall_returned.append(len(set(returned) & gold) / denominator if denominator else 0.0)
        page_recall10.append(len(set(top10) & gold) / denominator if denominator else 0.0)
        page_recall20.append(len(set(top20) & gold) / denominator if denominator else 0.0)
        page_hit10.append(float(bool(set(top10) & gold)))
        ndcg10.append(_ndcg(top10, qrels.get(query_id, {}), metric_page_k))
        file_recall_selected.append(
            len(set(selected) & gold_files) / len(gold_files) if gold_files else 0.0
        )
        derived_file_recall3.append(
            len(set(derived[:3]) & gold_files) / len(gold_files) if gold_files else 0.0
        )
        returned_pages.append(len(returned))
        returned_files.append(len({page_id.split("#page=", 1)[0] for page_id in returned}))
        selected_files_count.append(len(selected))
        candidate_pages.append(int(row["candidate_page_count"]))
        kf_values.append(int(row["kf"]))
        kp_values.append(int(row["kp"]))

    return {
        "queries": len(rows),
        "page_recall_at_returned_k": _mean(page_recall_returned),
        "page_recall@10": _mean(page_recall10),
        "page_recall@20": _mean(page_recall20),
        "page_hit@10": _mean(page_hit10),
        "ndcg@10": 100.0 * _mean(ndcg10),
        "file_recall_at_selected_kf": _mean(file_recall_selected),
        "derived_file_recall@3": _mean(derived_file_recall3),
        "cost": {
            "mean_returned_pages": _mean(returned_pages),
            "mean_returned_files": _mean(returned_files),
            "mean_selected_files": _mean(selected_files_count),
            "mean_candidate_pages": _mean(candidate_pages),
            "returned_page_distribution": _distribution(returned_pages),
            "kf_distribution": _distribution(kf_values),
            "kp_distribution": _distribution(kp_values),
        },
    }


def _rank_pages(
    pages: Sequence[RawPage],
    page_scores: Mapping[str, float],
) -> list[tuple[str, float]]:
    return sort_scores(page_scores)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fixed-file-k", type=int, default=3)
    parser.add_argument("--fixed-page-k", type=int, default=10)
    parser.add_argument("--metric-page-k", type=int, default=10)
    parser.add_argument("--ignore-extreme", type=_int_or_float, default=0.0)
    parser.add_argument("--ignore-extreme-tail", type=_int_or_float, default=0.1)
    parser.add_argument("--retrieve-more", type=_int_or_float, default=5)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cache = args.cache_dir.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    pages = [RawPage(**row) for row in read_jsonl(cache / "parsing/pages.jsonl")]
    queries = read_jsonl(dataset_root / "queries.jsonl")
    page_index = BM25Index.load(cache / "indexes/page_bm25.json")
    file_index = BM25Index.load(cache / "indexes/file_bm25.json")
    normalized_qrels = read_jsonl(cache / "qrels_page_level.jsonl")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    for row in normalized_qrels:
        qrels[str(row["query_id"])][str(row["page_id"])] = int(row.get("relevance", 1))

    documents = read_jsonl(dataset_root / "documents.jsonl")
    file_ids = [str(row["doc_id"]) for row in documents]
    pages_by_id = {page.page_id: page for page in pages}
    page_to_file = {page.page_id: page.doc_id for page in pages}
    page_by_file: dict[str, list[RawPage]] = defaultdict(list)
    for page in pages:
        page_by_file[page.doc_id].append(page)

    fixed10_rows: list[dict[str, Any]] = []
    fixed20_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for query in queries:
        query_id = str(query["query_id"])
        text = str(query["query"])
        page_hits = page_index.search(text, len(pages))
        page_bm25 = {page_index.chunk_ids[position]: float(score) for position, score in page_hits}
        page_norm = normalise_scores(page_bm25)
        page_base = {
            page.page_id: 0.70 * page_norm.get(page.page_id, 0.0)
            for page in pages
        }
        direct_hits = file_index.search(text, len(file_ids))
        direct = {file_index.chunk_ids[position]: float(score) for position, score in direct_hits}
        file_pool = aggregate_file_scores(page_base, page_to_file, file_ids)
        direct_norm = normalise_scores(direct)
        pool_norm = normalise_scores(file_pool)
        file_scores = {
            file_id: 0.50 * direct_norm.get(file_id, 0.0)
            + 0.50 * pool_norm.get(file_id, 0.0)
            for file_id in file_ids
        }
        file_ranked = sort_scores(file_scores)
        fixed_files = [file_id for file_id, _score in file_ranked[: args.fixed_file_k]]
        parent_norm = normalise_scores(file_scores)

        def page_ranking(selected_files: Sequence[str]) -> list[tuple[str, float]]:
            selected = set(selected_files)
            scores = {
                page.page_id: 0.85 * page_base[page.page_id]
                + 0.15 * parent_norm.get(page.doc_id, 0.0)
                for page in pages
                if page.doc_id in selected
            }
            return _rank_pages(pages, scores)

        fixed_ranking = page_ranking(fixed_files)
        adaptive_info = adaptive_k(
            [score for _page_id, score in fixed_ranking],
            ignore_extreme=args.ignore_extreme,
            ignore_extreme_tail=args.ignore_extreme_tail,
            retrieve_more=args.retrieve_more,
        )
        adaptive_ranking = fixed_ranking[: int(adaptive_info["k"])]

        def make_row(
            ranking: Sequence[tuple[str, float]],
            *,
            selected_files: Sequence[str],
            kp: int,
            policy: str,
            adaptive_metadata: Mapping[str, Any] | None,
        ) -> dict[str, Any]:
            chunks = []
            for rank, (page_id, score) in enumerate(ranking, start=1):
                page = pages_by_id[page_id]
                chunks.append(
                    {
                        "page_id": page_id,
                        "doc_id": page.doc_id,
                        "source": page.source,
                        "relative_path": page.relative_path,
                        "page_index": page.page_index,
                        "page_number": page.page_number,
                        "rank": rank,
                        "score": round(float(score), 8),
                        "components": {
                            "page_base": round(float(page_base[page_id]), 8),
                            "parent_file": round(float(parent_norm.get(page.doc_id, 0.0)), 8),
                        },
                    }
                )
            row: dict[str, Any] = {
                "query_id": query_id,
                "query": text,
                "policy": policy,
                "kf": len(selected_files),
                "kp": len(chunks),
                "selected_files": [
                    {"file_id": file_id, "score": round(float(file_scores[file_id]), 8)}
                    for file_id in selected_files
                ],
                "candidate_page_count": sum(len(page_by_file[file_id]) for file_id in selected_files),
                "returned_pages": [page_id for page_id, _score in ranking],
                "chunks": chunks,
            }
            if adaptive_metadata is not None:
                row["adaptive_k"] = dict(adaptive_metadata)
            return row

        fixed10_rows.append(
            make_row(
                fixed_ranking[: args.fixed_page_k],
                selected_files=fixed_files,
                kp=args.fixed_page_k,
                policy=f"fixed_kf{args.fixed_file_k}_kp{args.fixed_page_k}",
                adaptive_metadata=None,
            )
        )
        fixed20_rows.append(
            make_row(
                fixed_ranking[: max(args.fixed_page_k, 20)],
                selected_files=fixed_files,
                kp=max(args.fixed_page_k, 20),
                policy=f"fixed_kf{args.fixed_file_k}_kp{max(args.fixed_page_k, 20)}",
                adaptive_metadata=None,
            )
        )
        adaptive_rows.append(
            make_row(
                adaptive_ranking,
                selected_files=fixed_files,
                kp=int(adaptive_info["k"]),
                policy="adaptive_k_page_largest_gap",
                adaptive_metadata=adaptive_info,
            )
        )
        traces.append(
            {
                "query_id": query_id,
                "query": text,
                "candidate_page_count": sum(len(page_by_file[file_id]) for file_id in fixed_files),
                "file_scores": [
                    {"file_id": file_id, "score": round(float(score), 8)}
                    for file_id, score in file_ranked
                ],
                "page_scores_kf3": [
                    {"page_id": page_id, "score": round(float(score), 8)}
                    for page_id, score in fixed_ranking
                ],
                "adaptive_k": adaptive_info,
            }
        )

    fixed10_metrics = _evaluate(fixed10_rows, qrels, metric_page_k=args.metric_page_k)
    fixed20_metrics = _evaluate(fixed20_rows, qrels, metric_page_k=args.metric_page_k)
    adaptive_metrics = _evaluate(adaptive_rows, qrels, metric_page_k=args.metric_page_k)
    report = {
        "contract_version": "benchmark-adaptive-k-upstream-page-adapter-v1",
        "cache_dir": str(cache),
        "dataset_root": str(dataset_root),
        "queries": len(queries),
        "files": len(file_ids),
        "pages": len(pages),
        "retrieval": {
            "score_source": "cached hierarchical BM25-only",
            "visual_mode": "disabled",
            "file_selection": f"fixed top-{args.fixed_file_k}",
            "page_score": "0.85 * page_base + 0.15 * parent_file",
            "qrels_used_for_ranking": False,
        },
        "adaptive_k": {
            "implementation": "largest descending-score gap, matching upstream Retriever",
            "ignore_extreme": args.ignore_extreme,
            "ignore_extreme_tail": args.ignore_extreme_tail,
            "retrieve_more": args.retrieve_more,
            "min_k": 1,
            "max_k": "candidate_page_count",
        },
        "metrics": {
            "fixed_kp10": fixed10_metrics,
            "fixed_kp20": fixed20_metrics,
            "adaptive_k": adaptive_metrics,
        },
        "timing_seconds": {"offline_retrieval_and_evaluation": round(time.perf_counter() - started, 3)},
    }
    _write_json(output_dir / "report.json", report)
    _write_jsonl(output_dir / "fixed_kp10_per_query.jsonl", fixed10_rows)
    _write_jsonl(output_dir / "fixed_kp20_per_query.jsonl", fixed20_rows)
    _write_jsonl(output_dir / "fixed_per_query.jsonl", fixed10_rows)
    _write_jsonl(output_dir / "adaptive_per_query.jsonl", adaptive_rows)
    _write_jsonl(output_dir / "score_traces.jsonl", traces)

    fixed10 = fixed10_metrics
    fixed20 = fixed20_metrics
    adaptive = adaptive_metrics
    markdown = [
        "# Adaptive-k upstream adapter on the AXIOM benchmark",
        "",
        f"- Files: **{len(file_ids)}**; pages: **{len(pages)}**; queries: **{len(queries)}**.",
        "- Preparation: reused cached PDF-inspector/Tesseract + BM25 artifacts; no OCR, rendering, encoding, or API call.",
        f"- File stage: fixed top-{args.fixed_file_k}; page stage: cached hierarchical score; metric depth: top-{args.metric_page_k}.",
        f"- Adaptive-k: largest descending-score gap, tail ignored={args.ignore_extreme_tail}, retrieve_more={args.retrieve_more}.",
        "",
        "## Fixed versus Adaptive-k",
        "",
        "| Policy | Recall at returned K | Recall@10 | Recall@20 | nDCG@10 | File recall selected Kf | Mean returned pages | Median Kp | P90 Kp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| fixed Kf={args.fixed_file_k}, Kp={args.fixed_page_k} | {fixed10['page_recall_at_returned_k']:.2%} | {fixed10['page_recall@10']:.2%} | {fixed10['page_recall@20']:.2%} | {fixed10['ndcg@10']:.2f} | {fixed10['file_recall_at_selected_kf']:.2%} | {fixed10['cost']['mean_returned_pages']:.2f} | {fixed10['cost']['kp_distribution']['median']} | {fixed10['cost']['kp_distribution']['p90']} |",
        f"| fixed Kf={args.fixed_file_k}, Kp={max(args.fixed_page_k, 20)} | {fixed20['page_recall_at_returned_k']:.2%} | {fixed20['page_recall@10']:.2%} | {fixed20['page_recall@20']:.2%} | {fixed20['ndcg@10']:.2f} | {fixed20['file_recall_at_selected_kf']:.2%} | {fixed20['cost']['mean_returned_pages']:.2f} | {fixed20['cost']['kp_distribution']['median']} | {fixed20['cost']['kp_distribution']['p90']} |",
        f"| Adaptive-k page | {adaptive['page_recall_at_returned_k']:.2%} | {adaptive['page_recall@10']:.2%} | {adaptive['page_recall@20']:.2%} | {adaptive['ndcg@10']:.2f} | {adaptive['file_recall_at_selected_kf']:.2%} | {adaptive['cost']['mean_returned_pages']:.2f} | {adaptive['cost']['kp_distribution']['median']} | {adaptive['cost']['kp_distribution']['p90']} |",
        "",
        "## Interpretation",
        "",
        "- This is the direct page-level adapter of the upstream method; Kf remains fixed so changes are attributable to the amount of returned page evidence.",
        "- The underlying page order is unchanged, but queries where Adaptive-k returns fewer than 10 pages naturally have lower effective recall@10; the relevant comparison is recall at returned K and the Kp cost distribution.",
        "- Qrels are used only for the metrics above, never to choose K at query time.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
