"""Score-curve audit and unsupervised adaptive Kf/Kp screen for the new benchmark.

This is intentionally an offline analysis.  It loads the existing parsing and
BM25 artifacts, reconstructs the fixed hierarchical score exactly, and does
not parse, OCR, render, or encode anything.  Quantile thresholds are fitted
from score curves only; qrels are read only after ranking for diagnostics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import (
    RawPage,
    aggregate_file_scores,
    normalise_scores,
    read_jsonl,
    sort_scores,
)
from src.retrieval.sparse import BM25Index


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _gap(scores: Sequence[float], boundary: int) -> float:
    """Score at rank boundary minus the next score, using one-based ranks."""
    if len(scores) <= boundary:
        return 0.0
    return max(0.0, float(scores[boundary - 1]) - float(scores[boundary]))


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + end - 1) / 2.0 + 1.0
            for index in order[start:end]:
                result[index] = rank
            start = end
        return result

    x, y = ranks(left), ranks(right)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return numerator / denominator if denominator else 0.0


def _choose_budget(value: float, thresholds: Sequence[float], budgets: Sequence[int]) -> int:
    """High gap -> small budget; thresholds are q25..q75."""
    q25, q50, q75 = thresholds
    if value >= q75:
        return budgets[0]
    if value >= q50:
        return budgets[1]
    if value >= q25:
        return budgets[2]
    return budgets[3]


def _evaluate(
    rows: Sequence[dict[str, Any]],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    adaptive: bool,
) -> dict[str, Any]:
    file_recall: list[float] = []
    pool_recall: list[float] = []
    recall10: list[float] = []
    hits10: list[float] = []
    returned_pages: list[float] = []
    returned_files: list[float] = []
    for row in rows:
        gold = set(qrels.get(row["query_id"], {}))
        selected_files = row["selected_files"]
        gold_files = {page_id.split("#page=", 1)[0] for page_id in gold}
        returned = list(row["returned_pages"])
        found = set(returned) & gold
        top10 = set(returned[:10])
        file_recall.append(len(set(selected_files) & gold_files) / len(gold_files) if gold_files else 0.0)
        pool_recall.append(len(found) / len(gold) if gold else 0.0)
        recall10.append(len(top10 & gold) / len(gold) if gold else 0.0)
        hits10.append(float(bool(top10 & gold)))
        returned_pages.append(len(returned))
        returned_files.append(len(set(page_id.split("#page=", 1)[0] for page_id in returned)))
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "queries": len(rows),
        "adaptive": adaptive,
        "page_recall_at_returned_k": mean(pool_recall),
        "page_recall_at_10": mean(recall10),
        "page_hit_at_10": mean(hits10),
        "file_recall_at_selected_kf": mean(file_recall),
        "mean_returned_pages": mean(returned_pages),
        "mean_returned_files": mean(returned_files),
        "min_returned_pages": min(returned_pages) if returned_pages else 0,
        "max_returned_pages": max(returned_pages) if returned_pages else 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    started = time.perf_counter()
    cache = args.cache_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pages = [RawPage(**row) for row in read_jsonl(cache / "parsing/pages.jsonl")]
    queries = read_jsonl(args.dataset_root / "queries.jsonl")
    page_index = BM25Index.load(cache / "indexes/page_bm25.json")
    file_index = BM25Index.load(cache / "indexes/file_bm25.json")
    qrel_rows = read_jsonl(cache / "qrels_page_level.jsonl")
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    for row in qrel_rows:
        qrels[str(row["query_id"])][str(row["page_id"])] = float(row.get("relevance", 1))

    page_to_file = {page.page_id: page.doc_id for page in pages}
    page_by_id = {page.page_id: page for page in pages}
    file_ids = sorted({page.doc_id for page in pages})
    kfs = tuple(args.file_budgets)
    kps = tuple(args.page_budgets)
    traces: list[dict[str, Any]] = []
    for query in queries:
        qid = str(query["query_id"])
        text = str(query["query"])
        bm25_hits = page_index.search(text, len(pages))
        page_bm25 = {page_index.chunk_ids[position]: float(score) for position, score in bm25_hits}
        page_base = {
            page_id: 0.70 * score
            for page_id, score in normalise_scores(page_bm25).items()
        }
        direct_hits = file_index.search(text, len(file_ids))
        direct = {file_index.chunk_ids[position]: float(score) for position, score in direct_hits}
        file_pool = aggregate_file_scores(page_base, page_to_file, file_ids)
        file_scores = {
            file_id: 0.50 * normalise_scores(direct).get(file_id, 0.0)
            + 0.50 * normalise_scores(file_pool).get(file_id, 0.0)
            for file_id in file_ids
        }
        file_ranked = sort_scores(file_scores)
        parent_norm = normalise_scores(file_scores)
        curves: dict[str, Any] = {}
        page_rankings: dict[int, list[tuple[str, float]]] = {}
        for kf in kfs:
            selected = {file_id for file_id, _score in file_ranked[:kf]}
            scores = {
                page.page_id: 0.85 * page_base.get(page.page_id, 0.0)
                + 0.15 * parent_norm.get(page.doc_id, 0.0)
                for page in pages
                if page.doc_id in selected
            }
            page_rankings[kf] = sort_scores(scores)
        file_values = [score for _file_id, score in file_ranked]
        curves["file_scores"] = file_values[: max(kfs) + 5]
        curves["file_gaps"] = {str(k): _gap(file_values, k) for k in (3, 5, 10, 15, 20)}
        curves["page_gaps"] = {
            str(kf): {str(kp): _gap([score for _page_id, score in page_rankings[kf]], kp) for kp in (10, 20, 50, 100)}
            for kf in kfs
        }
        traces.append({
            "query_id": qid,
            "query": text,
            "file_ranked": [{"file_id": file_id, "score": score} for file_id, score in file_ranked],
            "curves": curves,
            "page_rankings": {
                str(kf): [{"page_id": page_id, "score": score} for page_id, score in ranking[: max(kps)]]
                for kf, ranking in page_rankings.items()
            },
        })

    file_gap_values = {k: [row["curves"]["file_gaps"][str(k)] for row in traces] for k in (3, 5, 10, 15, 20)}
    page_gap_values = {
        (kf, kp): [row["curves"]["page_gaps"][str(kf)][str(kp)] for row in traces]
        for kf in kfs for kp in (10, 20, 50, 100)
    }
    file_thresholds = _quantile(file_gap_values[3], 0.25), _quantile(file_gap_values[3], 0.50), _quantile(file_gap_values[3], 0.75)
    # Kp is selected from the first boundary with a locally strong gap.
    page_thresholds = {
        kp: (_quantile(page_gap_values[(3, kp)], 0.75),) for kp in (10, 20, 50)
    }

    baseline_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    for trace in traces:
        qid = trace["query_id"]
        ranked_files = trace["file_ranked"]
        fixed_kf = args.fixed_file_k
        fixed_ranking = trace["page_rankings"][str(fixed_kf)]
        fixed_pages = [item["page_id"] for item in fixed_ranking[: args.fixed_page_k]]
        baseline_rows.append({
            "query_id": qid,
            "selected_files": [item["file_id"] for item in ranked_files[:fixed_kf]],
            "returned_pages": fixed_pages,
            "kf": fixed_kf,
            "kp": args.fixed_page_k,
        })
        kf = _choose_budget(trace["curves"]["file_gaps"]["3"], file_thresholds, kfs)
        ranking = trace["page_rankings"][str(kf)]
        # Select Kp=10 when the rank-10 break is high; otherwise progressively
        # test later boundary breaks and fall back to the largest budget.
        kp = 100
        for boundary in (10, 20, 50):
            value = trace["curves"]["page_gaps"][str(kf)][str(boundary)]
            threshold = _quantile(page_gap_values[(3, boundary)], 0.75)
            if value >= threshold and threshold > 0:
                kp = boundary
                break
        adaptive_rows.append({
            "query_id": qid,
            "selected_files": [item["file_id"] for item in ranked_files[:kf]],
            "returned_pages": [item["page_id"] for item in ranking[:kp]],
            "kf": kf,
            "kp": kp,
            "file_gap_at_3": trace["curves"]["file_gaps"]["3"],
        })

    score_summary = {
        "file_score_rank": {
            str(rank): _summary(
                [row["curves"]["file_scores"][rank - 1] for row in traces if len(row["curves"]["file_scores"]) >= rank]
            )
            for rank in (1, 3, 4, 5, 10)
        },
        "page_score_rank_kf3": {
            str(rank): _summary(
                [row["page_rankings"]["3"][rank - 1]["score"] for row in traces if len(row["page_rankings"]["3"]) >= rank]
            )
            for rank in (1, 10, 20, 50, 100)
        },
        "file_gap_at": {str(k): _summary(values) for k, values in file_gap_values.items()},
        "page_gap_at": {f"kf{kf}_kp{kp}": _summary(values) for (kf, kp), values in page_gap_values.items()},
        "file_gap_thresholds_q25_q50_q75": list(file_thresholds),
        "page_gap_q75_thresholds": {str(kp): page_thresholds[kp][0] for kp in (10, 20, 50)},
        "file_gap_vs_gold_file_count_spearman": _spearman(
            file_gap_values[3], [len({page_id.split("#page=", 1)[0] for page_id in qrels.get(row["query_id"], {})}) for row in traces]
        ),
    }
    report = {
        "contract_version": "benchmark-adaptive-topk-score-audit-v1",
        "cache_dir": str(cache),
        "dataset_root": str(args.dataset_root),
        "queries": len(queries),
        "files": len(file_ids),
        "pages": len(pages),
        "visual_mode": "disabled/cache-BM25-only",
        "score_summary": score_summary,
        "budget_distribution": {
            "adaptive_kf": {str(kf): sum(row["kf"] == kf for row in adaptive_rows) for kf in kfs},
            "adaptive_kp": {str(kp): sum(row["kp"] == kp for row in adaptive_rows) for kp in (10, 20, 50, 100)},
        },
        "metrics": {
            "fixed": _evaluate(baseline_rows, qrels, adaptive=False),
            "adaptive_score_gap": _evaluate(adaptive_rows, qrels, adaptive=True),
        },
        "timing_seconds": {"offline_score_reconstruction": round(time.perf_counter() - started, 3)},
        "policy": {
            "file": "quantiles of score(file rank 3)-score(file rank 4): high gap -> Kf 3, then 5, 10, 15/20",
            "page": "first rank boundary in 10,20,50 with gap >= score-only q75; otherwise Kp 100",
            "qrels_used_for_thresholds": False,
        },
    }
    _write_json(args.output_dir / "report.json", report)
    _write_json(args.output_dir / "score_summary.json", score_summary)
    _write_jsonl(args.output_dir / "score_curves.jsonl", traces)
    _write_jsonl(args.output_dir / "fixed_per_query.jsonl", baseline_rows)
    _write_jsonl(args.output_dir / "adaptive_per_query.jsonl", adaptive_rows)
    fixed = report["metrics"]["fixed"]
    adaptive = report["metrics"]["adaptive_score_gap"]
    markdown = "\n".join([
        "# Benchmark adaptive TopK score audit",
        "",
        f"- Cache: `{cache}`; files={len(file_ids)}, pages={len(pages)}, queries={len(queries)}.",
        "- Visual signal: disabled; this is a cached PDF-inspector/Tesseract + BM25 audit.",
        "- Quantile thresholds use score curves only; qrels are used only for the metrics below.",
        "",
        "## Score-gap distribution",
        "",
        "| Gap | Q25 | Median | Q75 | Max |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| file score rank {key}→{int(key)+1} | {value['q25']:.6f} | {value['median']:.6f} | {value['q75']:.6f} | {value['max']:.6f} |"
            for key, value in score_summary["file_gap_at"].items()
        ],
        "",
        "## Fixed versus score-gap adaptive",
        "",
        "| Policy | Page recall at returned K | Page recall@10 | File recall at selected Kf | Mean pages | Mean files |",
        "|---|---:|---:|---:|---:|---:|",
        f"| fixed Kf={args.fixed_file_k}, Kp={args.fixed_page_k} | {fixed['page_recall_at_returned_k']:.2%} | {fixed['page_recall_at_10']:.2%} | {fixed['file_recall_at_selected_kf']:.2%} | {fixed['mean_returned_pages']:.2f} | {fixed['mean_returned_files']:.2f} |",
        f"| adaptive score-gap | {adaptive['page_recall_at_returned_k']:.2%} | {adaptive['page_recall_at_10']:.2%} | {adaptive['file_recall_at_selected_kf']:.2%} | {adaptive['mean_returned_pages']:.2f} | {adaptive['mean_returned_files']:.2f} |",
        "",
        "## Interpretation",
        "",
        "- File gaps are small and continuous: median gap at rank 3 is "
        f"`{score_summary['file_gap_at']['3']['median']:.6f}`, with Q25/Q75 "
        f"`{score_summary['file_gap_at']['3']['q25']:.6f}`/`{score_summary['file_gap_at']['3']['q75']:.6f}`.",
        "- The quick adaptive policy increases evidence coverage, but it does so by sending many more pages; it is not yet a cost-efficient cut-off.",
        "- The result is exploratory because this cache has V-SPLADE disabled; the same analysis should be repeated on a hybrid cache before drawing a final conclusion.",
        "",
    ])
    (args.output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-file-k", type=int, default=3)
    parser.add_argument("--fixed-page-k", type=int, default=10)
    parser.add_argument("--file-budgets", type=int, nargs="+", default=[3, 5, 10, 15, 20])
    parser.add_argument("--page-budgets", type=int, nargs="+", default=[10, 20, 50, 100])
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
