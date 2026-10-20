"""Describe hierarchical page/file similarity curves for all Physics queries.

The script is diagnostic only.  It reads the existing hierarchical stage
traces and Physics qrels, computes query-local score-curve statistics, and
writes a compact per-query artifact plus a report.  It does not rebuild an
index, encode a query, or tune a retrieval method.

The main trace is the current ``cascade-all_text-max-kf3-fusion`` arm.  Scores
are the normalized hierarchical scores already stored in the trace; they are
not raw BM25 or raw V-SPLADE scores and should not be compared across a
different scoring implementation without recalibration.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_TRACE = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/stage_traces.jsonl"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = ROOT / "data/benchmark/vidore_v3/results/physics_similarity_distribution"
METHOD = "cascade-all_text-max-kf3-fusion"
EXPECTED_QUERIES = 302


def _file_id(page_id: str) -> str:
    return str(page_id).split("#page=", 1)[0]


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {key: 0.0 for key in ("min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max", "mean", "std")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _at(scores: Sequence[float], rank: int) -> float:
    if not scores:
        return 0.0
    return float(scores[min(rank - 1, len(scores) - 1)])


def _curve_features(items: Sequence[Mapping[str, Any]], ranks: Sequence[int]) -> dict[str, float]:
    scores = [float(item.get("score") or 0.0) for item in items]
    top = _at(scores, 1)
    output: dict[str, float] = {"top1": top, "count": float(len(scores))}
    for rank in ranks:
        value = _at(scores, rank)
        suffix = str(rank)
        output[f"score{suffix}"] = value
        output[f"gap1_{suffix}"] = top - value
        output[f"ratio{suffix}"] = value / top if top else 0.0
    output["span"] = top - _at(scores, len(scores))
    if len(scores) >= 2:
        output["local_gap1_2"] = scores[0] - scores[1]
    else:
        output["local_gap1_2"] = 0.0
    return output


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    positives = [float(score) for score, label in zip(scores, labels) if label]
    negatives = [float(score) for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return 0.5
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _histogram(values: Sequence[float], edges: Sequence[float]) -> list[dict[str, Any]]:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=np.asarray(edges, dtype=float))
    total = max(int(sum(counts)), 1)
    output: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        output.append(
            {
                "lo": float(edges[index]),
                "hi": float(edges[index + 1]),
                "count": int(count),
                "share": float(count / total),
            }
        )
    return output


def _load_trace(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") != METHOD:
                continue
            qid = str(row["qid"])
            if qid in rows:
                raise ValueError(f"Duplicate {METHOD} trace for {qid}")
            rows[qid] = row
    if len(rows) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} {METHOD} traces, got {len(rows)}")
    return rows


def _group_summary(rows: Sequence[Mapping[str, Any]], feature: str) -> list[dict[str, Any]]:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][feature]))
    groups = np.array_split(np.asarray(order, dtype=int), 4)
    output: list[dict[str, Any]] = []
    for group_number, group in enumerate(groups, 1):
        selected = [rows[int(index)] for index in group]
        output.append(
            {
                "quartile": group_number,
                "queries": len(selected),
                "feature_mean": float(np.mean([float(row[feature]) for row in selected])),
                "page_recall@10_mean": float(np.mean([float(row["page_recall@10"]) for row in selected])),
                "page_recall@100_mean": float(np.mean([float(row["page_recall@100"]) for row in selected])),
                "page_full@10_share": float(np.mean([float(row["page_full@10"]) for row in selected])),
                "file_recall@3_mean": float(np.mean([float(row["file_recall@3"]) for row in selected])),
                "gold_pages_mean": float(np.mean([float(row["gold_page_count"]) for row in selected])),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    traces = _load_trace(args.trace)
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    if set(qids) != set(traces):
        missing = sorted(set(qids) - set(traces))
        extra = sorted(set(traces) - set(qids))
        raise RuntimeError(f"Trace/query mismatch; missing={missing[:5]}, extra={extra[:5]}")
    qrels = benchmark.qrels()

    per_query: list[dict[str, Any]] = []
    for qid in qids:
        trace = traces[qid]
        gold_pages = set(qrels[qid])
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        page_items = trace.get("page_candidates") or []
        file_items = trace.get("file_candidates") or []
        page_features = _curve_features(page_items, (5, 10, 20, 50, 100))
        file_features = _curve_features(file_items, (3, 5, 10, 15, 20))
        page_ids = [str(item["node_id"]) for item in page_items]
        file_ids = [str(item["node_id"]) for item in file_items]
        page_top10 = set(page_ids[:10])
        page_top100 = set(page_ids[:100])
        file_top3 = set(file_ids[:3])
        row: dict[str, Any] = {
            "qid": qid,
            "gold_page_count": len(gold_pages),
            "gold_file_count": len(gold_files),
            "page_candidate_count": len(page_items),
            "file_candidate_count": len(file_items),
            "page_recall@10": len(page_top10 & gold_pages) / len(gold_pages),
            "page_recall@100": len(page_top100 & gold_pages) / len(gold_pages),
            "file_recall@3": len(file_top3 & gold_files) / len(gold_files),
            "file_full@3": gold_files.issubset(file_top3),
            "page": page_features,
            "file": file_features,
        }
        for kp in (10, 20, 50, 100):
            page_at_k = set(page_ids[:kp])
            row[f"page_full@{kp}"] = len(page_at_k & gold_pages) == len(gold_pages)
        # Flat aliases make the JSONL easy to analyze without nested parsing.
        row.update({f"page_{key}": value for key, value in page_features.items()})
        row.update({f"file_{key}": value for key, value in file_features.items()})
        per_query.append(row)

    def values(prefix: str, key: str) -> list[float]:
        return [float(row[prefix][key]) for row in per_query]

    page_summary = {
        key: _stats(values("page", key))
        for key in ("top1", "score5", "score10", "score20", "score50", "score100", "gap1_10", "gap1_20", "gap1_50", "gap1_100", "ratio10", "ratio20", "ratio50", "ratio100", "span", "local_gap1_2")
    }
    file_summary = {
        key: _stats(values("file", key))
        for key in ("top1", "score3", "score5", "score10", "score15", "score20", "gap1_3", "gap1_5", "gap1_10", "gap1_15", "gap1_20", "ratio3", "ratio5", "ratio10", "ratio15", "ratio20", "span", "local_gap1_2")
    }

    page_auc: dict[str, float] = {}
    for kp in (10, 20, 50, 100):
        page_auc[f"gap1_20_for_full@{kp}"] = _auc(
            values("page", "gap1_20"),
            [bool(row[f"page_full@{kp}"]) for row in per_query],
        )
        page_auc[f"span_for_full@{kp}"] = _auc(
            values("page", "span"),
            [bool(row[f"page_full@{kp}"]) for row in per_query],
        )
    file_auc: dict[str, float] = {}
    for kf in (3, 5, 10, 15, 20):
        # ``file_full@kf`` is derived from the full file candidate list, not
        # the final page output, so this directly measures file-score shape.
        labels = []
        for row in per_query:
            file_ids = [
                str(item["node_id"])
                for item in traces[row["qid"]].get("file_candidates") or []
            ]
            file_ids = set(file_ids[:kf])
            gold = set(qrels[row["qid"]])
            gold = {_file_id(page_id) for page_id in gold}
            labels.append(gold.issubset(file_ids))
        file_auc[f"gap1_3_for_full@{kf}"] = _auc(values("file", "gap1_3"), labels)
        file_auc[f"top1_for_full@{kf}"] = _auc(values("file", "top1"), labels)

    report = {
        "dataset": "vidore_v3/physics",
        "method": METHOD,
        "queries": len(qids),
        "score_semantics": "stored normalized hierarchical page/file scores; page trace is Kf=3 and 100 page candidates",
        "page_score_summary": page_summary,
        "file_score_summary": file_summary,
        "page_auc": page_auc,
        "file_auc": file_auc,
        "page_gap_quartiles": _group_summary(per_query, "page_gap1_20"),
        "file_gap_quartiles": _group_summary(per_query, "file_gap1_3"),
        "histograms": {
            "page_top1": _histogram(values("page", "top1"), (0.0, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 1.000001)),
            "page_gap1_20": _histogram(values("page", "gap1_20"), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)),
            "file_top1": _histogram(values("file", "top1"), (0.0, 0.8, 0.9, 0.95, 0.99, 0.999, 1.000001)),
            "file_gap1_3": _histogram(values("file", "gap1_3"), (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)),
        },
        "counts": {
            "page_full@10": int(sum(bool(row["page_full@10"]) for row in per_query)),
            "page_full@20": int(sum(bool(row["page_full@20"]) for row in per_query)),
            "page_full@50": int(sum(bool(row["page_full@50"]) for row in per_query)),
            "page_full@100": int(sum(bool(row["page_full@100"]) for row in per_query)),
            "file_full@3": int(sum(bool(row["file_full@3"]) for row in per_query)),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def line_stats(summary: Mapping[str, Mapping[str, float]], key: str) -> str:
        value = summary[key]
        return f"{value['min']:.3f} / {value['p25']:.3f} / {value['median']:.3f} / {value['p75']:.3f} / {value['p90']:.3f} / {value['max']:.3f}"

    lines = [
        "# Physics similarity-score distribution",
        "",
        f"All **{len(qids)}** French Physics queries, using `{METHOD}`.",
        "Scores are the normalized hierarchical scores stored in the existing trace; they are query-local ranking scores, not calibrated probabilities.",
        "",
        "## Page score curve",
        "",
        "The columns show min / p25 / median / p75 / p90 / max over the 302 queries.",
        "",
        "| Feature | Distribution |",
        "|---|---:|",
    ]
    for key in ("top1", "score10", "score20", "score100", "gap1_10", "gap1_20", "gap1_50", "gap1_100", "ratio10", "ratio20", "span", "local_gap1_2"):
        lines.append(f"| page {key} | {line_stats(page_summary, key)} |")
    lines += [
        "",
        "## File score curve",
        "",
        "| Feature | Distribution |",
        "|---|---:|",
    ]
    for key in ("top1", "score3", "score10", "score20", "gap1_3", "gap1_5", "gap1_10", "gap1_20", "ratio3", "ratio10", "span", "local_gap1_2"):
        lines.append(f"| file {key} | {line_stats(file_summary, key)} |")
    lines += [
        "",
        "## Can score shape predict a useful budget?",
        "",
        "AUC=0.5 means no ranking signal; values above 0.5 mean that larger feature values tend to identify queries fully covered at that budget.",
        "",
        "| Page target | Gap top1–top20 AUC | Score span AUC |",
        "|---|---:|---:|",
    ]
    for kp in (10, 20, 50, 100):
        lines.append(f"| Full gold evidence by page {kp} | {page_auc[f'gap1_20_for_full@{kp}']:.3f} | {page_auc[f'span_for_full@{kp}']:.3f} |")
    lines += [
        "",
        "| File target | Gap top1–top3 AUC | Top score AUC |",
        "|---|---:|---:|",
    ]
    for kf in (3, 5, 10, 15, 20):
        lines.append(f"| All gold files by file {kf} | {file_auc[f'gap1_3_for_full@{kf}']:.3f} | {file_auc[f'top1_for_full@{kf}']:.3f} |")
    lines += [
        "",
        "## Quartile check",
        "",
        "| Page gap quartile | Mean gap | Recall@10 | Recall@100 | Full@10 | File recall@3 | Mean gold pages |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["page_gap_quartiles"]:
        lines.append(
            f"| {row['quartile']} | {row['feature_mean']:.3f} | {row['page_recall@10_mean']:.2%} | {row['page_recall@100_mean']:.2%} | {row['page_full@10_share']:.2%} | {row['file_recall@3_mean']:.2%} | {row['gold_pages_mean']:.1f} |"
        )
    lines += [
        "",
        "| File gap quartile | Mean gap | Recall@10 | Recall@100 | Full@10 | File recall@3 | Mean gold pages |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["file_gap_quartiles"]:
        lines.append(
            f"| {row['quartile']} | {row['feature_mean']:.3f} | {row['page_recall@10_mean']:.2%} | {row['page_recall@100_mean']:.2%} | {row['page_full@10_share']:.2%} | {row['file_recall@3_mean']:.2%} | {row['gold_pages_mean']:.1f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- The top score is saturated near 1.0, so it is not a useful universal threshold.",
        "- Page and file margins have real but overlapping separation. They can rank query confidence, but they do not define one global relevant/irrelevant breakpoint.",
        "- The signal is stronger for small budgets. At large Kf/Kp almost every query is covered, so AUC naturally becomes unstable or approaches 0.5.",
        "- A practical adaptive policy should use score shape as one feature, together with candidate coverage, query anchors and file/page agreement; it should use a held-out cost gate rather than a single raw-score threshold.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "queries": len(qids), "method": METHOD}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
