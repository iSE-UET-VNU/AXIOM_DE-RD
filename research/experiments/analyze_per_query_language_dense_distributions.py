"""Compare per-query dense file-score distributions by file language."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCORES = (
    ROOT
    / "data/work/ohr_replacement_20260920/retrieval/language_score_distribution_analysis"
    / "file_query_scores.jsonl"
)
OUTPUT = (
    ROOT
    / "data/work/ohr_replacement_20260920/retrieval"
    / "per_query_language_dense_distributions"
)


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: 0.0 for key in ("p10", "p25", "p50", "p75", "p90", "mean")}
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(np.mean(array)),
    }


def score_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return quantiles(float(row[key]) for row in rows)


def summarize_query_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "queries": len(rows),
        "raw_mean_gap_fr_minus_en": quantiles(row["raw_mean_gap_fr_minus_en"] for row in rows),
        "raw_median_gap_fr_minus_en": quantiles(row["raw_median_gap_fr_minus_en"] for row in rows),
        "raw_max_gap_fr_minus_en": quantiles(row["raw_max_gap_fr_minus_en"] for row in rows),
        "norm_mean_gap_fr_minus_en": quantiles(row["norm_mean_gap_fr_minus_en"] for row in rows),
        "norm_median_gap_fr_minus_en": quantiles(row["norm_median_gap_fr_minus_en"] for row in rows),
        "norm_max_gap_fr_minus_en": quantiles(row["norm_max_gap_fr_minus_en"] for row in rows),
        "auc_fr_over_en": quantiles(row["auc_fr_over_en"] for row in rows),
        "raw_mean_gap_positive_rate": float(np.mean([row["raw_mean_gap_fr_minus_en"] > 0 for row in rows])) if rows else 0.0,
        "raw_max_gap_positive_rate": float(np.mean([row["raw_max_gap_fr_minus_en"] > 0 for row in rows])) if rows else 0.0,
        "norm_mean_gap_positive_rate": float(np.mean([row["norm_mean_gap_fr_minus_en"] > 0 for row in rows])) if rows else 0.0,
        "norm_max_gap_positive_rate": float(np.mean([row["norm_max_gap_fr_minus_en"] > 0 for row in rows])) if rows else 0.0,
        "fr_top1_rate": float(np.mean([row["fr_top1"] for row in rows])) if rows else 0.0,
        "fr_top3_count": quantiles(row["fr_top3_count"] for row in rows),
        "fr_top10_count": quantiles(row["fr_top10_count"] for row in rows),
        "fr_best_rank": quantiles(row["fr_best_rank"] for row in rows),
        "fr_max_wins_rate": float(np.mean([row["fr_max_wins"] for row in rows])) if rows else 0.0,
    }


def fmt(values: dict[str, float], key: str = "p50") -> str:
    return f"{values[key]:.4f}"


def main() -> int:
    if not SCORES.is_file():
        raise FileNotFoundError(f"Missing prerequisite: {SCORES}")
    rows = [
        json.loads(line)
        for line in SCORES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_query[str(row["query_id"])].append(row)

    query_rows: list[dict[str, Any]] = []
    for qid, query_files in sorted(by_query.items()):
        english = [row for row in query_files if row["file_language"] == "en"]
        french = [row for row in query_files if row["file_language"] == "fr"]
        if len(english) != 81 or len(french) != 22:
            raise RuntimeError(f"Unexpected language split for {qid}: en={len(english)}, fr={len(french)}")

        dense_order = sorted(query_files, key=lambda row: (-float(row["dense_norm"]), str(row["file_id"])))
        dense_rank = {str(row["file_id"]): rank for rank, row in enumerate(dense_order, start=1)}
        fr_ranks = [dense_rank[str(row["file_id"])] for row in french]
        raw_en = [float(row["dense_raw"]) for row in english]
        raw_fr = [float(row["dense_raw"]) for row in french]
        norm_en = [float(row["dense_norm"]) for row in english]
        norm_fr = [float(row["dense_norm"]) for row in french]

        pair_total = 0.0
        pair_wins = 0.0
        for fr_score in norm_fr:
            for en_score in norm_en:
                pair_total += 1.0
                if fr_score > en_score:
                    pair_wins += 1.0
                elif fr_score == en_score:
                    pair_wins += 0.5

        query_rows.append(
            {
                "query_id": qid,
                "dataset": str(query_files[0]["dataset"]),
                "english_raw": score_summary(english, "dense_raw"),
                "french_raw": score_summary(french, "dense_raw"),
                "english_norm": score_summary(english, "dense_norm"),
                "french_norm": score_summary(french, "dense_norm"),
                "raw_mean_gap_fr_minus_en": float(np.mean(raw_fr) - np.mean(raw_en)),
                "raw_median_gap_fr_minus_en": float(np.median(raw_fr) - np.median(raw_en)),
                "raw_max_gap_fr_minus_en": float(max(raw_fr) - max(raw_en)),
                "norm_mean_gap_fr_minus_en": float(np.mean(norm_fr) - np.mean(norm_en)),
                "norm_median_gap_fr_minus_en": float(np.median(norm_fr) - np.median(norm_en)),
                "norm_max_gap_fr_minus_en": float(max(norm_fr) - max(norm_en)),
                "auc_fr_over_en": float(pair_wins / pair_total),
                "fr_top1": int(dense_order[0]["file_language"] == "fr"),
                "fr_top3_count": int(sum(row["file_language"] == "fr" for row in dense_order[:3])),
                "fr_top10_count": int(sum(row["file_language"] == "fr" for row in dense_order[:10])),
                "fr_best_rank": int(min(fr_ranks)),
                "fr_max_wins": int(max(norm_fr) > max(norm_en)),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        groups[str(row["dataset"])].append(row)

    all_file_rows = rows
    summary: dict[str, Any] = {
        "protocol": {
            "queries": len(query_rows),
            "files": 103,
            "english_files": 81,
            "french_files": 22,
            "file_score": "max page dense score per file",
            "raw_score_note": "dense_raw is the original cached dense score; dense_norm is query-max normalized",
            "qrels_used_for_scoring": False,
        },
        "all_file_pairs": {
            "english_raw": score_summary([row for row in all_file_rows if row["file_language"] == "en"], "dense_raw"),
            "french_raw": score_summary([row for row in all_file_rows if row["file_language"] == "fr"], "dense_raw"),
            "english_norm": score_summary([row for row in all_file_rows if row["file_language"] == "en"], "dense_norm"),
            "french_norm": score_summary([row for row in all_file_rows if row["file_language"] == "fr"], "dense_norm"),
        },
        "query_level": {name: summarize_query_groups(values) for name, values in sorted(groups.items())},
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "query_scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in query_rows) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Per-query English/French dense-score distributions",
        "",
        "Every query is compared against all 103 files: 81 English and 22 French. The file score is the maximum page dense score in that file.",
        "",
        "## All query-file pairs",
        "",
        "| Language | Raw p10 | Raw p50 | Raw p90 | Normalized p10 | Normalized p50 | Normalized p90 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("english", "English"), ("french", "French")):
        raw = summary["all_file_pairs"][f"{name}_raw"]
        norm = summary["all_file_pairs"][f"{name}_norm"]
        lines.append(
            f"| {label} | {fmt(raw, 'p10')} | {fmt(raw)} | {fmt(raw, 'p90')} | "
            f"{fmt(norm, 'p10')} | {fmt(norm)} | {fmt(norm, 'p90')} |"
        )

    lines.extend(
        [
            "",
            "## Per-query separation by benchmark",
            "",
            "Gaps are French minus English. A positive value means French files score higher for that query.",
            "",
            "| Dataset | Queries | Raw mean gap p50 | P(raw mean > 0) | Raw max gap p50 | P(raw max > 0) | Norm mean gap p50 | AUC(FR>EN) p50 | FR top-1 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name in ("physics", "industrial", "ohrbench"):
        group = summary["query_level"][dataset_name]
        lines.append(
            f"| {dataset_name} | {group['queries']} | {fmt(group['raw_mean_gap_fr_minus_en'])} | "
            f"{group['raw_mean_gap_positive_rate']:.2%} | {fmt(group['raw_max_gap_fr_minus_en'])} | "
            f"{group['raw_max_gap_positive_rate']:.2%} | {fmt(group['norm_mean_gap_fr_minus_en'])} | "
            f"{fmt(group['auc_fr_over_en'])} | {group['fr_top1_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Rank separation",
            "",
            "| Dataset | French wins max score | French top-1 | French best-rank p50 | French top-3 p50 | French top-10 p50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset_name in ("physics", "industrial", "ohrbench"):
        group = summary["query_level"][dataset_name]
        lines.append(
            f"| {dataset_name} | {group['fr_max_wins_rate']:.2%} | {group['fr_top1_rate']:.2%} | "
            f"{fmt(group['fr_best_rank'])} | {fmt(group['fr_top3_count'])} | {fmt(group['fr_top10_count'])} |"
        )

    lines.extend(
        [
            "",
            "The language comparison is descriptive only and does not use qrels for scoring or ranking.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
