"""Analyze dense/BM25/hybrid score distributions by language relation."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_language_routing_matrix import (  # noqa: E402
    load_file_languages,
)
from research.experiments.run_ohr_replacement_all_arms import load_inputs  # noqa: E402


BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/language_score_distribution_analysis"
BM25_ALPHA = 0.70


def max_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0.0 else np.zeros_like(values)


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ("p10", "p25", "p50", "p75", "p90", "mean")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(np.mean(array)),
    }


def dataset(qid: str) -> str:
    return qid.split("::", 1)[0].replace("vidore_", "")


def relation(file_language: str) -> str:
    return "same" if file_language == "en" else "cross"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    queries, _, _ = load_inputs()
    query_ids = [str(row["query_id"]) for row in queries]
    if len(query_ids) != 220 or len(page_ids) != 5091 or len(file_ids) != 103:
        raise RuntimeError("Canonical shape mismatch")
    if query_ids != [str(value) for value in metadata["query_ids"]]:
        raise RuntimeError("Query order mismatch")

    page_bm25 = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25 = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    file_languages, language_provenance = load_file_languages(file_ids)
    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_position[page.split("#page=", 1)[0]] for page in page_ids], dtype=np.int32)
    file_relations = {fid: relation(file_languages[fid]) for fid in file_ids}

    pair_rows: list[dict[str, Any]] = []
    for qi, qid in enumerate(query_ids):
        dense_norm = max_norm(dense[qi])
        page_bm_norm = max_norm(page_bm25[qi])
        direct_bm_norm = max_norm(file_bm25[qi])
        dense_file_norm = np.zeros(len(file_ids), dtype=np.float64)
        page_bm_file_norm = np.zeros(len(file_ids), dtype=np.float64)
        for fi in range(len(file_ids)):
            mask = page_file_pos == fi
            dense_file_norm[fi] = float(dense_norm[mask].max()) if np.any(mask) else 0.0
            page_bm_file_norm[fi] = float(page_bm_norm[mask].max()) if np.any(mask) else 0.0

        hybrid_page = BM25_ALPHA * page_bm_norm + (1.0 - BM25_ALPHA) * dense_norm
        hybrid_page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(hybrid_page_max, page_file_pos, hybrid_page)
        direct_hybrid = BM25_ALPHA * direct_bm_norm + (1.0 - BM25_ALPHA) * dense_file_norm
        file_hybrid = 0.50 * direct_hybrid + 0.50 * max_norm(hybrid_page_max)

        dense_order = np.argsort(-dense_file_norm, kind="stable")
        hybrid_order = np.argsort(-file_hybrid, kind="stable")
        dense_rank = {int(fi): rank + 1 for rank, fi in enumerate(dense_order)}
        hybrid_rank = {int(fi): rank + 1 for rank, fi in enumerate(hybrid_order)}
        source = dataset(qid)
        for fi, fid in enumerate(file_ids):
            direct_bm = float(file_bm25[qi, fi])
            bm_component = BM25_ALPHA * float(direct_bm_norm[fi])
            dense_component = (1.0 - BM25_ALPHA) * float(dense_file_norm[fi])
            pair_rows.append(
                {
                    "query_id": qid,
                    "dataset": source,
                    "file_id": fid,
                    "file_language": file_languages[fid],
                    "relation": file_relations[fid],
                    "dense_raw": float(dense_file_norm[fi] * float(dense[qi].max())),
                    "dense_norm": float(dense_file_norm[fi]),
                    "direct_bm25_raw": direct_bm,
                    "direct_bm25_norm": float(direct_bm_norm[fi]),
                    "page_bm25_max_norm": float(page_bm_file_norm[fi]),
                    "bm25_component": bm_component,
                    "dense_component": dense_component,
                    "direct_hybrid": float(direct_hybrid[fi]),
                    "file_hybrid": float(file_hybrid[fi]),
                    "delta_file_hybrid_minus_dense": float(file_hybrid[fi] - dense_file_norm[fi]),
                    "bm25_positive": direct_bm > 0.0,
                    "dense_rank": dense_rank[fi],
                    "hybrid_rank": hybrid_rank[fi],
                    "rank_delta_hybrid_minus_dense": hybrid_rank[fi] - dense_rank[fi],
                }
            )

    def group_rows(*, relation_name: str | None = None, dataset_name: str | None = None) -> list[dict[str, Any]]:
        return [
            row for row in pair_rows
            if (relation_name is None or row["relation"] == relation_name)
            and (dataset_name is None or row["dataset"] == dataset_name)
        ]

    def values(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "pairs": len(rows),
            "queries": len({row["query_id"] for row in rows}),
            "files": len({row["file_id"] for row in rows}),
            "dense_raw": quantiles(values(rows, "dense_raw")),
            "dense_norm": quantiles(values(rows, "dense_norm")),
            "direct_bm25_norm": quantiles(values(rows, "direct_bm25_norm")),
            "direct_hybrid": quantiles(values(rows, "direct_hybrid")),
            "file_hybrid": quantiles(values(rows, "file_hybrid")),
            "delta_file_hybrid_minus_dense": quantiles(values(rows, "delta_file_hybrid_minus_dense")),
            "bm25_positive_rate": float(np.mean([row["bm25_positive"] for row in rows])) if rows else 0.0,
            "bm25_component_mean": float(np.mean(values(rows, "bm25_component"))) if rows else 0.0,
            "dense_component_mean": float(np.mean(values(rows, "dense_component"))) if rows else 0.0,
            "rank_improved_rate": float(np.mean([row["rank_delta_hybrid_minus_dense"] < 0 for row in rows])) if rows else 0.0,
            "rank_worsened_rate": float(np.mean([row["rank_delta_hybrid_minus_dense"] > 0 for row in rows])) if rows else 0.0,
            "rank_unchanged_rate": float(np.mean([row["rank_delta_hybrid_minus_dense"] == 0 for row in rows])) if rows else 0.0,
        }

    summary: dict[str, Any] = {
        "protocol": {
            "benchmark": "80 Physics + 75 Industrial + 65 OHR-Bench",
            "queries": 220,
            "files": 103,
            "pages": 5091,
            "query_language": "English",
            "same_language_files": sum(file_languages[fid] == "en" for fid in file_ids),
            "cross_language_files": sum(file_languages[fid] == "fr" for fid in file_ids),
            "hybrid_formula": "0.70 normalized direct BM25 + 0.30 normalized dense",
            "file_formula": "0.50 direct hybrid + 0.50 normalized max page hybrid",
            "qrels_used_for_scoring": False,
            "language_provenance_counts": dict(Counter(language_provenance.values())),
        },
        "by_relation": {name: summarize(group_rows(relation_name=name)) for name in ("same", "cross")},
        "by_dataset_relation": {
            dataset_name: {name: summarize(group_rows(dataset_name=dataset_name, relation_name=name)) for name in ("same", "cross")}
            for dataset_name in ("physics", "industrial", "ohrbench")
        },
        "all_hypothetical_hybrid": {
            "same": summarize(group_rows(relation_name="same")),
            "cross": summarize(group_rows(relation_name="cross")),
        },
    }

    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "file_query_scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pair_rows) + "\n", encoding="utf-8"
    )

    def fmt(q: dict[str, float], key: str = "p50") -> str:
        return f"{q[key]:.4f}"

    lines = [
        "# Language-routed score distributions",
        "",
        "Canonical merged benchmark: English queries, 220 queries, 103 files, 5,091 pages. Same-language means English file; cross-language means French file.",
        "",
        "## Corpus relation",
        "",
        f"- Same-language files: `{summary['protocol']['same_language_files']}`; cross-language files: `{summary['protocol']['cross_language_files']}`.",
        "- BM25 is evaluated hypothetically for both relations; the production route uses hybrid for same-language and dense-only for cross-language.",
        "",
        "## File-level distribution by relation",
        "",
        "| Relation | Pairs | Dense p50 | Dense p90 | Direct BM25 p50 | Hybrid file p50 | Delta hybrid-dense p50 | BM25 positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("same", "cross"):
        row = summary["by_relation"][name]
        lines.append(
            f"| {name} | {row['pairs']:,} | {fmt(row['dense_norm'])} | {fmt(row['dense_norm'], 'p90')} | "
            f"{fmt(row['direct_bm25_norm'])} | {fmt(row['file_hybrid'])} | {fmt(row['delta_file_hybrid_minus_dense'])} | {row['bm25_positive_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Same-language BM25 effect by dataset",
            "",
            "| Dataset | Queries | Files | Dense p50 | BM25 p50 | Hybrid file p50 | Delta p50 | Delta mean | Rank improved | Rank worsened |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("physics", "industrial", "ohrbench"):
        row = summary["by_dataset_relation"][name]["same"]
        lines.append(
            f"| {name} | {row['queries']} | {row['files']} | {fmt(row['dense_norm'])} | {fmt(row['direct_bm25_norm'])} | "
            f"{fmt(row['file_hybrid'])} | {fmt(row['delta_file_hybrid_minus_dense'])} | {fmt(row['delta_file_hybrid_minus_dense'], 'mean')} | "
            f"{row['rank_improved_rate']:.2%} | {row['rank_worsened_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Same-language component means by dataset",
            "",
            "| Dataset | BM25 contribution mean | Dense contribution mean | BM25 positive rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("physics", "industrial", "ohrbench"):
        row = summary["by_dataset_relation"][name]["same"]
        lines.append(f"| {name} | {row['bm25_component_mean']:.4f} | {row['dense_component_mean']:.4f} | {row['bm25_positive_rate']:.2%} |")
    lines.extend(
        [
            "",
            "## Cross-language hypothetical BM25 effect",
            "",
            "| Dataset | Dense p50 | BM25 p50 | Hybrid file p50 | Delta p50 | Delta mean | BM25 positive rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("physics", "industrial", "ohrbench"):
        row = summary["by_dataset_relation"][name]["cross"]
        lines.append(
            f"| {name} | {fmt(row['dense_norm'])} | {fmt(row['direct_bm25_norm'])} | {fmt(row['file_hybrid'])} | "
            f"{fmt(row['delta_file_hybrid_minus_dense'])} | {fmt(row['delta_file_hybrid_minus_dense'], 'mean')} | {row['bm25_positive_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Definitions: dense file score is the maximum normalized page dense score per file; direct BM25 is the normalized cached file BM25 score; hybrid file score is the existing 0.50 direct + 0.50 max-page formula. Delta compares hybrid file score with dense file score on the same query-local normalized scale.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
