"""Audit routed-vs-hybrid score changes and max-normalization calibration."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METHOD_ROWS = ROOT / "data/work/ohr_replacement_20260920/retrieval/method_language_score_separation/scores.jsonl"
COMPONENT_ROWS = ROOT / "data/work/ohr_replacement_20260920/retrieval/language_score_distribution_analysis/file_query_scores.jsonl"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/routed_score_calibration_audit"


def source(file_id: str) -> str:
    return file_id.split("::", 1)[0].replace("vidore_", "")


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
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


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    method_rows = load_rows(METHOD_ROWS)
    component_rows = load_rows(COMPONENT_ROWS)
    methods = ("dense_only", "hybrid_all", "routed_hybrid")
    score_by_method_key = {
        (row["method"], row["query_id"], row["file_id"]): float(row["score"])
        for row in method_rows
    }

    source_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in method_rows:
        if row["dataset"] == "ohrbench":
            source_scores[row["method"]][source(str(row["file_id"]))].append(float(row["score"]))

    query_source_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in method_rows:
        if row["dataset"] == "ohrbench":
            query_source_scores[(row["method"], str(row["query_id"]), source(str(row["file_id"])))].append(float(row["score"]))

    ohr_summary: dict[str, Any] = {}
    for method in methods:
        gap_values: list[float] = []
        max_by_source: dict[str, list[float]] = defaultdict(list)
        qids = sorted({str(row["query_id"]) for row in method_rows if row["dataset"] == "ohrbench"})
        for qid in qids:
            ohr_max = max(query_source_scores[(method, qid, "ohrbench")])
            physics_max = max(query_source_scores[(method, qid, "physics")])
            gap_values.append(ohr_max - physics_max)
            for file_source in ("ohrbench", "physics", "industrial"):
                max_by_source[file_source].append(max(query_source_scores[(method, qid, file_source)]))
        ohr_summary[method] = {
            "pair_score_distribution": {name: quantiles(values) for name, values in source_scores[method].items()},
            "query_max_distribution": {name: quantiles(values) for name, values in max_by_source.items()},
            "ohr_minus_physics_max_gap": quantiles(gap_values),
            "ohr_max_higher_rate": float(np.mean(np.asarray(gap_values) > 0)),
        }

    delta_summary: dict[str, Any] = {}
    for query_dataset in ("physics", "industrial", "ohrbench"):
        delta_summary[query_dataset] = {}
        for language in ("en", "fr"):
            deltas = []
            for row in method_rows:
                if row["dataset"] != query_dataset or row["method"] != "routed_hybrid":
                    continue
                if row["file_language"] != language:
                    continue
                hybrid_score = score_by_method_key[("hybrid_all", row["query_id"], row["file_id"])]
                deltas.append(float(row["score"]) - float(hybrid_score))
            delta_summary[query_dataset][language] = {
                "count": len(deltas),
                "distribution": quantiles(deltas),
                "nonzero_rate": float(np.mean(np.abs(deltas) > 1e-12)) if deltas else 0.0,
                "max_abs": float(max((abs(value) for value in deltas), default=0.0)),
            }

    component_summary: dict[str, Any] = {}
    for query_dataset in ("physics", "industrial", "ohrbench"):
        component_summary[query_dataset] = {}
        for relation in ("same", "cross"):
            subset = [
                row for row in component_rows
                if row["dataset"] == query_dataset and row["relation"] == relation
            ]
            dense_values = [float(row["dense_norm"]) for row in subset]
            direct_bm_values = [float(row["direct_bm25_norm"]) for row in subset]
            page_bm_values = [float(row["page_bm25_max_norm"]) for row in subset]
            component_summary[query_dataset][relation] = {
                "pairs": len(subset),
                "dense_norm": quantiles(dense_values),
                "direct_bm25_norm": quantiles(direct_bm_values),
                "page_bm25_max_norm": quantiles(page_bm_values),
                "dense_direct_bm_correlation": correlation(dense_values, direct_bm_values),
            }

    english_deltas = delta_summary["physics"]["en"], delta_summary["industrial"]["en"], delta_summary["ohrbench"]["en"]
    english_max_abs = max(item["max_abs"] for item in english_deltas)
    report = {
        "protocol": {
            "qrels_used_for_scoring": False,
            "method_score_source": str(METHOD_ROWS),
            "ohr_source_comparison": "OHR query file scores grouped by file source: ohrbench vs physics vs industrial",
            "routed_formula": "English files use hybrid-all formula; French files use dense-only",
        },
        "ohr_query_source_separation": ohr_summary,
        "routed_minus_hybrid_all": {
            "english_max_abs_delta": english_max_abs,
            "by_query_dataset_and_language": delta_summary,
        },
        "normalized_component_distributions": component_summary,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.md").write_text(make_report(report), encoding="utf-8")
    image_path = make_png(method_rows)
    print(image_path.resolve())
    print(make_report(report))
    return 0


def make_report(report: dict[str, Any]) -> str:
    lines = [
        "# Routed score calibration audit",
        "",
        "All score analysis is query-local and qrels-free. OHR query source separation compares file-source groups within the full corpus.",
        "",
        "## OHR queries: OHR files versus Physics files",
        "",
        "| Method | OHR max p50 | Physics max p50 | OHR - Physics max p50 | OHR higher |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("dense_only", "hybrid_all", "routed_hybrid"):
        values = report["ohr_query_source_separation"][method]
        lines.append(
            f"| {method} | {values['query_max_distribution']['ohrbench']['p50']:.4f} | "
            f"{values['query_max_distribution']['physics']['p50']:.4f} | "
            f"{values['ohr_minus_physics_max_gap']['p50']:.4f} | "
            f"{values['ohr_max_higher_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Routed minus hybrid-all",
            "",
            "The direct hybrid component for English files is unchanged. The final file score can still change because the routed page-side max-normalization denominator is computed across both language groups.",
            f"Observed maximum absolute change in final English file score: `{report['routed_minus_hybrid_all']['english_max_abs_delta']:.3e}`.",
            "",
            "| Query dataset | File language | Delta p50 | Delta mean | Nonzero |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset in ("physics", "industrial", "ohrbench"):
        for language in ("en", "fr"):
            values = report["routed_minus_hybrid_all"]["by_query_dataset_and_language"][dataset][language]
            lines.append(
                f"| {dataset} | {language} | {values['distribution']['p50']:.4f} | "
                f"{values['distribution']['mean']:.4f} | {values['nonzero_rate']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Max-normalized component distributions",
            "",
            "| Query dataset | Relation | Dense p50 | Direct BM25 p50 | Page BM25 p50 | Dense/BM25 corr |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("physics", "industrial", "ohrbench"):
        for relation in ("same", "cross"):
            values = report["normalized_component_distributions"][dataset][relation]
            lines.append(
                f"| {dataset} | {relation} | {values['dense_norm']['p50']:.4f} | "
                f"{values['direct_bm25_norm']['p50']:.4f} | {values['page_bm25_max_norm']['p50']:.4f} | "
                f"{values['dense_direct_bm_correlation']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: max-normalization aligns only the query-level maxima. It does not align medians, tails, spread or semantic calibration between dense and BM25.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_png(method_rows: list[dict[str, Any]]) -> Path:
    colors = {"ohrbench": "#0f766e", "physics": "#c2410c", "industrial": "#2563eb"}
    methods = [("dense_only", "Dense-only"), ("hybrid_all", "Hybrid-all"), ("routed_hybrid", "Routed")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("OHR source separation and routed score changes", fontsize=15, fontweight="bold")
    pair_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    query_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in method_rows:
        if row["dataset"] != "ohrbench":
            continue
        method = str(row["method"])
        file_source = source(str(row["file_id"]))
        pair_values[(method, file_source)].append(float(row["score"]))
        query_values[(method, str(row["query_id"]), file_source)].append(float(row["score"]))
    for column, (method, label) in enumerate(methods):
        upper = axes[0, column]
        lower = axes[1, column]
        upper.set_title(f"{label}: OHR query file-source scores")
        upper.set_xlim(0, 1)
        upper.set_ylim(0, 1)
        upper.set_xlabel("File score")
        upper.set_ylabel("ECDF")
        upper.grid(alpha=0.2)
        for source_name in ("ohrbench", "physics", "industrial"):
            values = np.sort(np.asarray(pair_values[(method, source_name)], dtype=float))
            ecdf_y = np.arange(1, len(values) + 1) / len(values)
            upper.plot(values, ecdf_y, color=colors[source_name], linewidth=1.4, label=source_name)
        upper.legend(frameon=False, fontsize=8, loc="lower right")
        qids = sorted({str(row["query_id"]) for row in method_rows if row["dataset"] == "ohrbench" and row["method"] == method})
        gaps = np.asarray(
            [
                max(query_values[(method, qid, "ohrbench")])
                - max(query_values[(method, qid, "physics")])
                for qid in qids
            ],
            dtype=float,
        )
        bins = np.linspace(-1, 1, 15)
        lower.hist(gaps[gaps < 0], bins=bins, color=colors["physics"], alpha=0.75, label="Physics higher")
        lower.hist(gaps[gaps >= 0], bins=bins, color=colors["ohrbench"], alpha=0.75, label="OHR higher")
        lower.axhline(0, color="#374151", linewidth=1)
        lower.set_xlim(-1, 1)
        lower.set_xlabel("OHR max - Physics max")
        lower.set_ylabel("Queries")
        lower.set_title(f"OHR higher: {(gaps > 0).mean():.0%}")
        lower.legend(frameon=False, fontsize=8, loc="upper right")
        lower.grid(alpha=0.2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "ohr-source-separation.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
