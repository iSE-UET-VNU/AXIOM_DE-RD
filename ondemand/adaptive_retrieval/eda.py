"""Create a transparent EDA profile for the adaptive-K training features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, spearmanr

from .features import FEATURE_NAMES


FEATURE_DEFINITIONS: dict[str, tuple[str, str]] = {
    "score_mean_top_1": (
        "Mean fused retrieval score in the first 1 ranked page.",
        "sum(s_r for r=1..1) / 1",
    ),
    "score_mean_top_5": (
        "Cumulative prefix mean of the first 5 ranked-page scores.",
        "sum(s_r for r=1..5) / 5",
    ),
    "score_mean_top_10": (
        "Cumulative prefix mean of the first 10 ranked-page scores.",
        "sum(s_r for r=1..10) / 10",
    ),
    "score_mean_top_20": (
        "Cumulative prefix mean of the first 20 ranked-page scores.",
        "sum(s_r for r=1..20) / 20",
    ),
    "score_mean_top_50": (
        "Cumulative prefix mean of the first 50 ranked-page scores.",
        "sum(s_r for r=1..50) / 50",
    ),
    "score_mean_top_100": (
        "Cumulative prefix mean of the first 100 ranked-page scores.",
        "sum(s_r for r=1..100) / 100",
    ),
    "score_mean_tail": (
        "Mean score of the bottom quarter of the ranking (ranks 76–100 here).",
        "sum(s_r for r=76..100) / 25",
    ),
    "score_std": (
        "Population standard deviation of all 100 fused scores (not sample SD).",
        "sqrt(sum((s_r - mean(s))^2 for r=1..100) / 100)",
    ),
    "score_range": (
        "Difference between the highest and lowest fused retrieval scores.",
        "max(s_r for r=1..100) - min(s_r for r=1..100)",
    ),
    "gap_at_1": (
        "Score drop between rank 1 and rank 2.",
        "s[1] - s[2]",
    ),
    "gap_at_5": (
        "Score drop between rank 5 and rank 6.",
        "s[5] - s[6]",
    ),
    "gap_at_10": (
        "Score drop between rank 10 and rank 11.",
        "s[10] - s[11]",
    ),
    "gap_at_20": (
        "Score drop between rank 20 and rank 21.",
        "s[20] - s[21]",
    ),
    "gap_at_50": (
        "Score drop between rank 50 and rank 51.",
        "s[50] - s[51]",
    ),
    "max_gap": (
        "Largest adjacent score drop anywhere in the top-100 ranking.",
        "max(s[r] - s[r+1]), r=1..99",
    ),
    "max_gap_rank": (
        "1-based rank before the largest adjacent drop, divided by 100.",
        "r*/100, where r*=argmax_r(s_r-s_(r+1)), r=1..99",
    ),
    "slope_top_10": (
        "Least-squares score-versus-rank slope over ranks 1–10.",
        "OLS slope of s[r] ~ r, r=1..10",
    ),
    "slope_top_20": (
        "Least-squares score-versus-rank slope over ranks 1–20.",
        "OLS slope of s[r] ~ r, r=1..20",
    ),
    "slope_top_50": (
        "Least-squares score-versus-rank slope over ranks 1–50.",
        "OLS slope of s[r] ~ r, r=1..50",
    ),
    "slope_top_100": (
        "Least-squares score-versus-rank slope over ranks 1–100.",
        "OLS slope of s[r] ~ r, r=1..100",
    ),
    "softmax_entropy": (
        "Entropy of a temperature-1 softmax over the 100 fused retrieval scores.",
        "-sum(p_r*ln(p_r)); p_r=exp(s_r-max(s))/sum_j(exp(s_j-max(s)))",
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(values: np.ndarray) -> dict[str, float | int]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        raise ValueError("cannot profile an empty or entirely missing feature")
    return {
        "n": int(clean.size),
        "missing_n": int(values.size - clean.size),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "std_sample": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        "unique_count": int(np.unique(clean).size),
        "zero_count": int(np.count_nonzero(clean == 0.0)),
        "zero_share": float(np.mean(clean == 0.0)),
        "skewness": float(skew(clean, bias=False)) if clean.size > 2 else 0.0,
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if not np.isfinite(value):
        return "n/a"
    return f"{float(value):.6g}"


def _histogram_svg(values: np.ndarray, feature: str) -> str:
    counts, edges = np.histogram(values, bins="fd")
    if len(counts) > 24:
        counts, edges = np.histogram(values, bins=24)
    if len(counts) < 6 and np.unique(values).size > 6:
        counts, edges = np.histogram(values, bins=12)

    width, height = 580, 190
    left, right, top, bottom = 50, 12, 10, 40
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(int(counts.max()), 1)
    gap = 2
    bar_w = plot_w / max(len(counts), 1)
    bars = []
    for index, count in enumerate(counts):
        bar_h = plot_h * int(count) / max_count
        x = left + index * bar_w + gap / 2
        y = top + plot_h - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w-gap, 1):.2f}" '
            f'height="{bar_h:.2f}" fill="#4c78a8"><title>{int(count)} queries</title></rect>'
        )

    minimum, median, maximum = float(values.min()), float(np.median(values)), float(values.max())
    positions = []
    for value, anchor in ((minimum, "start"), (median, "middle"), (maximum, "end")):
        ratio = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
        x = left + ratio * plot_w
        positions.append(
            f'<text x="{x:.2f}" y="{height-10}" text-anchor="{anchor}">{_fmt(value)}</text>'
        )
    title = html.escape(feature)
    return (
        f'<svg class="histogram" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Histogram for {title}"><title>Distribution of {title}</title>'
        f'<desc>Histogram over {values.size} query rows. Horizontal axis is feature value; vertical axis is count.</desc>'
        f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#666" />'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#666" />'
        f'<text class="axis" transform="translate(15 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Queries</text>'
        f'{"".join(bars)}{"".join(positions)}'
        f'<text class="axis" x="{left+plot_w/2}" y="{height-1}" text-anchor="middle">Feature value</text></svg>'
    )


def _write_plots(
    output_dir: Path,
    feature_values: dict[str, np.ndarray],
    target: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, str, float]]]:
    names = [*FEATURE_NAMES, "target_k"]
    matrix = np.column_stack([*(feature_values[name] for name in FEATURE_NAMES), target])
    corr = np.corrcoef(matrix, rowvar=False)
    corr_path = output_dir / "feature_correlation.csv"
    _write_csv(
        corr_path,
        ["feature", *names],
        [
            {"feature": name, **{other: float(corr[i, j]) for j, other in enumerate(names)}}
            for i, name in enumerate(names)
        ],
    )

    fig, ax = plt.subplots(figsize=(18, 16), constrained_layout=True)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Pearson correlation: score-curve features and target K")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Pearson correlation")
    fig.savefig(output_dir / "feature_correlation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(7, 3, figsize=(17, 22), constrained_layout=True)
    for ax, name in zip(axes.flat, FEATURE_NAMES):
        values = feature_values[name]
        ax.hist(values, bins="fd", color="#4c78a8", edgecolor="white")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Value", fontsize=8)
        ax.set_ylabel("Queries", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Feature distributions (220 queries)", fontsize=15)
    fig.savefig(output_dir / "feature_distributions.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bins = np.arange(0.5, 101.5, 5)
    ax.hist(target, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_title("Training target distribution")
    ax.set_xlabel("Required K label (top-100 cap)")
    ax.set_ylabel("Queries")
    fig.savefig(output_dir / "target_distribution.png", dpi=180)
    plt.close(fig)

    pairs: list[tuple[str, str, float]] = []
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            pairs.append((left, names[j], float(corr[i, j])))
    pairs.sort(key=lambda row: abs(row[2]), reverse=True)
    return corr, pairs


def build_eda(feature_matrix: Path, training_table: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = _read_csv(feature_matrix)
    training_rows = _read_csv(training_table)
    if not matrix_rows:
        raise ValueError(f"no data rows in {feature_matrix}")
    qids = [row["qid"] for row in matrix_rows]
    training_by_qid = {row["qid"]: row for row in training_rows}
    if len(set(qids)) != len(qids) or set(qids) != set(training_by_qid):
        raise ValueError("feature_matrix and training_table must contain the same unique qids")

    missing_features = [name for name in FEATURE_NAMES if name not in matrix_rows[0]]
    if missing_features:
        raise ValueError(f"missing expected model inputs: {missing_features}")
    feature_values = {
        name: np.asarray([float(row[name]) for row in matrix_rows], dtype=np.float64)
        for name in FEATURE_NAMES
    }
    target = np.asarray([float(row["target_k"]) for row in matrix_rows], dtype=np.float64)
    prediction = np.asarray([float(training_by_qid[qid]["prediction_k_oof"]) for qid in qids])
    sources = [row["source"] for row in matrix_rows]
    if not np.isfinite(np.column_stack([*feature_values.values(), target, prediction])).all():
        raise ValueError("found non-finite feature, target, or OOF prediction")

    profile_rows: list[dict[str, Any]] = []
    source_profile_rows: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        values = feature_values[name]
        stats = _stats(values)
        pearson_target = float(np.corrcoef(values, target)[0, 1])
        spearman_target = float(spearmanr(values, target).statistic)
        description, formula = FEATURE_DEFINITIONS[name]
        profile_rows.append({
            "feature": name,
            "description": description,
            "formula": formula,
            **stats,
            "pearson_with_target_k": pearson_target,
            "spearman_with_target_k": spearman_target,
        })
        for source in sorted(set(sources)):
            source_values = values[np.asarray(sources) == source]
            source_stats = _stats(source_values)
            source_profile_rows.append({
                "source": source,
                "feature": name,
                "n": source_stats["n"],
                "min": source_stats["min"],
                "max": source_stats["max"],
                "mean": source_stats["mean"],
                "median": source_stats["median"],
                "std_sample": source_stats["std_sample"],
                "unique_count": source_stats["unique_count"],
                "zero_count": source_stats["zero_count"],
                "zero_share": source_stats["zero_share"],
            })

    profile_fields = [
        "feature", "description", "formula", "n", "missing_n", "min", "max",
        "mean", "median", "std_sample", "unique_count", "zero_count", "zero_share",
        "skewness", "pearson_with_target_k", "spearman_with_target_k",
    ]
    _write_csv(output_dir / "feature_profile.csv", profile_fields, profile_rows)
    _write_csv(
        output_dir / "feature_profile_by_source.csv",
        ["source", "feature", "n", "min", "max", "mean", "median", "std_sample",
         "unique_count", "zero_count", "zero_share"],
        source_profile_rows,
    )

    corr, pairs = _write_plots(output_dir, feature_values, target)
    label_counts: dict[str, int] = {}
    for value in target.astype(int):
        label_counts[str(value)] = label_counts.get(str(value), 0) + 1
    unreachable_by_label = {
        label: sum(
            int(row["target_k"]) == int(label) and row["reachable_at_cap"].lower() == "false"
            for row in matrix_rows
        )
        for label in label_counts
    }
    _write_csv(
        output_dir / "label_profile.csv",
        ["target_k", "query_count", "share", "unreachable_count"],
        [
            {
                "target_k": int(label),
                "query_count": count,
                "share": count / len(matrix_rows),
                "unreachable_count": unreachable_by_label[label],
            }
            for label, count in sorted(label_counts.items(), key=lambda pair: int(pair[0]))
        ],
    )
    source_counts = {source: sources.count(source) for source in sorted(set(sources))}
    source_target = {
        source: {
            "count": int(source_counts[source]),
            "target_mean": float(np.mean(target[np.asarray(sources) == source])),
            "target_median": float(np.median(target[np.asarray(sources) == source])),
            "unreachable_count": sum(
                row["source"] == source and row["reachable_at_cap"].lower() == "false"
                for row in matrix_rows
            ),
        }
        for source in sorted(set(sources))
    }
    target_stats = _stats(target)
    prediction_stats = _stats(prediction)
    summary = {
        "dataset_rows": len(matrix_rows),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "source_counts": source_counts,
        "target_distribution": label_counts,
        "target_stats": target_stats,
        "oof_prediction_k_stats": prediction_stats,
        "target_by_source": source_target,
        "top_correlated_pairs": [
            {"feature_a": left, "feature_b": right, "pearson_r": value}
            for left, right, value in pairs[:20]
        ],
        "highest_absolute_feature_target_correlations": sorted(
            [
                {"feature": row["feature"], "pearson_r": row["pearson_with_target_k"],
                 "spearman_r": row["spearman_with_target_k"]}
                for row in profile_rows
            ],
            key=lambda row: abs(row["spearman_r"]),
            reverse=True,
        ),
        "data_checks": {
            "all_feature_values_finite": True,
            "all_features_have_zero_missing": all(row["missing_n"] == 0 for row in profile_rows),
            "all_220_qids_have_oof_predictions": len(training_by_qid) == len(matrix_rows),
            "gold_or_qid_source_used_as_model_feature": False,
            "transformations_applied_during_eda": [],
        },
        "inputs": {
            "feature_matrix": str(feature_matrix),
            "feature_matrix_sha256": _sha256(feature_matrix),
            "training_table": str(training_table),
            "training_table_sha256": _sha256(training_table),
        },
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    source_rows_html = "".join(
        f"<tr><td>{html.escape(source)}</td><td>{item['count']}</td>"
        f"<td>{item['target_mean']:.2f}</td><td>{item['target_median']:.2f}</td>"
        f"<td>{item['unreachable_count']}</td></tr>"
        for source, item in source_target.items()
    )
    feature_sections = []
    for row in profile_rows:
        name = row["feature"]
        safe_name = html.escape(name)
        description = html.escape(row["description"])
        formula = html.escape(row["formula"])
        values = feature_values[name]
        stats_html = "".join(
            f"<tr><th>{label}</th><td>{value}</td></tr>"
            for label, value in (
                ("Min", _fmt(row["min"])),
                ("Max", _fmt(row["max"])),
                ("Mean", _fmt(row["mean"])),
                ("Median", _fmt(row["median"])),
                ("Std (sample)", _fmt(row["std_sample"])),
                ("Unique values", _fmt(row["unique_count"])),
                ("Exactly zero", f"{row['zero_count']} ({row['zero_share']:.1%})"),
                ("Missing", _fmt(row["missing_n"])),
                ("Pearson r with target K", _fmt(row["pearson_with_target_k"])),
                ("Spearman r with target K", _fmt(row["spearman_with_target_k"])),
            )
        )
        feature_sections.append(
            f'<section class="feature-row"><div><h2>{safe_name}</h2>'
            f'<p>{description}</p><p class="formula"><code>{formula}</code></p>'
            f'<table class="stats"><tbody>{stats_html}</tbody></table></div>'
            f'<div>{_histogram_svg(values, name)}</div></section>'
        )

    intermediate_count = len(matrix_rows) - label_counts.get("1", 0) - label_counts.get("100", 0)
    label_summary_text = (
        f"K=1: {label_counts.get('1', 0)} queries; K=100: {label_counts.get('100', 0)} queries; "
        f"intermediate labels: {intermediate_count} across "
        f"{len(label_counts) - int('1' in label_counts) - int('100' in label_counts)} distinct K values."
    )
    top_pairs_html = "".join(
        f"<tr><td>{html.escape(left)}</td><td>{html.escape(right)}</td><td>{value:.4f}</td></tr>"
        for left, right, value in pairs[:15]
    )
    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive-K feature EDA</title>
<style>
body{{font:15px/1.45 Segoe UI,Arial,sans-serif;color:#20252b;max-width:1500px;margin:24px auto;padding:0 20px}}
h1{{font-size:27px;margin-bottom:4px}}h2{{font-size:18px;margin:0 0 6px;overflow-wrap:anywhere}}
.intro{{color:#444;margin:6px 0 20px}}.feature-row{{display:grid;grid-template-columns:minmax(340px,0.95fr) minmax(360px,1.05fr);gap:24px;align-items:center;padding:18px 0;border-top:1px solid #ccd1d7}}
.feature-row p{{margin:5px 0}}.formula{{color:#454b51}}.stats{{border-collapse:collapse;margin-top:12px;width:100%;font-size:13px}}
.stats th,.stats td{{border-bottom:1px solid #e2e5e8;padding:4px 8px;text-align:left}}.stats th{{font-weight:500;width:58%}}.stats td{{font-variant-numeric:tabular-nums}}
.histogram{{width:100%;height:auto}}.histogram text{{font:12px Segoe UI,Arial,sans-serif;fill:#20252b}}.histogram text.axis{{font-size:11px}}
.section{{margin:24px 0}}table.summary{{border-collapse:collapse}}table.summary th,table.summary td{{padding:7px 10px;border-bottom:1px solid #d8dde2;text-align:left}}
code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}img{{display:block;max-width:100%;height:auto}}.note{{color:#4d535a}}
@media(max-width:800px){{.feature-row{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>Adaptive-K model feature profile</h1>
<p class="intro">{len(matrix_rows)} queries · 21 model inputs · top-100 retrieval scores. Each feature’s definition, formula, statistics and histogram appear together.</p>
<p>These are engineered summaries of the final fused retrieval-score curve. EDA did not scale, select, remove, discretize or otherwise alter any training value. Standard deviation in this report uses the sample definition (n−1); the model’s <code>score_std</code> feature itself was calculated as a population standard deviation.</p>
<div class="section"><h2>Dataset and target</h2>
<p>Source counts: {html.escape(json.dumps(source_counts, ensure_ascii=False))}. Target distribution: {html.escape(label_summary_text)}</p>
<p>Required-K target has a cap of 100: the rank needed to include every gold page, or 100 if one or more gold pages were not reachable in the top 100.</p>
<table class="summary"><thead><tr><th>Source</th><th>Queries</th><th>Mean target K</th><th>Median target K</th><th>Unreachable at 100</th></tr></thead><tbody>{source_rows_html}</tbody></table>
<img src="target_distribution.png" alt="Histogram of target K labels">
</div>
<div class="section"><h2>Feature-feature and feature-target correlation</h2>
<p class="note">Pearson correlation over all 220 rows. These are descriptive associations, not evidence of causal or out-of-sample predictive value. The source mix may contribute to correlations.</p>
<img src="feature_correlation.png" alt="Pearson correlation heatmap for all 21 model inputs and target K">
<table class="summary"><thead><tr><th>Feature A</th><th>Feature B</th><th>Pearson r</th></tr></thead><tbody>{top_pairs_html}</tbody></table>
</div>
<div class="section"><h2>Per-feature profiles</h2>{''.join(feature_sections)}</div>
<div class="section"><h2>Files</h2><ul><li><a href="feature_profile.csv">Global statistics, formulas, missing/zero counts and target correlations</a></li><li><a href="feature_profile_by_source.csv">Per-source feature statistics</a></li><li><a href="label_profile.csv">Counts for each required-K label and unreachable status</a></li><li><a href="feature_correlation.csv">Full Pearson correlation matrix</a></li><li><a href="feature_distributions.png">All feature histograms</a></li><li><a href="../feature_matrix.csv">Model feature matrix and labels</a></li><li><a href="../training_table.csv">Training table with nested OOF prediction</a></li><li><a href="eda_summary.json">Machine-readable EDA summary</a></li></ul></div>
</body></html>"""
    (output_dir / "feature_profile.html").write_text(html_doc, encoding="utf-8")

    report_lines = [
        "# Adaptive-K feature EDA",
        "",
        f"- Rows: {len(matrix_rows)}; features: {len(FEATURE_NAMES)}; sources: {source_counts}",
        f"- Target K: mean {target_stats['mean']:.2f}, median {target_stats['median']:.2f}, "
        f"sample SD {target_stats['std_sample']:.2f}, min {target_stats['min']:.0f}, max {target_stats['max']:.0f}.",
        f"- Target endpoints: K=1 has {label_counts.get('1', 0)} queries; K=100 has {label_counts.get('100', 0)} queries; "
        f"the other labels cover {intermediate_count} queries.",
        f"- OOF predicted K: mean {prediction_stats['mean']:.2f}, median {prediction_stats['median']:.2f}, "
        f"sample SD {prediction_stats['std_sample']:.2f}.",
        f"- Missing/non-finite feature values: {sum(row['missing_n'] for row in profile_rows)} / "
        f"{len(matrix_rows) * len(FEATURE_NAMES)}.",
        "- No feature scaling, selection, dimension reduction, sample reduction, binning or discretization was applied.",
        "- The global statistics pool three sources. Use the per-source file before interpreting a feature as source-agnostic.",
        "",
        "## Target distribution by source",
        "",
        "| Source | Queries | Mean target K | Median target K | Unreachable at cap |",
        "|---|---:|---:|---:|---:|",
    ]
    report_lines.extend(
        f"| {source} | {item['count']} | {item['target_mean']:.2f} | {item['target_median']:.2f} | {item['unreachable_count']} |"
        for source, item in source_target.items()
    )
    report_lines.extend([
        "",
        "## Highest absolute feature-feature correlations",
        "",
        "| Feature A | Feature B | Pearson r |",
        "|---|---|---:|",
    ])
    report_lines.extend(f"| {left} | {right} | {value:.4f} |" for left, right, value in pairs[:12])
    report_lines.extend([
        "",
        "## Most associated features with target K (Spearman)",
        "",
        "| Feature | Pearson r | Spearman r |",
        "|---|---:|---:|",
    ])
    report_lines.extend(
        f"| {row['feature']} | {row['pearson_with_target_k']:.4f} | {row['spearman_with_target_k']:.4f} |"
        for row in sorted(profile_rows, key=lambda r: abs(r["spearman_with_target_k"]), reverse=True)[:12]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("data/work/adaptive_retrieval_20261020_router_te3s_top100")
    parser.add_argument("--feature-matrix", type=Path, default=default_root / "feature_matrix.csv")
    parser.add_argument("--training-table", type=Path, default=default_root / "training_table.csv")
    parser.add_argument("--output-dir", type=Path, default=default_root / "eda")
    args = parser.parse_args()
    summary = build_eda(args.feature_matrix, args.training_table, args.output_dir)
    print(f"[adaptive-retrieval-eda] rows={summary['dataset_rows']} features={summary['feature_count']}")
    print(f"[adaptive-retrieval-eda] wrote {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
