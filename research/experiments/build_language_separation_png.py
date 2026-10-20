"""Build a static PNG for the English/French dense-score separation."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FILE_SCORES = ROOT / "data/work/ohr_replacement_20260920/retrieval/language_score_distribution_analysis/file_query_scores.jsonl"
QUERY_SCORES = ROOT / "data/work/ohr_replacement_20260920/retrieval/per_query_language_dense_distributions/query_scores.jsonl"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/per_query_language_dense_distributions/language-separation.png"


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def main() -> int:
    file_rows = [
        json.loads(line)
        for line in FILE_SCORES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    query_rows = [
        json.loads(line)
        for line in QUERY_SCORES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    file_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"en": [], "fr": []})
    query_gaps: dict[str, list[float]] = defaultdict(list)
    for row in file_rows:
        file_values[str(row["dataset"])][str(row["file_language"])].append(float(row["dense_raw"]))
    for row in query_rows:
        query_gaps[str(row["dataset"])].append(float(row["raw_max_gap_fr_minus_en"]))

    colors = {"en": "#0f766e", "fr": "#c2410c", "positive": "#c2410c", "negative": "#0f766e"}
    datasets = [("physics", "Physics"), ("industrial", "Industrial"), ("ohrbench", "OHR-Bench")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11})
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    fig.suptitle("Dense separation by document language", fontsize=15, fontweight="bold")

    for column, (dataset, label) in enumerate(datasets):
        upper = axes[0, column]
        lower = axes[1, column]
        english = file_values[dataset]["en"]
        french = file_values[dataset]["fr"]
        for language, values, language_label in (("en", english, "English"), ("fr", french, "French")):
            x, y = ecdf(values)
            upper.plot(x, y, color=colors[language], linewidth=2, label=language_label)
            upper.axvline(np.median(values), color=colors[language], linestyle="--", linewidth=1, alpha=0.7)
        upper.set_title(label)
        upper.set_xlabel("Raw max-page dense score")
        upper.set_ylabel("ECDF")
        upper.set_xlim(left=0)
        upper.set_ylim(0, 1.02)
        upper.grid(alpha=0.22)
        upper.legend(frameon=False, loc="lower right")

        gaps = np.asarray(query_gaps[dataset], dtype=float)
        bins = np.linspace(min(-0.7, float(gaps.min()) - 0.02), max(0.7, float(gaps.max()) + 0.02), 15)
        lower.hist(gaps[gaps < 0], bins=bins, color=colors["negative"], alpha=0.8, label="French lower")
        lower.hist(gaps[gaps >= 0], bins=bins, color=colors["positive"], alpha=0.8, label="French higher")
        lower.axvline(0, color="#374151", linewidth=1.2)
        lower.set_title(f"French max - English max; positive: {(gaps > 0).mean():.0%}")
        lower.set_xlabel("Per-query max-score gap")
        lower.set_ylabel("Queries")
        lower.set_xlim(bins[0], bins[-1])
        lower.grid(alpha=0.22)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, facecolor="white")
    plt.close(fig)
    print(OUTPUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
