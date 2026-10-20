"""Analyze dense/BM25/hybrid score scale and ranking bias."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "data/work/ohr_replacement_20260920/retrieval/combined_220_base"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/score_scale_rank_bias"
BM25_ALPHA = 0.70


def max_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values)


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


def rank_vector(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rank_vector(left)
    right_rank = rank_vector(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def source(value: str) -> str:
    return value.split("::", 1)[0].replace("vidore_", "")


def file_index_for_page(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def summarize_scores(dense: list[float], bm25: list[float], hybrid: list[float]) -> dict[str, Any]:
    return {
        "dense": quantiles(dense),
        "bm25": quantiles(bm25),
        "hybrid": quantiles(hybrid),
        "hybrid_minus_dense": quantiles([h - d for h, d in zip(hybrid, dense)]),
        "hybrid_minus_bm25": quantiles([h - b for h, b in zip(hybrid, bm25)]),
        "dense_bm25_spearman": spearman(np.asarray(dense), np.asarray(bm25)),
    }


def main() -> int:
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    query_ids = [str(value) for value in metadata["query_ids"]]
    dense_raw = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    bm25_raw = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_position[file_index_for_page(page)] for page in page_ids], dtype=np.int32)

    full_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"dense": [], "bm25": [], "hybrid": []})
    diagonal_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"dense": [], "bm25": [], "hybrid": []})
    query_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)

    source_file_positions = {
        dataset: [index for index, fid in enumerate(file_ids) if source(fid) == dataset]
        for dataset in ("physics", "industrial", "ohrbench")
    }
    source_page_positions = {
        dataset: np.asarray(
            [index for index, file_pos in enumerate(page_file_pos) if int(file_pos) in set(file_positions)],
            dtype=np.int32,
        )
        for dataset, file_positions in source_file_positions.items()
    }

    for qi, qid in enumerate(query_ids):
        query_dataset = source(qid)
        dense_page = max_norm(dense_raw[qi])
        bm_page = max_norm(bm25_raw[qi])
        hybrid_page = BM25_ALPHA * bm_page + (1.0 - BM25_ALPHA) * dense_page
        for key, values in (("dense", dense_page), ("bm25", bm_page), ("hybrid", hybrid_page)):
            full_values[query_dataset][key].extend(float(value) for value in values)

        dense_order = np.argsort(-dense_page, kind="stable")
        bm_order = np.argsort(-bm_page, kind="stable")
        hybrid_order = np.argsort(-hybrid_page, kind="stable")
        dense_top10 = set(int(index) for index in dense_order[:10])
        bm_top10 = set(int(index) for index in bm_order[:10])
        hybrid_top10 = set(int(index) for index in hybrid_order[:10])
        dense_rank = rank_vector(dense_page)
        bm_rank = rank_vector(bm_page)
        hybrid_rank = rank_vector(hybrid_page)
        query_metrics[query_dataset].append(
            {
                "query_id": qid,
                "page_spearman_hybrid_dense": spearman(dense_page, hybrid_page),
                "page_spearman_hybrid_bm25": spearman(bm_page, hybrid_page),
                "page_top10_overlap_hybrid_dense": len(hybrid_top10 & dense_top10) / 10.0,
                "page_top10_overlap_hybrid_bm25": len(hybrid_top10 & bm_top10) / 10.0,
                "dense_top1_rank_in_hybrid": int(hybrid_rank[int(dense_order[0])]),
                "bm25_top1_rank_in_hybrid": int(hybrid_rank[int(bm_order[0])]),
                "hybrid_top1_rank_in_dense": int(dense_rank[int(hybrid_order[0])]),
                "hybrid_top1_rank_in_bm25": int(bm_rank[int(hybrid_order[0])]),
            }
        )

        selected_pages = source_page_positions.get(query_dataset, np.asarray([], dtype=np.int32))
        if selected_pages.size:
            diagonal_dense = max_norm(dense_raw[qi, selected_pages])
            diagonal_bm = max_norm(bm25_raw[qi, selected_pages])
            diagonal_hybrid = BM25_ALPHA * diagonal_bm + (1.0 - BM25_ALPHA) * diagonal_dense
            for key, values in (("dense", diagonal_dense), ("bm25", diagonal_bm), ("hybrid", diagonal_hybrid)):
                diagonal_values[query_dataset][key].extend(float(value) for value in values)

    summary: dict[str, Any] = {
        "protocol": {
            "score_formula": "page_hybrid = 0.70 * max_norm(BM25) + 0.30 * max_norm(dense)",
            "full_corpus": "query-local normalization over all 5,091 pages",
            "diagonal_control": "query-local normalization restricted to pages from the query source",
            "qrels_used_for_scoring": False,
        },
        "full_corpus": {},
        "diagonal_control": {},
        "ranking": {},
    }
    for dataset in ("physics", "industrial", "ohrbench"):
        summary["full_corpus"][dataset] = summarize_scores(
            full_values[dataset]["dense"], full_values[dataset]["bm25"], full_values[dataset]["hybrid"]
        )
        summary["diagonal_control"][dataset] = summarize_scores(
            diagonal_values[dataset]["dense"], diagonal_values[dataset]["bm25"], diagonal_values[dataset]["hybrid"]
        )
        metrics = query_metrics[dataset]
        summary["ranking"][dataset] = {
            "queries": len(metrics),
            "page_spearman_hybrid_dense": quantiles([row["page_spearman_hybrid_dense"] for row in metrics]),
            "page_spearman_hybrid_bm25": quantiles([row["page_spearman_hybrid_bm25"] for row in metrics]),
            "page_top10_overlap_hybrid_dense": quantiles([row["page_top10_overlap_hybrid_dense"] for row in metrics]),
            "page_top10_overlap_hybrid_bm25": quantiles([row["page_top10_overlap_hybrid_bm25"] for row in metrics]),
            "dense_top1_rank_in_hybrid": quantiles([row["dense_top1_rank_in_hybrid"] for row in metrics]),
            "bm25_top1_rank_in_hybrid": quantiles([row["bm25_top1_rank_in_hybrid"] for row in metrics]),
            "hybrid_top1_rank_in_dense": quantiles([row["hybrid_top1_rank_in_dense"] for row in metrics]),
            "hybrid_top1_rank_in_bm25": quantiles([row["hybrid_top1_rank_in_bm25"] for row in metrics]),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "query_ranking_metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"dataset": dataset, **row}, ensure_ascii=False)
            for dataset, rows in query_metrics.items()
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.md").write_text(make_report(summary), encoding="utf-8")
    page_doc_path = make_png(full_values, diagonal_values)
    print(page_doc_path.resolve())
    print(make_report(summary))
    return 0


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Score scale and ranking bias",
        "",
        "Dense, BM25 and hybrid are evaluated on the same page/document candidates. Scores are query-local max-normalized; qrels are not used.",
        "",
        "## Full-corpus page score scale",
        "",
        "| Dataset | Dense p50 | BM25 p50 | Hybrid p50 | Hybrid-Dense p50 | Hybrid-BM25 p50 | Dense/BM25 rank corr |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohrbench"):
        values = summary["full_corpus"][dataset]
        lines.append(
            f"| {dataset} | {values['dense']['p50']:.4f} | {values['bm25']['p50']:.4f} | {values['hybrid']['p50']:.4f} | "
            f"{values['hybrid_minus_dense']['p50']:.4f} | {values['hybrid_minus_bm25']['p50']:.4f} | {values['dense_bm25_spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagonal source-restricted control",
            "",
            "| Dataset | Dense p50 | BM25 p50 | Hybrid p50 | Hybrid-Dense p50 | Hybrid-BM25 p50 | Dense/BM25 rank corr |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("physics", "industrial", "ohrbench"):
        values = summary["diagonal_control"][dataset]
        lines.append(
            f"| {dataset} | {values['dense']['p50']:.4f} | {values['bm25']['p50']:.4f} | {values['hybrid']['p50']:.4f} | "
            f"{values['hybrid_minus_dense']['p50']:.4f} | {values['hybrid_minus_bm25']['p50']:.4f} | {values['dense_bm25_spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Page ranking influence of hybrid",
            "",
            "| Dataset | Hybrid-Dense Spearman p50 | Hybrid-BM25 Spearman p50 | Top-10 overlap with Dense | Top-10 overlap with BM25 | Dense top-1 rank in Hybrid | BM25 top-1 rank in Hybrid |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("physics", "industrial", "ohrbench"):
        values = summary["ranking"][dataset]
        lines.append(
            f"| {dataset} | {values['page_spearman_hybrid_dense']['p50']:.4f} | {values['page_spearman_hybrid_bm25']['p50']:.4f} | "
            f"{values['page_top10_overlap_hybrid_dense']['p50']:.2%} | {values['page_top10_overlap_hybrid_bm25']['p50']:.2%} | "
            f"{values['dense_top1_rank_in_hybrid']['p50']:.1f} | {values['bm25_top1_rank_in_hybrid']['p50']:.1f} |"
        )
    lines.extend(
        [
            "",
            "A hybrid closer to BM25 than dense in rank correlation, overlap or top-1 survival indicates that the 0.70 BM25 weight dominates the common ranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_png(
    full_values: dict[str, dict[str, list[float]]],
    diagonal_values: dict[str, dict[str, list[float]]],
) -> Path:
    # Build a compact 2x3 figure: full-corpus page scale and diagonal control scale.
    datasets = [("physics", "Physics"), ("industrial", "Industrial"), ("ohrbench", "OHR-Bench")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Dense, BM25 and hybrid score scale", fontsize=15, fontweight="bold")
    colors = {"dense": "#0f766e", "bm25": "#2563eb", "hybrid": "#c2410c"}
    for column, (dataset, label) in enumerate(datasets):
        ax = axes[0, column]
        values = full_values[dataset]
        ax.hexbin(values["dense"], values["bm25"], gridsize=38, extent=(0, 1, 0, 1), mincnt=1, cmap="Blues", bins="log")
        ax.plot([0, 1], [0, 1], color="#374151", linestyle="--", linewidth=1, label="equal scale")
        ax.set_title(f"{label}: full-corpus page scores")
        ax.set_xlabel("Dense score")
        ax.set_ylabel("BM25 score")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8, loc="upper left")

        ax = axes[1, column]
        values = diagonal_values[dataset]
        ax.hist(
            np.asarray(values["hybrid"], dtype=float) - np.asarray(values["dense"], dtype=float),
            bins=np.linspace(-1, 1, 21), color=colors["hybrid"], alpha=0.75, label="Hybrid - Dense",
        )
        ax.hist(
            np.asarray(values["hybrid"], dtype=float) - np.asarray(values["bm25"], dtype=float),
            bins=np.linspace(-1, 1, 21), color=colors["bm25"], alpha=0.55, label="Hybrid - BM25",
        )
        ax.axvline(0, color="#374151", linewidth=1)
        ax.set_title(f"{label}: diagonal score deltas")
        ax.set_xlabel("Score difference")
        ax.set_ylabel("Page-query pairs")
        ax.set_xlim(-1, 1)
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    path = OUTPUT / "score-scale-rank-bias.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
