"""Compare dense and hybrid scores on source-restricted query/file blocks."""

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
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/diagonal_method_calibration"
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


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_order = np.argsort(np.argsort(np.asarray(left), kind="stable"), kind="stable")
    right_order = np.argsort(np.argsort(np.asarray(right), kind="stable"), kind="stable")
    if np.std(left_order) == 0 or np.std(right_order) == 0:
        return 0.0
    return float(np.corrcoef(left_order, right_order)[0, 1])


def file_source(file_id: str) -> str:
    return file_id.split("::", 1)[0].replace("vidore_", "")


def main() -> int:
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    query_ids = [str(value) for value in metadata["query_ids"]]
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    page_bm25 = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25 = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray(
        [file_position[page.split("#page=", 1)[0]] for page in page_ids], dtype=np.int32
    )
    file_sources = {fid: file_source(fid) for fid in file_ids}
    source_files = {
        source: [index for index, fid in enumerate(file_ids) if file_sources[fid] == source]
        for source in ("physics", "industrial", "ohrbench")
    }
    source_page_indices = {
        source: np.asarray(
            [index for index, fi in enumerate(page_file_pos) if int(fi) in set(file_indices)],
            dtype=np.int32,
        )
        for source, file_indices in source_files.items()
    }

    rows: list[dict[str, Any]] = []
    query_deltas: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qi, qid in enumerate(query_ids):
        query_dataset = qid.split("::", 1)[0].replace("vidore_", "")
        if query_dataset not in source_files:
            continue
        selected_files = source_files[query_dataset]
        selected_pages = source_page_indices[query_dataset]
        dense_page = max_norm(dense[qi, selected_pages])
        bm_page = max_norm(page_bm25[qi, selected_pages])
        dense_file = np.zeros(len(selected_files), dtype=np.float64)
        bm_page_file = np.zeros(len(selected_files), dtype=np.float64)
        selected_file_position = {fi: index for index, fi in enumerate(selected_files)}
        selected_page_file_pos = np.asarray(
            [selected_file_position[int(fi)] for fi in page_file_pos[selected_pages]], dtype=np.int32
        )
        for fi in range(len(selected_files)):
            mask = selected_page_file_pos == fi
            dense_file[fi] = float(dense_page[mask].max())
            bm_page_file[fi] = float(bm_page[mask].max())
        direct_bm = max_norm(file_bm25[qi, selected_files])
        hybrid_page = BM25_ALPHA * bm_page + (1.0 - BM25_ALPHA) * dense_page
        hybrid_page_max = np.zeros(len(selected_files), dtype=np.float64)
        np.maximum.at(hybrid_page_max, selected_page_file_pos, hybrid_page)
        direct_hybrid = BM25_ALPHA * direct_bm + (1.0 - BM25_ALPHA) * dense_file
        hybrid_file = 0.50 * direct_hybrid + 0.50 * max_norm(hybrid_page_max)
        delta = hybrid_file - dense_file
        dense_order = np.argsort(-dense_file, kind="stable")
        hybrid_order = np.argsort(-hybrid_file, kind="stable")
        dense_rank = {int(index): rank for rank, index in enumerate(dense_order, start=1)}
        hybrid_rank = {int(index): rank for rank, index in enumerate(hybrid_order, start=1)}
        query_deltas[query_dataset].append(
            {
                "query_id": qid,
                "delta_p50": float(np.median(delta)),
                "delta_max": float(delta.max()),
                "delta_min": float(delta.min()),
                "top1_changed": int(int(dense_order[0]) != int(hybrid_order[0])),
            }
        )
        for local_index, global_file_index in enumerate(selected_files):
            rows.append(
                {
                    "query_id": qid,
                    "dataset": query_dataset,
                    "file_id": file_ids[global_file_index],
                    "dense_score": float(dense_file[local_index]),
                    "hybrid_score": float(hybrid_file[local_index]),
                    "delta_hybrid_minus_dense": float(delta[local_index]),
                    "dense_rank": dense_rank[local_index],
                    "hybrid_rank": hybrid_rank[local_index],
                }
            )

    summary: dict[str, Any] = {
        "protocol": {
            "blocks": "Physics queries x Physics files; Industrial queries x Industrial files; OHR queries x OHR files",
            "file_counts": {source: len(indices) for source, indices in source_files.items()},
            "bm25_alpha": BM25_ALPHA,
            "normalization": "dense, page BM25 and direct BM25 are max-normalized inside each source-restricted query corpus",
            "qrels_used_for_scoring": False,
        },
        "blocks": {},
    }
    for dataset in ("physics", "industrial", "ohrbench"):
        subset = [row for row in rows if row["dataset"] == dataset]
        dense_values = [float(row["dense_score"]) for row in subset]
        hybrid_values = [float(row["hybrid_score"]) for row in subset]
        deltas = [float(row["delta_hybrid_minus_dense"]) for row in subset]
        query_rows = query_deltas[dataset]
        summary["blocks"][dataset] = {
            "queries": len(query_rows),
            "files": len(source_files[dataset]),
            "dense_distribution": quantiles(dense_values),
            "hybrid_distribution": quantiles(hybrid_values),
            "delta_hybrid_minus_dense": quantiles(deltas),
            "rank_spearman": spearman(dense_values, hybrid_values),
            "query_delta_p50": quantiles([row["delta_p50"] for row in query_rows]),
            "top1_changed_rate": float(np.mean([row["top1_changed"] for row in query_rows])) if query_rows else 0.0,
            "query_delta_positive_rate": float(np.mean(np.asarray([row["delta_p50"] for row in query_rows]) > 0)) if query_rows else 0.0,
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "report.md").write_text(make_report(summary), encoding="utf-8")
    image_path = make_png(rows)
    print(image_path.resolve())
    print(make_report(summary))
    return 0


def make_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Diagonal method calibration",
        "",
        "Each query is scored only against files from its own benchmark source. Dense and hybrid are both applied to the same file set.",
        "",
        "| Block | Queries | Files | Dense p50 | Hybrid p50 | Delta p50 | Spearman | Top-1 changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohrbench"):
        values = summary["blocks"][dataset]
        lines.append(
            f"| {dataset} | {values['queries']} | {values['files']} | {values['dense_distribution']['p50']:.4f} | "
            f"{values['hybrid_distribution']['p50']:.4f} | {values['delta_hybrid_minus_dense']['p50']:.4f} | "
            f"{values['rank_spearman']:.4f} | {values['top1_changed_rate']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The delta is hybrid minus dense on the same restricted file set. This is a method-calibration diagnostic, not a production retrieval score.",
        ]
    )
    return "\n".join(lines) + "\n"


def make_png(rows: list[dict[str, Any]]) -> Path:
    colors = {"dense": "#0f766e", "hybrid": "#c2410c", "positive": "#c2410c", "negative": "#0f766e"}
    datasets = [("physics", "Physics files"), ("industrial", "Industrial files"), ("ohrbench", "OHR files")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Dense versus hybrid on source-restricted file sets", fontsize=15, fontweight="bold")
    for column, (dataset, label) in enumerate(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        dense_values = np.sort(np.asarray([row["dense_score"] for row in subset], dtype=float))
        hybrid_values = np.sort(np.asarray([row["hybrid_score"] for row in subset], dtype=float))
        for values, color, name in ((dense_values, colors["dense"], "Dense"), (hybrid_values, colors["hybrid"], "Hybrid")):
            axes[0, column].plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=1.7, label=name)
            axes[0, column].axvline(np.median(values), color=color, linestyle="--", linewidth=0.8, alpha=0.7)
        axes[0, column].set_title(label)
        axes[0, column].set_xlim(0, 1)
        axes[0, column].set_ylim(0, 1.02)
        axes[0, column].set_xlabel("File score")
        axes[0, column].set_ylabel("ECDF")
        axes[0, column].grid(alpha=0.2)
        axes[0, column].legend(frameon=False, loc="lower right")

        deltas = np.asarray([row["delta_hybrid_minus_dense"] for row in subset], dtype=float)
        bins = np.linspace(-1, 1, 17)
        axes[1, column].hist(deltas[deltas < 0], bins=bins, color=colors["negative"], alpha=0.8, label="Hybrid lower")
        axes[1, column].hist(deltas[deltas >= 0], bins=bins, color=colors["positive"], alpha=0.8, label="Hybrid higher")
        axes[1, column].axvline(0, color="#374151", linewidth=1)
        axes[1, column].set_xlim(-1, 1)
        axes[1, column].set_xlabel("Hybrid - dense")
        axes[1, column].set_ylabel("File-query pairs")
        axes[1, column].grid(alpha=0.2)
        axes[1, column].legend(frameon=False, loc="upper right")

    path = OUTPUT / "diagonal-method-calibration.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
