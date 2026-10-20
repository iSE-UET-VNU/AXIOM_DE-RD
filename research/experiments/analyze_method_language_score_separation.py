"""Compare language-score separation under several cached scoring policies."""

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
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/method_language_score_separation"

BM25_ALPHA = 0.70


def max_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values)


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "mean": float(np.mean(array)),
    } if array.size else {key: 0.0 for key in ("p10", "p25", "p50", "p75", "p90", "mean")}


def file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def load_languages(file_ids: list[str]) -> dict[str, str]:
    from research.experiments.evaluate_language_routing_matrix import load_file_languages

    languages, _ = load_file_languages(file_ids)
    return languages


def build_scores() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = json.loads((BASE / "metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(value) for value in metadata["page_ids"]]
    file_ids = [str(value) for value in metadata["file_ids"]]
    query_ids = [str(value) for value in metadata["query_ids"]]
    dense = np.load(BASE / "query_page_dense.npy", mmap_mode="r")
    page_bm25 = np.load(BASE / "page_bm25_raw.npy", mmap_mode="r")
    file_bm25 = np.load(BASE / "file_bm25_raw.npy", mmap_mode="r")
    languages = load_languages(file_ids)
    file_position = {fid: index for index, fid in enumerate(file_ids)}
    page_file_pos = np.asarray([file_position[file_id(page)] for page in page_ids], dtype=np.int32)
    source_by_query = {qid: qid.split("::", 1)[0].replace("vidore_", "") for qid in query_ids}

    rows: list[dict[str, Any]] = []
    query_gaps: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    method_names = ("dense_only", "bm25_only", "hybrid_all", "routed_hybrid")
    for qi, qid in enumerate(query_ids):
        dense_norm = max_norm(dense[qi])
        page_bm_norm = max_norm(page_bm25[qi])
        direct_bm_norm = max_norm(file_bm25[qi])
        dense_file = np.zeros(len(file_ids), dtype=np.float64)
        page_bm_file = np.zeros(len(file_ids), dtype=np.float64)
        for fi in range(len(file_ids)):
            mask = page_file_pos == fi
            dense_file[fi] = float(dense_norm[mask].max())
            page_bm_file[fi] = float(page_bm_norm[mask].max())

        hybrid_page = BM25_ALPHA * page_bm_norm + (1.0 - BM25_ALPHA) * dense_norm
        hybrid_page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(hybrid_page_max, page_file_pos, hybrid_page)
        hybrid_direct = BM25_ALPHA * direct_bm_norm + (1.0 - BM25_ALPHA) * dense_file
        hybrid_all = 0.50 * hybrid_direct + 0.50 * max_norm(hybrid_page_max)
        bm25_only = 0.50 * direct_bm_norm + 0.50 * page_bm_file

        routed_page = np.zeros(len(page_ids), dtype=np.float64)
        routed_direct = np.zeros(len(file_ids), dtype=np.float64)
        for fi, fid in enumerate(file_ids):
            mask = page_file_pos == fi
            if languages[fid] == "en":
                routed_page[mask] = hybrid_page[mask]
                routed_direct[fi] = hybrid_direct[fi]
            else:
                routed_page[mask] = dense_norm[mask]
                routed_direct[fi] = dense_file[fi]
        routed_page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(routed_page_max, page_file_pos, routed_page)
        routed_hybrid = 0.50 * routed_direct + 0.50 * max_norm(routed_page_max)

        method_scores = {
            "dense_only": dense_file,
            "bm25_only": bm25_only,
            "hybrid_all": hybrid_all,
            "routed_hybrid": routed_hybrid,
        }
        for method in method_names:
            values = method_scores[method]
            for fi, fid in enumerate(file_ids):
                rows.append(
                    {
                        "query_id": qid,
                        "dataset": source_by_query[qid],
                        "method": method,
                        "file_id": fid,
                        "file_language": languages[fid],
                        "score": float(values[fi]),
                    }
                )
            french = [float(values[fi]) for fi, fid in enumerate(file_ids) if languages[fid] == "fr"]
            english = [float(values[fi]) for fi, fid in enumerate(file_ids) if languages[fid] == "en"]
            query_gaps[qid][method] = [max(french) - max(english)]

    protocol = {
        "queries": len(query_ids),
        "files": len(file_ids),
        "english_files": sum(languages[fid] == "en" for fid in file_ids),
        "french_files": sum(languages[fid] == "fr" for fid in file_ids),
        "bm25_alpha": BM25_ALPHA,
        "bm25_only_definition": "0.50 normalized direct BM25 + 0.50 max normalized page BM25",
        "hybrid_all_definition": "0.50 direct hybrid + 0.50 normalized max page hybrid",
        "routed_definition": "English files use hybrid; French files use dense-only",
        "qrels_used_for_scoring": False,
    }
    return rows, {"protocol": protocol, "query_gaps": query_gaps}


def summarize(rows: list[dict[str, Any]], query_gaps: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("dense_only", "bm25_only", "hybrid_all", "routed_hybrid"):
        method_rows = [row for row in rows if row["method"] == method]
        method_result: dict[str, Any] = {"all_pairs": {}}
        for language in ("en", "fr"):
            method_result["all_pairs"][language] = quantiles(
                [float(row["score"]) for row in method_rows if row["file_language"] == language]
            )
        for dataset in ("physics", "industrial", "ohrbench"):
            gaps = [query_gaps[qid][method][0] for qid in query_gaps if qid.split("::", 1)[0].replace("vidore_", "") == dataset]
            method_result[dataset] = {
                "queries": len(gaps),
                "max_gap_fr_minus_en": quantiles(gaps),
                "french_max_higher_rate": float(np.mean(np.asarray(gaps) > 0)) if gaps else 0.0,
            }
        result[method] = method_result
    return result


def make_png(rows: list[dict[str, Any]], query_gaps: dict[str, dict[str, list[float]]]) -> Path:
    colors = {"en": "#0f766e", "fr": "#c2410c", "positive": "#c2410c", "negative": "#0f766e"}
    methods = [
        ("dense_only", "Dense-only"),
        ("bm25_only", "BM25-only"),
        ("hybrid_all", "Hybrid-all"),
        ("routed_hybrid", "Routed: same hybrid / cross dense"),
    ]
    datasets = [("physics", "Physics"), ("industrial", "Industrial"), ("ohrbench", "OHR-Bench")]
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["dataset"], row["file_language"])].append(float(row["score"]))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 10})
    fig = plt.figure(figsize=(15, 16), constrained_layout=True)
    grid = fig.add_gridspec(8, 3, height_ratios=[3, 1, 3, 1, 3, 1, 3, 1], hspace=0.35)
    fig.suptitle("Language separation after score combination", fontsize=16, fontweight="bold")

    for method_index, (method, method_label) in enumerate(methods):
        upper_row = method_index * 2
        lower_row = upper_row + 1
        for column, (dataset, dataset_label) in enumerate(datasets):
            upper = fig.add_subplot(grid[upper_row, column])
            lower = fig.add_subplot(grid[lower_row, column])
            english = grouped[(method, dataset, "en")]
            french = grouped[(method, dataset, "fr")]
            for language, values, label in (("en", english, "English"), ("fr", french, "French")):
                sorted_values = np.sort(np.asarray(values, dtype=float))
                ecdf_y = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
                upper.plot(sorted_values, ecdf_y, color=colors[language], linewidth=1.6, label=label)
                upper.axvline(np.median(values), color=colors[language], linestyle="--", linewidth=0.8, alpha=0.75)
            upper.set_title(f"{method_label} - {dataset_label}")
            upper.set_xlim(0, 1)
            upper.set_ylim(0, 1.02)
            upper.set_ylabel("ECDF" if column == 0 else "")
            upper.set_xlabel("File score")
            upper.grid(alpha=0.2)
            if method_index == 0 and column == 2:
                upper.legend(frameon=False, loc="lower right")

            qids = [qid for qid in query_gaps if qid.split("::", 1)[0].replace("vidore_", "") == dataset]
            gaps = np.asarray([query_gaps[qid][method][0] for qid in qids], dtype=float)
            bins = np.linspace(-1, 1, 17)
            lower.hist(gaps[gaps < 0], bins=bins, color=colors["negative"], alpha=0.8)
            lower.hist(gaps[gaps >= 0], bins=bins, color=colors["positive"], alpha=0.8)
            lower.axvline(0, color="#374151", linewidth=1.0)
            lower.set_xlim(-1, 1)
            lower.set_ylabel("Queries" if column == 0 else "")
            lower.set_xlabel("French max - English max")
            lower.set_title(f"French higher: {(gaps > 0).mean():.0%}")
            lower.grid(alpha=0.2)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "method-language-score-separation.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    return path


def main() -> int:
    rows, context = build_scores()
    summary = {**context["protocol"], "methods": summarize(rows, context["query_gaps"])}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "scores.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    image_path = make_png(rows, context["query_gaps"])
    print(image_path.resolve())
    for method, values in summary["methods"].items():
        print(method, "physics_gap_p50=", values["physics"]["max_gap_fr_minus_en"]["p50"], "industrial_gap_p50=", values["industrial"]["max_gap_fr_minus_en"]["p50"], "ohr_gap_p50=", values["ohrbench"]["max_gap_fr_minus_en"]["p50"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
