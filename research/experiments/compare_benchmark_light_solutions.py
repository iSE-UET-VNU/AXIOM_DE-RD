"""Compare light-retrieval solutions by benchmark dataset.

This is an offline comparison over cached 220-query runs.  It deliberately
recomputes macro page/file recall from the same gold pages so that runs with
different report formats can be compared under one protocol.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
BASELINE_REPORT = BUNDLE / "retrieval/report.json"


METHOD_ORDER = [
    "baseline_bm25",
    "embedded_image_ocr_bm25",
    "embeddings",
    "embedded_image_ocr_embeddings",
    "pymupdf",
]

METHOD_LABELS = {
    "baseline_bm25": "Baseline BM25 (PDF-inspector + Tesseract)",
    "embedded_image_ocr_bm25": "Embedded-image OCR text + BM25",
    "embeddings": "Embeddings (PDF-inspector + Tesseract text)",
    "embedded_image_ocr_embeddings": "Embedded-image OCR text + embeddings",
    "pymupdf": "PyMuPDF + BM25",
}

DATASET_LABELS = OrderedDict(
    [
        ("vidore_physics", "Physics"),
        ("vidore_industrial", "Industrial"),
        ("mpdocvqa", "MPDocVQA"),
    ]
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_ids(values: Iterable[Any]) -> list[str]:
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, dict):
            file_id = value.get("file_id") or value.get("doc_id")
            if file_id:
                result.append(file_id)
    return result


def page_ids_from_chunks(row: dict[str, Any]) -> list[str]:
    chunks = row.get("chunks", [])
    return [chunk["page_id"] for chunk in chunks if chunk.get("page_id")]


def normalize_run_row(row: dict[str, Any]) -> dict[str, Any]:
    if "top20_pages" in row:
        pages = list(row["top20_pages"])
    elif "returned_pages" in row:
        pages = list(row["returned_pages"])
    else:
        pages = page_ids_from_chunks(row)

    selected = row.get("selected_files", [])
    return {
        "query_id": row["query_id"],
        "selected_files": file_ids(selected),
        "top20_pages": pages[:20],
    }


def query_metrics(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, float]:
    gold_pages = set(gold["gold_pages"])
    gold_files = set(gold["gold_files"])
    top20 = row["top20_pages"][:20]
    selected = set(row["selected_files"][:3])
    return {
        "page_recall@10": len(set(top20[:10]) & gold_pages) / len(gold_pages),
        "page_recall@20": len(set(top20[:20]) & gold_pages) / len(gold_pages),
        "file_recall@3": len(selected & gold_files) / len(gold_files),
    }


def aggregate(rows: dict[str, dict[str, Any]], gold: dict[str, dict[str, Any]]) -> dict[str, Any]:
    per_query = {}
    for query_id, row in rows.items():
        per_query[query_id] = query_metrics(row, gold[query_id])

    by_dataset: dict[str, dict[str, Any]] = OrderedDict()
    for prefix, label in DATASET_LABELS.items():
        query_ids = [query_id for query_id in gold if query_id.startswith(prefix + "::")]
        values = [per_query[query_id] for query_id in query_ids]
        by_dataset[prefix] = {
            "label": label,
            "queries": len(query_ids),
            "page_recall@10": mean(item["page_recall@10"] for item in values),
            "page_recall@20": mean(item["page_recall@20"] for item in values),
            "file_recall@3": mean(item["file_recall@3"] for item in values),
        }

    all_values = list(per_query.values())
    overall = {
        "queries": len(all_values),
        "page_recall@10": mean(item["page_recall@10"] for item in all_values),
        "page_recall@20": mean(item["page_recall@20"] for item in all_values),
        "file_recall@3": mean(item["file_recall@3"] for item in all_values),
    }
    return {"overall": overall, "by_dataset": by_dataset, "per_query": per_query}


def load_inputs() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    baseline = load_json(BASELINE_REPORT)
    baseline_rows = {row["query_id"]: normalize_run_row(row) for row in baseline["per_query"]}
    gold = {
        row["query_id"]: {
            "gold_pages": row["gold_pages"],
            "gold_files": row["gold_files"],
        }
        for row in baseline["per_query"]
    }

    ocr_rows = {
        row["query_id"]: normalize_run_row(row)
        for row in load_jsonl(
            BUNDLE
            / "embedded_image_ocr_eng_w48/retrieval_bm25_global_vs_hierarchical_kf3/runs/hierarchical_bm25_kf3_top20.jsonl"
        )
    }
    embedding_rows = {
        row["query_id"]: normalize_run_row(row)
        for row in load_jsonl(
            ROOT / "data/work/benchmark_hierarchical_local/no_ocr_embedding_hier_top20_from_cache.jsonl"
        )
    }
    ocr_embedding_rows = {
        row["query_id"]: normalize_run_row(row)
        for row in load_jsonl(
            ROOT
            / "data/work/benchmark_hierarchical_local/embedding_full_embedded_ocr_openrouter_te3s_clean_eng_w48/hierarchical_top20.jsonl"
        )
    }
    pymupdf_rows = {
        row["query_id"]: normalize_run_row(row)
        for row in load_jsonl(BUNDLE / "pymupdf_ablation/pymupdf_first/per_query.jsonl")
    }

    runs = {
        "baseline_bm25": baseline_rows,
        "embedded_image_ocr_bm25": ocr_rows,
        "embeddings": embedding_rows,
        "embedded_image_ocr_embeddings": ocr_embedding_rows,
        "pymupdf": pymupdf_rows,
    }
    for method, rows in runs.items():
        missing = set(gold) - set(rows)
        extra = set(rows) - set(gold)
        if missing or extra:
            raise ValueError(f"{method}: qid mismatch; missing={sorted(missing)}, extra={sorted(extra)}")

    sources = {
        "baseline_bm25": str(BASELINE_REPORT),
        "embedded_image_ocr_bm25": str(
            BUNDLE
            / "embedded_image_ocr_eng_w48/retrieval_bm25_global_vs_hierarchical_kf3/runs/hierarchical_bm25_kf3_top20.jsonl"
        ),
        "embeddings": str(
            ROOT / "data/work/benchmark_hierarchical_local/no_ocr_embedding_hier_top20_from_cache.jsonl"
        ),
        "embedded_image_ocr_embeddings": str(
            ROOT
            / "data/work/benchmark_hierarchical_local/embedding_full_embedded_ocr_openrouter_te3s_clean_eng_w48/hierarchical_top20.jsonl"
        ),
        "pymupdf": str(BUNDLE / "pymupdf_ablation/pymupdf_first/per_query.jsonl"),
    }
    return {method: (rows, gold) for method, rows in runs.items()}, sources


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta_pct(value: float, baseline: float) -> str:
    delta = (value - baseline) * 100
    if abs(delta) < 0.005:
        return "0.00 pp"
    return f"{delta:+.2f} pp"


def render_markdown(results: dict[str, Any], sources: dict[str, str]) -> str:
    lines = [
        "# Light retrieval solution comparison",
        "",
        "Offline comparison over the cached 220-query benchmark.",
        "",
        "Protocol: macro-average over queries within each dataset; page recall uses the first 10/20 returned pages; file recall uses the selected top 3 files; all methods use the same gold pages and `Kf=3`/top-20 output budget.",
        "",
        "The embeddings-only arm uses the cached dense index built from the original PDF-inspector/Tesseract text. The OCR+embeddings arm uses `openai/text-embedding-3-small` over embedded-image OCR text.",
        "",
    ]

    lines.extend(["## Overall", "", "| Method | Queries | Page R@10 | Page R@20 | File R@3 |", "|---|---:|---:|---:|---:|"])
    for method in METHOD_ORDER:
        row = results[method]["overall"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['queries']} | {pct(row['page_recall@10'])} | {pct(row['page_recall@20'])} | {pct(row['file_recall@3'])} |"
        )

    for prefix, label in DATASET_LABELS.items():
        lines.extend(
            [
                "",
                f"## {label}",
                "",
                "| Method | Page R@10 | Δ vs baseline | Page R@20 | Δ vs baseline | File R@3 | Δ vs baseline |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        baseline = results["baseline_bm25"]["by_dataset"][prefix]
        for method in METHOD_ORDER:
            row = results[method]["by_dataset"][prefix]
            lines.append(
                f"| {METHOD_LABELS[method]} | {pct(row['page_recall@10'])} | {delta_pct(row['page_recall@10'], baseline['page_recall@10'])} | {pct(row['page_recall@20'])} | {delta_pct(row['page_recall@20'], baseline['page_recall@20'])} | {pct(row['file_recall@3'])} | {delta_pct(row['file_recall@3'], baseline['file_recall@3'])} |"
            )
        lines.append(f"\n{label}: {results['baseline_bm25']['by_dataset'][prefix]['queries']} queries.")

    lines.extend(["", "## Sources", ""])
    for method in METHOD_ORDER:
        lines.append(f"- {METHOD_LABELS[method]}: `{sources[method]}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BUNDLE / "solution_comparison_by_dataset",
    )
    args = parser.parse_args()

    loaded, sources = load_inputs()
    results = {}
    for method in METHOD_ORDER:
        rows, gold = loaded[method]
        results[method] = aggregate(rows, gold)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "contract_version": "light-solution-comparison-v1",
        "queries": 220,
        "protocol": {
            "page_metrics": ["page_recall@10", "page_recall@20"],
            "file_metric": "file_recall@3",
            "macro_average": "per query within dataset",
            "file_k": 3,
            "saved_page_k": 20,
        },
        "methods": {method: {"label": METHOD_LABELS[method], "source": sources[method], **results[method]} for method in METHOD_ORDER},
    }
    (output_dir / "report.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_markdown(results, sources), encoding="utf-8")

    print(render_markdown(results, sources))


if __name__ == "__main__":
    main()
