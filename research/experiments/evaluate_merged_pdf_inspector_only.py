"""Evaluate PDF-inspector native text only on the merged 220-query benchmark.

Pages whose preparation source is Tesseract or empty are assigned an empty
text field. Embedded-image OCR is also excluded because it is a separate
preparation stage.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    build_bm25_arm,
    load_inputs,
    metric_rows,
    read_jsonl,
    write_jsonl,
)
from research.experiments.evaluate_merged_pymupdf_native_only import (  # noqa: E402
    DEFAULT_OUTPUT as PYMUPDF_OUTPUT,
    load_current_rows,
)


DEFAULT_OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/pdf_inspector_only"
MERGED_OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms"


def main() -> int:
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    rows = load_current_rows()
    page_ids = [str(row["page_id"]) for row in rows]
    page_text = {
        str(row["page_id"]): (
            str(row.get("text") or "")
            if str(row.get("text_source") or "") == "pdf_inspector"
            else ""
        )
        for row in rows
    }
    status_rows = [
        {
            "page_id": str(row["page_id"]),
            "source": str(row.get("source") or ""),
            "relative_path": str(row.get("relative_path") or ""),
            "page_index": int(row["page_index"]),
            "parser_text_source": str(row.get("text_source") or ""),
            "included": str(row.get("text_source") or "") == "pdf_inspector",
            "chars": len(page_text[str(row["page_id"])]),
            "words": len(page_text[str(row["page_id"])] .split()),
        }
        for row in rows
    ]
    write_jsonl(output / "page_native_status.jsonl", status_rows)

    queries, qrels, languages = load_inputs()
    started = time.perf_counter()
    run_rows = build_bm25_arm(page_text, queries, languages)
    retrieval_seconds = time.perf_counter() - started
    write_jsonl(output / "run_top20.jsonl", run_rows)

    baseline_rows = read_jsonl(MERGED_OUTPUT / "runs/baseline_bm25.jsonl")
    ocr_rows = read_jsonl(MERGED_OUTPUT / "runs/embedded_image_ocr_bm25.jsonl")
    report = {
        "contract_version": "merged-pdf-inspector-only-v1",
        "protocol": {
            "queries": 220,
            "files": 103,
            "pages": 5091,
            "file_k": 3,
            "candidate_depth": 200,
            "saved_depth": 20,
            "qrels_used_for_ranking": False,
            "included_text_source": "pdf_inspector only",
            "excluded_text_sources": ["tesseract", "empty", "embedded_image_ocr"],
        },
        "counts": {
            "pages": len(rows),
            "included_pages": sum(bool(row["included"]) for row in status_rows),
            "excluded_pages": sum(not bool(row["included"]) for row in status_rows),
            "included_by_source": dict(
                Counter(row["source"] for row in status_rows if row["included"])
            ),
            "excluded_by_source": dict(
                Counter(row["source"] for row in status_rows if not row["included"])
            ),
            "parser_source_counts": dict(Counter(row["parser_text_source"] for row in status_rows)),
            "included_chars": sum(len(value) for value in page_text.values()),
        },
        "timing_seconds": {"retrieval": retrieval_seconds},
        "metrics": {
            "baseline_bm25_with_selective_tesseract": {
                "@10": metric_rows(baseline_rows, qrels, 10),
                "@20": metric_rows(baseline_rows, qrels, 20),
            },
            "embedded_image_ocr_bm25": {
                "@10": metric_rows(ocr_rows, qrels, 10),
                "@20": metric_rows(ocr_rows, qrels, 20),
            },
            "pdf_inspector_only": {
                "@10": metric_rows(run_rows, qrels, 10),
                "@20": metric_rows(run_rows, qrels, 20),
            },
        },
        "artifacts": {
            "page_status": str((output / "page_native_status.jsonl").resolve()),
            "run_top20": str((output / "run_top20.jsonl").resolve()),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Merged PDF-inspector-only ablation",
        "",
        "| Arm | Recall@10 | Recall@20 | nDCG@10 | File recall@3 | Candidate recall@200 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "baseline_bm25_with_selective_tesseract": "Baseline BM25 + selective Tesseract",
        "embedded_image_ocr_bm25": "Embedded-image OCR + BM25",
        "pdf_inspector_only": "PDF-inspector only",
    }
    for key, label in labels.items():
        m10 = report["metrics"][key]["@10"]
        m20 = report["metrics"][key]["@20"]
        lines.append(
            f"| {label} | {m10['page_recall@10']:.2%} | "
            f"{m20['page_recall@20']:.2%} | {m10['ndcg@10']:.2f} | "
            f"{m10['file_recall@3']:.2%} | {m10['candidate_pool_recall@200']:.2%} |"
        )
    lines += [
        "",
        f"Included pages: {report['counts']['included_pages']}/"
        f"{report['counts']['pages']}; excluded pages: "
        f"{report['counts']['excluded_pages']}.",
        f"BM25 retrieval: {retrieval_seconds:.3f}s.",
        "",
        "This arm excludes full-page Tesseract and embedded-image OCR. Pages "
        "without PDF-inspector text remain empty.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
