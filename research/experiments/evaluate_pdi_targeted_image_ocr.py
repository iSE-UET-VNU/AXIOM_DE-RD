"""Evaluate marker-aware embedded-image OCR only on PDF-inspector OCR targets.

Protocol:

* all pages retain PDF-inspector full-page native text;
* a page is an OCR target only if PDF-inspector's cached or fresh region
  decision says ``needs_ocr`` or its document is classified ``scanned``;
* on target pages, PDF-inspector image markers identify image occurrences;
  pypdf extracts the matching PDF image object and Tesseract OCR text is
  inserted immediately after that marker.

PDF-inspector does not expose decoded embedded-image bytes through its public
Python API.  It is still the parser and reading-order authority here; pypdf is
used solely to materialise the PDF image object that the marker names.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_pdi_native_preserve_ocr import (  # noqa: E402
    BASELINE_RUN,
    by_dataset,
    extract_pdi_native,
)
from research.experiments.evaluate_merged_pymupdf_native_only import load_current_rows  # noqa: E402
from research.experiments.ocr_benchmark_embedded_images import (  # noqa: E402
    completed_keys,
    extract_and_schedule,
    jsonl,
    merged_pages,
)
from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    build_bm25_arm,
    load_inputs,
    read_jsonl,
    write_jsonl,
)


DEFAULT_OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison/pdi_targeted_image_ocr"
DATASET_ROOT = ROOT / "data/raw/0.1. BENCHMARK"
DEFAULT_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def target_reason(row: Mapping[str, Any], fresh: Mapping[str, Any]) -> list[str]:
    """Return parser-derived reasons why this page receives image OCR."""
    reasons: list[str] = []
    if bool(row.get("needs_ocr")):
        reasons.append("cached_needs_ocr")
    if bool(fresh.get("pdi_region_needs_ocr")):
        reasons.append("fresh_region_needs_ocr")
    if str(row.get("pdf_type") or "").casefold() == "scanned":
        reasons.append("document_pdf_type_scanned")
    return reasons


def render_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# PDF-inspector native text + targeted embedded-image OCR",
        "",
        "Target pages are selected solely from PDF-inspector OCR/scanned flags. "
        "On each target page, Tesseract text is inserted after the corresponding "
        "PDF-inspector `[Image: ...]` marker; no full-page Tesseract fallback is used.",
        "",
        f"Targets: {report['counts']['target_pages']}/{report['counts']['pages']} pages; "
        f"target markers: {report['counts']['target_markers']}; image OCR records: {report['counts']['ocr_records']}; "
        f"non-empty OCR records: {report['counts']['ocr_nonempty']}; inserted images: {report['counts']['inserted_images']}.",
        "",
        "| Dataset | Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (
            ("baseline_pdi_full_page_tesseract", "Existing PDI + full-page Tesseract"),
            ("pdi_native", "PDI native only"),
            ("pdi_targeted_image_ocr", "PDI + target-page image OCR"),
        ):
            m10 = report["metrics"][key][dataset]["@10"]
            m20 = report["metrics"][key][dataset]["@20"]
            lines.append(
                f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | "
                f"{m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |"
            )
    lines += [
        "",
        f"Timing: PDF-inspector native extraction {report['timing_seconds']['native_extraction']:.2f}s; "
        f"embedded-image OCR wall time {report['timing_seconds']['image_ocr_wall']:.2f}s; "
        f"BM25 retrieval {report['timing_seconds']['retrieval']:.2f}s.",
        "",
        "A marker with no pypdf image object is logged as `missing_image_object`; it remains the native marker and is not silently replaced.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-pending", type=int, default=192)
    parser.add_argument("--language", default="eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--tesseract", type=Path, default=DEFAULT_TESSERACT)
    args = parser.parse_args()
    if not args.tesseract.is_file():
        raise FileNotFoundError(f"Tesseract executable not found: {args.tesseract}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_current_rows()
    native, native_meta = extract_pdi_native(rows)
    native_pages = [{**row, "text": native[str(row["page_id"])]} for row in rows]
    fresh_flags = native_meta["flags"]
    targets: list[dict[str, Any]] = []
    target_reasons: Counter[str] = Counter()
    for page in native_pages:
        reasons = target_reason(page, fresh_flags[str(page["page_id"])])
        if reasons:
            page["image_ocr_target_reasons"] = reasons
            targets.append(page)
            target_reasons.update(reasons)
    write_jsonl(args.output_dir / "pages_before_image_ocr.jsonl", native_pages)
    write_jsonl(args.output_dir / "target_pages.jsonl", targets)
    write_jsonl(
        args.output_dir / "native_page_metadata.jsonl",
        [{"page_id": page_id, **metadata} for page_id, metadata in fresh_flags.items()],
    )

    results_path = args.output_dir / "ocr_results.jsonl"
    done = completed_keys(results_path)
    counters: Counter[str] = Counter()
    started_ocr = time.perf_counter()
    events_path = args.output_dir / "events.jsonl"
    config = {
        "dataset_root": str(DATASET_ROOT.resolve()),
        "pages": len(rows),
        "target_policy": "cached needs_ocr OR fresh PDF-inspector region needs_ocr OR document pdf_type=scanned",
        "target_pages": len(targets),
        "target_reason_counts": dict(target_reasons),
        "tesseract": str(args.tesseract),
        "language": args.language,
        "psm": args.psm,
        "workers": args.workers,
        "max_pending": args.max_pending,
        "merge_policy": "insert OCR immediately after matching PDF-inspector [Image: marker]",
    }
    with events_path.open("a", encoding="utf-8") as events, results_path.open("a", encoding="utf-8") as result_stream:
        events.write(json.dumps({"event": "targeted-image-ocr", "status": "started", "resume_completed": len(done), **config}, ensure_ascii=False) + "\n")
        events.flush()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            extract_and_schedule(
                targets, DATASET_ROOT, args.output_dir, done, executor, set(), result_stream,
                args.tesseract, args.language, args.psm, args.max_pending, counters,
            )
        image_ocr_seconds = time.perf_counter() - started_ocr
        events.write(json.dumps({"event": "targeted-image-ocr", "status": "complete", "seconds": image_ocr_seconds, "counters": dict(counters)}, ensure_ascii=False) + "\n")

    results = list(jsonl(results_path))
    merged_targets, merge_stats = merged_pages(targets, results)
    merged_by_id = {str(row["page_id"]): row for row in merged_targets}
    page_texts = {
        str(page["page_id"]): str(merged_by_id.get(str(page["page_id"]), page).get("text") or "")
        for page in native_pages
    }
    write_jsonl(args.output_dir / "pages_after_image_ocr.jsonl", [merged_by_id.get(str(page["page_id"]), page) for page in native_pages])

    queries, qrels, languages = load_inputs()
    started_retrieval = time.perf_counter()
    run_rows = build_bm25_arm(page_texts, queries, languages)
    retrieval_seconds = time.perf_counter() - started_retrieval
    write_jsonl(args.output_dir / "run_top20.jsonl", run_rows)
    native_run = build_bm25_arm(native, queries, languages)
    write_jsonl(args.output_dir / "run_pdi_native_top20.jsonl", native_run)
    baseline = read_jsonl(BASELINE_RUN)

    report = {
        "protocol": config,
        "counts": {
            "pages": len(rows),
            "target_pages": len(targets),
            "target_markers": sum(str(page.get("text") or "").count("[Image:") for page in targets),
            "target_reason_counts": dict(target_reasons),
            "ocr_records": len(results),
            "ocr_nonempty": sum(row.get("status") == "ok" and bool(str(row.get("text") or "").strip()) for row in results),
            **dict(counters),
            **merge_stats,
        },
        "timing_seconds": {
            "native_extraction": native_meta["seconds"],
            "image_ocr_wall": image_ocr_seconds,
            "retrieval": retrieval_seconds,
        },
        "metrics": {
            "baseline_pdi_full_page_tesseract": by_dataset(baseline, qrels),
            "pdi_native": by_dataset(native_run, qrels),
            "pdi_targeted_image_ocr": by_dataset(run_rows, qrels),
        },
        "artifacts": {
            "target_pages": str((args.output_dir / "target_pages.jsonl").resolve()),
            "ocr_results": str(results_path.resolve()),
            "pages_after_image_ocr": str((args.output_dir / "pages_after_image_ocr.jsonl").resolve()),
            "run": str((args.output_dir / "run_top20.jsonl").resolve()),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md = render_report(report)
    (args.output_dir / "report.md").write_text(report_md, encoding="utf-8")
    print(report_md)
    print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
