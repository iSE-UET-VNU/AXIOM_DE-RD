"""Augment PyMuPDF native text with image OCR on OCR-flagged pages.

The base corpus is the established PyMuPDF mixed arm: native PyMuPDF text when
available, otherwise the canonical PDF-inspector/Tesseract preparation text.
Only pages satisfying both ``needs_ocr=true`` and non-empty PyMuPDF native text
are augmented.  Their PDF-inspector image markers define the image sequence;
the already-computed Tesseract image OCR is appended after the native text in
that sequence.  No qrel is used for selection or ranking.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import pymupdf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_pdi_native_preserve_ocr import by_dataset  # noqa: E402
from research.experiments.evaluate_merged_pymupdf_native_only import load_current_rows, page_path  # noqa: E402
from research.experiments.ocr_benchmark_embedded_images import norm_label  # noqa: E402
from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    build_bm25_arm,
    load_inputs,
    load_page_corpora,
    read_jsonl,
    write_jsonl,
)


OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison/pymupdf_plus_image_ocr_on_native_ocr_pages"
BASELINE_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
PYMUPDF_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/pymupdf.jsonl"
PDI_MARKER_PAGES = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison/pdi_targeted_image_ocr/pages_before_image_ocr.jsonl"
IMAGE_OCR = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison/pdi_targeted_image_ocr/ocr_results.jsonl"
IMAGE_MARKER = re.compile(r"\[Image:\s*([^\]]+?)\]")


def load_native(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[page_path(row)].append(row)
    native: dict[str, str] = {}
    for path, page_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with pymupdf.open(path) as document:
            for row in page_rows:
                native[str(row["page_id"])] = (document[int(row["page_index"])].get_text("text") or "").strip()
    return native


def ordered_image_ocr(marker_text: str, results: list[Mapping[str, Any]], page_id: str) -> list[str]:
    """Return successful image OCR in the reading order of PDI markers."""
    lookup: dict[tuple[str, int], deque[str]] = defaultdict(deque)
    for row in results:
        if str(row.get("page_id")) != page_id or row.get("status") != "ok":
            continue
        text = str(row.get("text") or "").strip()
        if text:
            lookup[(str(row.get("image_label") or ""), int(row.get("occurrence_on_page") or 0))].append(text)
    seen: Counter[str] = Counter()
    ordered: list[str] = []
    for match in IMAGE_MARKER.finditer(marker_text):
        label = norm_label(match.group(1).strip())
        occurrence = seen[label]
        seen[label] += 1
        values = lookup.get((label, occurrence))
        if values:
            ordered.append(values.popleft())
    return ordered


def main() -> int:
    if not PDI_MARKER_PAGES.is_file() or not IMAGE_OCR.is_file():
        raise FileNotFoundError("Run evaluate_pdi_targeted_image_ocr.py before this augmentation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_current_rows()
    native = load_native(rows)
    pdi_marker_text = {str(row["page_id"]): str(row.get("text") or "") for row in read_jsonl(PDI_MARKER_PAGES)}
    ocr_results = read_jsonl(IMAGE_OCR)
    _, corpora = load_page_corpora()
    base = {page_id: str(row.get("text") or "") for page_id, row in corpora["pymupdf"].items()}

    augmented = dict(base)
    target_rows: list[dict[str, Any]] = []
    inserted_images = 0
    inserted_words = 0
    for row in rows:
        page_id = str(row["page_id"])
        native_text = native[page_id]
        if not bool(row.get("needs_ocr")) or not native_text:
            continue
        image_texts = ordered_image_ocr(pdi_marker_text.get(page_id, ""), ocr_results, page_id)
        if image_texts:
            appended = "\n\n".join(image_texts)
            augmented[page_id] = f"{native_text}\n\n[OCR from embedded images]\n{appended}"
            inserted_images += len(image_texts)
            inserted_words += sum(len(text.split()) for text in image_texts)
        target_rows.append({
            "page_id": page_id,
            "doc_id": str(row.get("doc_id") or ""),
            "page_index": int(row["page_index"]),
            "native_chars": len(native_text),
            "image_ocr_segments": len(image_texts),
            "image_ocr_words": sum(len(text.split()) for text in image_texts),
        })

    queries, qrels, languages = load_inputs()
    started = time.perf_counter()
    augmented_run = build_bm25_arm(augmented, queries, languages)
    retrieval_seconds = time.perf_counter() - started
    write_jsonl(OUTPUT / "target_pages.jsonl", target_rows)
    write_jsonl(OUTPUT / "run_top20.jsonl", augmented_run)
    write_jsonl(OUTPUT / "pages_changed.jsonl", [
        {"page_id": row["page_id"], "text": augmented[str(row["page_id"])]}
        for row in target_rows if row["image_ocr_segments"]
    ])

    baseline = read_jsonl(BASELINE_RUN)
    pymupdf = read_jsonl(PYMUPDF_RUN)
    report = {
        "protocol": {
            "base": "PyMuPDF mixed fallback: PyMuPDF native text when nonempty, canonical PDI/Tesseract text otherwise",
            "augmentation_target": "needs_ocr=true AND PyMuPDF native text nonempty",
            "augmentation": "append successful Tesseract OCR segments from PDI-marked embedded images in marker reading order",
            "qrels_used_for_ranking": False,
        },
        "counts": {
            "pages": len(rows),
            "target_pages": len(target_rows),
            "target_pages_with_image_ocr": sum(bool(row["image_ocr_segments"]) for row in target_rows),
            "inserted_images": inserted_images,
            "inserted_words": inserted_words,
        },
        "timing_seconds": {"pymupdf_native_extraction": None, "retrieval": retrieval_seconds, "image_ocr": 0.0},
        "cache_note": "Tesseract image OCR was reused from pdi_targeted_image_ocr/ocr_results.jsonl; no image was OCRed twice.",
        "metrics": {
            "baseline_pdi_full_page_tesseract": by_dataset(baseline, qrels),
            "pymupdf_mixed": by_dataset(pymupdf, qrels),
            "pymupdf_mixed_plus_image_ocr": by_dataset(augmented_run, qrels),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PyMuPDF native text + image OCR on OCR-flagged native pages", "",
        f"Target pages: {report['counts']['target_pages']}; pages with appended image OCR: {report['counts']['target_pages_with_image_ocr']}; inserted images: {inserted_images}; inserted words: {inserted_words}.",
        "", "| Dataset | Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (("baseline_pdi_full_page_tesseract", "PDI + full-page Tesseract"), ("pymupdf_mixed", "PyMuPDF mixed"), ("pymupdf_mixed_plus_image_ocr", "PyMuPDF mixed + image OCR")):
            m10 = report["metrics"][key][dataset]["@10"]
            m20 = report["metrics"][key][dataset]["@20"]
            lines.append(f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    lines.append(f"\nRetrieval time: {retrieval_seconds:.2f}s. Image OCR is cached from the preceding targeted-image run.")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
