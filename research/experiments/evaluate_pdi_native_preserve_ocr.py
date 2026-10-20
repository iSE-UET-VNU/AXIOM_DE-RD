"""Evaluate preserving PDF-inspector native text before selective OCR.

Arm A keeps PDF-inspector's full-page native extraction on every page,
including pages flagged ``needs_ocr``.  Arm B adds cached full-page Tesseract
text only on pages with ``needs_ocr=true``.  The existing baseline is reported
for reference.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import pymupdf
import pdf_inspector

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_merged_pymupdf_native_only import (  # noqa: E402
    load_current_rows,
)
from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    build_bm25_arm,
    load_inputs,
    metric_rows,
    read_jsonl,
    write_jsonl,
)


OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison/pdi_native_preserve_ocr"
BASELINE_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
IMAGE_MARKER = re.compile(r"\[Image:\s*[^\]]+\]", re.IGNORECASE)


def page_path(row: Mapping[str, Any]) -> Path:
    path = ROOT / "data/raw/0.1. BENCHMARK" / str(row.get("relative_path") or "")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_pdi_native(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[page_path(row)].append(row)
    native: dict[str, str] = {}
    flagged: dict[str, Any] = {}
    started = time.perf_counter()
    for path, page_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with pymupdf.open(path) as document:
            requests = []
            for row in page_rows:
                index = int(row["page_index"])
                page = document[index]
                requests.append((index, [[0.0, 0.0, float(page.rect.width), float(page.rect.height)]]))
            results = pdf_inspector.extract_text_in_regions(str(path), requests)
            if len(results) != len(page_rows):
                raise RuntimeError(f"pdf-inspector returned {len(results)} pages for {len(page_rows)} requests: {path}")
            for row, result in zip(page_rows, results, strict=True):
                page_id = str(row["page_id"])
                region = result.regions[0] if result.regions else None
                native[page_id] = str(region.text or "").strip() if region else ""
                flagged[page_id] = {
                    "pdi_region_needs_ocr": bool(region.needs_ocr) if region else None,
                    "pdi_region_ocr_reason": getattr(region, "ocr_reason", None) if region else None,
                    "cached_needs_ocr": bool(row.get("needs_ocr")),
                    "cached_text_source": str(row.get("text_source") or ""),
                    "native_chars": len(native[page_id]),
                }
    return native, {"seconds": time.perf_counter() - started, "pages": len(native), "flags": flagged}


def merge_native_and_ocr(native: str, ocr: str) -> tuple[str, str]:
    """Merge full-page OCR without discarding native text.

    Marker-only native pages replace the image marker with OCR.  If native
    text is substantive, OCR is appended as an auxiliary segment; exact
    block-level ordering requires OCR word boxes and is a later arm.
    """
    if not ocr.strip():
        return native, "native_only"
    native_core = IMAGE_MARKER.sub(" ", native)
    if len(native_core.strip()) <= 32 or len(native_core.split()) <= 8:
        merged = IMAGE_MARKER.sub(lambda _match: ocr.strip(), native, count=1) if IMAGE_MARKER.search(native) else ocr.strip()
        return merged, "replace_marker_with_ocr"
    return f"{native.rstrip()}\n{ocr.strip()}", "native_plus_ocr_appended"


def by_dataset(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]]) -> dict[str, Any]:
    prefixes = {"physics": "vidore_physics::", "industrial": "vidore_industrial::", "ohr": "ohrbench::", "all": ""}
    result: dict[str, Any] = {}
    for dataset, prefix in prefixes.items():
        subset = [row for row in rows if not prefix or str(row["query_id"]).startswith(prefix)]
        subqrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
        result[dataset] = {"@10": metric_rows(subset, subqrels, 10), "@20": metric_rows(subset, subqrels, 20)}
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_current_rows()
    native, native_meta = extract_pdi_native(rows)
    ocr_text = {str(row["page_id"]): (str(row.get("text") or "") if str(row.get("text_source") or "") == "tesseract" and bool(row.get("needs_ocr")) else "") for row in rows}
    native_plus_ocr: dict[str, str] = {}
    merge_modes = defaultdict(int)
    for row in rows:
        page_id = str(row["page_id"])
        merged, mode = merge_native_and_ocr(native.get(page_id, ""), ocr_text[page_id])
        native_plus_ocr[page_id] = merged
        merge_modes[mode] += 1

    queries, qrels, languages = load_inputs()
    started = time.perf_counter()
    native_run = build_bm25_arm(native, queries, languages)
    native_seconds = time.perf_counter() - started
    started = time.perf_counter()
    native_ocr_run = build_bm25_arm(native_plus_ocr, queries, languages)
    native_ocr_seconds = time.perf_counter() - started
    write_jsonl(OUTPUT / "run_pdi_native_all_top20.jsonl", native_run)
    write_jsonl(OUTPUT / "run_pdi_native_plus_ocr_top20.jsonl", native_ocr_run)

    flagged = [row for row in rows if bool(row.get("needs_ocr"))]
    ocr_used = [row for row in flagged if ocr_text[str(row["page_id"])].strip()]
    baseline_rows = read_jsonl(BASELINE_RUN)
    report = {
        "protocol": {
            "pages": len(rows),
            "queries": len(queries),
            "file_k": 3,
            "candidate_depth": 200,
            "native_extraction": "pdf_inspector.extract_text_in_regions(full_page) for every page",
            "ocr_policy": "cached Tesseract text only when cached needs_ocr=true and text_source=tesseract",
            "qrels_used_for_ranking": False,
        },
        "counts": {
            "pages_flagged_needs_ocr": len(flagged),
            "flagged_pages_with_cached_ocr": len(ocr_used),
            "tesseract_pages_excluded_because_needs_ocr_false": sum(str(row.get("text_source") or "") == "tesseract" and not bool(row.get("needs_ocr")) for row in rows),
            "merge_modes": dict(merge_modes),
        },
        "native_extraction": {key: value for key, value in native_meta.items() if key != "flags"},
        "timing_seconds": {"native_extraction": native_meta["seconds"], "native_retrieval": native_seconds, "native_plus_ocr_retrieval": native_ocr_seconds},
        "metrics": {
            "baseline_pdi_tesseract": by_dataset(baseline_rows, qrels),
            "pdi_native_all_pages": by_dataset(native_run, qrels),
            "pdi_native_plus_selective_ocr": by_dataset(native_ocr_run, qrels),
        },
        "artifacts": {
            "native_run": str((OUTPUT / "run_pdi_native_all_top20.jsonl").resolve()),
            "native_ocr_run": str((OUTPUT / "run_pdi_native_plus_ocr_top20.jsonl").resolve()),
            "native_page_metadata": str((OUTPUT / "native_page_metadata.jsonl").resolve()),
        },
    }
    write_jsonl(OUTPUT / "native_page_metadata.jsonl", [{"page_id": page_id, **meta} for page_id, meta in native_meta["flags"].items()])
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Preserve PDF-inspector native text + selective OCR",
        "",
        f"Flagged pages: {len(flagged)}; flagged pages with cached Tesseract text: {len(ocr_used)}; Tesseract pages excluded because needs_ocr=false: {report['counts']['tesseract_pages_excluded_because_needs_ocr_false']}.",
        f"Merge modes: {dict(merge_modes)}.",
        "",
        "| Dataset | Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (("baseline_pdi_tesseract", "Current baseline"), ("pdi_native_all_pages", "PDI native all pages"), ("pdi_native_plus_selective_ocr", "PDI native + selective OCR")):
            m10 = report["metrics"][key][dataset]["@10"]
            m20 = report["metrics"][key][dataset]["@20"]
            lines.append(f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    lines += [
        "",
        f"Timing: native extraction {native_meta['seconds']:.3f}s; native retrieval {native_seconds:.3f}s; native+OCR retrieval {native_ocr_seconds:.3f}s.",
        "",
        "For substantive native text, cached full-page OCR is appended as an auxiliary segment; marker-only pages replace the first image marker. Exact bbox-level ordering requires a separate OCR TSV/region arm.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
