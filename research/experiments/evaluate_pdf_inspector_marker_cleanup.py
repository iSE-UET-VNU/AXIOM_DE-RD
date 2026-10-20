"""Evaluate removing PDF-inspector image markers from the BM25 text stream."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

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


OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison"
BASELINE_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
IMAGE_MARKER = re.compile(r"\[Image:\s*[^\]]+\]", re.IGNORECASE)


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
    original: dict[str, str] = {}
    cleaned: dict[str, str] = {}
    removed_chars = 0
    removed_markers = 0
    for row in rows:
        page_id = str(row["page_id"])
        text = str(row.get("text") or "")
        source = str(row.get("text_source") or "")
        original[page_id] = text
        if source == "pdf_inspector":
            removed_markers += len(IMAGE_MARKER.findall(text))
            cleaned_text = IMAGE_MARKER.sub(" ", text)
            removed_chars += len(text) - len(cleaned_text)
            cleaned[page_id] = cleaned_text
        else:
            cleaned[page_id] = text

    queries, qrels, languages = load_inputs()
    started = time.perf_counter()
    cleaned_rows = build_bm25_arm(cleaned, queries, languages)
    retrieval_seconds = time.perf_counter() - started
    write_jsonl(OUTPUT / "run_pdi_tesseract_marker_cleaned_top20.jsonl", cleaned_rows)
    baseline_rows = read_jsonl(BASELINE_RUN)
    report = {
        "protocol": {
            "pages": len(rows),
            "queries": len(queries),
            "file_k": 3,
            "candidate_depth": 200,
            "cleaned_field": "PDF-inspector text only; Tesseract text unchanged",
            "qrels_used_for_ranking": False,
        },
        "cleanup": {"markers_removed": removed_markers, "characters_removed": removed_chars},
        "retrieval_seconds": retrieval_seconds,
        "metrics": {
            "baseline_pdi_tesseract": by_dataset(baseline_rows, qrels),
            "pdi_tesseract_markers_removed": by_dataset(cleaned_rows, qrels),
        },
        "artifacts": {"run": str((OUTPUT / "run_pdi_tesseract_marker_cleaned_top20.jsonl").resolve())},
    }
    (OUTPUT / "marker_cleanup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# PDF-inspector image-marker cleanup",
        "",
        f"Removed {removed_markers:,} markers and {removed_chars:,} characters from PDF-inspector text. Tesseract text was unchanged.",
        "",
        "| Dataset | Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (("baseline_pdi_tesseract", "Original"), ("pdi_tesseract_markers_removed", "Markers removed")):
            m10 = report["metrics"][key][dataset]["@10"]
            m20 = report["metrics"][key][dataset]["@20"]
            lines.append(f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    lines += ["", f"BM25 retrieval time: {retrieval_seconds:.3f}s.", "", "This is an offline text-index ablation; it does not remove image records or OCR text."]
    (OUTPUT / "marker_cleanup_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
