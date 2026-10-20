"""Compare PyMuPDF and Tesseract as fallback for PDF-inspector pages.

The primary text remains cached PDF-inspector text.  Only pages whose cached
source is not ``pdf_inspector`` use either native PyMuPDF or cached Tesseract
text.  This isolates the fallback decision from the extractor choice on the
already successful pages.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata
from typing import Any, Mapping, Sequence

import pymupdf

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
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower()))


def page_path(row: Mapping[str, Any]) -> Path:
    path = ROOT / "data/raw/0.1. BENCHMARK" / str(row.get("relative_path") or "")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def extract_native(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    grouped: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[page_path(row)].append(row)
    native: dict[str, str] = {}
    for path, page_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with pymupdf.open(path) as document:
            for row in page_rows:
                native[str(row["page_id"])] = (
                    document[int(row["page_index"])].get_text("text") or ""
                ).strip()
    return native


def summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": float(statistics.fmean(values)), "median": float(statistics.median(values))}


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
    native = extract_native(rows)
    pdi_text: dict[str, str] = {}
    tesseract_text: dict[str, str] = {}
    pdi_native_fallback: dict[str, str] = {}
    for row in rows:
        page_id = str(row["page_id"])
        source = str(row.get("text_source") or "")
        cached = str(row.get("text") or "")
        pdi = cached if source == "pdf_inspector" else ""
        tess = cached if source == "tesseract" else ""
        pdi_text[page_id] = pdi
        tesseract_text[page_id] = tess
        pdi_native_fallback[page_id] = pdi if pdi.strip() else native.get(page_id, "")

    queries, qrels, languages = load_inputs()
    started = time.perf_counter()
    fallback_run = build_bm25_arm(pdi_native_fallback, queries, languages)
    retrieval_seconds = time.perf_counter() - started
    write_jsonl(OUTPUT / "run_pdi_pymupdf_fallback_top20.jsonl", fallback_run)

    fallback_pages = [row for row in rows if str(row.get("text_source") or "") != "pdf_inspector"]
    both = []
    for row in fallback_pages:
        page_id = str(row["page_id"])
        tess = tesseract_text[page_id]
        py = native.get(page_id, "")
        if tess.strip() and py.strip():
            t = token_set(tess)
            p = token_set(py)
            both.append(
                {
                    "jaccard": len(t & p) / len(t | p) if t | p else 1.0,
                    "tess_in_pymupdf": len(t & p) / len(t) if t else 0.0,
                    "pymupdf_in_tess": len(t & p) / len(p) if p else 0.0,
                    "tess_chars": len(tess),
                    "pymupdf_chars": len(py),
                    "tess_words": len(t),
                    "pymupdf_words": len(p),
                }
            )
    fallback_stats = {
        "pages": len(fallback_pages),
        "tesseract_nonempty": sum(bool(tesseract_text[str(row["page_id"])].strip()) for row in fallback_pages),
        "pymupdf_nonempty": sum(bool(native.get(str(row["page_id"]), "").strip()) for row in fallback_pages),
        "both_nonempty": len(both),
        "tesseract_only": sum(bool(tesseract_text[str(row["page_id"])].strip()) and not bool(native.get(str(row["page_id"]), "").strip()) for row in fallback_pages),
        "pymupdf_only": sum(bool(native.get(str(row["page_id"]), "").strip()) and not bool(tesseract_text[str(row["page_id"])].strip()) for row in fallback_pages),
        "both_empty": sum(not bool(tesseract_text[str(row["page_id"])].strip()) and not bool(native.get(str(row["page_id"]), "").strip()) for row in fallback_pages),
        "both_nonempty_overlap": {
            "token_jaccard": summary([float(item["jaccard"]) for item in both]),
            "tesseract_token_coverage_in_pymupdf": summary([float(item["tess_in_pymupdf"]) for item in both]),
            "pymupdf_token_coverage_in_tesseract": summary([float(item["pymupdf_in_tess"]) for item in both]),
            "tesseract_chars": summary([float(item["tess_chars"]) for item in both]),
            "pymupdf_chars": summary([float(item["pymupdf_chars"]) for item in both]),
            "tesseract_unique_words": summary([float(item["tess_words"]) for item in both]),
            "pymupdf_unique_words": summary([float(item["pymupdf_words"]) for item in both]),
        },
    }

    baseline_rows = read_jsonl(BASELINE_RUN)
    report = {
        "protocol": {
            "pages": len(rows),
            "queries": len(queries),
            "file_k": 3,
            "candidate_depth": 200,
            "fallback_policy": "PDF-inspector text first; native PyMuPDF only on non-pdf_inspector pages",
            "qrels_used_for_ranking": False,
        },
        "fallback_page_comparison": fallback_stats,
        "retrieval_seconds": retrieval_seconds,
        "metrics": {
            "baseline_pdi_plus_tesseract": by_dataset(baseline_rows, qrels),
            "pdi_plus_pymupdf_fallback": by_dataset(fallback_run, qrels),
        },
        "artifacts": {"run": str((OUTPUT / "run_pdi_pymupdf_fallback_top20.jsonl").resolve())},
    }
    (OUTPUT / "fallback_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PDF-inspector fallback: PyMuPDF versus Tesseract",
        "",
        "| Fallback arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (("baseline_pdi_plus_tesseract", "PDI + Tesseract"), ("pdi_plus_pymupdf_fallback", "PDI + PyMuPDF fallback")):
            m10 = report["metrics"][key][dataset]["@10"]
            m20 = report["metrics"][key][dataset]["@20"]
            lines.append(f"| {dataset} / {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    lines += [
        "",
        "## Fallback pages",
        "",
        f"- Non-PDF-inspector pages: {fallback_stats['pages']}; Tesseract non-empty: {fallback_stats['tesseract_nonempty']}; PyMuPDF native non-empty: {fallback_stats['pymupdf_nonempty']}.",
        f"- Both non-empty: {fallback_stats['both_nonempty']}; Tesseract-only: {fallback_stats['tesseract_only']}; PyMuPDF-only: {fallback_stats['pymupdf_only']}; both empty: {fallback_stats['both_empty']}.",
        f"- On the {fallback_stats['both_nonempty']} pages where both have text, median token Jaccard is {fallback_stats['both_nonempty_overlap']['token_jaccard']['median']:.3f}.",
        f"- Median Tesseract token coverage in PyMuPDF is {fallback_stats['both_nonempty_overlap']['tesseract_token_coverage_in_pymupdf']['median']:.3f}; inverse is {fallback_stats['both_nonempty_overlap']['pymupdf_token_coverage_in_tesseract']['median']:.3f}.",
        "",
        "The retrieval arm keeps PDF-inspector text unchanged and substitutes only the fallback pages, so this is the direct replacement test for Tesseract.",
    ]
    (OUTPUT / "fallback_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
