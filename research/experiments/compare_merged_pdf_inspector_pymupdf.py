"""Compare cached PDF-inspector text with native PyMuPDF text page by page.

This is an explanatory audit for the merged 220-query corpus.  It does not
use qrels to build either text representation.  The retrieval comparison uses
the already generated strict PDF-inspector-only and strict PyMuPDF-native-only
runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    load_inputs,
    metric_rows,
    read_jsonl,
    write_jsonl,
)


DATASET_ROOT = ROOT / "data/raw/0.1. BENCHMARK"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison"
PDI_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/pdf_inspector_only/run_top20.jsonl"
PYMUPDF_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/pymupdf_native_only/run_top20.jsonl"

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
IMAGE_MARKER_RE = re.compile(r"\[Image:[^\]]+\]", re.IGNORECASE)
MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â.|�)")


def tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return set(TOKEN_RE.findall(normalized))


def words(text: str) -> int:
    return len(TOKEN_RE.findall(unicodedata.normalize("NFKC", text)))


def page_path(row: Mapping[str, Any]) -> Path:
    path = DATASET_ROOT / str(row.get("relative_path") or "")
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resolve {row['page_id']}: {path}")
    return path


def text_stats(text: str) -> dict[str, Any]:
    return {
        "chars": len(text),
        "words": words(text),
        "lines": len(text.splitlines()),
        "image_markers": len(IMAGE_MARKER_RE.findall(text)),
        "mojibake_markers": len(MOJIBAKE_RE.findall(text)),
        "control_chars": sum(1 for char in text if ord(char) < 32 and char not in "\n\t\r"),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def summary_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
    }


def compare_page_texts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[Path, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[page_path(row)].append(row)

    native_text: dict[str, str] = {}
    started = time.perf_counter()
    for path, page_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with pymupdf.open(path) as document:
            for row in page_rows:
                native_text[str(row["page_id"])] = (
                    document[int(row["page_index"])].get_text("text") or ""
                ).strip()
    extraction_seconds = time.perf_counter() - started

    page_records: list[dict[str, Any]] = []
    for row in rows:
        page_id = str(row["page_id"])
        pdi_text = str(row.get("text") or "") if str(row.get("text_source") or "") == "pdf_inspector" else ""
        py_text = native_text.get(page_id, "")
        pdi_tokens = tokens(pdi_text)
        py_tokens = tokens(py_text)
        intersection = pdi_tokens & py_tokens
        union = pdi_tokens | py_tokens
        pdi_stats = text_stats(pdi_text)
        py_stats = text_stats(py_text)
        page_records.append(
            {
                "page_id": page_id,
                "source": str(row.get("source") or ""),
                "relative_path": str(row.get("relative_path") or ""),
                "page_index": int(row["page_index"]),
                "pdi_text_source": str(row.get("text_source") or ""),
                "pdi_nonempty": bool(pdi_text.strip()),
                "pymupdf_nonempty": bool(py_text.strip()),
                "pdi_chars": pdi_stats["chars"],
                "pymupdf_chars": py_stats["chars"],
                "pdi_words": pdi_stats["words"],
                "pymupdf_words": py_stats["words"],
                "pdi_lines": pdi_stats["lines"],
                "pymupdf_lines": py_stats["lines"],
                "pdi_image_markers": pdi_stats["image_markers"],
                "pdi_mojibake_markers": pdi_stats["mojibake_markers"],
                "pymupdf_mojibake_markers": py_stats["mojibake_markers"],
                "pdi_control_chars": pdi_stats["control_chars"],
                "pymupdf_control_chars": py_stats["control_chars"],
                "shared_unique_tokens": len(intersection),
                "pdi_unique_tokens": len(pdi_tokens),
                "pymupdf_unique_tokens": len(py_tokens),
                "token_jaccard": len(intersection) / len(union) if union else 1.0,
                "pdi_token_coverage_in_pymupdf": len(intersection) / len(pdi_tokens) if pdi_tokens else 0.0,
                "pymupdf_token_coverage_in_pdi": len(intersection) / len(py_tokens) if py_tokens else 0.0,
            }
        )

    def aggregate(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        both = [row for row in group if row["pdi_nonempty"] and row["pymupdf_nonempty"]]
        pdi_only = [row for row in group if row["pdi_nonempty"] and not row["pymupdf_nonempty"]]
        py_only = [row for row in group if row["pymupdf_nonempty"] and not row["pdi_nonempty"]]
        return {
            "pages": len(group),
            "pdi_nonempty": sum(bool(row["pdi_nonempty"]) for row in group),
            "pymupdf_nonempty": sum(bool(row["pymupdf_nonempty"]) for row in group),
            "both_nonempty": len(both),
            "pdi_only_nonempty": len(pdi_only),
            "pymupdf_only_nonempty": len(py_only),
            "both_empty": len(group) - len(both) - len(pdi_only) - len(py_only),
            "pdi_chars": summary_stats([float(row["pdi_chars"]) for row in group]),
            "pymupdf_chars": summary_stats([float(row["pymupdf_chars"]) for row in group]),
            "pdi_words": summary_stats([float(row["pdi_words"]) for row in group]),
            "pymupdf_words": summary_stats([float(row["pymupdf_words"]) for row in group]),
            "token_jaccard_both": summary_stats([float(row["token_jaccard"]) for row in both]),
            "pdi_token_coverage_in_pymupdf_both": summary_stats(
                [float(row["pdi_token_coverage_in_pymupdf"]) for row in both]
            ),
            "pymupdf_token_coverage_in_pdi_both": summary_stats(
                [float(row["pymupdf_token_coverage_in_pdi"]) for row in both]
            ),
            "pages_with_pdi_image_markers": sum(bool(row["pdi_image_markers"]) for row in group),
            "pdi_image_markers": sum(int(row["pdi_image_markers"]) for row in group),
            "pages_with_pdi_mojibake": sum(bool(row["pdi_mojibake_markers"]) for row in group),
            "pages_with_pymupdf_mojibake": sum(bool(row["pymupdf_mojibake_markers"]) for row in group),
        }

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in page_records:
        groups["all"].append(row)
        groups[str(row["source"])].append(row)
    page_summary = {key: aggregate(value) for key, value in groups.items()}
    return page_records, {"extraction_seconds": extraction_seconds, "groups": page_summary}


def retrieval_metrics() -> dict[str, Any]:
    _, qrels, _ = load_inputs()
    result: dict[str, Any] = {}
    prefixes = {
        "physics": "vidore_physics::",
        "industrial": "vidore_industrial::",
        "ohr": "ohrbench::",
        "all": "",
    }
    for label, run_path in {"pdf_inspector_only": PDI_RUN, "pymupdf_native_only": PYMUPDF_RUN}.items():
        rows = read_jsonl(run_path)
        result[label] = {}
        for dataset, prefix in prefixes.items():
            subset = [row for row in rows if not prefix or str(row["query_id"]).startswith(prefix)]
            subqrels = {str(row["query_id"]): qrels[str(row["query_id"])] for row in subset}
            result[label][dataset] = {
                "@10": metric_rows(subset, subqrels, 10),
                "@20": metric_rows(subset, subqrels, 20),
            }
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_current_rows()
    page_records, page_stats = compare_page_texts(rows)
    write_jsonl(OUTPUT / "per_page.jsonl", page_records)
    report = {
        "protocol": {
            "pages": len(rows),
            "queries": 220,
            "pdi_text_policy": "text only when text_source=pdf_inspector; Tesseract/OCR pages blank",
            "pymupdf_call": "Page.get_text('text')",
            "qrels_used_for_text": False,
        },
        "page_comparison": page_stats,
        "retrieval_metrics": retrieval_metrics(),
        "artifacts": {"per_page": str((OUTPUT / "per_page.jsonl").resolve())},
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_stats = page_stats["groups"]["all"]
    lines = [
        "# PDF-inspector vs PyMuPDF native page audit",
        "",
        "Both extractors were compared on the same 5,091 merged pages. PDF-inspector text is included only when the cached source is `pdf_inspector`; Tesseract/OCR pages are blank in this comparison.",
        "",
        "## Page-level extraction summary",
        "",
        "| Statistic | PDF-inspector | PyMuPDF native |",
        "|---|---:|---:|",
        f"| Non-empty pages | {all_stats['pdi_nonempty']:,} | {all_stats['pymupdf_nonempty']:,} |",
        f"| Mean chars/page | {all_stats['pdi_chars']['mean']:.1f} | {all_stats['pymupdf_chars']['mean']:.1f} |",
        f"| Median chars/page | {all_stats['pdi_chars']['median']:.1f} | {all_stats['pymupdf_chars']['median']:.1f} |",
        f"| Mean words/page | {all_stats['pdi_words']['mean']:.1f} | {all_stats['pymupdf_words']['mean']:.1f} |",
        f"| Pages with PDI image markers | {all_stats['pages_with_pdi_image_markers']:,} | — |",
        f"| PDI pages with mojibake markers | {all_stats['pages_with_pdi_mojibake']:,} | — |",
        f"| PyMuPDF pages with mojibake markers | — | {all_stats['pages_with_pymupdf_mojibake']:,} |",
        "",
        "### Non-empty overlap",
        "",
        f"- Both non-empty: {all_stats['both_nonempty']:,}; PDF-inspector-only: {all_stats['pdi_only_nonempty']:,}; PyMuPDF-only: {all_stats['pymupdf_only_nonempty']:,}; both empty: {all_stats['both_empty']:,}.",
        f"- On pages where both are non-empty, median token Jaccard is {all_stats['token_jaccard_both']['median']:.3f}.",
        f"- Median PDF-inspector token coverage inside PyMuPDF is {all_stats['pdi_token_coverage_in_pymupdf_both']['median']:.3f}; inverse coverage is {all_stats['pymupdf_token_coverage_in_pdi_both']['median']:.3f}.",
        "",
        "## Retrieval comparison by dataset",
        "",
        "| Dataset | Extractor | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for extractor, label in (("pdf_inspector_only", "PDF-inspector"), ("pymupdf_native_only", "PyMuPDF")):
            m10 = report["retrieval_metrics"][extractor][dataset]["@10"]
            m20 = report["retrieval_metrics"][extractor][dataset]["@20"]
            lines.append(
                f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- PDF-inspector preserves layout-oriented markers and structured reading information, but its page text can contain image placeholders, encoding artefacts and less directly searchable text.",
        "- PyMuPDF native extraction is lighter and often exposes a cleaner continuous text stream, but `Page.get_text('text')` is still only the PDF native text layer: it does not OCR scanned pages or understand image/chart/table semantics.",
        "- A higher PyMuPDF score does not prove that every PyMuPDF page is better. The overlap and per-page records show where one extractor is empty, where token coverage diverges, and where noise/reading order changed.",
        "- This audit compares native text only. It intentionally does not include Tesseract or embedded-image OCR.",
        "",
        f"Native PyMuPDF extraction wall time: {page_stats['extraction_seconds']:.3f}s.",
        "",
        "Per-page details are in `per_page.jsonl`; machine-readable results are in `report.json`.",
    ]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
