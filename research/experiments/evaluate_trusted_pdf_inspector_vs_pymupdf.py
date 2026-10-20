"""Compare PDF-inspector and PyMuPDF only on trusted PDF-inspector pages.

The eligible page set is exactly the cached pages with non-empty
``text_source=pdf_inspector`` and ``needs_ocr=False``.  All other pages are
kept in the corpus with empty text for both arms, so the candidate universe and
qrel ceiling remain identical.
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
    write_jsonl,
)


OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/parser_comparison/trusted_pdi_pages"
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> set[str]:
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
    result: dict[str, str] = {}
    for path, page_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        with pymupdf.open(path) as document:
            for row in page_rows:
                result[str(row["page_id"])] = (
                    document[int(row["page_index"])].get_text("text") or ""
                ).strip()
    return result


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
    eligible = [
        row
        for row in rows
        if str(row.get("text_source") or "") == "pdf_inspector"
        and not bool(row.get("needs_ocr"))
        and str(row.get("text") or "").strip()
    ]
    eligible_ids = {str(row["page_id"]) for row in eligible}
    native = extract_native(rows)
    pdi_text = {str(row["page_id"]): (str(row.get("text") or "") if str(row["page_id"]) in eligible_ids else "") for row in rows}
    pymupdf_text = {str(row["page_id"]): (native.get(str(row["page_id"]), "") if str(row["page_id"]) in eligible_ids else "") for row in rows}

    page_comparison = []
    for row in eligible:
        page_id = str(row["page_id"])
        pdi = pdi_text[page_id]
        py = pymupdf_text[page_id]
        p = tokens(pdi)
        m = tokens(py)
        inter = p & m
        union = p | m
        page_comparison.append(
            {
                "page_id": page_id,
                "source": str(row.get("source") or ""),
                "page_index": int(row["page_index"]),
                "pdi_chars": len(pdi),
                "pymupdf_chars": len(py),
                "pdi_words": len(p),
                "pymupdf_words": len(m),
                "shared_unique_tokens": len(inter),
                "token_jaccard": len(inter) / len(union) if union else 1.0,
                "pdi_token_coverage_in_pymupdf": len(inter) / len(p) if p else 0.0,
                "pymupdf_token_coverage_in_pdi": len(inter) / len(m) if m else 0.0,
            }
        )
    write_jsonl(OUTPUT / "per_page.jsonl", page_comparison)

    queries, qrels, languages = load_inputs()
    started = time.perf_counter()
    pdi_run = build_bm25_arm(pdi_text, queries, languages)
    pdi_seconds = time.perf_counter() - started
    started = time.perf_counter()
    pymupdf_run = build_bm25_arm(pymupdf_text, queries, languages)
    pymupdf_seconds = time.perf_counter() - started
    write_jsonl(OUTPUT / "run_pdi_top20.jsonl", pdi_run)
    write_jsonl(OUTPUT / "run_pymupdf_top20.jsonl", pymupdf_run)

    def stats(field: str) -> dict[str, float]:
        values = [float(row[field]) for row in page_comparison]
        return {"mean": float(statistics.fmean(values)), "median": float(statistics.median(values))}

    gold_total = sum(len(values) for values in qrels.values())
    gold_eligible = sum(page in eligible_ids for values in qrels.values() for page in values)
    queries_all_gold_eligible = sum(all(page in eligible_ids for page in values) for values in qrels.values())
    report = {
        "protocol": {
            "pages": len(rows),
            "eligible_pages": len(eligible),
            "queries": len(queries),
            "eligible_policy": "text_source=pdf_inspector, needs_ocr=false, non-empty text",
            "non_eligible_pages_blank_in_both_arms": True,
            "qrels_used_for_ranking": False,
        },
        "gold_coverage_of_eligible_set": {
            "gold_page_records": gold_total,
            "gold_pages_in_eligible_set": gold_eligible,
            "gold_page_coverage": gold_eligible / max(gold_total, 1),
            "queries_with_all_gold_pages_eligible": queries_all_gold_eligible,
        },
        "page_text_comparison": {
            "token_jaccard": stats("token_jaccard"),
            "pdi_token_coverage_in_pymupdf": stats("pdi_token_coverage_in_pymupdf"),
            "pymupdf_token_coverage_in_pdi": stats("pymupdf_token_coverage_in_pdi"),
            "pdi_chars": stats("pdi_chars"),
            "pymupdf_chars": stats("pymupdf_chars"),
            "pdi_words": stats("pdi_words"),
            "pymupdf_words": stats("pymupdf_words"),
        },
        "timing_seconds": {"pdi_retrieval": pdi_seconds, "pymupdf_retrieval": pymupdf_seconds},
        "metrics": {"pdf_inspector": by_dataset(pdi_run, qrels), "pymupdf": by_dataset(pymupdf_run, qrels)},
        "artifacts": {
            "per_page": str((OUTPUT / "per_page.jsonl").resolve()),
            "pdi_run": str((OUTPUT / "run_pdi_top20.jsonl").resolve()),
            "pymupdf_run": str((OUTPUT / "run_pymupdf_top20.jsonl").resolve()),
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Trusted PDF-inspector pages: PDF-inspector versus PyMuPDF",
        "",
        f"Eligible pages: {len(eligible):,}/{len(rows):,}; all non-eligible pages are blank in both arms.",
        f"Gold page coverage of the eligible set: {gold_eligible:,}/{gold_total:,} ({gold_eligible / max(gold_total, 1):.2%}); queries with all gold pages eligible: {queries_all_gold_eligible}/{len(qrels)}.",
        "",
        "## Text comparison on the same eligible pages",
        "",
        f"- Mean characters/page: PDF-inspector {stats('pdi_chars')['mean']:.1f}; PyMuPDF {stats('pymupdf_chars')['mean']:.1f}.",
        f"- Median characters/page: PDF-inspector {stats('pdi_chars')['median']:.1f}; PyMuPDF {stats('pymupdf_chars')['median']:.1f}.",
        f"- Median token Jaccard: {stats('token_jaccard')['median']:.3f}.",
        f"- Median PDF-inspector token coverage in PyMuPDF: {stats('pdi_token_coverage_in_pymupdf')['median']:.3f}.",
        f"- Median PyMuPDF token coverage in PDF-inspector: {stats('pymupdf_token_coverage_in_pdi')['median']:.3f}.",
        "",
        "## Controlled retrieval comparison",
        "",
        "| Dataset | Extractor | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (("pdf_inspector", "PDF-inspector"), ("pymupdf", "PyMuPDF")):
            m10 = report["metrics"][key][dataset]["@10"]
            m20 = report["metrics"][key][dataset]["@20"]
            lines.append(f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    lines += ["", f"Retrieval time: PDF-inspector {pdi_seconds:.3f}s; PyMuPDF {pymupdf_seconds:.3f}s.", "", "This is the cleanest two-extractor comparison because both arms see exactly the same eligible page set and all other pages are blank in both."]
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Output: {OUTPUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
