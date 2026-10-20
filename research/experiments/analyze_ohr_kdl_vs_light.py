"""Forensic OHR audit: KDL parsing/chunk retrieval versus light preparation."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
KDL_PAGES = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/kdl/kdl_pages.jsonl"
LIGHT_PAGES = ROOT / "data/work/ohr_replacement_20260920/embedded_ocr_filtered/size_filter_only/pages.jsonl"
QRELS = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"
LIGHT_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/hybrid_bm25_embeddings.jsonl"
KDL_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/kdl_global_chunk_retrieval/per_query.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/analysis/ohr_kdl_vs_light"
WORD = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)
NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def terms(text: str) -> set[str]:
    value = unicodedata.normalize("NFKD", text).casefold()
    return {item for item in WORD.findall(value) if len(item) > 1}


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def rank(pages: list[str], page: str) -> int | None:
    try:
        return pages.index(page) + 1
    except ValueError:
        return None


def page_type(light: dict[str, Any], kdl_text: str) -> str:
    labels = []
    if str(light.get("pdf_type")) == "scanned": labels.append("scan")
    if "![" in kdl_text or "[Image:" in kdl_text: labels.append("figure")
    if "```latex" in kdl_text or "$$" in kdl_text: labels.append("formula")
    if kdl_text.count("|") >= 6: labels.append("table")
    return "+".join(labels) if labels else "native_text"


def main() -> None:
    light = {str(row["page_id"]): row for row in rows(LIGHT_PAGES)}
    kdl = {str(row["page_id"]): row for row in rows(KDL_PAGES) if str(row["page_id"]).startswith("ohrbench::")}
    if set(light) != set(kdl):
        raise RuntimeError(f"OHR page alignment mismatch: light={len(light)} kdl={len(kdl)}")
    gold: dict[str, set[str]] = defaultdict(set)
    for row in rows(QRELS):
        gold[str(row["query_id"])].add(f"{row['doc_id']}#page={int(row['page_index'])}")
    light_runs = {str(row["query_id"]): row for row in rows(LIGHT_RUN) if str(row["query_id"]).startswith("ohrbench::")}
    kdl_runs = {str(row["query_id"]): row for row in rows(KDL_RUN) if str(row["query_id"]).startswith("ohrbench::")}
    if set(gold) != set(light_runs) or set(gold) != set(kdl_runs):
        raise RuntimeError("OHR query alignment mismatch")

    aligned = []
    for page_id in sorted(kdl):
        lrow, krow = light[page_id], kdl[page_id]
        ltext, ktext = str(lrow.get("text") or ""), str(krow.get("text") or "")
        lt, kt = terms(ltext), terms(ktext)
        both = lt & kt
        kdl_figure_text = re.findall(r"!\[([^\]]+)\]", ktext, flags=re.S)
        aligned.append({
            "page_id": page_id, "doc_id": str(lrow["doc_id"]), "page_index": int(lrow["page_index"]),
            "source_path": lrow.get("source_path"), "page_type": page_type(lrow, ktext),
            "light_text_source": lrow.get("text_source"), "light_needs_ocr": bool(lrow.get("needs_ocr")),
            "light_ocr_applied": bool(lrow.get("ocr_applied")), "light_ocr_words": int(lrow.get("ocr_word_count") or 0),
            "light_ocr_confidence": float(lrow.get("ocr_mean_confidence") or 0.0), "light_chars": len(ltext), "kdl_chars": len(ktext),
            "light_terms": len(lt), "kdl_terms": len(kt), "light_numbers": len(set(NUMBER.findall(ltext))), "kdl_numbers": len(set(NUMBER.findall(ktext))),
            "light_term_covered_by_kdl": len(both) / len(lt) if lt else 1.0,
            "kdl_term_covered_by_light": len(both) / len(kt) if kt else 1.0,
            "kdl_markdown_figures": len(kdl_figure_text), "kdl_image_markers": ktext.count("[Image:"),
            "kdl_table_pipes": ktext.count("|"), "kdl_latex_blocks": ktext.count("```latex") + ktext.count("$$"),
            "kdl_only_terms": sorted(kt - lt)[:40], "light_only_terms": sorted(lt - kt)[:40],
            "kdl_figure_text_sample": [value.replace("\n", " ")[:180] for value in kdl_figure_text[:3]],
        })
    page_map = {row["page_id"]: row for row in aligned}

    query_rows = []
    for qid in sorted(gold):
        light_row, kdl_row = light_runs[qid], kdl_runs[qid]
        light10 = list(light_row["returned_pages"][:10])
        light20 = list(light_row["returned_pages"][:20])
        selected_files = set(light_row.get("selected_files") or [])
        kdl10 = [item["page_id"] for item in kdl_row["runs"]["kdl_hybrid_dense07"][:10]]
        kdl20 = [item["page_id"] for item in kdl_row["runs"]["kdl_hybrid_dense07"][:20]]
        gold_pages = sorted(gold[qid])
        details = []
        causes = set()
        for page in gold_pages:
            item = page_map[page]
            lr, kr = rank(light20, page), rank(kdl20, page)
            in_selected = page.split("#page=", 1)[0] in selected_files
            if not in_selected: causes.add("file_selection_miss")
            elif lr is None: causes.add("candidate_pool_miss")
            elif lr > 10: causes.add("light_page_ranking_miss")
            if kr is not None and lr is None: causes.add("kdl_only_retrieval")
            if (item["kdl_markdown_figures"] or item["kdl_table_pipes"] or item["kdl_latex_blocks"]) and item["kdl_term_covered_by_light"] < 0.65:
                causes.add("kdl_only_structured_signal")
            if item["light_needs_ocr"] and (item["light_ocr_words"] == 0 or item["light_ocr_confidence"] < 55): causes.add("weak_light_ocr")
            details.append({"page_id": page, "light_rank_top20": lr, "kdl_rank_top20": kr, "in_light_selected_files": in_selected, "page_type": item["page_type"], "kdl_light_term_coverage": item["kdl_term_covered_by_light"], "light_chars": item["light_chars"], "kdl_chars": item["kdl_chars"]})
        query_rows.append({"query_id": qid, "query": light_row["query"], "gold_pages": gold_pages, "light_top10_hit": bool(set(light10) & gold[qid]), "kdl_top10_hit": bool(set(kdl10) & gold[qid]), "light_top10": light10, "kdl_top10": kdl10, "gold_page_details": details, "cause_labels": sorted(causes)})

    counts = Counter(row["page_type"] for row in aligned)
    gold_pages = {page for pages in gold.values() for page in pages}
    def describe(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {"pages": len(subset), "light_chars": mean([row["light_chars"] for row in subset]), "kdl_chars": mean([row["kdl_chars"] for row in subset]), "light_empty": sum(row["light_chars"] == 0 for row in subset), "kdl_empty": sum(row["kdl_chars"] == 0 for row in subset), "mean_kdl_term_coverage_in_light": mean([row["kdl_term_covered_by_light"] for row in subset]), "mean_light_term_coverage_in_kdl": mean([row["light_term_covered_by_kdl"] for row in subset]), "scanned": sum(row["light_needs_ocr"] for row in subset), "low_or_empty_ocr": sum(row["light_needs_ocr"] and (row["light_ocr_words"] == 0 or row["light_ocr_confidence"] < 55) for row in subset), "pages_with_kdl_figures": sum(row["kdl_markdown_figures"] > 0 for row in subset), "pages_with_kdl_tables": sum(row["kdl_table_pipes"] >= 6 for row in subset), "pages_with_kdl_formula": sum(row["kdl_latex_blocks"] > 0 for row in subset)}
    kdl_only = [row for row in query_rows if row["kdl_top10_hit"] and not row["light_top10_hit"]]
    both_miss = [row for row in query_rows if not row["kdl_top10_hit"] and not row["light_top10_hit"]]
    samples = sorted(kdl_only, key=lambda row: (-max((detail["kdl_chars"] - detail["light_chars"] for detail in row["gold_page_details"]), default=0), row["query_id"]))[:8]
    report = {"protocol": {"ohr_queries": len(gold), "ohr_pages": len(aligned), "light": "official language-routed hybrid final top-10", "kdl": "global cached KDL chunk hybrid, dense alpha .70, chunk depth 100, page max-pool", "caveat": "KDL is global retrieval while light has Kf=3 file stage; this is diagnostic, not a parser-only ablation."}, "page_stats": {"all": describe(aligned), "gold": describe([row for row in aligned if row["page_id"] in gold_pages]), "types": dict(counts)}, "retrieval": {"light_top10_hits": sum(row["light_top10_hit"] for row in query_rows), "kdl_top10_hits": sum(row["kdl_top10_hit"] for row in query_rows), "kdl_only_hits": len(kdl_only), "both_miss": len(both_miss), "cause_counts_for_kdl_only": dict(Counter(cause for row in kdl_only for cause in row["cause_labels"]))}, "visual_review_candidates": [{"query_id": row["query_id"], "query": row["query"], "gold_page_details": row["gold_page_details"], "cause_labels": row["cause_labels"]} for row in samples]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "page_alignment.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned), encoding="utf-8")
    (OUT / "per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in query_rows), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# OHR KDL vs light preparation audit", "", "## Page evidence", "", "| Scope | Pages | Light chars | KDL chars | KDL terms covered by light | Light empty | KDL figure pages |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("all", "gold"):
        value = report["page_stats"][name]
        lines.append(f"| {name} | {value['pages']} | {value['light_chars']:.0f} | {value['kdl_chars']:.0f} | {100*value['mean_kdl_term_coverage_in_light']:.1f}% | {value['light_empty']} | {value['pages_with_kdl_figures']} |")
    lines += ["", "## Retrieval", "", f"- Light hybrid top-10 hit: {report['retrieval']['light_top10_hits']}/{len(gold)}.", f"- Global KDL chunk-hybrid top-10 hit: {report['retrieval']['kdl_top10_hits']}/{len(gold)}.", f"- KDL-only hits: {report['retrieval']['kdl_only_hits']}; both miss: {report['retrieval']['both_miss']}.", "", "`visual_review_candidates` in `report.json` lists gold pages for PDF rendering/review."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
