"""Page-by-page audit: cached KDL text versus OCR-augmented light text.

This uses the exact text sources supplied to the current merged hybrid/Cohere
arm, not a re-render or a new OCR/KDL run.  It writes one aligned record for
every page and aggregate/source/gold-page statistics for diagnosis.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OLD_LIGHT = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl"
OHR_LIGHT = ROOT / "data/work/ohr_replacement_20260920/embedded_ocr_filtered/size_filter_only/pages.jsonl"
KDL = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/kdl/kdl_pages.jsonl"
OLD_QRELS = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
OHR_QRELS = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/analysis/kdl_vs_light_parsing"
TOKEN = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokens(text: str) -> set[str]:
    value = unicodedata.normalize("NFKD", text).casefold()
    return {token for token in TOKEN.findall(value) if len(token) > 1 and not token.isdigit()}


def source(page_id: str) -> str:
    return page_id.split("::", 1)[0]


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def main() -> None:
    light_all = {str(row["page_id"]): row for row in jsonl(OLD_LIGHT) + jsonl(OHR_LIGHT)}
    kdl = {str(row["page_id"]): row for row in jsonl(KDL)}
    light = {page_id: light_all[page_id] for page_id in kdl if page_id in light_all}
    if set(light) != set(kdl):
        raise RuntimeError(f"merged page mismatch: light-only={len(set(light)-set(kdl))}, kdl-only={len(set(kdl)-set(light))}")
    gold_pages: set[str] = set()
    for row in jsonl(OLD_QRELS):
        if str(row["query_id"]).startswith(("vidore_physics::", "vidore_industrial::")):
            gold_pages.add(str(row["page_id"]))
    for row in jsonl(OHR_QRELS):
        gold_pages.add(f"{row['doc_id']}#page={int(row['page_index'])}")

    aligned = []
    for page_id in sorted(light):
        lrow, krow = light[page_id], kdl[page_id]
        ltext, ktext = str(lrow.get("text") or ""), str(krow.get("text") or "")
        lt, kt = tokens(ltext), tokens(ktext)
        overlap = lt & kt
        aligned.append({
            "page_id": page_id, "source": source(page_id), "gold_page": page_id in gold_pages,
            "light_chars": len(ltext), "kdl_chars": len(ktext), "light_tokens": len(lt), "kdl_tokens": len(kt),
            "light_token_coverage_in_kdl": len(overlap) / len(lt) if lt else 1.0,
            "kdl_token_coverage_in_light": len(overlap) / len(kt) if kt else 1.0,
            "light_text_source": lrow.get("text_source"), "light_ocr_applied": bool(lrow.get("ocr_applied")),
            # The supplied rows expose only a pipeline/policy flag, not a
            # per-page nonempty embedded-OCR payload.  Treat this as artifact
            # provenance, not evidence that OCR added tokens on that page.
            "light_embedded_ocr_artifact": bool(lrow.get("embedded_image_ocr_inserted")) or "embedded_ocr_policy" in lrow,
            "kdl_status": krow.get("kdl_status"), "kdl_markdown_images": ktext.count("!["),
            "kdl_image_markers": ktext.count("[Image:"), "light_image_markers": ltext.count("[Image:"),
            "kdl_only_token_sample": sorted(kt - lt)[:30], "light_only_token_sample": sorted(lt - kt)[:30],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "page_alignment.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in aligned), encoding="utf-8")

    report: dict[str, Any] = {"pages": len(aligned), "gold_pages": len(gold_pages), "sources": {}}
    for group in ("all", "vidore_physics", "vidore_industrial", "ohrbench"):
        subset = aligned if group == "all" else [row for row in aligned if row["source"] == group]
        gold_subset = [row for row in subset if row["gold_page"]]
        def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
            return {
                "pages": len(rows), "mean_light_chars": mean([row["light_chars"] for row in rows]),
                "mean_kdl_chars": mean([row["kdl_chars"] for row in rows]),
                "light_empty": sum(row["light_chars"] == 0 for row in rows), "kdl_empty": sum(row["kdl_chars"] == 0 for row in rows),
                "mean_light_coverage_in_kdl": mean([row["light_token_coverage_in_kdl"] for row in rows]),
                "mean_kdl_coverage_in_light": mean([row["kdl_token_coverage_in_light"] for row in rows]),
                "light_ocr_pages": sum(row["light_ocr_applied"] for row in rows), "embedded_ocr_artifact_rows": sum(row["light_embedded_ocr_artifact"] for row in rows),
                "kdl_markdown_image_pages": sum(row["kdl_markdown_images"] > 0 for row in rows),
            }
        report["sources"][group] = {"all_pages": describe(subset), "gold_pages": describe(gold_subset)}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# KDL versus OCR-augmented light page text", "", "One row per aligned page is in `page_alignment.jsonl`. Token coverage is lexical overlap after Unicode normalization; it is not semantic equivalence. The embedded-OCR field records artifact provenance only: supplied rows do not expose a per-page nonempty embedded-OCR payload.", "", "| Source | Pages | Light chars | KDL chars | Light tokens covered by KDL | KDL tokens covered by light | Light OCR pages | Embedded OCR artifact rows | KDL image pages |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group, values in report["sources"].items():
        item = values["all_pages"]
        lines.append(f"| {group} | {item['pages']} | {item['mean_light_chars']:.0f} | {item['mean_kdl_chars']:.0f} | {100*item['mean_light_coverage_in_kdl']:.1f}% | {100*item['mean_kdl_coverage_in_light']:.1f}% | {item['light_ocr_pages']} | {item['embedded_ocr_artifact_rows']} | {item['kdl_markdown_image_pages']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
