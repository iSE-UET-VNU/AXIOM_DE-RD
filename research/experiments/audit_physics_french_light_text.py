"""Audit French Physics light-preparation text against gold evidence pages.

This is a retrieval-text diagnostic, not a new ranking experiment.  It
compares the cached PDF-inspector/Tesseract text with the embedded-image-OCR
variant and measures native French query-token coverage before and after
accent folding on the gold pages.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import unicodedata
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking_embedding.lexical import analyze  # noqa: E402


BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval"
LANGUAGE_PATH = BUNDLE / "reports/language_routed_bm25/file_language_detection.jsonl"
BASELINE_PATH = BUNDLE / "preparation/baseline_pdf_inspector_tesseract/pages.jsonl"
EMBEDDED_PATH = BUNDLE / "preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl"
QRELS_PATH = BUNDLE / "input/qrels_page_level.jsonl"
MAPPING_PATH = ROOT / "data/work/ohr_replacement_20260920/retrieval/physics_native_french_language_route/query_mapping_audit.jsonl"
OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/physics_french_light_text_audit"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fold_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(text))
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def coverage(query: str, texts: list[str], folded: bool) -> dict[str, Any]:
    query_tokens = set(analyze(fold_accents(query) if folded else query))
    text_tokens = set(
        token
        for text in texts
        for token in analyze(fold_accents(text) if folded else text)
    )
    matched = sorted(query_tokens & text_tokens)
    missing = sorted(query_tokens - text_tokens)
    return {
        "query_unique_tokens": len(query_tokens),
        "matched_tokens": len(matched),
        "missing_tokens": len(missing),
        "coverage": len(matched) / len(query_tokens) if query_tokens else 0.0,
        "matched": matched,
        "missing": missing,
    }


def page_token_coverage(query: str, texts: list[str], folded: bool) -> float:
    query_tokens = set(analyze(fold_accents(query) if folded else query))
    if not query_tokens:
        return 0.0
    values = []
    for text in texts:
        text_tokens = set(analyze(fold_accents(text) if folded else text))
        values.append(len(query_tokens & text_tokens) / len(query_tokens))
    return sum(values) / len(values) if values else 0.0


def summarize_text(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sources = Counter(str(row.get("text_source") or "unknown") for row in rows)
    pdf_types = Counter(str(row.get("pdf_type") or "unknown") for row in rows)
    ocr_rows = [row for row in rows if row.get("ocr_applied")]
    lengths = [len(str(row.get("text") or "")) for row in rows]
    confidence = [float(row["ocr_mean_confidence"]) for row in ocr_rows if row.get("ocr_mean_confidence") is not None]
    return {
        "pages": len(rows),
        "empty_pages": sum(not str(row.get("text") or "").strip() for row in rows),
        "mean_characters": statistics.mean(lengths) if lengths else 0.0,
        "median_characters": statistics.median(lengths) if lengths else 0.0,
        "sources": dict(sources),
        "pdf_types": dict(pdf_types),
        "ocr_pages": len(ocr_rows),
        "ocr_confidence_mean": statistics.mean(confidence) if confidence else None,
        "ocr_confidence_min": min(confidence) if confidence else None,
        "replacement_char_pages": sum("\ufffd" in str(row.get("text") or "") for row in rows),
        "replacement_characters": sum(str(row.get("text") or "").count("\ufffd") for row in rows),
    }


def main() -> int:
    languages = {
        str(row["file_id"])
        for row in read_jsonl(LANGUAGE_PATH)
        if str(row.get("detected_language")) == "fr"
    }
    baseline_all = read_jsonl(BASELINE_PATH)
    embedded_all = read_jsonl(EMBEDDED_PATH)
    baseline = {
        str(row["page_id"]): row
        for row in baseline_all
        if str(row.get("doc_id")) in languages and str(row.get("source")) == "vidore_physics"
    }
    embedded = {
        str(row["page_id"]): row
        for row in embedded_all
        if str(row.get("doc_id")) in languages and str(row.get("source")) == "vidore_physics"
    }
    if set(baseline) != set(embedded) or len(baseline) != 1024:
        raise RuntimeError(f"French Physics page mismatch: baseline={len(baseline)} embedded={len(embedded)}")

    mapping = {str(row["canonical_query_id"]): row for row in read_jsonl(MAPPING_PATH)}
    qrels: dict[str, set[str]] = {}
    for row in read_jsonl(QRELS_PATH):
        qid = str(row.get("query_id"))
        if qid.startswith("vidore_physics::"):
            qrels.setdefault(qid, set()).add(str(row["page_id"]))
    if len(mapping) != 80 or len(qrels) != 80:
        raise RuntimeError(f"Expected 80 Physics queries: mapping={len(mapping)} qrels={len(qrels)}")

    page_deltas = []
    for page_id in sorted(baseline):
        old_text = str(baseline[page_id].get("text") or "")
        new_text = str(embedded[page_id].get("text") or "")
        page_deltas.append(
            {
                "page_id": page_id,
                "baseline_characters": len(old_text),
                "embedded_characters": len(new_text),
                "character_delta": len(new_text) - len(old_text),
                "changed": old_text != new_text,
                "text_source": embedded[page_id].get("text_source"),
                "ocr_applied": bool(embedded[page_id].get("ocr_applied")),
            }
        )

    query_rows = []
    for qid in sorted(qrels):
        query = str(mapping[qid]["french_query"])
        gold_pages = sorted(qrels[qid])
        if not set(gold_pages).issubset(baseline):
            missing = sorted(set(gold_pages) - set(baseline))
            raise RuntimeError(f"Gold pages missing from French corpus for {qid}: {missing}")
        row: dict[str, Any] = {"query_id": qid, "french_query": query, "gold_pages": gold_pages}
        for name, corpus in (("baseline", baseline), ("embedded", embedded)):
            texts = [str(corpus[page]["text"] or "") for page in gold_pages]
            exact = coverage(query, texts, folded=False)
            folded = coverage(query, texts, folded=True)
            row[name] = {
                "gold_page_count": len(texts),
                "exact": exact,
                "accent_folded": folded,
                "mean_page_coverage_exact": page_token_coverage(query, texts, folded=False),
                "mean_page_coverage_folded": page_token_coverage(query, texts, folded=True),
            }
        row["accent_recovered_tokens_baseline"] = sorted(
            set(row["baseline"]["accent_folded"]["matched"])
            - set(row["baseline"]["exact"]["matched"])
        )
        row["accent_recovered_tokens_embedded"] = sorted(
            set(row["embedded"]["accent_folded"]["matched"])
            - set(row["embedded"]["exact"]["matched"])
        )
        query_rows.append(row)

    def mean(path: str) -> float:
        values = []
        for row in query_rows:
            value: Any = row
            for part in path.split("."):
                value = value[part]
            values.append(float(value))
        return sum(values) / len(values) if values else 0.0

    summary = {
        "corpus": {
            "language_files": len(languages),
            "physics_pages": len(baseline),
            "gold_queries": len(query_rows),
        },
        "baseline_text": summarize_text(list(baseline.values())),
        "embedded_text": summarize_text(list(embedded.values())),
        "embedded_delta": {
            "changed_pages": sum(item["changed"] for item in page_deltas),
            "pages_with_positive_delta": sum(item["character_delta"] > 0 for item in page_deltas),
            "mean_character_delta": statistics.mean(item["character_delta"] for item in page_deltas),
            "median_character_delta": statistics.median(item["character_delta"] for item in page_deltas),
            "positive_delta_characters": sum(max(item["character_delta"], 0) for item in page_deltas),
        },
        "query_token_coverage": {
            "baseline_exact": mean("baseline.exact.coverage"),
            "baseline_accent_folded": mean("baseline.accent_folded.coverage"),
            "embedded_exact": mean("embedded.exact.coverage"),
            "embedded_accent_folded": mean("embedded.accent_folded.coverage"),
            "baseline_mean_page_exact": mean("baseline.mean_page_coverage_exact"),
            "embedded_mean_page_exact": mean("embedded.mean_page_coverage_exact"),
        },
        "queries_with_accent_recovery": {
            "baseline": sum(bool(row["accent_recovered_tokens_baseline"]) for row in query_rows),
            "embedded": sum(bool(row["accent_recovered_tokens_embedded"]) for row in query_rows),
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "page_deltas.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in page_deltas) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "query_coverage.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in query_rows) + "\n",
        encoding="utf-8",
    )

    top_recovery = sorted(
        query_rows,
        key=lambda row: len(row["accent_recovered_tokens_embedded"]),
        reverse=True,
    )[:10]
    top_embedded_delta = sorted(page_deltas, key=lambda row: row["character_delta"], reverse=True)[:10]
    lines = [
        "# Physics French light-text audit",
        "",
        "The audit uses the 22 language-detected French Physics files, 1,024 pages, 80 native French queries, and the existing Physics qrels only for selecting evidence pages.",
        "",
        "## Preparation coverage",
        "",
        f"- Baseline pages: `{summary['baseline_text']['pages']}`; empty `{summary['baseline_text']['empty_pages']}`; PDF-inspector `{summary['baseline_text']['sources'].get('pdf_inspector', 0)}`; Tesseract `{summary['baseline_text']['sources'].get('tesseract', 0)}`.",
        f"- Embedded-OCR pages: `{summary['embedded_text']['pages']}`; empty `{summary['embedded_text']['empty_pages']}`; replacement-character pages `{summary['embedded_text']['replacement_char_pages']}`.",
        f"- Embedded variant changed `{summary['embedded_delta']['changed_pages']}` pages and added a net `{summary['embedded_delta']['positive_delta_characters']:,}` characters where its text became longer.",
        f"- Tesseract pages have mean confidence `{summary['baseline_text']['ocr_confidence_mean']}` (min `{summary['baseline_text']['ocr_confidence_min']}`).",
        "",
        "## Native French query-token coverage on gold pages",
        "",
        "| Text variant | Exact accents | Accent-folded comparison |",
        "|---|---:|---:|",
        f"| Baseline PDF-inspector/Tesseract | {summary['query_token_coverage']['baseline_exact']:.2%} | {summary['query_token_coverage']['baseline_accent_folded']:.2%} |",
        f"| Embedded-image-OCR variant | {summary['query_token_coverage']['embedded_exact']:.2%} | {summary['query_token_coverage']['embedded_accent_folded']:.2%} |",
        "",
        f"Accent folding recovers at least one query token on `{summary['queries_with_accent_recovery']['baseline']}/80` baseline-query cases and `{summary['queries_with_accent_recovery']['embedded']}/80` embedded-query cases.",
        "",
        "## Queries with largest accent recovery",
        "",
        "| Query ID | Recovered tokens | Missing exact tokens |",
        "|---|---:|---:|",
    ]
    for row in top_recovery:
        lines.append(
            f"| {row['query_id']} | {len(row['accent_recovered_tokens_embedded'])} | "
            f"{len(row['embedded']['exact']['missing'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest embedded-text additions",
            "",
            "| Page | Character delta | OCR applied | Source |",
            "|---|---:|---:|---|",
        ]
    )
    for row in top_embedded_delta:
        lines.append(
            f"| {row['page_id']} | {row['character_delta']:+d} | {row['ocr_applied']} | {row['text_source']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: query-token coverage is a diagnostic of lexical availability, not a semantic-relevance score. Low coverage can be legitimate when the French query paraphrases the evidence, but a large exact-versus-folded gap indicates analyzer mismatch rather than missing document content.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Output: {OUTPUT.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
