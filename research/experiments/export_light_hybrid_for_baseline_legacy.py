"""Export the merged-220 language-aware light-retrieval handoff bundle.

The bundle is deliberately slim enough to hand to the accurate-retrieval
owner: it contains the query-only input and the 20 page IDs selected by the
light stage, plus the corresponding light-preparation text cache for audit.
Reference answers/qrels are isolated under ``evaluation/`` and must not be
provided to the answer generator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[2]
OLD_PAGES = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl"
OHR_PAGES = ROOT / "data/work/ohr_replacement_20260920/embedded_ocr_filtered/size_filter_only/pages.jsonl"
RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/hybrid_bm25_embeddings.jsonl"
DOCUMENTS = ROOT / "data/raw/0.1. BENCHMARK/documents.jsonl"
QUERIES = ROOT / "data/raw/0.1. BENCHMARK/queries.jsonl"
DOCUMENT_QRELS = ROOT / "data/raw/0.1. BENCHMARK/qrels.jsonl"
OLD_PAGE_QRELS = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
OHR_PAGE_QRELS = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/work/exports/merged_220_light_hybrid_language_aware_top20_for_baseline_legacy",
    )
    parser.add_argument("--zip", action="store_true", help="Also create a ZIP next to output-dir.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    run_rows = read_jsonl(RUN)
    query_rows = read_jsonl(QUERIES)
    document_rows = read_jsonl(DOCUMENTS)
    if len(run_rows) != 220 or len(query_rows) != 220 or len(document_rows) != 103:
        raise RuntimeError("Expected the canonical merged benchmark: 220 queries and 103 documents.")

    query_only = [{key: row[key] for key in ("query_id", "source", "source_query_id", "query", "metadata")} for row in query_rows]
    write_jsonl(out / "input/queries.jsonl", query_only)
    write_jsonl(out / "input/documents.jsonl", document_rows)

    rows = []
    for row in run_rows:
        pages = row["returned_pages"]
        scores = row["returned_scores"]
        if not 1 <= len(pages) <= 20 or len(pages) != len(scores):
            raise RuntimeError(f"Invalid top-20 record for {row['query_id']}")
        rows.append({
            "query_id": row["query_id"],
            "query": row["query"],
            "selected_files": row["selected_files"],
            "pages": [
                {
                    "rank": index + 1,
                    "page_id": page_id,
                    "doc_id": page_id.rsplit("#page=", 1)[0],
                    "page_index": int(page_id.rsplit("#page=", 1)[1]),
                    "light_score": float(score),
                }
                for index, (page_id, score) in enumerate(zip(pages, scores))
            ],
        })
    write_jsonl(out / "runs/hybrid_language_aware_top20.jsonl", rows)

    # The historic Physics/Industrial cache includes removed MPDocVQA pages.
    # Keep only the sources in the canonical merged corpus, then append OHR.
    old_pages = [row for row in read_jsonl(OLD_PAGES) if row["source"] in {"vidore_physics", "vidore_industrial"}]
    prepared_pages = old_pages + read_jsonl(OHR_PAGES)
    page_ids = [row["page_id"] for row in prepared_pages]
    if len(prepared_pages) != 5091 or len(page_ids) != len(set(page_ids)):
        raise RuntimeError("Expected 5,091 unique prepared pages in the merged corpus.")
    write_jsonl(out / "light_preparation/pages.jsonl", prepared_pages)

    page_qrels = []
    for row in read_jsonl(OLD_PAGE_QRELS):
        if str(row["query_id"]).startswith(("vidore_physics::", "vidore_industrial::")):
            page_qrels.append(row)
    for row in read_jsonl(OHR_PAGE_QRELS):
        page_qrels.append({
            "query_id": row["query_id"],
            "page_id": f"{row['doc_id']}#page={int(row['page_index'])}",
            "relevance": row["relevance"],
        })
    write_jsonl(out / "evaluation/page_qrels.jsonl", page_qrels)
    write_jsonl(out / "evaluation/document_qrels.jsonl", read_jsonl(DOCUMENT_QRELS))
    write_jsonl(out / "evaluation/queries_with_reference_answers.jsonl", query_rows)

    manifest = {
        "protocol": "merged-220 light retrieval handoff for baseline legacy",
        "benchmark": {"queries": 220, "files": 103, "pages": 5091, "sources": {"physics": 80, "industrial": 75, "ohr": 65}},
        "light_preparation": "PDF-inspector; full-page Tesseract for scanned/empty pages; Tesseract embedded-image OCR text appended in reading order",
        "light_retrieval": {
            "returned_depth": 20,
            "file_k": 3,
            "same_language": "0.70 normalized BM25 + 0.30 normalized text-embedding-3-small",
            "cross_language": "text-embedding-3-small dense score only",
            "query_for_dense": "original query",
            "qrels_used_for_ranking": False,
        },
        "artifacts": {},
    }
    for path in sorted(out.rglob("*.jsonl")):
        manifest["artifacts"][str(path.relative_to(out)).replace("\\", "/")] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(
        """# Light retrieval handoff: language-aware hybrid, top-20

Use `runs/hybrid_language_aware_top20.jsonl` as the input to baseline legacy's
accurate stage. Each query has up to 20 ordered pages, their document/page IDs,
and the three files selected by the light stage. The page ID convention is
`<doc_id>#page=<zero-based-index>`.

## What was run

- Light preparation: PDF-inspector; full-page Tesseract only for pages that
  were scanned/empty; Tesseract OCR of embedded images, with the OCR text
  inserted into the page text.
- Light retrieval: language-aware hybrid. For query/page text in the same
  language: `0.70 * normalized BM25 + 0.30 * normalized
  text-embedding-3-small`. For cross-language pairs: dense score only.
- File budget: 3; output: top 20 pages. Qrels/reference answers were not used
  in light preparation or ranking.

## Handoff contents

- `input/`: query-only input and document-to-raw-file mapping.
- `runs/`: the compact ranked top-20 candidate list. This is the only file
  needed by an accurate KDL/legacy parser when raw PDFs are available.
- `light_preparation/pages.jsonl`: full 5,091-page cached light text for audit
  or reproduction; not required by KDL accurate parsing.
- `evaluation/`: qrels and reference answers. Keep this directory out of the
  answering prompt; it exists only for offline evaluation.

The colleague should resolve `doc_id` against `input/documents.jsonl`, extract
the requested zero-based page from the raw file, run KDL/legacy accurate
parsing, then rerank/retrieve only inside each query's supplied 20 pages.
""",
        encoding="utf-8",
    )
    if args.zip:
        archive = out.with_suffix(".zip")
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(out))
        print(archive)
    print(out)


if __name__ == "__main__":
    main()
