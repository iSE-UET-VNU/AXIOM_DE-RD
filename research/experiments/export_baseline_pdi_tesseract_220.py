"""Package the canonical PDF-inspector + Tesseract BM25 baseline for sharing.

The export is deliberately self-contained at the light-preparation/retrieval
layer.  It contains all 5,091 parsed page records, all 220 queries and qrels,
the compact top-20 run, and complete score matrices over the 220 x 5,091
query-page space.  Raw PDFs are intentionally excluded.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    file_id,
    load_inputs,
    load_page_corpora,
    normalise,
    read_jsonl,
    write_jsonl,
)


BASELINE_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
DEFAULT_OUTPUT = ROOT / "data/work/share_baseline_pdf_inspector_tesseract_220"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def full_scores(page_texts: Mapping[str, str], queries: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Recompute the canonical hierarchical BM25 scores, retaining every score."""
    from src.retrieval.sparse import BM25Index

    page_ids = list(page_texts)
    file_ids = list(dict.fromkeys(file_id(page) for page in page_ids))
    page_files = np.asarray([file_ids.index(file_id(page)) for page in page_ids], dtype=np.int32)
    file_to_positions = [np.flatnonzero(page_files == index) for index in range(len(file_ids))]
    page_index = BM25Index().build([
        {"chunk_id": page, "doc_id": file_id(page), "text": page_texts[page]} for page in page_ids
    ])
    grouped = {fid: [] for fid in file_ids}
    for page in page_ids:
        grouped[file_id(page)].append(page_texts[page])
    file_index = BM25Index().build([
        {"chunk_id": fid, "doc_id": fid, "text": "\n".join(grouped[fid])} for fid in file_ids
    ])

    q_count, p_count, f_count = len(queries), len(page_ids), len(file_ids)
    page_raw = np.zeros((q_count, p_count), dtype=np.float32)
    page_norm = np.zeros_like(page_raw)
    file_raw = np.zeros((q_count, f_count), dtype=np.float32)
    file_score = np.zeros((q_count, f_count), dtype=np.float32)
    page_final = np.full((q_count, p_count), -np.inf, dtype=np.float32)
    selected_files: list[list[str]] = []
    complete_candidates: list[dict[str, Any]] = []

    for qi, query in enumerate(queries):
        text = str(query["query"])
        raw = np.zeros(p_count, dtype=np.float64)
        raw_file = np.zeros(f_count, dtype=np.float64)
        for position, score in page_index.search(text, p_count):
            raw[int(position)] = float(score)
        for position, score in file_index.search(text, f_count):
            raw_file[int(position)] = float(score)
        norm = np.asarray(normalise(raw), dtype=np.float64)
        norm_file = np.asarray(normalise(raw_file), dtype=np.float64)
        page_base = 0.70 * norm
        file_max = np.asarray([float(page_base[positions].max()) if positions.size else 0.0 for positions in file_to_positions])
        combined_file = 0.50 * norm_file + 0.50 * np.asarray(normalise(file_max), dtype=np.float64)
        selected_indices = sorted(range(f_count), key=lambda index: (-float(combined_file[index]), file_ids[index]))[:3]
        selected = [file_ids[index] for index in selected_indices]
        parent = normalise({file_ids[index]: float(combined_file[index]) for index in range(f_count)})
        # Mapping normalisation returns a dict; preserve explicit file order.
        parent_scores = np.asarray([float(parent.get(fid, 0.0)) for fid in file_ids], dtype=np.float64)
        keep = np.isin(page_files, np.asarray(selected_indices, dtype=np.int32))
        final = 0.85 * page_base + 0.15 * parent_scores[page_files]
        page_final[qi, keep] = final[keep].astype(np.float32)
        candidate_positions = np.flatnonzero(keep)
        candidate_positions = candidate_positions[np.lexsort((np.asarray([page_ids[index] for index in candidate_positions]), -final[candidate_positions]))]
        complete_candidates.append({
            "query_id": str(query["query_id"]),
            "selected_files": selected,
            "ranked_candidate_pages": [page_ids[index] for index in candidate_positions],
            "ranked_candidate_scores": [float(final[index]) for index in candidate_positions],
        })
        selected_files.append(selected)
        page_raw[qi] = raw.astype(np.float32)
        page_norm[qi] = norm.astype(np.float32)
        file_raw[qi] = raw_file.astype(np.float32)
        file_score[qi] = combined_file.astype(np.float32)

    return {
        "page_ids": page_ids,
        "file_ids": file_ids,
        "query_ids": [str(query["query_id"]) for query in queries],
        "page_bm25_raw": page_raw,
        "page_bm25_normalized": page_norm,
        "file_bm25_raw": file_raw,
        "file_score": file_score,
        "page_final_selected_files": page_final,
        "selected_files": selected_files,
    }, complete_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse an interrupted export directory and refresh its files")
    args = parser.parse_args()
    output = args.output_dir
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    output.mkdir(parents=True, exist_ok=args.resume)

    started = time.perf_counter()
    page_texts, corpora = load_page_corpora()
    baseline_rows = list(corpora["baseline_bm25"].values())
    queries, qrels, languages = load_inputs()
    if len(baseline_rows) != 5091 or len(queries) != 220 or len(qrels) != 220:
        raise RuntimeError("Expected canonical 5,091 pages and 220 queries")
    if len({str(row["page_id"]) for row in baseline_rows}) != 5091:
        raise RuntimeError("Baseline preparation has duplicate page IDs")

    write_jsonl(output / "preparation/pages.jsonl", baseline_rows)
    files: dict[str, dict[str, Any]] = {}
    for row in baseline_rows:
        fid = file_id(str(row["page_id"]))
        files.setdefault(fid, {
            "file_id": fid, "doc_id": str(row.get("doc_id") or fid), "source": str(row.get("source") or ""),
            "relative_path": str(row.get("relative_path") or ""), "pdf_type": str(row.get("pdf_type") or ""),
            "detected_language": languages.get(fid, ""), "page_count": 0,
        })["page_count"] += 1
    write_jsonl(output / "preparation/files.jsonl", list(files.values()))
    write_jsonl(output / "inputs/queries.jsonl", queries)
    write_jsonl(output / "inputs/qrels_page_level.jsonl", [
        {"query_id": qid, "page_id": page_id, "relevance": 1}
        for qid in sorted(qrels) for page_id in sorted(qrels[qid])
    ])

    arrays, candidate_rows = full_scores(page_texts["baseline_bm25"], queries)
    canonical = read_jsonl(BASELINE_RUN)
    canonical_by_qid = {str(row["query_id"]): row for row in canonical}
    exported_by_qid = {str(row["query_id"]): row for row in candidate_rows}
    mismatches = []
    for qid, row in canonical_by_qid.items():
        computed = exported_by_qid[qid]
        if row.get("selected_files") != computed["selected_files"] or row.get("returned_pages") != computed["ranked_candidate_pages"][:20]:
            mismatches.append(qid)
    if mismatches:
        raise RuntimeError(f"Full-score recomputation disagrees with canonical baseline for {len(mismatches)} queries, e.g. {mismatches[:3]}")

    retrieval = output / "retrieval"
    retrieval.mkdir(exist_ok=args.resume)
    np.savez_compressed(
        retrieval / "full_score_matrices.npz",
        page_bm25_raw=arrays["page_bm25_raw"],
        page_bm25_normalized=arrays["page_bm25_normalized"],
        file_bm25_raw=arrays["file_bm25_raw"],
        file_score=arrays["file_score"],
        page_final_selected_files=arrays["page_final_selected_files"],
    )
    write_json(retrieval / "score_matrix_metadata.json", {
        "query_ids": arrays["query_ids"], "page_ids": arrays["page_ids"], "file_ids": arrays["file_ids"],
        "shapes": {key: list(value.shape) for key, value in arrays.items() if isinstance(value, np.ndarray)},
        "score_definition": {
            "page_bm25_raw": "BM25(query, page text), for every query-page pair",
            "page_bm25_normalized": "raw score divided by that query's maximum page BM25 score",
            "file_bm25_raw": "BM25(query, concatenated file page text), for every query-file pair",
            "file_score": "0.50 normalized direct file BM25 + 0.50 normalized max(0.70 normalized page BM25)",
            "page_final_selected_files": "0.85*(0.70 normalized page BM25) + 0.15 normalized file score; -inf outside selected top-3 files",
        },
    })
    write_jsonl(retrieval / "run_top20.jsonl", canonical)
    with gzip.open(retrieval / "all_selected_file_pages_ranked.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
        for row in candidate_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    source_script = ROOT / "research/experiments/run_ohr_replacement_all_arms.py"
    (output / "reproduce").mkdir(exist_ok=True)
    shutil.copy2(source_script, output / "reproduce/run_ohr_replacement_all_arms.py")
    write_json(output / "manifest.json", {
        "arm": "canonical PDF-inspector + selective full-page Tesseract + hierarchical BM25",
        "dataset": "merged benchmark: 80 Physics + 75 Industrial + 65 OHR",
        "files": len(files), "pages": len(baseline_rows), "queries": len(queries), "qrel_records": sum(len(value) for value in qrels.values()),
        "file_k": 3, "metric_depth": 10, "saved_depth": 20, "candidate_depth": 200,
        "ranking_verified_against": str(BASELINE_RUN.resolve()),
        "ranking_verification": "exact selected-file and top-20-page match for all 220 queries",
        "elapsed_seconds": time.perf_counter() - started,
    })
    readme = """# Canonical PDF-inspector + Tesseract baseline — 220 queries

This package contains the light-preparation and light-retrieval outputs for
the canonical baseline on 103 PDFs / 5,091 pages and 220 queries (80 Physics,
75 Industrial, 65 OHR).  It does **not** include raw PDFs.

## Preparation

- `preparation/pages.jsonl`: all 5,091 final page records. `text` is the
  exact text indexed by BM25; `text_source`, `needs_ocr`, `pdf_type`, and page
  provenance are retained.
- `preparation/files.jsonl`: the 103 file records derived from those pages.

## Retrieval

- `retrieval/run_top20.jsonl`: canonical top-20 result for every one of the
  220 queries.
- `retrieval/all_selected_file_pages_ranked.jsonl.gz`: every page in each
  query's selected top-3 files, in final retrieval order. This is the complete
  hierarchical candidate ranking, not only its top-20 prefix.
- `retrieval/full_score_matrices.npz`: all 220 x 5,091 global page BM25
  scores, normalized page scores, all 220 x 103 file scores, and final page
  scores. A final score of `-inf` means that its file was outside that query's
  selected top-3 files.
- `retrieval/score_matrix_metadata.json`: exact row/column ID order and score
  definitions for the NPZ arrays.

`inputs/qrels_page_level.jsonl` is included for evaluation only and was not
used in parsing, indexing, file selection, or ranking.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    if not args.no_zip:
        shutil.make_archive(str(output), "zip", root_dir=output)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
