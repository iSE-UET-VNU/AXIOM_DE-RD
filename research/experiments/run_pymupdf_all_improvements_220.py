"""Evaluate every agreed light-retrieval enhancement over PyMuPDF mixed text.

Pipeline: PyMuPDF mixed page text + cached embedded-image Tesseract OCR ->
OpenRouter text embeddings -> dense-only scoring for cross-language files and
0.70 BM25 + 0.30 dense scoring for same-language files -> top-3 files ->
Cohere rerank-v3.5 over the top-200 page candidates.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_pdi_native_preserve_ocr import by_dataset  # noqa: E402
from research.experiments.ocr_benchmark_embedded_images import norm_label  # noqa: E402
from research.experiments.run_ohr_replacement_all_arms import (  # noqa: E402
    BUNDLE,
    OHR_ROOT,
    build_bm25_arm,
    file_id,
    load_inputs,
    load_page_corpora,
    read_jsonl,
)
from research.experiments.run_ohr_replacement_hybrid_cohere import (  # noqa: E402
    build_scores,
    clean_short_ocr_noise,
    run_cohere,
)


OUTPUT = ROOT / "data/work/ohr_replacement_20260920/retrieval/pymupdf_all_improvements_220"
PDI_MARKER_OLD = BUNDLE / "preparation/baseline_pdf_inspector_tesseract/pages.jsonl"
PDI_MARKER_OHR = OHR_ROOT / "ohr_preparation/parsing/pages.jsonl"
OCR_OLD = BUNDLE / "preparation/embedded_image_ocr/ocr_results.jsonl"
OCR_OHR = OHR_ROOT / "embedded_image_ocr/ocr_results.jsonl"
PYMUPDF_RUN = OHR_ROOT / "retrieval/merged_220_all_arms/runs/pymupdf.jsonl"
IMAGE_MARKER = re.compile(r"\[Image:\s*([^\]]+?)\]")


def ordered_ocr(marker_text: str, page_id: str, results: list[Mapping[str, Any]]) -> list[str]:
    lookup: dict[tuple[str, int], deque[str]] = defaultdict(deque)
    for row in results:
        if str(row.get("page_id")) != page_id or row.get("status") != "ok":
            continue
        value, _ = clean_short_ocr_noise(str(row.get("text") or ""))
        if value.strip():
            lookup[(str(row.get("image_label") or ""), int(row.get("occurrence_on_page") or 0))].append(value.strip())
    seen: Counter[str] = Counter()
    output: list[str] = []
    for match in IMAGE_MARKER.finditer(marker_text):
        label = norm_label(match.group(1).strip())
        occurrence = seen[label]
        seen[label] += 1
        values = lookup.get((label, occurrence))
        if values:
            output.append(values.popleft())
    return output


def build_pymupdf_plus_image_text() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    _, corpora = load_page_corpora()
    base_rows = corpora["pymupdf"]
    marker_rows = read_jsonl(PDI_MARKER_OLD) + read_jsonl(PDI_MARKER_OHR)
    marker_by_id = {str(row["page_id"]): str(row.get("text") or "") for row in marker_rows}
    results = read_jsonl(OCR_OLD) + read_jsonl(OCR_OHR)
    output_rows: list[dict[str, Any]] = []
    text_by_id: dict[str, str] = {}
    changed = 0
    segments = 0
    words = 0
    for page_id, row in base_rows.items():
        additions = ordered_ocr(marker_by_id.get(page_id, ""), page_id, results)
        text = str(row.get("text") or "")
        if additions:
            text = f"{text}\n\n[OCR from embedded images]\n" + "\n\n".join(additions)
            changed += 1
            segments += len(additions)
            words += sum(len(value.split()) for value in additions)
        output_rows.append({**row, "text": text, "embedded_image_ocr_segments": len(additions)})
        text_by_id[page_id] = text
    if len(output_rows) != 5091:
        raise RuntimeError(f"Expected 5,091 pages, got {len(output_rows)}")
    return output_rows, text_by_id, {"pages_with_appended_image_ocr": changed, "ocr_segments": segments, "ocr_words": words, "ocr_result_records": len(results)}


def encode(pages: list[dict[str, Any]], queries: list[dict[str, Any]], output: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    vectors = output / "embeddings/page_embeddings.npy"
    query_vectors = output / "embeddings/query_embeddings.npy"
    meta_path = output / "embeddings/metadata.json"
    fingerprint = hashlib.sha256("\0".join(str(row["text"]) for row in pages).encode("utf-8")).hexdigest()
    if vectors.is_file() and query_vectors.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("text_sha256") == fingerprint:
            return np.load(vectors, mmap_mode="r"), np.load(query_vectors, mmap_mode="r"), meta

    from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
    from src.utils.env import load_dotenv_file
    load_dotenv_file(ROOT)
    texts = [str(row["text"]) for row in pages]
    chunks_by_page = [[text[start : start + 6000] for start in range(0, len(text), 6000)] or [""] for text in texts]
    chunks = [chunk for values in chunks_by_page for chunk in values]
    output.joinpath("embeddings").mkdir(parents=True, exist_ok=True)
    embedder = OpenRouterEmbedder(model="openai/text-embedding-3-small", batch_size=256, cache_dir=output / "embeddings/api_cache")
    started = time.perf_counter()
    chunk_vectors = np.asarray(embedder.embed(chunks), dtype=np.float32)
    q_vectors = np.asarray(embedder.embed([str(row["query"]) for row in queries]), dtype=np.float32)
    pooled: list[np.ndarray] = []
    offset = 0
    for values in chunks_by_page:
        vector = chunk_vectors[offset : offset + len(values)].mean(axis=0)
        norm = float(np.linalg.norm(vector))
        pooled.append(vector / norm if norm else vector)
        offset += len(values)
    page_vectors = np.vstack(pooled).astype(np.float32)
    np.save(vectors, page_vectors)
    np.save(query_vectors, q_vectors)
    meta = {
        "model": "openai/text-embedding-3-small", "text_sha256": fingerprint,
        "pages": len(pages), "page_chunks": len(chunks), "queries": len(queries),
        "dimension": int(page_vectors.shape[1]), "seconds": time.perf_counter() - started,
        "api_calls": int(embedder.stats["api_calls"]), "cache_hits": int(embedder.stats["cache_hits"]), "tokens": int(embedder.stats["tokens"]),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return page_vectors, q_vectors, meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cohere", action="store_true")
    parser.add_argument("--cohere-workers", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pages, page_texts, ocr_counts = build_pymupdf_plus_image_text()
    queries, qrels, languages = load_inputs()
    write_path = output / "preparation/pages_pymupdf_plus_embedded_ocr.jsonl"
    if not write_path.is_file():
        from research.experiments.run_ohr_replacement_all_arms import write_jsonl
        write_jsonl(write_path, pages)
    page_vectors, query_vectors, embed_meta = encode(pages, queries, output)
    started = time.perf_counter()
    hybrid_rows, score_counts = build_scores(pages=pages, queries=queries, page_vectors=np.asarray(page_vectors), query_vectors=np.asarray(query_vectors), language_by_file=languages, output_dir=output / "hybrid")
    from research.experiments.run_ohr_replacement_all_arms import write_jsonl
    write_jsonl(output / "hybrid/run_top20.jsonl", hybrid_rows)
    hybrid_seconds = time.perf_counter() - started
    reranked = None
    if not args.skip_cohere:
        cohere = run_cohere(hybrid_rows, queries, pages, qrels, output / "cohere", args.cohere_workers)
        reranked = [{**row, "returned_pages": row["reranked_pages"]} for row in read_jsonl(output / "cohere/cohere_top20.jsonl")]
    baseline_pymupdf = read_jsonl(PYMUPDF_RUN)
    report = {
        "protocol": {
            "base": "PyMuPDF mixed fallback", "embedded_ocr": "append cached Tesseract OCR from every PDI-marked embedded image",
            "cross_language": "dense embedding only", "same_language": "0.70 BM25 + 0.30 dense embedding",
            "file_stage": "top-3 files, 0.50 direct score + 0.50 max page score", "reranking": "Cohere rerank-v3.5 over top-200 candidate pages",
            "qrels_used_for_ranking": False,
        },
        "counts": {"pages": len(pages), "queries": len(queries), **ocr_counts, **score_counts},
        "embedding": embed_meta,
        "timing_seconds": {"hybrid_retrieval": hybrid_seconds},
        "metrics": {"pymupdf_mixed_bm25": by_dataset(baseline_pymupdf, qrels), "all_improvements_before_rerank": by_dataset(hybrid_rows, qrels)},
    }
    if reranked is not None:
        report["metrics"]["all_improvements_cohere_rerank"] = by_dataset(reranked, qrels)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# PyMuPDF baseline + all light-retrieval enhancements", "", "| Dataset | Arm | Recall@10 | nDCG@10 | Recall@20 | nDCG@20 | File recall@3 |", "|---|---|---:|---:|---:|---:|---:|"]
    for dataset in ("physics", "industrial", "ohr", "all"):
        for key, label in (("pymupdf_mixed_bm25", "PyMuPDF mixed BM25"), ("all_improvements_before_rerank", "+ OCR + language-routed hybrid"), ("all_improvements_cohere_rerank", "+ Cohere rerank")):
            if key not in report["metrics"]:
                continue
            m10, m20 = report["metrics"][key][dataset]["@10"], report["metrics"][key][dataset]["@20"]
            lines.append(f"| {dataset} | {label} | {m10['page_recall@10']:.2%} | {m10['ndcg@10']:.2f} | {m20['page_recall@20']:.2%} | {m20['ndcg@20']:.2f} | {m10['file_recall@3']:.2%} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
