"""Run the replacement OHR(65) hybrid light-retrieval + Cohere experiment.

The corpus is the 38 cached Physics/Industrial files plus the newly prepared
OHR files.  Only OHR queries are evaluated.  Existing Physics/Industrial page
preparation and text-embedding vectors are reused; only OHR page/query vectors
are requested from OpenRouter.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OLD_BUNDLE = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval"
OLD_PAGES = OLD_BUNDLE / "preparation/embedded_image_ocr/pages_with_embedded_ocr.jsonl"
OLD_PAGE_META = OLD_BUNDLE / "runs/dense_embedded_image_ocr/page_metadata.json"
OLD_PAGE_VECTORS = OLD_BUNDLE / "runs/dense_embedded_image_ocr/embeddings/page_embeddings.npy"
OLD_LANGUAGE = OLD_BUNDLE / "reports/language_routed_bm25/file_language_detection.jsonl"
OLD_DOCS = OLD_BUNDLE / "input/documents.jsonl"

KEEP_SHORT = {
    "a", "i", "an", "as", "at", "be", "by", "do", "if", "in", "is", "it", "no", "of", "on", "or", "to", "up", "us", "we",
    "au", "de", "du", "en", "et", "la", "le", "les", "un", "une",
}
SHORT_ALPHA = re.compile(r"^[A-Za-z]{1,2}$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_short_ocr_noise(text: str) -> tuple[str, int]:
    removed = 0
    output: list[str] = []
    for token in str(text).split():
        stripped = token.strip(".,;:!?()[]{}<>\"'`|~+=*/\\")
        if SHORT_ALPHA.fullmatch(stripped) and stripped.casefold() not in KEEP_SHORT:
            removed += 1
            continue
        output.append(token)
    return " ".join(output), removed


def normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values)


def file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def ndcg(ranked: Sequence[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 2.0) for rank, page in enumerate(ranked[:k]) if page in gold)
    ideal = sum(1.0 / math.log2(rank + 2.0) for rank in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def load_inputs(input_dir: Path, prep_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]]:
    queries = read_jsonl(input_dir / "queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(input_dir / "qrels.jsonl"):
        qid = str(row["query_id"])
        page = f"{row['doc_id']}#page={int(row['page_index'])}"
        qrels[qid].add(page)
    pages = read_jsonl(prep_dir / "embedded_ocr_filtered/size_filter_only/pages.jsonl")
    if not pages:
        raise RuntimeError("OHR filtered embedded-OCR pages are missing")
    return pages, queries, qrels


def load_merged_pages(ohr_pages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], np.ndarray]:
    old_pages_all = read_jsonl(OLD_PAGES)
    old_sources = {str(row["doc_id"]): str(row.get("source", "")) for row in read_jsonl(OLD_DOCS)}
    keep_sources = {"vidore_physics", "vidore_industrial"}
    old_pages = [row for row in old_pages_all if old_sources.get(str(row["doc_id"])) in keep_sources]
    if len(old_pages) != 4682:
        raise RuntimeError(f"Expected 4682 cached Physics/Industrial pages, got {len(old_pages)}")
    old_meta = json.loads(OLD_PAGE_META.read_text(encoding="utf-8"))
    old_ids = [str(value) for value in old_meta["page_ids"]]
    old_vectors = np.load(OLD_PAGE_VECTORS, mmap_mode="r")
    vector_pos = {page_id: index for index, page_id in enumerate(old_ids)}
    missing = [str(row["page_id"]) for row in old_pages if str(row["page_id"]) not in vector_pos]
    if missing:
        raise RuntimeError(f"Cached page vectors miss {len(missing)} old pages")
    old_selected_vectors = np.asarray([old_vectors[vector_pos[str(row["page_id"])]] for row in old_pages], dtype=np.float32)
    page_ids = {str(row["page_id"]) for row in old_pages}
    overlap = page_ids & {str(row["page_id"]) for row in ohr_pages}
    if overlap:
        raise RuntimeError(f"Duplicate page IDs after merge: {len(overlap)}")
    return old_pages + ohr_pages, old_selected_vectors


def encode_new_pages_and_queries(
    *,
    pages: list[dict[str, Any]],
    old_count: int,
    old_vectors: np.ndarray,
    queries: list[dict[str, Any]],
    output_dir: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    page_path = output_dir / "page_embeddings.npy"
    query_path = output_dir / "query_embeddings.npy"
    meta_path = output_dir / "embedding_metadata.json"
    if page_path.is_file() and query_path.is_file() and meta_path.is_file() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        page_vectors = np.load(page_path, mmap_mode="r")
        query_vectors = np.load(query_path, mmap_mode="r")
        if page_vectors.shape[0] == len(pages) and query_vectors.shape[0] == len(queries):
            return np.asarray(page_vectors), np.asarray(query_vectors), meta

    from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder
    from src.utils.env import load_dotenv_file

    load_dotenv_file(ROOT)
    model = "openai/text-embedding-3-small"
    cleaned_new: list[str] = []
    removed = 0
    for row in pages[old_count:]:
        text, count = clean_short_ocr_noise(str(row.get("text") or ""))
        cleaned_new.append(text)
        removed += count
    chunk_chars = 6000
    page_chunks: list[list[str]] = []
    chunk_inputs: list[str] = []
    for text in cleaned_new:
        chunks = [text[start : start + chunk_chars] for start in range(0, len(text), chunk_chars)] if text else []
        page_chunks.append(chunks)
        chunk_inputs.extend(chunks)
    embedder = OpenRouterEmbedder(model=model, batch_size=256, cache_dir=output_dir / "api_cache")
    started = time.perf_counter()
    chunk_vectors = np.asarray(embedder.embed(chunk_inputs), dtype=np.float32)
    query_vectors = np.asarray(embedder.embed([str(row["query"]) for row in queries]), dtype=np.float32)
    if chunk_vectors.ndim != 2:
        raise RuntimeError(f"Unexpected new-page embedding shape: {chunk_vectors.shape}")
    new_vectors: list[np.ndarray] = []
    cursor = 0
    for chunks in page_chunks:
        if not chunks:
            new_vectors.append(np.zeros(chunk_vectors.shape[1], dtype=np.float32))
            continue
        pooled = chunk_vectors[cursor : cursor + len(chunks)].mean(axis=0)
        norm = float(np.linalg.norm(pooled))
        new_vectors.append(pooled / norm if norm else pooled)
        cursor += len(chunks)
    page_vectors = np.vstack([old_vectors, np.vstack(new_vectors)]).astype(np.float32)
    meta = {
        "model": model,
        "provider": "openrouter",
        "dimension": int(page_vectors.shape[1]),
        "old_cached_pages": old_count,
        "new_ohr_pages": len(cleaned_new),
        "new_page_chunk_count": len(chunk_inputs),
        "query_count": len(queries),
        "chunk_chars": chunk_chars,
        "cleaning": "remove isolated 1-2 alphabetic OCR tokens except English/French function words",
        "removed_short_tokens_new_ohr": removed,
        "api_calls": int(embedder.stats["api_calls"]),
        "cache_hits": int(embedder.stats["cache_hits"]),
        "tokens": int(embedder.stats["tokens"]),
        "seconds": time.perf_counter() - started,
        "old_vectors_reused": str(OLD_PAGE_VECTORS.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(page_path, page_vectors)
    np.save(query_path, query_vectors)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return page_vectors, query_vectors, meta


def build_scores(
    *, pages: list[dict[str, Any]], queries: list[dict[str, Any]], page_vectors: np.ndarray,
    query_vectors: np.ndarray, language_by_file: dict[str, str], output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from src.retrieval.sparse import BM25Index

    page_ids = [str(row["page_id"]) for row in pages]
    file_ids = sorted({file_id(page_id) for page_id in page_ids})
    file_pos = {value: index for index, value in enumerate(file_ids)}
    page_file_pos = np.asarray([file_pos[file_id(page_id)] for page_id in page_ids], dtype=np.int64)
    page_index = BM25Index().build([{"chunk_id": page_id, "doc_id": file_id(page_id), "text": str(row.get("text") or "")} for page_id, row in zip(page_ids, pages)])
    file_texts: dict[str, list[str]] = defaultdict(list)
    for page_id, row in zip(page_ids, pages):
        file_texts[file_id(page_id)].append(str(row.get("text") or ""))
    file_index = BM25Index().build([{"chunk_id": file_id, "doc_id": file_id, "text": "\n".join(file_texts[file_id])} for file_id in file_ids])

    page_bm_raw = np.zeros((len(queries), len(pages)), dtype=np.float32)
    file_bm_raw = np.zeros((len(queries), len(file_ids)), dtype=np.float32)
    dense = np.asarray(query_vectors @ page_vectors.T, dtype=np.float32)
    for qi, query in enumerate(queries):
        for position, score in page_index.search(str(query["query"]), len(pages)):
            page_bm_raw[qi, position] = float(score)
        for position, score in file_index.search(str(query["query"]), len(file_ids)):
            file_bm_raw[qi, position] = float(score)

    rows: list[dict[str, Any]] = []
    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        page_bm = normalise(page_bm_raw[qi])
        dense_page = normalise(dense[qi])
        direct = np.zeros(len(file_ids), dtype=np.float64)
        page_score = np.zeros(len(pages), dtype=np.float64)
        for fi, file_value in enumerate(file_ids):
            mask = page_file_pos == fi
            language = language_by_file.get(file_value, "en")
            if language == "en":
                page_score[mask] = 0.70 * page_bm[mask] + 0.30 * dense_page[mask]
                direct[fi] = 0.70 * float(normalise(file_bm_raw[qi])[fi]) + 0.30 * float(dense_page[mask].max(initial=0.0))
            else:
                page_score[mask] = dense_page[mask]
                direct[fi] = float(dense_page[mask].max(initial=0.0))
        page_max = np.zeros(len(file_ids), dtype=np.float64)
        np.maximum.at(page_max, page_file_pos, page_score)
        file_score = 0.50 * direct + 0.50 * normalise(page_max)
        selected = np.argsort(-file_score, kind="stable")[:3]
        selected_mask = np.isin(page_file_pos, selected)
        final_score = 0.85 * page_score + 0.15 * normalise(file_score)[page_file_pos]
        selected_positions = np.flatnonzero(selected_mask)
        order = selected_positions[np.argsort(-final_score[selected_positions], kind="stable")]
        candidate_positions = order[:200]
        returned_positions = order[:20]
        rows.append({
            "query_id": qid,
            "query": str(query["query"]),
            "selected_files": [file_ids[int(index)] for index in selected],
            "candidate_pages": [page_ids[int(index)] for index in candidate_positions],
            "candidate_scores": [float(final_score[int(index)]) for index in candidate_positions],
            "returned_pages": [page_ids[int(index)] for index in returned_positions],
            "returned_scores": [float(final_score[int(index)]) for index in returned_positions],
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "page_bm25_raw.npy", page_bm_raw)
    np.save(output_dir / "file_bm25_raw.npy", file_bm_raw)
    np.save(output_dir / "query_page_dense.npy", dense)
    (output_dir / "metadata.json").write_text(json.dumps({
        "page_ids": page_ids, "file_ids": file_ids,
        "query_ids": [str(row["query_id"]) for row in queries],
        "page_count": len(pages), "file_count": len(file_ids), "query_count": len(queries),
        "file_k": 3, "candidate_depth": 200, "page_bm25_weight_same_language": 0.70,
        "dense_weight_same_language": 0.30, "file_score": "0.50 direct + 0.50 max page",
        "page_final": "0.85 page score + 0.15 normalized file score",
        "qrels_used_for_ranking": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, {"pages": len(pages), "files": len(file_ids), "queries": len(queries)}


def metrics(rows: Sequence[Mapping[str, Any]], qrels: Mapping[str, set[str]], depth: int) -> dict[str, float]:
    recalls, hits, ndcgs, file_recalls, ceilings = [], [], [], [], []
    for row in rows:
        gold = qrels[str(row["query_id"])]
        ranked = list(row["returned_pages"])
        candidate = set(row["candidate_pages"])
        top = ranked[:depth]
        recalls.append(len(set(top) & gold) / max(len(gold), 1))
        hits.append(float(bool(set(top) & gold)))
        ndcgs.append(ndcg(ranked, gold, depth))
        gold_files = {file_id(page) for page in gold}
        file_recalls.append(len(set(row["selected_files"]) & gold_files) / max(len(gold_files), 1))
        ceilings.append(len(candidate & gold) / max(len(gold), 1))
    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {"queries": len(rows), f"page_recall@{depth}": mean(recalls), f"page_hit@{depth}": mean(hits), f"ndcg@{depth}": 100.0 * mean(ndcgs), "file_recall@3": mean(file_recalls), "candidate_pool_recall@200": mean(ceilings)}


def get_key() -> str:
    from src.utils.env import load_dotenv_file
    load_dotenv_file(ROOT)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return key


def rerank_one(qid: str, query: str, page_ids: list[str], texts: Mapping[str, str], key: str) -> dict[str, Any]:
    documents = [texts[page_id][:12000] for page_id in page_ids]
    payload = {"model": "cohere/rerank-v3.5", "query": query, "documents": documents, "top_n": len(documents), "return_documents": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "X-Title": "AXIOM OHR replacement rerank"}
    last_error = ""
    for attempt in range(3):
        try:
            response = requests.post("https://openrouter.ai/api/v1/rerank", headers=headers, json=payload, timeout=180)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                time.sleep(min(2 ** attempt, 8))
                continue
            response.raise_for_status()
            result = response.json().get("results")
            if not isinstance(result, list) or len(result) != len(page_ids):
                raise RuntimeError(f"Expected {len(page_ids)} rerank results")
            ranked = []
            seen: set[int] = set()
            for rank, item in enumerate(result, start=1):
                index = int(item["index"])
                if index in seen or not 0 <= index < len(page_ids):
                    raise RuntimeError("Invalid reranker index")
                seen.add(index)
                ranked.append({"page_id": page_ids[index], "rank": rank, "score": float(item["relevance_score"])})
            return {"query_id": qid, "candidate_pages": page_ids, "ranked": ranked}
        except Exception as error:  # noqa: BLE001
            last_error = f"{type(error).__name__}: {error}"
            if attempt < 2:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Cohere rerank failed for {qid}: {last_error}")


def run_cohere(rows: list[dict[str, Any]], queries: list[dict[str, Any]], pages: list[dict[str, Any]], qrels: dict[str, set[str]], output_dir: Path, workers: int) -> dict[str, Any]:
    key = get_key()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "cohere_cache.jsonl"
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        for row in read_jsonl(cache_path):
            cached[str(row["query_id"])] = row
    texts = {str(row["page_id"]): str(row.get("text") or "") for row in pages}
    query_map = {str(row["query_id"]): str(row["query"]) for row in queries}
    pending = [row for row in rows if str(row["query_id"]) not in cached or cached[str(row["query_id"])].get("candidate_pages") != row["candidate_pages"]]
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(rerank_one, str(row["query_id"]), query_map[str(row["query_id"])], list(row["candidate_pages"]), texts, key): str(row["query_id"]) for row in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                cached[futures[future]] = future.result()
                with cache_path.open("w", encoding="utf-8", newline="\n") as handle:
                    for qid in query_map:
                        if qid in cached:
                            handle.write(json.dumps(cached[qid], ensure_ascii=False, separators=(",", ":")) + "\n")
                if index % 5 == 0 or index == len(pending):
                    print(f"Cohere rerank {index}/{len(pending)}", flush=True)
    reranked_rows = []
    for row in rows:
        cached_row = cached[str(row["query_id"])]
        ranked = [str(item["page_id"]) for item in cached_row["ranked"]]
        reranked_rows.append({**row, "reranked_pages": ranked, "rerank_scores": [float(item["score"]) for item in cached_row["ranked"]]})
    write_jsonl(output_dir / "cohere_top20.jsonl", reranked_rows)
    report = {
        "method": "fixed top-3 file stage -> hybrid page candidate top-200 -> Cohere rerank-v3.5",
        "queries": len(reranked_rows), "file_k": 3, "candidate_depth": 200,
        "reranker_workers": workers, "qrels_used_for_ranking": False,
        "metrics": {"@10": metrics([{**row, "returned_pages": row["reranked_pages"]} for row in reranked_rows], qrels, 10), "@20": metrics([{**row, "returned_pages": row["reranked_pages"]} for row in reranked_rows], qrels, 20)},
    }
    (output_dir / "cohere_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/work/ohr_replacement_20260920/inputs")
    parser.add_argument("--prep-dir", type=Path, default=ROOT / "data/work/ohr_replacement_20260920")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/work/ohr_replacement_20260920/retrieval")
    parser.add_argument("--force-embed", action="store_true")
    parser.add_argument("--cohere-workers", type=int, default=32)
    args = parser.parse_args()
    started = time.perf_counter()
    ohr_pages, queries, qrels = load_inputs(args.input_dir, args.prep_dir)
    pages, old_vectors = load_merged_pages(ohr_pages)
    embed_dir = args.output_dir / "embeddings"
    page_vectors, query_vectors, embed_meta = encode_new_pages_and_queries(pages=pages, old_count=len(pages) - len(ohr_pages), old_vectors=old_vectors, queries=queries, output_dir=embed_dir, force=args.force_embed)
    old_languages = {str(row["file_id"]): str(row["detected_language"]) for row in read_jsonl(OLD_LANGUAGE) if str(row["file_id"]).startswith(("vidore_physics::", "vidore_industrial::"))}
    language_by_file = {**old_languages, **{file_id(str(row["page_id"])): "en" for row in ohr_pages}}
    retrieval_dir = args.output_dir / "base"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = build_scores(pages=pages, queries=queries, page_vectors=page_vectors, query_vectors=query_vectors, language_by_file=language_by_file, output_dir=retrieval_dir)
    write_jsonl(retrieval_dir / "hybrid_top20.jsonl", rows)
    base_report = {"method": "same-language 0.70 BM25 + 0.30 OpenRouter text embedding; cross-language embedding only", "counts": counts, "embedding": embed_meta, "metrics": {"@10": metrics(rows, qrels, 10), "@20": metrics(rows, qrels, 20)}, "timing_seconds": {"total_to_base": time.perf_counter() - started}, "qrels_used_for_ranking": False}
    (retrieval_dir / "report.json").write_text(json.dumps(base_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cohere_dir = args.output_dir / "cohere"
    cohere_report = run_cohere(rows, queries, pages, qrels, cohere_dir, args.cohere_workers)
    report = {"base": base_report, "cohere": cohere_report, "total_seconds": time.perf_counter() - started, "output_dir": str(args.output_dir.resolve())}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
