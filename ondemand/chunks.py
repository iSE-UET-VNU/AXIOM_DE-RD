import hashlib
import json
from time import perf_counter

import numpy as np

from .bench import BENCH, fingerprint, load_jsonl, queries as load_queries, work
from .openrouter import embed
from .text import windows
from .timing import Timings

MODEL, N_WORDS, OVERLAP, SLICE = "openai/text-embedding-3-small", 512, 128, 256


def page_chunks(text):
    return [text[s:e] for s, e in windows(text, N_WORDS, OVERLAP) if text[s:e].strip()]


def run(bench=BENCH):
    pages_path = work("kdl", bench=bench) / "kdl_pages.jsonl"
    out = work("embeddings", "kdl", bench=bench)
    timings = Timings(out / "timings.jsonl", stage_group="embed")
    owners, texts = [], []
    for page in load_jsonl(pages_path):
        for chunk in page_chunks(page["text"]):
            owners.append(page["page_id"])
            texts.append(chunk)
    cache = work("embedding_cache", "te3s", bench=bench)
    vectors = np.zeros((len(texts), 1536), dtype=np.float32)
    started = perf_counter()
    for s in range(0, len(texts), SLICE):
        t = perf_counter()
        vectors[s:s + SLICE] = embed(texts[s:s + SLICE], MODEL, cache)
        timings.record("embed_chunks", "chunk", len(texts[s:s + SLICE]), perf_counter() - t)
    queries = list(load_queries(bench).values())
    t = perf_counter()
    qvecs = embed([q["query"] for q in queries], MODEL, cache)
    timings.record("embed_queries", "query", len(queries), perf_counter() - t)
    np.save(out / "chunk_vectors.npy", vectors)
    np.save(out / "query_vectors.npy", qvecs)
    (out / "chunk_owners.json").write_text(json.dumps(owners), encoding="utf-8")
    (out / "chunk_texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    (out / "query_ids.json").write_text(json.dumps([q["query_id"] for q in queries]), encoding="utf-8")
    (out / "embed_meta.json").write_text(json.dumps({
        "model": MODEL, "dim": 1536, "chunker": "fixed_overlap", "n_words": N_WORDS, "overlap": OVERLAP,
        "n_chunks": len(texts), "n_queries": len(queries),
        "pages_sha256": hashlib.sha256(pages_path.read_bytes()).hexdigest(), "bundle_fingerprint": fingerprint(bench),
        "wall_seconds": round(perf_counter() - started, 2)}, indent=1), encoding="utf-8")
    return out
