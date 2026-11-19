"""How this bundle's inputs/ and cache/ were assembled (run once on the author's machine; kept for provenance).

Sources (all under D:\\AXIOM_DE-RD):
- pages (read in place, not copied):   data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/pages_ppocrv5.jsonl
- queries: data/raw/0.1. BENCHMARK/queries.jsonl (220 rows, includes reference answers for later QA)
- documents: data/raw/0.1. BENCHMARK/documents.jsonl (file manifest; the PDFs themselves stay in data/raw/0.1. BENCHMARK/data)
- qrels:   Physics + Industrial page qrels data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/
           bundle_debug_light-retrieval/input/qrels_page_level.jsonl (graded 2/1) + OHR data/work/ohr_replacement_20260920/
           inputs/qrels.jsonl (relevance 1); merged as max grade per (query, page); 220 queries.
- vectors (page vectors read in place, query vectors copied): data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/light_v2/dense_qwen8b_ppocr_{pages,queries}.npy
           and _meta.json (qwen/qwen3-embedding-8b via OpenRouter).
- LLM decomposition cache: cache/llm_decomposition/DeepInfra/ was produced by this bundle itself
           (python run_all.py --live, Qwen3-32B pinned to DeepInfra, reasoning off, 2026-11-14); not copied from the repo.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\AXIOM_DE-RD")
B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B))
from data_discovery import common as C  # noqa: E402

PAGES = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/pages_ppocrv5.jsonl"
BENCH = ROOT / "data/raw/0.1. BENCHMARK"
QRELS = [ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl",
         ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"]
LV2 = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/light_v2"
KEEP = ("page_id", "doc_id", "source", "source_path", "page_index", "page_number", "text", "text_source", "pdf_type")


def main():
    # The page text (PAGES) and page vectors (LV2/dense_qwen8b_ppocr_pages.npy) are NOT copied (size); the bundle
    # reads them from these repo paths (common.PAGES_PATH / common.PAGE_VECTORS). Only their page-id order is kept.
    pages = [json.loads(l) for l in open(PAGES, encoding="utf-8") if l.strip()]
    shutil.copy(BENCH / "queries.jsonl", C.INPUTS / "queries.jsonl")
    shutil.copy(BENCH / "documents.jsonl", C.INPUTS / "documents.jsonl")
    page_ids = {p["page_id"] for p in pages}
    grade = {}
    for path in QRELS:
        for r in map(json.loads, open(path, encoding="utf-8")):
            q = r["query_id"]
            if not q.startswith(C.SOURCES):
                continue
            pid = r.get("page_id") or f"{r['doc_id']}#page={r['page_index']}"
            if pid in page_ids:
                grade[(q, pid)] = max(grade.get((q, pid), 0), int(r.get("relevance") or 1))
    C.write_jsonl(C.INPUTS / "qrels.jsonl", [{"query_id": q, "page_id": p, "relevance": g} for (q, p), g in sorted(grade.items())])

    meta = json.loads((LV2 / "dense_qwen8b_ppocr_meta.json").read_text())
    assert meta["units"] == [p["page_id"] for p in pages], "page vectors not in page order"
    shutil.copy(LV2 / "dense_qwen8b_ppocr_queries.npy", C.QUERY_VECTORS)
    C.VECTOR_META.write_text(json.dumps({
        "model": meta["model"], "page_ids": meta["units"], "qids": meta["qids"], "dim": 4096,
        "page_text": "real_text(page text) (image placeholders removed), clipped to the first 6,000 cl100k tokens; "
                     "empty pages get a zero vector (154 pages)",
        "query_text": C.QWEN_PREFIX + "<question>",
        "timing": {"page_embedding_seconds_total": meta.get("page_seconds"), "query_embedding_seconds_each": meta.get("query_seconds_each")},
    }), encoding="utf-8")

    print(f"pages {len(pages)} (not copied) | queries 220 | qrels {len(grade)}")

if __name__ == "__main__":
    main()
