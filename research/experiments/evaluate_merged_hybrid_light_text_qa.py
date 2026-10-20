"""QA directly from the top-10 light-hybrid pages and cached light text.

This is deliberately a no-accurate-retrieval arm: it does not call KDL, does
not rechunk, and does not embed or rerank.  It only feeds the already-ranked
top-ten pages from the 52.37% hybrid run into the repository's standard
DeepSeek V4 generator and GPT-4o-mini judge.  Completed query rows are a
checkpoint; provider-error rows are retried only with ``--retry-errors``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_merged_baseline_vs_kdl_qa import (  # noqa: E402
    GENERATOR, JUDGE, TOP_K, evaluate_one, query_map, read_jsonl, report, write_json, write_jsonl,
)
from research.experiments.run_ohr_replacement_all_arms import load_page_corpora  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402

RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/hybrid_bm25_embeddings.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/e2e_light_hybrid_direct_top10"
ARM = "light_hybrid_language_aware_top10_pages"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args()
    load_dotenv_file(ROOT)

    queries = query_map()
    rows = {str(row["query_id"]): row for row in read_jsonl(args.run)}
    if set(rows) != set(queries) or len(rows) != 220:
        raise RuntimeError(f"Expected exact matching 220 qids, got run={len(rows)}, queries={len(queries)}")
    page_texts, _ = load_page_corpora()
    texts = page_texts["embedded_image_ocr_bm25"]
    contexts: dict[str, list[dict[str, Any]]] = {}
    for qid, row in rows.items():
        pages = list(row.get("returned_pages") or [])[:TOP_K]
        scores = list(row.get("returned_scores") or [])[:TOP_K]
        if len(pages) != TOP_K or len(set(pages)) != TOP_K or any(page not in texts for page in pages):
            raise RuntimeError(f"{qid}: invalid light top-{TOP_K} context")
        contexts[qid] = [
            {"chunk_id": f"light_page:{page}", "page_id": page, "rank": rank,
             "score": float(scores[rank - 1]) if rank <= len(scores) else 0.0, "text": texts[page]}
            for rank, page in enumerate(pages, 1)
        ]

    context_path = args.output_dir / "contexts" / f"{ARM}.jsonl"
    write_jsonl(context_path, [{"query_id": qid, "pages": contexts[qid]} for qid in queries])
    manifest = {
        "arm": ARM, "queries": len(queries), "pages_per_query": TOP_K,
        "retrieval_run": str(args.run.resolve()),
        "page_text_source": "embedded_image_ocr_bm25 (PDF-inspector + selective full-page Tesseract + embedded-image OCR)",
        "accurate_retrieval": "disabled", "generator": GENERATOR, "judge": JUDGE,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    if args.materialize_only:
        print(json.dumps({**manifest, "context_path": str(context_path.resolve())}, ensure_ascii=False, indent=2))
        return 0

    out = args.output_dir / "qa" / f"{ARM}.jsonl"
    existing = {str(row["query_id"]): row for row in read_jsonl(out)} if out.is_file() else {}
    pending = [
        qid for qid in queries
        if qid not in existing or (args.retry_errors and (existing[qid].get("generation_error") or existing[qid].get("judge_error")))
    ]
    print(f"{ARM}: pending={len(pending)}/{len(queries)}, workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(evaluate_one, ARM, qid, queries[qid], contexts[qid]): qid for qid in pending}
        for number, future in enumerate(as_completed(futures), 1):
            qid = futures[future]
            existing[qid] = future.result()
            write_jsonl(out, [existing[item] for item in queries if item in existing])
            if number % 10 == 0 or number == len(pending):
                print(f"{ARM}: {number}/{len(pending)}", flush=True)
    summary = report(ARM, [existing[qid] for qid in queries])
    summary.update(manifest)
    write_json(args.output_dir / "qa" / f"{ARM}.report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
