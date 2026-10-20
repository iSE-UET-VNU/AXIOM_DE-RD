"""QA only from KDL page text after two top-20 light candidate gates.

Light retrieval is used solely to constrain accurate retrieval.  It provides
no context text and its scores are never shown to the generator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any
import argparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.evaluate_merged_baseline_vs_kdl_qa import (
    GENERATOR, JUDGE, TOP_K, WORKERS, evaluate_one, query_map, read_jsonl,
    report, write_json, write_jsonl,
)
from src.utils.env import load_dotenv_file

BUNDLE = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803"
OUT = ROOT / "data/work/ohr_replacement_20260920/e2e_kdl_page_cascades_top10"
ARMS = {
    "light_hybrid_top20_then_kdl_pages": ROOT / "data/work/ohr_replacement_20260920/retrieval/light_hybrid_top20_kdl_colvec/per_query.jsonl",
    "light_hybrid_cohere_top20_then_kdl_pages": ROOT / "data/work/ohr_replacement_20260920/retrieval/cohere_top20_kdl_text/per_query.jsonl",
}


def kdl_pages() -> dict[str, str]:
    pages = {}
    for row in read_jsonl(BUNDLE / "kdl/kdl_pages.jsonl"):
        page = str(row["page_id"])
        # Preserve a selected empty KDL page as empty rather than falling
        # back to light text; ``generate`` will omit it from its fixed budget.
        pages[page] = str(row.get("text") or "")
    if len(pages) != 5091:
        raise RuntimeError(f"Expected 5,091 KDL pages, got {len(pages)}")
    return pages


def contexts(path: Path, page_text: dict[str, str], qids: set[str]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for row in read_jsonl(path):
        qid = str(row["query_id"])
        scores = {str(page): float(value) for page, value in row["kdl_text_scores"].items()}
        ranked = sorted(scores, key=lambda page: (-scores[page], page))[:TOP_K]
        if len(ranked) != TOP_K or any(page not in page_text for page in ranked):
            raise RuntimeError(f"Invalid KDL top-{TOP_K} pages for {qid}")
        result[qid] = [
            {"chunk_id": f"kdl_page:{page}", "page_id": page, "rank": rank, "score": scores[page], "text": page_text[page]}
            for rank, page in enumerate(ranked, 1)
        ]
    if set(result) != qids:
        raise RuntimeError(f"QID mismatch for {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    load_dotenv_file(ROOT)
    queries = query_map()
    page_text = kdl_pages()
    for arm, run_path in ARMS.items():
        context = contexts(run_path, page_text, set(queries))
        write_jsonl(OUT / "contexts" / f"{arm}.jsonl", [{"query_id": qid, "pages": context[qid]} for qid in queries])
        rows_path = OUT / "qa" / f"{arm}.jsonl"
        existing = {str(row["query_id"]): row for row in read_jsonl(rows_path)} if rows_path.exists() else {}
        # Provider-error rows remain auditable; a later explicit retry can
        # replace them without repeating successful calls.
        pending = [qid for qid in queries if qid not in existing]
        print(f"{arm}: {len(pending)} pending / {len(queries)}", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(evaluate_one, arm, qid, queries[qid], context[qid]): qid for qid in pending}
            for number, future in enumerate(as_completed(futures), 1):
                existing[futures[future]] = future.result()
                write_jsonl(rows_path, [existing[qid] for qid in queries if qid in existing])
                if number % 10 == 0 or number == len(pending):
                    print(f"{arm}: {number}/{len(pending)}", flush=True)
        summary = report(arm, [existing[qid] for qid in queries])
        summary.update({"context_representation": "KDL page text only; light candidate pages/text/scores excluded from QA", "accurate_retrieval_run": str(run_path.relative_to(ROOT))})
        write_json(OUT / "qa" / f"{arm}.report.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
