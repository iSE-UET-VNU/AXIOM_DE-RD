"""End-to-end QA for the canonical light baseline and its KDL cascade.

The retrieval comparison is page-level.  The light arm passes its top ten
page texts to the generator; the KDL arm preserves that page ranking but
passes the highest-scoring KDL chunk for each selected page.  Both arms use
the frozen ViDoRe prompt, 12k-character packing budget, generator and judge.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any
import argparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.generate import ContextChunk, generate
from src.evaluation.benchmarks.vidore_v3_judge import ViDoreStyle, ViDoreVerdict, judge_answer, score
from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index
from src.utils.env import load_dotenv_file

BUNDLE = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803"
QUERIES = ROOT / "data/raw/0.1. BENCHMARK/queries.jsonl"
BASELINE_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
KDL_RUN = ROOT / "data/work/ohr_replacement_20260920/retrieval/baseline_top20_kdl_colvec/per_query.jsonl"
OUT = ROOT / "data/work/ohr_replacement_20260920/e2e_baseline_vs_kdl_top10"
GENERATOR = "deepseek/deepseek-v4-flash"
JUDGE = "openai/gpt-4o-mini"
TOP_K = 10
WORKERS = 8


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def source(qid: str) -> str:
    return qid.split("::", 1)[0]


def minmax(values: dict[str, float]) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    return {key: 1.0 if high - low <= 1e-12 else (value - low) / (high - low) for key, value in values.items()}


def query_map() -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(QUERIES):
        answers = [str(value) for value in row.get("answers") or []]
        if not answers:
            raise ValueError(f"Query {row.get('query_id')} has no reference answer")
        result[str(row["query_id"])] = {"query": str(row["query"]), "answers": answers, "source": str(row.get("source") or source(str(row["query_id"]))) }
    return result


def baseline_page_text() -> dict[str, str]:
    # Reuse the exact canonical page-text loader rather than reconstruct text.
    from research.experiments.run_ohr_replacement_all_arms import load_page_corpora
    texts, _ = load_page_corpora()
    return {str(page): str(text) for page, text in texts["baseline_bm25"].items()}


def kdl_contexts(baseline: dict[str, dict[str, Any]], queries: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return one winning KDL chunk for every final top-ten KDL page."""
    expected = {str(row["query_id"]): row for row in read_jsonl(KDL_RUN)}
    emb = BUNDLE / "embeddings/kdl"
    owners = np.asarray(json.loads((emb / "chunk_owners.json").read_text(encoding="utf-8")))
    texts = json.loads((emb / "chunk_texts.json").read_text(encoding="utf-8"))
    vectors = np.load(emb / "chunk_vectors.npy").astype(np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    qids = json.loads((emb / "query_ids.json").read_text(encoding="utf-8"))
    qvectors = np.load(emb / "query_vectors.npy").astype(np.float32)
    qvectors /= np.clip(np.linalg.norm(qvectors, axis=1, keepdims=True), 1e-12, None)
    qvector = {qid: qvectors[index] for index, qid in enumerate(qids)}
    union = {page for row in baseline.values() for page in row["returned_pages"][:20]}
    keep = np.asarray([index for index, page in enumerate(owners) if page in union], dtype=np.int32)
    local_owners, local_vectors = owners[keep], vectors[keep]
    index = BM25Index(analyzer_name="plain").build([
        {"chunk_id": f"c{position}", "doc_id": str(local_owners[position]), "text": texts[int(original)]}
        for position, original in enumerate(keep)
    ])
    positions: dict[str, list[int]] = defaultdict(list)
    for position, page in enumerate(local_owners):
        positions[str(page)].append(position)
    result: dict[str, list[dict[str, Any]]] = {}
    for qid, row in baseline.items():
        candidate_pages = row["returned_pages"][:20]
        allowed = {position for page in candidate_pages for position in positions[page]}
        lexical = index.search(queries[qid]["query"], 100, allowed)
        dense_positions = np.asarray(sorted(allowed), dtype=np.int32)
        dense_scores = local_vectors[dense_positions] @ qvector[qid]
        dense_selected = dense_positions[np.argsort(-dense_scores)[:min(100, len(dense_positions))]]
        dense = [(int(position), float(local_vectors[position] @ qvector[qid])) for position in dense_selected]
        fused = alpha_fuse(lexical, dense, alpha=0.7, top_k=100)
        winner: dict[str, tuple[int, float]] = {}
        for position, chunk_score in fused:
            page = str(local_owners[position])
            if page not in winner or chunk_score > winner[page][1]:
                winner[page] = (position, float(chunk_score))
        page_scores = {page: winner.get(page, (-1, 0.0))[1] for page in candidate_pages}
        ranked_pages = sorted(page_scores, key=lambda page: (-page_scores[page], page))[:TOP_K]
        expected_pages = sorted(expected[qid]["kdl_text_scores"], key=lambda page: (-expected[qid]["kdl_text_scores"][page], page))[:TOP_K]
        if ranked_pages != expected_pages:
            raise RuntimeError(f"KDL replay differs from stored page rank for {qid}")
        chunks = []
        for rank, page in enumerate(ranked_pages, 1):
            position, chunk_score = winner[page]
            if position < 0:
                raise RuntimeError(f"No KDL chunk for selected page {page}")
            chunks.append({"chunk_id": f"kdl:{int(keep[position])}", "page_id": page, "rank": rank, "score": chunk_score, "text": texts[int(keep[position])]})
        result[qid] = chunks
    return result


def judge_all(query: dict[str, Any], answer: str) -> tuple[ViDoreVerdict, list[dict[str, Any]]]:
    labels = {"Incorrect": 0, "Partially Correct": 1, "Correct": 2}
    alternatives, best = [], None
    for gold in query["answers"]:
        verdict = judge_answer("", query["query"], gold, answer, model=JUDGE, generator_model=GENERATOR)
        alternatives.append({"gold_answer": gold, "judgment": verdict.judgment, "error": verdict.error})
        if best is None or labels[verdict.judgment] > labels[best.judgment]:
            best = verdict
    assert best is not None
    return best, alternatives


def evaluate_one(arm: str, qid: str, query: dict[str, Any], context: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = [ContextChunk(item["chunk_id"], item["page_id"], item["text"], float(item.get("score", 0.0))) for item in context]
    started = time.perf_counter()
    generated = generate(qid, query["query"], chunks, model=GENERATOR, render_prompt=ViDoreStyle().render_prompt)
    generation_seconds = time.perf_counter() - started
    judged_at = time.perf_counter()
    if generated.error:
        verdict, alternatives = ViDoreVerdict(qid, "Incorrect", error=generated.error), []
    else:
        verdict, alternatives = judge_all(query, generated.answer)
    return {"arm": arm, "query_id": qid, "source": query["source"], "answer": generated.answer, "generation_error": generated.error, "judgment": verdict.judgment, "judge_error": verdict.error, "judge_alternatives": alternatives, "correct": verdict.judgment == "Correct" and not verdict.error, "credited": verdict.judgment in ("Correct", "Partially Correct") and not verdict.error, "context_chunks": [{key: item[key] for key in ("chunk_id", "page_id", "rank", "score")} for item in context], "chunks_used": generated.chunks_used, "chars_used": generated.chars_used, "generation_seconds": round(generation_seconds, 3), "judge_seconds": round(time.perf_counter() - judged_at, 3)}


def report(arm: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("generation_error") and not row.get("judge_error")]
    verdicts = [ViDoreVerdict(row["query_id"], row["judgment"]) for row in valid]
    metrics = score(verdicts) if verdicts else {"correct_only": 0.0, "correct_plus_partial": 0.0, "n": 0}
    by_source = {}
    for group in ("all", "vidore_physics", "vidore_industrial", "ohrbench"):
        subset = valid if group == "all" else [row for row in valid if row["source"] == group]
        subset_score = score([ViDoreVerdict(row["query_id"], row["judgment"]) for row in subset]) if subset else {"correct_only": 0.0, "correct_plus_partial": 0.0}
        by_source[group] = {"queries": len(subset), "correct_only": round(100 * subset_score["correct_only"], 2), "correct_plus_partial": round(100 * subset_score["correct_plus_partial"], 2)}
    return {"arm": arm, "queries": len(rows), "valid": len(valid), "errors": len(rows) - len(valid), "generator": GENERATOR, "judge": JUDGE, "correct_only": round(100 * metrics["correct_only"], 2), "correct_plus_partial": round(100 * metrics["correct_plus_partial"], 2), "by_source": by_source, "mean_chunks": round(sum(row["chunks_used"] for row in rows) / max(1, len(rows)), 2), "mean_context_chars": round(sum(row["chars_used"] for row in rows) / max(1, len(rows))), "generation_seconds_sum": round(sum(row["generation_seconds"] for row in rows), 2), "judge_seconds_sum": round(sum(row["judge_seconds"] for row in rows), 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retry-errors", action="store_true", help="Retry provider-error checkpoints as well as missing rows.")
    args = parser.parse_args()
    load_dotenv_file(ROOT)
    queries = query_map()
    baseline = {str(row["query_id"]): row for row in read_jsonl(BASELINE_RUN)}
    if set(baseline) != set(queries) or len(baseline) != 220:
        raise RuntimeError("Expected matching 220-query merged inputs")
    pages = baseline_page_text()
    light_contexts = {qid: [{"chunk_id": f"page:{page}", "page_id": page, "rank": rank, "score": 0.0, "text": pages[page]} for rank, page in enumerate(row["returned_pages"][:TOP_K], 1)] for qid, row in baseline.items()}
    accurate_contexts = kdl_contexts(baseline, queries)
    write_jsonl(OUT / "contexts" / "light_top10_pages.jsonl", [{"query_id": qid, "chunks": value} for qid, value in light_contexts.items()])
    write_jsonl(OUT / "contexts" / "kdl_top10_winning_chunks.jsonl", [{"query_id": qid, "chunks": value} for qid, value in accurate_contexts.items()])
    for arm, contexts in (("light_baseline_top10_pages", light_contexts), ("kdl_cascade_top10_winning_chunks", accurate_contexts)):
        rows_path = OUT / "qa" / f"{arm}.jsonl"
        existing = {str(row["query_id"]): row for row in read_jsonl(rows_path)} if rows_path.exists() else {}
        pending = [
            qid for qid in queries
            if qid not in existing or (args.retry_errors and (existing[qid].get("generation_error") or existing[qid].get("judge_error")))
        ]
        print(f"{arm}: {len(pending)} pending / {len(queries)}", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(evaluate_one, arm, qid, queries[qid], contexts[qid]): qid for qid in pending}
            for number, future in enumerate(as_completed(futures), 1):
                existing[futures[future]] = future.result()
                write_jsonl(rows_path, [existing[qid] for qid in queries if qid in existing])
                if number % 10 == 0 or number == len(pending):
                    print(f"{arm}: {number}/{len(pending)}", flush=True)
        summary = report(arm, [existing[qid] for qid in queries])
        write_json(OUT / "qa" / f"{arm}.report.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
