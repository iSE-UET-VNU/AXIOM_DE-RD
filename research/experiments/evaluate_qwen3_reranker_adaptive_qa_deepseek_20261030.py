"""Run DeepSeek QA on the Qwen3-reranker + OOF adaptive-K arm."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/adaptive/llm/scripts"))
sys.path.insert(0, str(ROOT / "research/experiments"))
import run_adaptive_accurate_qa as qa  # noqa: E402
import evaluate_reranker_adaptive_qa_20261026 as rerank_qa  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402

OUT = ROOT / "data/work/reranker_adaptive_k_qwen3_deepseek_qa_20261030"
ANSWER_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_MODEL = "openai/gpt-4o-mini"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    load_dotenv_file(ROOT)
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is unset")

    qa.OUT = OUT
    qa.ANSWER_MODEL = ANSWER_MODEL
    qa.ANSWER_REASONING = {"effort": "none"}
    qa.MAX_ANSWER_TOKENS = 1024
    qa.JUDGE_MODEL = JUDGE_MODEL

    rows, qa_rows = rerank_qa.build_rows(["qwen3_8b_xgb"])
    if len(rows) != 220:
        raise RuntimeError(f"Expected 220 qids; got {len(rows)}")
    for row in rows:
        # Map the reranker-QA runner's names onto the shared adaptive-QA metric helper.
        row["candidate_prefix_recall"] = row["prefix_recall"]
        row["qa_context_gold_recall"] = row["accurate_recall_at_10"]
        row["accurate_rerank_seconds"] = 0.0  # cached EVIE matrix; inference is excluded
    jobs: dict[str, dict] = {}
    for row in rows:
        key = row["answer_prompt_hash"]
        jobs.setdefault(key, {"hash": key, "qid": row["qid"], "prompt": row["answer_prompt"],
                              "ctx": row["ctx"], "legacy_key": qa.answer_cache_key(row["qid"], row["ctx"])})
    OUT.mkdir(parents=True, exist_ok=True)
    write_json = lambda path, obj: path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(OUT / "manifest.json", {
        "queries": 220,
        "arm": "Qwen3 Reranker top-100 ranking -> XGBoost OOF adaptive K -> EVIE rerank within prefix -> top-10 pages",
        "mean_k_expected": 57.06363636363636,
        "answer_model": ANSWER_MODEL,
        "answer_reasoning": {"effort": "none"},
        "answer_max_tokens": 1024,
        "judge_model": JUDGE_MODEL,
        "qa_context": "one first-512-word accurate KDL excerpt per EVIE-ranked page, up to 10 pages; same adaptive QA prompt",
        "qrels_usage": "evaluation only; not passed to generator",
        "workers": args.workers,
    })

    started = perf_counter()
    answers = {}
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(qa.cached_or_call_answer, job, OUT): key for key, job in jobs.items()}
        for i, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            answers[key] = future.result()
            if i % 50 == 0 or i == len(futures):
                print(f"answers {i}/{len(futures)}", flush=True)
    for row in rows:
        result = answers[row["answer_prompt_hash"]]
        row.update(answer=result.get("answer", ""), answer_cache=result.get("cache"),
                   answer_error=result.get("error"), answer_latency_seconds=result.get("latency_seconds"),
                   answer_cost_usd=result.get("cost_usd"), answer_prompt_tokens=result.get("prompt_tokens"),
                   answer_completion_tokens=result.get("completion_tokens"))
    errors = [r for r in rows if r.get("answer_error") or not r.get("answer")]
    if errors:
        write_rows(OUT / "per_query_partial.jsonl", rows)
        raise RuntimeError(f"{len(errors)} answer failures. Partial results saved; rerun to resume after checking errors.")

    qa.run_judges(rows, {"qa_rows": qa_rows}, OUT, args.workers)
    write_rows(OUT / "per_query.jsonl", rows)
    sources = {}
    for source in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == source]
        sources[source] = qa.metric_summary(group)
    report = {
        "protocol": {"queries": 220, "answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL,
                     "arm": "qwen3_8b_xgb", "qa_context": "top-10 EVIE pages; first 512 KDL words/page",
                     "adaptive_and_rerank_metrics": "OOF adaptive K trained on Qwen3 reranked candidate ranking; EVIE reranks selected prefix"},
        "wall_seconds": perf_counter() - started,
        "all": qa.metric_summary(rows),
        "by_source": sources,
        "api_calls": qa.unique_call_summary(OUT, {r["answer_prompt_hash"] for r in rows}, {r["judge_prompt_hash"] for r in rows}),
    }
    write_json(OUT / "report.json", report)
    allm = report["all"]
    md = ["# QA: Qwen3 reranker + XGBoost adaptive K, DeepSeek V4 Flash", "",
          "| Measure | Result |", "|---|---:|",
          f"| Queries | {allm['n']} |", f"| Mean K | {allm['mean_k']:.2f} |",
          f"| Candidate-prefix recall | {allm['mean_candidate_prefix_recall']*100:.2f}% |",
          f"| EVIE Recall@10 | {allm['macro_recall_at_10']*100:.2f}% |",
          f"| EVIE nDCG@10 | {allm['macro_ndcg_at_10']*100:.2f} |",
          f"| QA Correct | {allm['correct_only']*100:.2f}% |",
          f"| QA Correct + Partial | {allm['correct_plus_partial']*100:.2f}% |",
          f"| Mean pages supplied to QA | {allm['mean_accurate_pages_to_qa']:.2f} |",
          f"| Answer failures / judge failures | {allm['answer_failures']} / {allm['judge_failures']} |",
          f"| Live API cost | ${report['api_calls']['total_unique_reported_cost_usd']:.4f} |",
          f"| Wall time ({args.workers} workers) | {report['wall_seconds']:.1f}s |", "",
          "## By source", "", "| Dataset | n | Mean K | Prefix recall | EVIE Recall@10 | nDCG@10 | QA Correct | Correct + Partial |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for source, m in sources.items():
        md.append(f"| {source} | {m['n']} | {m['mean_k']:.2f} | {m['mean_candidate_prefix_recall']*100:.2f}% | {m['macro_recall_at_10']*100:.2f}% | {m['macro_ndcg_at_10']*100:.2f} | {m['correct_only']*100:.2f}% | {m['correct_plus_partial']*100:.2f}% |")
    md += ["", "The existing Qwen-generator run for the same arm is in `data/work/reranker_adaptive_k_qa_20261026/report.json`; this DeepSeek run uses the same retrieval ranking, predicted K, EVIE top-10 pages, 512-word/page context and judge policy. The XGBoost K predictions are frozen OOF predictions. QA accuracy is reference-answer agreement, not a grounding check.", ""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
