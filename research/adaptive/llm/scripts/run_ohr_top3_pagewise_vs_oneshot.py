"""Compare pagewise top-3 and one-shot top-3 LLM stopping on current OHR ranking.

The LLM sees no qrels, gold page IDs, reference answers, target K, or XGBoost K.
If it does not produce an accepted STOP by rank 3, deployment falls back to
the nested-OOF calibrated XGBoost K from the frozen text-embedding-3-small run.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_adaptive_llm_stopping import (  # noqa: E402
    Budget,
    GENERAL_SYSTEM,
    OpenRouterClient,
    accepted_stop,
    next_state,
    normalize_result,
    parse_json_object,
    user_prompt,
)
from src.utils.env import load_dotenv_file  # noqa: E402


MODEL = "deepseek/deepseek-v4-flash"
RUN_ROOT = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100"
HANDOFF = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803"
GATE_PATH = HANDOFF / "light_prep/gate_ppocrv5_ungated.json"
PAGES_PATH = HANDOFF / "light_prep/pages_ppocrv5.jsonl"
QUERIES_PATH = ROOT / "data/raw/0.1. BENCHMARK/queries.jsonl"
DEFAULT_OUT = ROOT / "research/adaptive/llm/results/ohr_top3_pagewise_vs_oneshot_te3s_20261103"
ARMS = ("pagewise_1_to_3", "oneshot_top3")
SOURCE_COUNTS = {"ohrbench": 65, "vidore_industrial": 75, "vidore_physics": 80}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def load_inputs(sources: tuple[str, ...] = ("ohrbench",)) -> tuple[list[dict[str, Any]], dict[str, str]]:
    queries = {str(row["query_id"]): str(row["query"]) for row in read_jsonl(QUERIES_PATH)}
    pages = {str(row["page_id"]): str(row.get("text") or "") for row in read_jsonl(PAGES_PATH)}
    rankings = {str(row["qid"]): row for row in read_jsonl(RUN_ROOT / "retrieval_scores.jsonl")}
    predictions = {str(row["qid"]): row for row in read_jsonl(RUN_ROOT / "oof_predictions.jsonl")}
    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))

    rows = []
    for qid in sorted(qid for qid in rankings if qid.split("::", 1)[0] in sources):
        ranked = [str(page_id) for page_id in rankings[qid]["ranked_pages"]]
        if len(ranked) != 100:
            raise ValueError(f"{qid}: expected 100 ranked pages, got {len(ranked)}")
        missing = [page_id for page_id in ranked[:3] if page_id not in pages]
        if missing:
            raise KeyError(f"{qid}: missing top-3 page text: {missing}")
        gold_ids = list(gate["gold_rank"][qid].keys())
        gold_count = int(predictions[qid]["gold_count"])
        if len(gold_ids) != gold_count:
            raise ValueError(f"{qid}: gold count mismatch {len(gold_ids)} != {gold_count}")
        rows.append({
            "qid": qid,
            "source": qid.split("::", 1)[0],
            "query": queries[qid],
            "ranked_pages": ranked,
            "scores": [float(value) for value in rankings[qid]["scores"]],
            "gold_ids": gold_ids,
            "gold_count": gold_count,
            "target_k": int(predictions[qid]["target_k"]),
            "xgb_k": int(predictions[qid]["prediction_k_oof"]),
        })
    expected = sum(SOURCE_COUNTS[source] for source in sources)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} queries for {sources}, got {len(rows)}")
    return rows, pages


def block(row: Mapping[str, Any], pages: Mapping[str, str], start: int, end: int) -> str:
    chunks = []
    for index in range(start, end):
        page_id = str(row["ranked_pages"][index])
        chunks.append(
            f'<page rank="{index + 1}" page_id="{page_id}">\n'
            f'{pages[page_id]}\n</page>'
        )
    return "\n\n".join(chunks)


def recall_at(row: Mapping[str, Any], k: int) -> float:
    gold = set(row["gold_ids"])
    hits = sum(page_id in gold for page_id in row["ranked_pages"][:k])
    return hits / int(row["gold_count"])


def call_controller(
    *,
    client: OpenRouterClient,
    row: Mapping[str, Any],
    pages: Mapping[str, str],
    arm: str,
    start: int,
    end: int,
    state: Mapping[str, Any],
    seen: set[str],
) -> dict[str, Any]:
    page_text = block(row, pages, start, end)
    prompt = user_prompt("general", str(row["query"]), state, page_text, end)
    response = client.complete(
        system=GENERAL_SYSTEM,
        user=prompt,
        session_id=f"ohr-top3-{arm}-{row['qid']}",
    )
    parse_error = ""
    try:
        result = normalize_result(parse_json_object(str(response["text"])), "general")
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        parse_error = str(error)
        result = normalize_result({}, "general")
    stop, rejection_reasons = accepted_stop(result, seen, "general")
    return {
        "rank_start": start + 1,
        "rank_end": end,
        "new_page_ids": list(row["ranked_pages"][start:end]),
        "parse_error": parse_error,
        "accepted_stop": stop,
        "stop_rejection_reasons": rejection_reasons,
        "result": result,
        "raw_response": str(response.get("text") or ""),
        "request": {
            key: response.get(key)
            for key in (
                "request_hash", "cache_reused", "requested_model", "resolved_model",
                "provider", "prompt_tokens", "completion_tokens", "cached_tokens",
                "cost_usd", "cost_is_estimated", "ttft_seconds", "total_seconds",
            )
        },
    }


def run_one(
    *,
    arm: str,
    row: Mapping[str, Any],
    pages: Mapping[str, str],
    client: OpenRouterClient,
    oneshot_k: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    state: dict[str, Any] = {}
    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    schedule = (
        [(0, 1), (1, 2), (2, 3)]
        if arm == "pagewise_1_to_3"
        else [(0, oneshot_k)]
    )
    error = ""
    try:
        for start, end in schedule:
            seen.update(str(value) for value in row["ranked_pages"][start:end])
            step = call_controller(
                client=client,
                row=row,
                pages=pages,
                arm=arm,
                start=start,
                end=end,
                state=state,
                seen=seen,
            )
            steps.append(step)
            state = next_state(step["result"], "general")
            if step["accepted_stop"]:
                break
    except Exception as exc:  # fallback remains safe and the error is persisted
        error = f"{type(exc).__name__}: {exc}"

    accepted = bool(steps and steps[-1]["accepted_stop"])
    stop_k = int(steps[-1]["rank_end"]) if accepted else None
    deployed_k = stop_k if stop_k is not None else int(row["xgb_k"])
    stop_recall = recall_at(row, stop_k) if stop_k is not None else None
    baseline_recall = recall_at(row, int(row["xgb_k"]))
    deployed_recall = recall_at(row, deployed_k)
    return {
        "arm": arm,
        "qid": row["qid"],
        "source": row["source"],
        "query": row["query"],
        "target_k": row["target_k"],
        "xgb_k": row["xgb_k"],
        "gold_count": row["gold_count"],
        "gold_ranks": [
            rank
            for rank, page_id in enumerate(row["ranked_pages"], 1)
            if page_id in set(row["gold_ids"])
        ],
        "oracle_prefix_k": oneshot_k,
        "oracle_prefix_safe": recall_at(row, oneshot_k) == 1.0,
        "accepted_stop": accepted,
        "stop_k": stop_k,
        "stop_recall": stop_recall,
        "safe_vs_xgb": accepted and stop_recall == baseline_recall,
        "full_gold_at_stop": accepted and stop_recall == 1.0,
        "deployed_k": deployed_k,
        "baseline_recall": baseline_recall,
        "deployed_recall": deployed_recall,
        "calls": len(steps),
        "error": error,
        "wall_seconds": time.perf_counter() - started,
        "steps": steps,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [step for row in rows for step in row["steps"]]
    requests = [step["request"] for step in calls]
    stops = [row for row in rows if row["accepted_stop"]]
    oracle_safe = [row for row in rows if row["oracle_prefix_safe"]]
    return {
        "queries": len(rows),
        "mean_deployed_k": statistics.fmean(row["deployed_k"] for row in rows),
        "macro_recall": statistics.fmean(row["deployed_recall"] for row in rows),
        "baseline_mean_xgb_k": statistics.fmean(row["xgb_k"] for row in rows),
        "baseline_macro_recall": statistics.fmean(row["baseline_recall"] for row in rows),
        "oracle_prefix_k": rows[0]["oracle_prefix_k"] if rows else None,
        "oracle_prefix_safe_queries": len(oracle_safe),
        "accepted_stops": len(stops),
        "safe_stops_vs_xgb": sum(row["safe_vs_xgb"] for row in stops),
        "full_gold_stops": sum(row["full_gold_at_stop"] for row in stops),
        "unsafe_stops_vs_xgb": sum(not row["safe_vs_xgb"] for row in stops),
        "safe_stop_precision": (
            sum(row["safe_vs_xgb"] for row in stops) / len(stops) if stops else 0.0
        ),
        "oracle_safe_stop_coverage": (
            sum(row["accepted_stop"] and row["oracle_prefix_safe"] for row in rows) / len(oracle_safe)
            if oracle_safe else 0.0
        ),
        "fallback_queries": sum(not row["accepted_stop"] for row in rows),
        "stop_k_distribution": {
            str(k): sum(row["stop_k"] == k for row in rows) for k in (1, 2, 3)
        },
        "errors": sum(bool(row["error"]) for row in rows),
        "parse_errors": sum(bool(step["parse_error"]) for step in calls),
        "api_calls": len(calls),
        "new_api_calls": sum(not bool(req.get("cache_reused")) for req in requests),
        "prompt_tokens": sum(int(req.get("prompt_tokens") or 0) for req in requests),
        "completion_tokens": sum(int(req.get("completion_tokens") or 0) for req in requests),
        "cached_tokens": sum(int(req.get("cached_tokens") or 0) for req in requests),
        "cost_usd": sum(float(req.get("cost_usd") or 0.0) for req in requests),
        "sum_api_seconds": sum(float(req.get("total_seconds") or 0.0) for req in requests),
        "mean_request_seconds": statistics.fmean(
            float(req.get("total_seconds") or 0.0) for req in requests
        ) if requests else 0.0,
        "mean_query_wall_seconds": statistics.fmean(row["wall_seconds"] for row in rows),
    }


def write_chat(path: Path, row: Mapping[str, Any]) -> None:
    lines = [
        f"# {row['qid']} — {row['arm']}", "",
        f"**Question:** {row['query']}", "",
        f"- Target K (evaluation only): {row['target_k']}",
        f"- Gold ranks (evaluation only): {row['gold_ranks']}",
        f"- XGBoost fallback K: {row['xgb_k']}",
        f"- Accepted STOP: {row['accepted_stop']}",
        f"- Deployed K: {row['deployed_k']}",
        f"- Deployed recall: {100 * row['deployed_recall']:.2f}%", "",
    ]
    for index, step in enumerate(row["steps"], 1):
        result = step["result"]
        lines.extend([
            f"## Call {index}: ranks {step['rank_start']}–{step['rank_end']}", "",
            f"- Pages: {', '.join(step['new_page_ids'])}",
            f"- Decision: {result['decision']}",
            f"- Confidence: {result['confidence']}",
            f"- Accepted: {step['accepted_stop']}",
            f"- Rejection reasons: {step['stop_rejection_reasons']}",
            f"- Cited pages: {result['cited_page_ids']}",
            f"- Missing information: {result['missing_information']}",
            f"- Provisional answer: {result['provisional_answer']}",
            f"- State summary: {result['state_summary']}", "",
            "### Raw JSON", "", "```json", step["raw_response"], "```", "",
        ])
    if row["error"]:
        lines.extend(["## Error", "", row["error"], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--sources", nargs="+", choices=sorted(SOURCE_COUNTS), default=["ohrbench"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--oneshot-k", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = tuple(dict.fromkeys(args.sources))
    arms = tuple(dict.fromkeys(args.arms))
    rows, pages = load_inputs(sources)
    load_dotenv_file(ROOT)
    client = OpenRouterClient(
        model=MODEL,
        max_output_tokens=350,
        cache_path=args.output_dir / "request_cache.jsonl",
        budget=Budget(max_calls=300, max_cost_usd=args.max_cost_usd),
        resume=args.resume,
    )
    log_lock = threading.Lock()
    log_path = args.output_dir / "console.log"

    def log(message: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        with log_lock:
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")

    manifest = {
        "model": MODEL,
        "ranking": "adaptive_retrieval_20261020_router_te3s_top100 / text-embedding-3-small",
        "queries": len(rows),
        "sources": list(sources),
        "arms": list(arms),
        "fallback": "nested-OOF calibrated XGBoost K",
        "acceptance": "STOP, confidence>=0.85, non-empty answer, valid citation, empty missing_information",
        "qrels_in_prompt": False,
        "reference_answers_in_prompt": False,
        "xgb_k_in_prompt": False,
        "workers": args.workers,
        "oneshot_k": args.oneshot_k,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "general_system_prompt.txt").write_text(GENERAL_SYSTEM + "\n", encoding="utf-8")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    log(f"starting {len(rows) * len(arms)} trajectories with workers={args.workers}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                arm=arm,
                row=row,
                pages=pages,
                client=client,
                oneshot_k=args.oneshot_k,
            ): (arm, row["qid"])
            for arm in arms for row in rows
        }
        for completed, future in enumerate(as_completed(futures), 1):
            arm, qid = futures[future]
            result = future.result()
            results.append(result)
            log(
                f"{completed}/{len(futures)} {arm} {qid} "
                f"stop={result['accepted_stop']} k={result['deployed_k']} "
                f"recall={result['deployed_recall']:.4f} calls={result['calls']}"
            )

    results.sort(key=lambda row: (str(row["arm"]), str(row["qid"])))
    write_jsonl(args.output_dir / "trajectories.jsonl", results)
    for row in results:
        safe_qid = str(row["qid"]).replace("::", "__")
        write_chat(args.output_dir / "chats" / str(row["arm"]) / f"{safe_qid}.md", row)

    summaries = {arm: summarize([row for row in results if row["arm"] == arm]) for arm in arms}
    per_query = []
    by_key = {(row["arm"], row["qid"]): row for row in results}
    for row in rows:
        qid = str(row["qid"])
        record = {
            "qid": qid,
            "target_k": row["target_k"],
            "xgb_k": row["xgb_k"],
            "oracle_prefix_k": by_key[(arms[0], qid)]["oracle_prefix_k"],
            "oracle_prefix_safe": by_key[(arms[0], qid)]["oracle_prefix_safe"],
        }
        for arm in arms:
            result = by_key[(arm, qid)]
            record.update({
                f"{arm}_stop": result["accepted_stop"],
                f"{arm}_stop_k": result["stop_k"],
                f"{arm}_deployed_k": result["deployed_k"],
                f"{arm}_recall": result["deployed_recall"],
            })
        per_query.append(record)
    with (args.output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)

    paired_agreement = None
    if set(ARMS).issubset(arms):
        paired_agreement = statistics.fmean(
            by_key[("pagewise_1_to_3", row["qid"])]["accepted_stop"]
            == by_key[("oneshot_top3", row["qid"])]["accepted_stop"]
            for row in rows
        )

    report = {
        "manifest": manifest,
        "wall_seconds": time.perf_counter() - started,
        "summaries": summaries,
        "paired_decision_agreement": paired_agreement,
        "new_api_budget": {
            "calls": client.budget.calls,
            "cost_usd": client.budget.cost_usd,
        },
    }
    write_json(args.output_dir / "report.json", report)
    lines = [
        f"# Top-{args.oneshot_k} LLM stopping: {', '.join(sources)}", "",
        "Current text-embedding-3-small top-100 ranking; accepted STOPs replace the calibrated XGBoost K. CONTINUE falls back to XGBoost.", "",
        f"| Arm | Mean K | Recall | Stops | Safe stops | Unsafe stops | Coverage of top-{args.oneshot_k}-safe | Calls | Cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        summary = summaries[arm]
        lines.append(
            f"| {arm} | {summary['mean_deployed_k']:.2f} | {100*summary['macro_recall']:.2f}% | "
            f"{summary['accepted_stops']} | {summary['safe_stops_vs_xgb']} | {summary['unsafe_stops_vs_xgb']} | "
            f"{100*summary['oracle_safe_stop_coverage']:.2f}% | {summary['api_calls']} | ${summary['cost_usd']:.6f} |"
        )
    lines.extend(["", f"- Wall time: {report['wall_seconds']:.2f}s."])
    if paired_agreement is not None:
        lines.append(f"- Paired STOP/CONTINUE agreement: {100*paired_agreement:.2f}%.")
    lines.extend(["- Full per-query decisions and raw JSON are under `chats/`.", ""])
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"finished wall={report['wall_seconds']:.2f}s calls={client.budget.calls} cost=${client.budget.cost_usd:.6f}")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
