"""Compare 3-page and 10-page iterative LLM rescue on XGBoost-hard queries.

This is an exploratory, source-balanced pilot.  XGBoost OOF predictions route
queries into the experiment.  Qrels are used only to construct a balanced
diagnostic sample and to evaluate the completed trajectories; they never enter
an LLM prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "research" / "experiments"
for candidate in (str(ROOT), str(EXPERIMENTS)):
    if candidate not in sys.path:
        sys.path.append(candidate)
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from adaptive_k_optimization_core import load_frozen_data, recall_at, read_jsonl  # noqa: E402
from run_adaptive_llm_stopping import (  # noqa: E402
    Budget,
    DEFAULT_MODEL,
    HARD_SYSTEM,
    OpenRouterClient,
    run_trajectory,
    write_json,
    write_jsonl,
)
from src.utils.env import load_dotenv_file  # noqa: E402


DEFAULT_OUTPUT = ROOT / "research/adaptive/llm/results/xgb_hard_rescue_b3_vs_b10_20261019"
XGB_ARTIFACT = ROOT / "data/work/score_only_model_focus_20261016/per_query.jsonl"
SEED = 42
EASY_K = 30
HARD_K = 100
RESCUE_CAP = 50
XGB_THRESHOLD = 0.2


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("xgb_hard_llm_rescue")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "console.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def xgb_hard_mask(data: Any) -> tuple[np.ndarray, np.ndarray]:
    records = read_jsonl(XGB_ARTIFACT)
    by_qid = {str(row["qid"]): float(row["xgb_hard_probability"]) for row in records}
    missing = [str(row["qid"]) for row in data.rows if str(row["qid"]) not in by_qid]
    if missing:
        raise ValueError(f"XGBoost artifact misses {len(missing)} frozen qids")
    probabilities = np.asarray([by_qid[str(row["qid"])] for row in data.rows], dtype=float)
    hard = probabilities >= XGB_THRESHOLD
    if int(hard.sum()) != 134:
        raise ValueError(f"Expected 134 XGBoost-hard rows at threshold 0.2, found {hard.sum()}")
    return hard, probabilities


def select_pilot(data: Any, hard: np.ndarray, probabilities: np.ndarray) -> list[int]:
    """Pick four XGB-hard rows/source, balancing safe/unsafe where possible."""
    rng = random.Random(SEED)
    sources = np.asarray([str(row["source"]) for row in data.rows])
    safe = np.asarray([
        recall_at(row, HARD_K) > 0
        and math.isclose(recall_at(row, RESCUE_CAP), recall_at(row, HARD_K), abs_tol=1e-12)
        for row in data.rows
    ])
    unsafe = np.asarray([
        recall_at(row, RESCUE_CAP) < recall_at(row, HARD_K) - 1e-12
        for row in data.rows
    ])
    selected: list[int] = []
    for source in sorted(set(sources)):
        safe_candidates = sorted(int(index) for index in np.flatnonzero(hard & safe & (sources == source)))
        unsafe_candidates = sorted(int(index) for index in np.flatnonzero(hard & unsafe & (sources == source)))
        safe_count = 2 if len(unsafe_candidates) >= 2 else 4
        unsafe_count = 4 - safe_count
        if len(safe_candidates) < safe_count or len(unsafe_candidates) < unsafe_count:
            raise ValueError(f"Not enough XGBoost-hard pilot candidates for {source}")
        selected.extend(rng.sample(safe_candidates, safe_count))
        selected.extend(rng.sample(unsafe_candidates, unsafe_count))
    return selected


def request_cost(trajectory: Mapping[str, Any]) -> float:
    requests: dict[str, Mapping[str, Any]] = {}
    for step in trajectory["steps"]:
        request = step["request"]
        requests.setdefault(str(request["request_hash"]), request)
    return float(sum(float(item.get("cost_usd") or 0.0) for item in requests.values()))


def render_chat(output: Path, data: Any, row: Mapping[str, Any], trajectory: Mapping[str, Any]) -> None:
    lines = [
        f"# {trajectory['qid']} — batch {trajectory['batch_size']}",
        "",
        f"**Question:** {row['query']}",
        "",
        f"**Source:** `{row['source']}`  ",
        f"**LLM stop:** `{trajectory['stopped_by_llm']}`  ",
        f"**Retrieval position inspected:** `{trajectory['chosen_k']}`  ",
        f"**Final policy K:** `{trajectory['final_k']}`  ",
        "",
        "## System prompt",
        "",
        "```text",
        HARD_SYSTEM,
        "```",
    ]
    for step in trajectory["steps"]:
        lines += [
            "",
            f"## Step {step['step'] + 1}: ranks {step['rank_start']}–{step['rank_end']}",
            "",
        ]
        for position in range(int(step["rank_start"]) - 1, int(step["rank_end"])):
            page_index = int(row["ranked_indices"][position])
            page_id = str(data.page_ids[page_index])
            lines += [
                f"### Rank {position + 1}: `{page_id}`",
                "",
                "```text",
                str(data.page_texts[page_index]),
                "```",
                "",
            ]
        lines += [
            "### Parsed LLM response",
            "",
            "```json",
            json.dumps(step["result"], ensure_ascii=False, indent=2),
            "```",
            "",
            f"Accepted STOP: `{step['accepted_stop']}`  ",
            f"Rejection reasons: `{json.dumps(step['stop_rejection_reasons'])}`  ",
            f"Parse error: `{step['parse_error']}`  ",
            f"Cost: `${float(step['request'].get('cost_usd') or 0.0):.6f}`  ",
            f"Latency: `{float(step['request'].get('total_seconds') or 0.0):.2f}s`",
        ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def policy_metrics(data: Any, hard: np.ndarray, trajectories: Sequence[Mapping[str, Any]], batch: int) -> dict[str, Any]:
    by_qid = {str(item["qid"]): item for item in trajectories if int(item["batch_size"]) == batch}
    choices = np.where(hard, HARD_K, EASY_K).astype(int)
    pilot_stops = []
    for index, row in enumerate(data.rows):
        item = by_qid.get(str(row["qid"]))
        if item is not None and bool(item["stopped_by_llm"]):
            choices[index] = int(item["chosen_k"])
            pilot_stops.append((row, item))
    recalls = np.asarray([recall_at(row, int(k)) for row, k in zip(data.rows, choices)], dtype=float)
    baseline = np.asarray([
        recall_at(row, HARD_K if hard[index] else EASY_K)
        for index, row in enumerate(data.rows)
    ])
    safe = [
        math.isclose(recall_at(row, int(item["chosen_k"])), recall_at(row, HARD_K), abs_tol=1e-12)
        for row, item in pilot_stops
    ]
    losses = [
        recall_at(row, HARD_K) - recall_at(row, int(item["chosen_k"]))
        for row, item in pilot_stops
    ]
    arm = [item for item in trajectories if int(item["batch_size"]) == batch]
    return {
        "batch_size": batch,
        "pilot_queries": len(arm),
        "llm_stops": len(pilot_stops),
        "safe_stops": int(sum(safe)),
        "unsafe_stops": int(len(safe) - sum(safe)),
        "safe_stop_precision": float(sum(safe) / len(safe)) if safe else 1.0,
        "recall_loss_on_stops": float(sum(max(loss, 0.0) for loss in losses)),
        "pilot_mean_final_k": float(np.mean([int(item["final_k"]) for item in arm])),
        "pilot_calls": int(sum(int(item["calls"]) for item in arm)),
        "pilot_cost_usd": float(sum(request_cost(item) for item in arm)),
        "full_policy_mean_k_with_unrun_hard_unchanged": float(np.mean(choices)),
        "full_policy_macro_recall_with_unrun_hard_unchanged": float(np.mean(recalls)),
        "baseline_macro_recall": float(np.mean(baseline)),
    }


def render_report(output: Path, report: Mapping[str, Any], trajectories: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# XGBoost-hard LLM rescue: batch 3 versus batch 10",
        "",
        f"Exploratory iterative run on {report['manifest']['pilot_queries']} of the {report['manifest']['xgb_hard_queries']} XGBoost-hard queries (source filter: {report['manifest']['source_filter']}). Gold is evaluation-only and never enters prompts.",
        "",
        "## Arm summary",
        "",
        "| Batch | Stops | Safe | Unsafe | Safe precision | Pilot mean final K | Calls | Cost | Full-policy Mean K* | Full-policy recall* |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["arms"]:
        lines.append(
            f"| {item['batch_size']} | {item['llm_stops']} | {item['safe_stops']} | {item['unsafe_stops']} | "
            f"{item['safe_stop_precision']:.2%} | {item['pilot_mean_final_k']:.2f} | {item['pilot_calls']} | "
            f"${item['pilot_cost_usd']:.6f} | {item['full_policy_mean_k_with_unrun_hard_unchanged']:.2f} | "
            f"{item['full_policy_macro_recall_with_unrun_hard_unchanged']:.2%} |"
        )
    lines += [
        "",
        f"\\* Only the {report['manifest']['pilot_queries']} selected hard queries are eligible for LLM rescue; all other queries keep their baseline action.",
        "",
        "## Per-query comparison",
        "",
        "| QID | Source | Safe by K50 | Batch | Inspected K | Final K | Stop | Recall final | Recall@100 | Calls | Cost | Chat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in sorted(trajectories, key=lambda value: (str(value["qid"]), int(value["batch_size"]))):
        filename = f"b{item['batch_size']}_{str(item['qid']).replace('::', '_')}.md"
        lines.append(
            f"| {item['qid']} | {item['source']} | {item['safe_50_vs_100']} | {item['batch_size']} | "
            f"{item['chosen_k']} | {item['final_k']} | {item['stopped_by_llm']} | "
            f"{item['final_recall']:.2%} | {item['recall_at_100']:.2%} | {item['calls']} | "
            f"${request_cost(item):.6f} | [chat](chats/{filename}) |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-calls", type=int, default=350)
    parser.add_argument("--max-cost-usd", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=350)
    parser.add_argument("--source", choices=("all", "ohrbench", "vidore_physics", "vidore_industrial"), default="all")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent queries")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "chats").mkdir(exist_ok=True)
    logger = setup_logger(args.output_dir)
    data = load_frozen_data()
    hard, probabilities = xgb_hard_mask(data)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.source == "all":
        selected = select_pilot(data, hard, probabilities)
    else:
        selected = [int(i) for i in np.flatnonzero(hard) if str(data.rows[int(i)]["source"]) == args.source]
    selection = []
    for index in selected:
        row = data.rows[index]
        selection.append({
            "qid": str(row["qid"]),
            "source": str(row["source"]),
            "xgb_hard_probability": float(probabilities[index]),
            "safe_50_vs_100": recall_at(row, HARD_K) > 0 and math.isclose(recall_at(row, RESCUE_CAP), recall_at(row, HARD_K)),
            "recall_at_50": recall_at(row, RESCUE_CAP),
            "recall_at_100": recall_at(row, HARD_K),
        })
    manifest = {
        "model": args.model,
        "router": "XGBoost OOF hard probability >= 0.2",
        "xgb_hard_queries": int(hard.sum()),
        "pilot_queries": len(selected),
        "source_filter": args.source,
        "workers": args.workers,
        "batches": [3, 10],
        "rescue_cap": RESCUE_CAP,
        "fallback_k": HARD_K,
        "selection": selection,
        "qrels_in_prompt": False,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    (args.output_dir / "hard_only_system_prompt.txt").write_text(HARD_SYSTEM + "\n", encoding="utf-8")
    logger.info("materialized %d XGBoost-hard queries (source=%s); 134 hard total", len(selected), args.source)
    if args.materialize_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    load_dotenv_file(ROOT)
    client = OpenRouterClient(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        cache_path=args.output_dir / "request_cache.jsonl",
        budget=Budget(args.max_calls, args.max_cost_usd),
        resume=args.resume,
    )
    completed: dict[tuple[int, str], dict[str, Any]] = {}
    trajectory_path = args.output_dir / "trajectories.jsonl"
    if args.resume and trajectory_path.is_file():
        for item in read_jsonl(trajectory_path):
            completed[(int(item["batch_size"]), str(item["qid"]))] = item

    def run_query(index: int) -> list[dict[str, Any]]:
        row = data.rows[index]
        qid = str(row["qid"])
        results = []
        for batch in (3, 10):
            if (batch, qid) in completed:
                logger.info("reuse batch=%d qid=%s", batch, qid)
                continue
            logger.info("start batch=%d qid=%s", batch, qid)
            trajectory = run_trajectory(
                data=data,
                row=row,
                mode="hard-only",
                batch_size=batch,
                initial_k=batch,
                cap=RESCUE_CAP,
                client=client,
            )
            trajectory["xgb_hard_probability"] = float(probabilities[index])
            trajectory["safe_50_vs_100"] = recall_at(row, HARD_K) > 0 and math.isclose(recall_at(row, RESCUE_CAP), recall_at(row, HARD_K))
            trajectory["final_k"] = int(trajectory["chosen_k"]) if trajectory["stopped_by_llm"] else HARD_K
            trajectory["final_recall"] = recall_at(row, int(trajectory["final_k"]))
            results.append(trajectory)
        return results

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_query, index): index for index in selected}
        for future in as_completed(futures):
            index = futures[future]
            row = data.rows[index]
            for trajectory in future.result():
                batch = int(trajectory["batch_size"])
                qid = str(trajectory["qid"])
                completed[(batch, qid)] = trajectory
                write_jsonl(trajectory_path, completed.values())
                filename = f"b{batch}_{qid.replace('::', '_')}.md"
                render_chat(args.output_dir / "chats" / filename, data, row, trajectory)
                logger.info(
                    "done batch=%d qid=%s inspected=%d final_k=%d stop=%s calls=%d cost=$%.6f",
                    batch, qid, trajectory["chosen_k"], trajectory["final_k"],
                    trajectory["stopped_by_llm"], trajectory["calls"], request_cost(trajectory),
                )

    trajectories = list(completed.values())
    report = {
        "manifest": manifest,
        "baseline": {"mean_k": 72.63636363636364, "macro_recall": 0.8639312066828464},
        "arms": [policy_metrics(data, hard, trajectories, batch) for batch in (3, 10)],
        "new_api_calls": client.budget.calls,
        "new_cost_usd": client.budget.cost_usd,
    }
    write_json(args.output_dir / "report.json", report)
    render_report(args.output_dir / "report.md", report, trajectories)
    logger.info("completed comparison; report=%s", args.output_dir / "report.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
