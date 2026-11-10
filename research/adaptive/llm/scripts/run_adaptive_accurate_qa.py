"""Evaluate adaptive candidate budgets followed by accurate top-10 reranking and QA.

This is an end-to-end, resumable evaluation for the two existing adaptive
policies. It reuses frozen PP-OCRv5 candidates, ColVec/EVIE score matrices,
page text, and the established ViDoRe answer/judge prompts. Qrels are used only
for offline evaluation, never in prompts or model features.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[4]
FINGERPRINT = "ef7f11811803"
HANDOFF = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2" / FINGERPRINT
EVAL = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2" / FINGERPRINT
GATE_PATH = HANDOFF / "light_prep/gate_ppocrv5_ungated.json"
DATA_PATH = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100/feature_labels.jsonl"
TRAJECTORY_PATH = ROOT / "research/adaptive/llm/results/general_all_220_b1_cap20_xgbfallback_20261021/trajectories.jsonl"
TOP3_OHR_TRAJECTORY_PATH = ROOT / "research/adaptive/llm/results/ohr_top3_pagewise_vs_oneshot_te3s_20261103/trajectories.jsonl"
TOP3_PI_TRAJECTORY_PATH = ROOT / "research/adaptive/llm/results/top3_oneshot_physics_industrial_te3s_20261106/trajectories.jsonl"
OOF_PATH = ROOT / "research/adaptive/llm/results/xgb_llm_features_nested_oof_20261021/oof_predictions.csv"
QA_RECORDS_PATH = EVAL / "results/best_armAB/per_query.jsonl"
KDL_PATH = EVAL / "kdl/kdl_pages.jsonl"
COLVEC_DIR = EVAL / "colvec"
EVIE_DIR = HANDOFF / "routing_axes/k3/k3_out_shard0"
OUT = ROOT / "research/adaptive/llm/results/adaptive_accurate_top10_qa_20261021"

ANSWER_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
ANSWER_REASONING: dict[str, Any] | None = None
JUDGE_MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = 0.0
MAX_ANSWER_TOKENS = 512
MAX_JUDGE_TOKENS = 256
TOP_ACCURATE = 10
PAGE_WORD_LIMIT = 512

ANSWER_PROMPT = (
    "You are an expert at answering query based on documents.\n"
    "Here is a list of relevant documents:\n"
    "{documents}\n\n"
    "Based on the above documents, answer the following query:\n"
    "{query}\n\n"
    "Keep the response short when appropriate. Output the answer only."
)
JUDGE_PROMPT = (
    'You are an expert judge evaluating the accuracy of a test answer against a '
    'gold-standard true answer. Your goal is to determine if the test answer '
    'captures the essential "core information."\n'
    "\n"
    "### Evaluation Criteria:\n"
    "- Correct: The test answer contains all core information of the true answer. "
    "Minor omissions of non-essential details or the addition of minor, "
    'non-contradictory information should still be marked as "Correct."\n'
    "- Partially Correct: The test answer captures some of the core information, "
    "but suffers from significant omissions or includes substantial extra "
    "information that was not requested or present in the true answer.\n"
    "- Incorrect: The test answer is fundamentally wrong, contradicts the true "
    "answer, or misses the core information entirely.\n"
    "\n"
    "### Input Data:\n"
    "Query: {query}\n"
    "True Answer: {true_answer}\n"
    "Test Answer: {test_answer}\n"
    "\n"
    "### Output Format:\n"
    "Provide a very brief explanation for your judgment. You must output your "
    'final response in a JSON format with two fields: "explanation" and '
    '"judgment" (which must be "Correct", "Partially Correct", or "Incorrect").'
)
LABELS = ("Correct", "Partially Correct", "Incorrect")
_JUDGMENT_RE = re.compile(r'"judgment"\s*:\s*"([^"]+)"')
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
_PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "console.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "OPENROUTER_API_KEY":
                value = value.strip().strip('"').strip("'")
                if value:
                    os.environ["OPENROUTER_API_KEY"] = value
                    return value
    raise RuntimeError("OPENROUTER_API_KEY was not found in the environment or workspace .env")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def answer_cache_key(qid: str, ctx: list[str]) -> str:
    # Compatible with the existing light_v2 Qwen text-QA cache convention.
    return stable_hash(json.dumps([ANSWER_MODEL, "kdl", qid, ctx], ensure_ascii=False))


def format_gold(answer: str | list[str] | None) -> str:
    if isinstance(answer, str):
        items = [_BULLET_RE.sub("", line).strip() for line in answer.splitlines()]
    else:
        items = [str(item).strip() for item in (answer or [])]
    items = list(dict.fromkeys(item for item in items if item))
    if len(items) <= 1:
        return items[0] if items else ""
    return "Any one of the following is fully correct: " + " | ".join(items)


def parse_judgment(reply: str) -> str:
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", reply.strip(), flags=re.I)
    try:
        label = str(json.loads(text)["judgment"]).strip()
    except (ValueError, KeyError, TypeError):
        match = _JUDGMENT_RE.search(text)
        if not match:
            raise ValueError(f"could not parse judge label from: {reply[:300]!r}")
        label = match.group(1).strip()
    if label not in LABELS:
        raise ValueError(f"judge label {label!r} is not in {LABELS}")
    return label


def call_openrouter(model: str, prompt: str, max_tokens: int,
                    reasoning: dict[str, Any] | None = None) -> dict[str, Any]:
    key = load_api_key()
    started = perf_counter()
    response = requests.post(
        API_URL,
        timeout=180,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True},
            **({"reasoning": reasoning} if reasoning else {}),
        },
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    answer = (choices[0].get("message") or {}).get("content") if choices else None
    if isinstance(answer, list):
        answer = "".join(str(part.get("text", "")) for part in answer if isinstance(part, dict))
    if not str(answer or "").strip():
        raise RuntimeError(f"empty response from {model}: {str(payload)[:500]}")
    usage = payload.get("usage") or {}
    return {
        "text": str(answer).strip(),
        "model": payload.get("model", model),
        "provider": payload.get("provider"),
        "response_id": payload.get("id"),
        "latency_seconds": perf_counter() - started,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cost_usd": usage.get("cost", usage.get("total_cost")),
        "raw_usage": usage,
        "raw_response": payload,
    }


def load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    for path in (GATE_PATH, DATA_PATH, TRAJECTORY_PATH, TOP3_OHR_TRAJECTORY_PATH,
                 TOP3_PI_TRAJECTORY_PATH, OOF_PATH, QA_RECORDS_PATH, KDL_PATH,
                 COLVEC_DIR / "colvec_qids.json", COLVEC_DIR / "colvec_keys.json",
                 COLVEC_DIR / "colvec_scores.npy", EVIE_DIR / "evie_scores_2048.npy",
                 EVIE_DIR / "meta.json"):
        if not path.exists():
            raise FileNotFoundError(path)

    gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
    labels = {row["qid"]: row for row in read_jsonl(DATA_PATH)}
    trajectories = {row["qid"]: row for row in read_jsonl(TRAJECTORY_PATH)}
    top3_trajectories = {
        row["qid"]: row
        for path in (TOP3_OHR_TRAJECTORY_PATH, TOP3_PI_TRAJECTORY_PATH)
        for row in read_jsonl(path)
        if row.get("arm") == "oneshot_top3"
    }
    qa_rows = {row["query_id"]: row for row in read_jsonl(QA_RECORDS_PATH)}
    xgb_llm: dict[str, int] = {}
    xgb_plain: dict[str, int] = {}
    xgb_plain60: dict[str, int] = {}
    with OOF_PATH.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["calibration_budget"] == "50":
                if row["arm"] == "xgb_plus_llm_trajectory":
                    xgb_llm[row["qid"]] = int(float(row["prediction_k_oof"]))
                elif row["arm"] == "xgb_score_only":
                    xgb_plain[row["qid"]] = int(float(row["prediction_k_oof"]))
            elif row["calibration_budget"] == "60" and row["arm"] == "xgb_score_only":
                xgb_plain60[row["qid"]] = int(float(row["prediction_k_oof"]))
    expected = set(labels)
    if any(set(part) != expected for part in (
        trajectories, top3_trajectories, qa_rows, xgb_llm, xgb_plain, xgb_plain60
    )):
        raise ValueError("qid mismatch among frozen feature labels, trajectories, OOF predictions, and QA records")
    return gate, {
        "labels": labels,
        "trajectories": trajectories,
        "top3_trajectories": top3_trajectories,
        "qa_rows": qa_rows,
        "xgb_llm": xgb_llm,
        "xgb_plain": xgb_plain,
        "xgb_plain60": xgb_plain60,
    }


def get_context(page_ids: list[str], page_text: dict[str, str]) -> list[str]:
    # One bounded evidence excerpt per accurately ranked page, up to top 10.
    return [" ".join(page_text.get(page_id, "").split()[:PAGE_WORD_LIMIT]) for page_id in page_ids]


def make_jobs(gate: dict[str, Any], inputs: dict[str, Any], page_text: dict[str, str],
              score_mats: dict[str, tuple[np.ndarray, dict[str, int], dict[str, int]]],
              selected_policy: str = "all", selected_scorer: str = "all") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policies = {
        "fixed_k100": {qid: 100 for qid in inputs["labels"]},
        "fixed_k58": {qid: 58 for qid in inputs["labels"]},
        "llm_stop_xgb_fallback_b1_cap20": {
            qid: int(row["deployed_k"]) for qid, row in inputs["trajectories"].items()
        },
        "llm_top3_xgb_fallback": {
            qid: int(row["deployed_k"]) for qid, row in inputs["top3_trajectories"].items()
        },
        "xgb_plus_llm_trajectory_oof_budget50": inputs["xgb_llm"],
        "xgb_score_only_oof_budget50": inputs["xgb_plain"],
        "xgb_score_only_oof_budget60": inputs["xgb_plain60"],
    }
    if selected_policy == "all":
        policies.pop("fixed_k100")
        policies.pop("llm_top3_xgb_fallback")
        policies.pop("fixed_k58")
        policies.pop("xgb_score_only_oof_budget50")
        policies.pop("xgb_score_only_oof_budget60")
    else:
        policies = {selected_policy: policies[selected_policy]}
    if selected_scorer != "all":
        score_mats = {selected_scorer: score_mats[selected_scorer]}
    answer_jobs: dict[str, dict[str, Any]] = {}
    judge_jobs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for policy_name, chosen_ks in policies.items():
        for scorer_name, (score_matrix, q_index, p_index) in score_mats.items():
            for qid in sorted(inputs["labels"]):
                k = max(1, min(100, int(chosen_ks[qid])))
                candidates = gate["light_run_top100"][qid][:k]
                if qid not in q_index or any(page_id not in p_index for page_id in candidates):
                    raise ValueError(f"unmapped query/page in frozen ranker matrices: {qid}")
                # Stable tie handling retains the frozen candidate order.
                rerank_started = perf_counter()
                reranked = sorted(candidates, key=lambda page_id: float(score_matrix[q_index[qid], p_index[page_id]]), reverse=True)
                rerank_seconds = perf_counter() - rerank_started
                top_pages = reranked[:min(TOP_ACCURATE, k)]
                ctx = get_context(top_pages, page_text)
                qa = inputs["qa_rows"][qid]
                prompt = ANSWER_PROMPT.format(
                    documents="\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(ctx)),
                    query=qa["query"],
                )
                prompt_hash = stable_hash(ANSWER_MODEL + "\n" + prompt)
                answer_jobs.setdefault(prompt_hash, {
                    "hash": prompt_hash, "qid": qid, "prompt": prompt, "ctx": ctx,
                    "legacy_key": answer_cache_key(qid, ctx),
                })
                gold = gate["gold_rank"][qid]
                # Every key is a qrel gold page; a None rank means it was not
                # reachable in the frozen top-100, not that it is nonrelevant.
                relevant = set(gold)
                prefix_hits = len(set(candidates) & relevant)
                accurate_hits = len(set(top_pages) & relevant)
                dcg = sum(1.0 / math.log2(rank + 2) for rank, page_id in enumerate(top_pages) if page_id in relevant)
                ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), TOP_ACCURATE)))
                row = {
                    "policy": policy_name, "scorer": scorer_name, "qid": qid,
                    "source": inputs["labels"][qid]["source"], "chosen_k": k,
                    "accurate_rerank_seconds": rerank_seconds,
                    "target_k": int(inputs["labels"][qid]["target_k"]),
                    "gold_page_count": len(relevant),
                    "candidate_prefix_recall": prefix_hits / max(1, len(relevant)),
                    "candidate_pages": candidates,
                    "accurate_top_pages": top_pages,
                    "accurate_recall_at_10": accurate_hits / max(1, len(relevant)),
                    "accurate_ndcg_at_10": dcg / ideal if ideal else 0.0,
                    "qa_context_gold_recall": accurate_hits / max(1, len(relevant)),
                    "qa_context_pages": top_pages,
                    "context_words_per_page": [len(text.split()) for text in ctx],
                    "answer_prompt_hash": prompt_hash,
                }
                rows.append(row)
    return list(answer_jobs.values()), rows


def legacy_answer_path(qid: str, ctx: list[str]) -> Path:
    return EVAL / "qa_cache" / ANSWER_MODEL.replace("/", "_") / f"{answer_cache_key(qid, ctx)}.json"


def load_old_judge(prompt: str) -> dict[str, Any] | None:
    key = stable_hash(JUDGE_MODEL + "\n" + prompt)
    path = ROOT / "data/work/Cached-Ondemand-Eval/data/cache/judge" / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("judgment") in LABELS:
            return {"judgment": cached["judgment"], "reply": cached.get("reply", ""), "cache": "legacy"}
    return None


def cached_or_call_answer(job: dict[str, Any], output: Path) -> dict[str, Any]:
    path = output / "cache/answers" / f"{job['hash']}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache") != "error" and cached.get("answer"):
            if "prompt" not in cached:
                cached["prompt"] = job["prompt"]
                atomic_json(path, cached)
            return cached
    legacy = legacy_answer_path(job["qid"], job["ctx"])
    if legacy.exists():
        item = json.loads(legacy.read_text(encoding="utf-8"))
        result = {"hash": job["hash"], "answer": item["answer"], "prompt": job["prompt"], "cache": "legacy",
                  "latency_seconds": item.get("gen_seconds", 0.0), "prompt_tokens": None,
                  "completion_tokens": None, "cost_usd": 0.0}
    else:
        try:
            response = call_openrouter(ANSWER_MODEL, job["prompt"], MAX_ANSWER_TOKENS, ANSWER_REASONING)
            result = {"hash": job["hash"], "answer": response.pop("text"), "prompt": job["prompt"], "cache": "live", **response}
        except Exception as exc:  # recorded so successful jobs can be resumed
            result = {"hash": job["hash"], "error": f"{type(exc).__name__}: {exc}", "cache": "error"}
    atomic_json(path, result)
    return result


def cached_or_call_judge(job: dict[str, Any], output: Path) -> dict[str, Any]:
    # Keep benchmark-exact judgments separate from an earlier exploratory pass
    # whose prompt was a paraphrase rather than the pinned benchmark wording.
    path = output / "cache/judgments_pinned" / f"{job['hash']}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache") != "error" and cached.get("judgment") in LABELS:
            if "prompt" not in cached:
                cached["prompt"] = job["prompt"]
                atomic_json(path, cached)
            return cached
    old = load_old_judge(job["prompt"])
    if old:
        result = {"hash": job["hash"], "prompt": job["prompt"], **old, "latency_seconds": 0.0, "prompt_tokens": None,
                  "completion_tokens": None, "cost_usd": 0.0}
    else:
        try:
            response = call_openrouter(JUDGE_MODEL, job["prompt"], MAX_JUDGE_TOKENS)
            reply = response.pop("text")
            result = {"hash": job["hash"], "judgment": parse_judgment(reply), "reply": reply, "prompt": job["prompt"],
                      "cache": "live", **response}
        except Exception as exc:
            result = {"hash": job["hash"], "error": f"{type(exc).__name__}: {exc}",
                      "judgment": "Incorrect", "cache": "error"}
    atomic_json(path, result)
    return result


def make_judge_jobs(base_rows: list[dict[str, Any]], inputs: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        qa = inputs["qa_rows"][row["qid"]]
        prompt = JUDGE_PROMPT.format(
            query=qa["query"], true_answer=format_gold(qa.get("answers") or qa.get("gold")),
            test_answer=row.get("answer", ""),
        )
        key = stable_hash(JUDGE_MODEL + "\n" + prompt)
        row["judge_prompt_hash"] = key
        jobs.setdefault(key, {"hash": key, "qid": row["qid"], "prompt": prompt})
    return list(jobs.values())


def run_judges(base_rows: list[dict[str, Any]], inputs: dict[str, Any], output: Path, workers: int) -> list[dict[str, Any]]:
    jobs = make_judge_jobs(base_rows, inputs)
    done: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cached_or_call_judge, job, output): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            done[job["hash"]] = future.result()
            if index % 50 == 0 or index == len(futures):
                log(f"[judge] {index}/{len(futures)} complete")
    for row in base_rows:
        result = done[row["judge_prompt_hash"]]
        row.update({
            "judgment": result.get("judgment", "Incorrect"),
            "judgment_cache": result.get("cache"),
            "judge_latency_seconds": result.get("latency_seconds"),
            "judge_prompt_tokens": result.get("prompt_tokens"),
            "judge_completion_tokens": result.get("completion_tokens"),
            "judge_cost_usd": result.get("cost_usd"),
            "judge_error": result.get("error"),
            "correct": result.get("judgment") == "Correct",
            "credited": result.get("judgment") in ("Correct", "Partially Correct"),
        })
    return base_rows


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_k": float(np.mean([r["chosen_k"] for r in rows])),
        "mean_candidate_prefix_recall": float(np.mean([r["candidate_prefix_recall"] for r in rows])),
        "macro_recall_at_10": float(np.mean([r["accurate_recall_at_10"] for r in rows])),
        "macro_ndcg_at_10": float(np.mean([r["accurate_ndcg_at_10"] for r in rows])),
        "mean_accurate_pages_to_qa": float(np.mean([len(r["accurate_top_pages"]) for r in rows])),
        "accurate_rerank_seconds_total": float(sum(r.get("accurate_rerank_seconds", 0.0) for r in rows)),
        "accurate_rerank_seconds_mean_per_query": float(np.mean([r.get("accurate_rerank_seconds", 0.0) for r in rows])),
        "correct_only": float(np.mean([bool(r.get("correct", False)) for r in rows])),
        "correct_plus_partial": float(np.mean([bool(r.get("credited", False)) for r in rows])),
        "qa_judged": sum(r.get("judgment") in LABELS for r in rows),
        "answer_failures": sum(bool(r.get("answer_error")) for r in rows),
        "judge_failures": sum(bool(r.get("judge_error")) for r in rows),
        "all_qrels_covered_by_prefix": sum(r["candidate_prefix_recall"] >= 1.0 for r in rows),
        "all_qrels_covered_by_qa_context": sum(r["qa_context_gold_recall"] >= 1.0 for r in rows),
        "qa_credited_despite_missing_qrel_pages": sum(r["credited"] and r["qa_context_gold_recall"] < 1.0 for r in rows),
        "qa_correct_despite_missing_qrel_pages": sum(r["correct"] and r["qa_context_gold_recall"] < 1.0 for r in rows),
        "qa_partial_despite_missing_qrel_pages": sum(r.get("judgment") == "Partially Correct" and r["qa_context_gold_recall"] < 1.0 for r in rows),
        "prefix_credited_despite_missing_qrel_pages": sum(r["credited"] and r["candidate_prefix_recall"] < 1.0 for r in rows),
        "all_qrels_covered_by_qa_but_qa_not_credited": sum((r["qa_context_gold_recall"] >= 1.0) and not r["credited"] for r in rows),
        "mean_live_answer_seconds": float(np.mean([r["answer_latency_seconds"] for r in rows if r.get("answer_cache") == "live" and r.get("answer_latency_seconds") is not None])) if any(r.get("answer_cache") == "live" for r in rows) else 0.0,
        "mean_live_judge_seconds": float(np.mean([r["judge_latency_seconds"] for r in rows if r.get("judgment_cache") == "live" and r.get("judge_latency_seconds") is not None])) if any(r.get("judgment_cache") == "live" for r in rows) else 0.0,
        "answer_cost_usd": float(sum(float(r.get("answer_cost_usd") or 0) for r in rows)),
        "judge_cost_usd": float(sum(float(r.get("judge_cost_usd") or 0) for r in rows)),
        "answer_prompt_tokens": sum(int(r["answer_prompt_tokens"]) for r in rows if r.get("answer_prompt_tokens") is not None),
        "answer_completion_tokens": sum(int(r["answer_completion_tokens"]) for r in rows if r.get("answer_completion_tokens") is not None),
        "judge_prompt_tokens": sum(int(r["judge_prompt_tokens"]) for r in rows if r.get("judge_prompt_tokens") is not None),
        "judge_completion_tokens": sum(int(r["judge_completion_tokens"]) for r in rows if r.get("judge_completion_tokens") is not None),
    }


def estimate_list_price(rows: list[dict[str, Any]]) -> float:
    """Fallback estimate when a provider response omits its actual billed cost."""
    answer_in = sum(int(r.get("answer_prompt_tokens") or 0) for r in rows)
    answer_out = sum(int(r.get("answer_completion_tokens") or 0) for r in rows)
    judge_in = sum(int(r.get("judge_prompt_tokens") or 0) for r in rows)
    judge_out = sum(int(r.get("judge_completion_tokens") or 0) for r in rows)
    return answer_in * 0.20 / 1_000_000 + answer_out * 0.88 / 1_000_000 + judge_in * 0.15 / 1_000_000 + judge_out * 0.60 / 1_000_000


def unique_call_summary(output: Path, answer_hashes: set[str] | None = None,
                        judgment_hashes: set[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    total_cost = 0.0
    total_prompt = total_completion = 0
    for category, folder in (("answers", "answers"), ("judgments", "judgments_pinned"),
                             ("discarded_preliminary_judgments", "judgments")):
        records = [json.loads(path.read_text(encoding="utf-8")) for path in (output / "cache" / folder).glob("*.json")]
        relevant_hashes = answer_hashes if category == "answers" else judgment_hashes if category == "judgments" else None
        if relevant_hashes is not None:
            records = [record for record in records if record.get("hash") in relevant_hashes]
        live = [record for record in records if record.get("cache") == "live"]
        errors = [record for record in records if record.get("cache") == "error"]
        cost = sum(float(record.get("cost_usd") or 0.0) for record in live)
        prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in live)
        completion_tokens = sum(int(record.get("completion_tokens") or 0) for record in live)
        summary[category] = {
            "unique_cache_entries": len(records), "live_api_calls": len(live), "errors": len(errors),
            "reported_cost_usd": cost, "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        if category != "discarded_preliminary_judgments":
            total_cost += cost
            total_prompt += prompt_tokens
            total_completion += completion_tokens
        else:
            summary[category]["cost_usd_not_used_for_final_scores"] = cost
    summary["total_unique_reported_cost_usd"] = total_cost
    summary["total_unique_prompt_tokens"] = total_prompt
    summary["total_unique_completion_tokens"] = total_completion
    summary["total_unique_api_calls"] = summary["answers"]["live_api_calls"] + summary["judgments"]["live_api_calls"]
    summary["total_billed_including_discarded_first_judge_pass_usd"] = total_cost + summary["discarded_preliminary_judgments"]["reported_cost_usd"]
    return summary


def main() -> None:
    global ANSWER_MODEL, ANSWER_REASONING, MAX_ANSWER_TOKENS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="load all artifacts and print counts; no API calls")
    parser.add_argument("--run", action="store_true", help="run or resume QA and judging through OpenRouter")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="optional query limit for a pilot; default all 220")
    parser.add_argument("--policy", choices=("all", "fixed_k100", "fixed_k58", "llm_stop_xgb_fallback_b1_cap20", "llm_top3_xgb_fallback", "xgb_plus_llm_trajectory_oof_budget50", "xgb_score_only_oof_budget50", "xgb_score_only_oof_budget60"), default="all")
    parser.add_argument("--scorer", choices=("all", "colvec", "evie"), default="all")
    parser.add_argument("--answer-model", default=ANSWER_MODEL,
                        help="OpenRouter answer/generator model; judge remains fixed to the benchmark model")
    parser.add_argument("--answer-max-tokens", type=int, default=MAX_ANSWER_TOKENS,
                        help="Maximum answer completion tokens; raise for reasoning models that spend tokens before answering")
    parser.add_argument("--answer-reasoning-effort", choices=("none", "low", "high", "max"), default=None,
                        help="Optional OpenRouter reasoning effort for the answer model")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.preflight and not args.run:
        parser.error("choose --preflight or --run")

    ANSWER_MODEL = args.answer_model
    MAX_ANSWER_TOKENS = args.answer_max_tokens
    ANSWER_REASONING = {"effort": args.answer_reasoning_effort} if args.answer_reasoning_effort else None
    OUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    previous_summary_path = OUT / "summary.json"
    previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8")) if previous_summary_path.exists() else {}
    gate, inputs = load_inputs()
    col_qids = json.loads((COLVEC_DIR / "colvec_qids.json").read_text(encoding="utf-8"))
    col_pages = json.loads((COLVEC_DIR / "colvec_keys.json").read_text(encoding="utf-8"))
    q_index = {qid: i for i, qid in enumerate(col_qids)}
    p_index = {page: i for i, page in enumerate(col_pages)}
    colvec = np.load(COLVEC_DIR / "colvec_scores.npy", mmap_mode="r")
    evie = np.load(EVIE_DIR / "evie_scores_2048.npy", mmap_mode="r")
    if colvec.shape != (220, 5091) or evie.shape != (220, 5091):
        raise ValueError(f"unexpected score matrix dimensions: ColVec={colvec.shape} EVIE={evie.shape}")
    if len(q_index) != 220 or len(p_index) != 5091:
        raise ValueError("unexpected ColVec index dimensions")
    score_mats = {"colvec": (colvec, q_index, p_index), "evie": (evie, q_index, p_index)}
    page_text = {row["page_id"]: row.get("text", "") for row in read_jsonl(KDL_PATH)}

    _, base_rows = make_jobs(gate, inputs, page_text, score_mats, args.policy, args.scorer)
    if args.limit:
        chosen = set(sorted(inputs["labels"])[:args.limit])
        base_rows = [row for row in base_rows if row["qid"] in chosen]
    answer_job_map: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        # Recreate the exact answer prompt for stable content-addressed caching.
        qa = inputs["qa_rows"][row["qid"]]
        ctx = get_context(row["accurate_top_pages"], page_text)
        prompt = ANSWER_PROMPT.format(
            documents="\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(ctx)), query=qa["query"]
        )
        key = stable_hash(ANSWER_MODEL + "\n" + prompt)
        row["answer_prompt_hash"] = key
        answer_job_map.setdefault(key, {"hash": key, "qid": row["qid"], "ctx": ctx, "prompt": prompt})

    OUT.mkdir(parents=True, exist_ok=True)
    preflight = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": len({row["qid"] for row in base_rows}),
        "policy_scorer_rows": len(base_rows),
        "answer_prompts_unique": len(answer_job_map),
        "answer_cache_hits_local": sum((OUT / "cache/answers" / f"{key}.json").exists() for key in answer_job_map),
        "answer_cache_hits_legacy": sum(legacy_answer_path(job["qid"], job["ctx"]).exists() for job in answer_job_map.values()),
        "old_judge_cache_files": sum(1 for _ in (ROOT / "data/work/Cached-Ondemand-Eval/data/cache/judge").glob("*.json")),
        "mean_input_characters_per_prompt": float(np.mean([len(job["prompt"]) for job in answer_job_map.values()])) if answer_job_map else 0,
        "approx_answer_input_tokens_char_div_4": int(sum(len(job["prompt"]) for job in answer_job_map.values()) / 4),
        "models": {"answer": ANSWER_MODEL, "judge": JUDGE_MODEL},
        "workers": args.workers,
        "answer_max_tokens": MAX_ANSWER_TOKENS,
        "answer_reasoning": ANSWER_REASONING,
        "limit": args.limit or None,
        "policy": args.policy,
        "scorer": args.scorer,
        "accurate_rerank_timing_note": "Measures local sorting of each query's already cached ColVec/EVIE scores within the adaptive prefix; it excludes model inference/score-matrix generation, which was completed before this experiment.",
        "protocol": f"For each policy, use first K PP-OCRv5 candidates; rerank only this prefix by ColVec or EVIE; send first 10 ranked pages (one 512-word excerpt/page) to {ANSWER_MODEL}; judge with GPT-4o-mini. Qrel/gold page IDs are never sent; reference answers go only to the judge, not the QA answer model.",
    }
    atomic_json(OUT / "preflight.json", preflight)
    log(json.dumps(preflight, indent=2))
    if args.preflight:
        return
    expected_rows = 220 * (2 if args.policy == "all" else 1) * (1 if args.scorer != "all" else 2)
    if args.limit == 0 and len(base_rows) != expected_rows:
        raise ValueError(f"full run expected {expected_rows} policy/scorer rows, got {len(base_rows)}")
    load_api_key()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": FINGERPRINT,
        "query_count": len({row["qid"] for row in base_rows}),
        "policies": sorted({row["policy"] for row in base_rows}),
        "scorers": ["ColVec 1.1 8B", "EVIE-4.5B"],
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_prompt_note": "Pinned byte-identical ViDoRe Figure 24 prompt; preliminary near-match judgments are excluded from final scores.",
        "candidate_ranker": gate_variant if (gate_variant := gate.get("variant")) else "unknown",
        "candidate_cap": 100,
        "accurate_top_k": TOP_ACCURATE,
        "qa_context": f"one first-{PAGE_WORD_LIMIT}-word excerpt from each available accurately ranked page, up to K=10",
        "answer_prompt": ANSWER_PROMPT,
        "judge_prompt": JUDGE_PROMPT,
        "qrels_usage": "evaluation only; never passed to answer or judge prompts",
        "workers": args.workers,
        "answer_max_tokens": MAX_ANSWER_TOKENS,
        "answer_reasoning": ANSWER_REASONING,
    }
    atomic_json(OUT / "manifest.json", manifest)
    started = perf_counter()
    answer_results: dict[str, dict[str, Any]] = {}
    answer_jobs = list(answer_job_map.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(cached_or_call_answer, job, OUT): job for job in answer_jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            answer_results[job["hash"]] = future.result()
            if index % 50 == 0 or index == len(futures):
                log(f"[answer] {index}/{len(futures)} complete")
    log(f"[answer] elapsed {perf_counter() - started:.1f}s")
    for row in base_rows:
        answer = answer_results[row["answer_prompt_hash"]]
        row.update({
            "answer": answer.get("answer", ""),
            "answer_cache": answer.get("cache"),
            "answer_latency_seconds": answer.get("latency_seconds"),
            "answer_prompt_tokens": answer.get("prompt_tokens"),
            "answer_completion_tokens": answer.get("completion_tokens"),
            "answer_cost_usd": answer.get("cost_usd"),
            "answer_error": answer.get("error"),
        })
    run_judges(base_rows, inputs, OUT, args.workers)
    elapsed = perf_counter() - started

    summary: dict[str, Any] = {"elapsed_wall_seconds": elapsed, "rows": len(base_rows), "by_arm": {}, "by_source": {}}
    for policy in sorted({row["policy"] for row in base_rows}):
        for scorer in sorted({row["scorer"] for row in base_rows}):
            subset = [row for row in base_rows if row["policy"] == policy and row["scorer"] == scorer]
            if subset:
                summary["by_arm"][f"{policy}__{scorer}"] = metric_summary(subset)
                summary["by_source"][f"{policy}__{scorer}"] = {
                    source: metric_summary([row for row in subset if row["source"] == source])
                    for source in sorted({row["source"] for row in subset})
                }
    summary["unique_api_calls"] = unique_call_summary(
        OUT,
        {row["answer_prompt_hash"] for row in base_rows},
        {row["judge_prompt_hash"] for row in base_rows},
    )
    if args.policy != "all":
        full_first_pass = elapsed
        correction_pass = 0.0
    elif previous_summary.get("end_to_end_full_run_wall_seconds") is not None:
        full_first_pass = float(previous_summary["end_to_end_full_run_wall_seconds"])
        correction_pass = float(previous_summary.get("benchmark_prompt_correction_wall_seconds", previous_summary.get("elapsed_wall_seconds", 0.0)))
    else:
        full_first_pass = float(previous_summary.get("elapsed_wall_seconds", elapsed)) if previous_summary.get("rows") == 880 else elapsed
        correction_pass = 0.0
    summary["end_to_end_full_run_wall_seconds"] = full_first_pass
    summary["benchmark_prompt_correction_wall_seconds"] = correction_pass
    summary["cumulative_experiment_wall_seconds"] = full_first_pass + correction_pass
    summary["list_price_estimate_usd_if_provider_cost_missing"] = estimate_list_price(base_rows)
    atomic_json(OUT / "summary.json", summary)
    with (OUT / "per_query.jsonl").open("w", encoding="utf-8") as stream:
        for row in base_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = [
        "# Adaptive retrieval + Accurate top-10 QA evaluation", "",
        f"- Queries: {len({row['qid'] for row in base_rows})}; policy/scorer rows: {len(base_rows)}.",
        f"- Answer model: `{ANSWER_MODEL}`; judge: `{JUDGE_MODEL}`.",
        f"- Initial full pass: {summary['end_to_end_full_run_wall_seconds']:.1f}s; exact-prompt judge correction: {summary['benchmark_prompt_correction_wall_seconds']:.1f}s; cumulative experiment wall time: {summary['cumulative_experiment_wall_seconds']:.1f}s.",
        f"- Unique live API calls: {summary['unique_api_calls']['total_unique_api_calls']} (QA {summary['unique_api_calls']['answers']['live_api_calls']}, judge {summary['unique_api_calls']['judgments']['live_api_calls']}); provider-reported total cost ${summary['unique_api_calls']['total_unique_reported_cost_usd']:.4f}.",
        f"- Billed spend including the discarded preliminary judge pass: ${summary['unique_api_calls']['total_billed_including_discarded_first_judge_pass_usd']:.4f}; that pass is excluded from final scores.",
        "- Each K restricts the frozen top-100 PP-OCRv5 candidate list; ColVec/EVIE reranks only that prefix; up to 10 pages are sent to QA (one first-512-word excerpt per page).",
        "- Accurate-rerank latency measures local sorting of already cached score matrices; it excludes EVIE model inference/score generation, which was run earlier and has no timing record in this cache artifact.",
        f"- Gold page IDs/qrels were not sent to models; reference answers went only to the answer judge, never to the `{ANSWER_MODEL}` QA generator.", "",
        "## Overall", "",
        "| Adaptive policy | Accurate scorer | Mean K | Prefix recall | Recall@10 | nDCG@10 | QA Correct | Correct + Partial | Mean pages to QA | Rerank ms/query | Est./reported cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, m in summary["by_arm"].items():
        cost = m["answer_cost_usd"] + m["judge_cost_usd"]
        md.append(f"| {arm.rsplit('__', 1)[0]} | {arm.rsplit('__', 1)[1]} | {m['mean_k']:.2f} | {100*m['mean_candidate_prefix_recall']:.2f}% | {100*m['macro_recall_at_10']:.2f}% | {100*m['macro_ndcg_at_10']:.2f}% | {100*m['correct_only']:.2f}% | {100*m['correct_plus_partial']:.2f}% | {m['mean_accurate_pages_to_qa']:.2f} | {1000*m['accurate_rerank_seconds_mean_per_query']:.3f} | ${cost:.4f}* |")
    md += ["", "`*` Sum of provider-reported answer + benchmark-exact judge costs for one arm if run standalone; values overlap across arms because identical prompts were deduplicated. It excludes the discarded preliminary judge pass. If a provider omitted cost, consult `summary.json` for token counts and the list-price fallback estimate.", "", "## By dataset", ""]
    for arm, sources in summary["by_source"].items():
        md += [f"### {arm}", "", "| Dataset | n | Mean K | Prefix recall | Recall@10 | nDCG@10 | QA Correct | Correct + Partial | Incomplete-qrel QA credited | Strict Correct among those |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for source, m in sources.items():
            md.append(f"| {source} | {m['n']} | {m['mean_k']:.2f} | {100*m['mean_candidate_prefix_recall']:.2f}% | {100*m['macro_recall_at_10']:.2f}% | {100*m['macro_ndcg_at_10']:.2f}% | {100*m['correct_only']:.2f}% | {100*m['correct_plus_partial']:.2f}% | {m['qa_credited_despite_missing_qrel_pages']} | {m['qa_correct_despite_missing_qrel_pages']} |")
        md.append("")
    md += [
        "## Reading whether retrieved qrels are sufficient for answering", "",
        "`qa_credited_despite_missing_qrel_pages` counts answers judged Correct/Partially Correct even though the actual top-10 pages shown to QA omit one or more qrel gold pages; `qa_correct_despite_missing_qrel_pages` is the stricter Correct-only subset. This shows that full qrel coverage is not always necessary for the benchmark answer, but does not prove an inference-time LLM can safely stop. Conversely, `all_qrels_covered_by_qa_but_qa_not_credited` shows complete qrel coverage does not guarantee successful QA. Qrels are offline evaluation only.",
        "", "## Files", "", "- `per_query.jsonl`: every policy x scorer x query with K, page IDs, retrieval metrics, answer, judgment, token/cost/time details.", "- `cache/answers/` and `cache/judgments_pinned/`: final resumable request/response records; API keys are never saved.", "- `cache/judgments/`: preliminary near-match judge pass, retained for audit but excluded from final scores.", "- `preflight.json`, `manifest.json`, `summary.json`: protocol and aggregate results.",
    ]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    log(f"[done] wrote {OUT}")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
