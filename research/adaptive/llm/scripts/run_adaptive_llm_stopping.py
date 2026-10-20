"""Adaptive LLM stopping and historical false-hard rescue.

The runner consumes the frozen PP-OCRv5/Qwen top-200 artifact. Qrels and
reference answers are used only after a trajectory is complete; neither is
included in an OpenRouter request.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "research" / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from adaptive_k_optimization_core import (  # noqa: E402
    BENCHMARK_ROOT,
    FrozenData,
    load_frozen_data,
    read_jsonl,
    recall_at,
    required_k,
)
from src.utils.env import load_dotenv_file  # noqa: E402


DEFAULT_OUTPUT = ROOT / "research/adaptive/llm/results/adaptive_llm_stopping_20261019"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_VERSION = "adaptive-llm-stopping-v1.1"
INPUT_USD_PER_M = 0.0679
OUTPUT_USD_PER_M = 0.168
SEED = 42


GENERAL_SYSTEM = """You are an evidence-sufficiency controller for adaptive document retrieval.

You receive:
1. a question;
2. a compact evidence state produced at the previous step;
3. a new batch of retrieved pages in descending retrieval rank.

Use only the supplied pages. Do not use unstated world knowledge as evidence.

Your task is to decide whether the accumulated evidence is sufficient to
produce a correct, specific, fully grounded answer to the question.

STOP is allowed only when:
- every material part of the question is resolved;
- the answer is directly supported by one or more seen pages;
- the cited page IDs actually contain the supporting evidence;
- no essential value, condition, entity, comparison, step, or qualification
  remains missing;
- the answer does not depend on a plausible but unsupported inference.

Do not require more pages merely because more corroboration might exist.
If the current evidence is partial, ambiguous, contradictory, or only
topically related, choose CONTINUE.

Return JSON only. Keep the evidence state compact and reusable in the next step."""


HARD_SYSTEM = """You are a conservative false-hard detector for an adaptive retrieval system.

A statistical model routed this query to a large retrieval budget, but that
routing can be wrong. Determine whether the pages seen so far already contain
enough evidence for a correct and complete grounded answer.

The statistical model's hard label is not evidence. Do not preserve it
automatically, but do not downgrade the query merely because an answer sounds
plausible.

Choose STOP only when the current prefix can safely replace the larger
retrieval budget:
- all requested parts are answered;
- supporting statements are explicit in the seen pages;
- at least one valid page ID is cited;
- no required fact, condition, number, comparison, or explanation is missing.

Otherwise choose CONTINUE and state exactly what evidence is still needed.
Use only supplied pages and return JSON only."""


GENERAL_SCHEMA = {
    "decision": "STOP | CONTINUE",
    "confidence": "number from 0 to 1",
    "provisional_answer": "string",
    "cited_page_ids": ["string"],
    "resolved_claims": [{"claim": "string", "page_ids": ["string"]}],
    "missing_information": ["string"],
    "contradictions": ["string"],
    "state_summary": "string",
}


HARD_SCHEMA = {
    "decision": "STOP | CONTINUE",
    "safe_to_use_current_k": "boolean",
    "confidence": "number from 0 to 1",
    "provisional_answer": "string",
    "cited_page_ids": ["string"],
    "resolved_claims": [{"claim": "string", "page_ids": ["string"]}],
    "missing_information": ["string"],
    "false_hard_reason": "string",
    "state_summary": "string",
}


@dataclass
class Budget:
    max_calls: int
    max_cost_usd: float
    calls: int = 0
    cost_usd: float = 0.0

    def check(self) -> None:
        if self.calls >= self.max_calls:
            raise RuntimeError(f"API call budget exhausted ({self.calls}/{self.max_calls})")
        if self.cost_usd >= self.max_cost_usd:
            raise RuntimeError(f"API cost budget exhausted (${self.cost_usd:.6f}/${self.max_cost_usd:.2f})")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    return parsed


def clipped_text(value: Any, limit: int = 1800) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def string_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clipped_text(item, 500) for item in value[:limit] if str(item or "").strip()]


def normalize_result(parsed: Mapping[str, Any], mode: str) -> dict[str, Any]:
    claims = []
    for item in parsed.get("resolved_claims") or []:
        if isinstance(item, Mapping) and str(item.get("claim") or "").strip():
            claims.append({
                "claim": clipped_text(item.get("claim"), 700),
                "page_ids": string_list(item.get("page_ids"), 8),
            })
        if len(claims) >= 8:
            break
    result = {
        "decision": str(parsed.get("decision") or "CONTINUE").upper(),
        "confidence": max(0.0, min(1.0, float(parsed.get("confidence") or 0.0))),
        "provisional_answer": clipped_text(parsed.get("provisional_answer"), 2200),
        "cited_page_ids": string_list(parsed.get("cited_page_ids"), 20),
        "resolved_claims": claims,
        "missing_information": string_list(parsed.get("missing_information"), 8),
        "state_summary": clipped_text(parsed.get("state_summary"), 1500),
    }
    if mode == "general":
        result["contradictions"] = string_list(parsed.get("contradictions"), 8)
    else:
        result["safe_to_use_current_k"] = bool(parsed.get("safe_to_use_current_k", False))
        result["false_hard_reason"] = clipped_text(parsed.get("false_hard_reason"), 1000)
    return result


def next_state(result: Mapping[str, Any], mode: str) -> dict[str, Any]:
    keys = [
        "provisional_answer", "cited_page_ids", "resolved_claims",
        "missing_information", "state_summary",
    ]
    keys.append("contradictions" if mode == "general" else "false_hard_reason")
    return {key: result.get(key) for key in keys}


def accepted_stop(result: Mapping[str, Any], seen_page_ids: set[str], mode: str) -> tuple[bool, list[str]]:
    reasons = []
    citations = {str(item) for item in result.get("cited_page_ids") or []}
    invalid = sorted(citations - seen_page_ids)
    if str(result.get("decision")) != "STOP":
        reasons.append("decision_not_stop")
    if float(result.get("confidence") or 0.0) < 0.85:
        reasons.append("confidence_below_0.85")
    if not str(result.get("provisional_answer") or "").strip():
        reasons.append("missing_answer")
    if not citations:
        reasons.append("missing_citation")
    if invalid:
        reasons.append("invalid_citation")
    if result.get("missing_information"):
        reasons.append("missing_information_nonempty")
    if mode == "hard-only" and not bool(result.get("safe_to_use_current_k")):
        reasons.append("safe_to_use_current_k_false")
    return not reasons, reasons


def page_block(data: FrozenData, row: Mapping[str, Any], start: int, end: int) -> str:
    blocks = []
    for position in range(start, end):
        page_index = int(row["ranked_indices"][position])
        blocks.append(
            f"<page rank=\"{position + 1}\" page_id=\"{data.page_ids[page_index]}\">\n"
            f"{data.page_texts[page_index]}\n</page>"
        )
    return "\n\n".join(blocks)


def user_prompt(
    mode: str,
    query: str,
    state: Mapping[str, Any],
    pages: str,
    current_k: int,
) -> str:
    state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if mode == "general":
        return (
            f"QUESTION:\n{query}\n\nPREVIOUS EVIDENCE STATE:\n{state_json}\n\n"
            f"NEW RETRIEVED PAGES:\n{pages}\n\nReturn this JSON shape:\n"
            f"{json.dumps(GENERAL_SCHEMA, ensure_ascii=False)}"
        )
    return (
        f"QUESTION:\n{query}\n\nCURRENT RETRIEVAL PREFIX:\nranks 1 through {current_k}\n\n"
        f"PREVIOUS EVIDENCE STATE:\n{state_json}\n\nNEW PAGES IN THIS STEP:\n{pages}\n\n"
        f"Return this JSON shape:\n{json.dumps(HARD_SCHEMA, ensure_ascii=False)}"
    )


class OpenRouterClient:
    def __init__(
        self,
        *,
        model: str,
        max_output_tokens: int,
        cache_path: Path,
        budget: Budget,
        resume: bool,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.cache_path = cache_path
        self.budget = budget
        self.timeout = timeout
        self._lock = threading.RLock()
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is unset")
        self.cache: dict[str, dict[str, Any]] = {}
        if resume and cache_path.is_file():
            for row in read_jsonl(cache_path):
                if row.get("ok") and row.get("request_hash"):
                    self.cache[str(row["request_hash"])] = row

    def complete(self, *, system: str, user: str, session_id: str) -> dict[str, Any]:
        identity = {
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
            "system": system,
            "user": user,
            "max_output_tokens": self.max_output_tokens,
            "reasoning": {"enabled": False},
        }
        request_hash = stable_hash(identity)
        with self._lock:
            if request_hash in self.cache:
                return {**self.cache[request_hash], "cache_reused": True}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
            "session_id": session_id,
            "stream": True,
            "stream_options": {"include_usage": True},
            "metadata": {"experiment": "adaptive_llm_stopping", "prompt_version": PROMPT_VERSION},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "AXIOM_DE-RD adaptive LLM stopping",
        }
        last_error = ""
        for attempt in range(1, 4):
            with self._lock:
                self.budget.check()
                self.budget.calls += 1
            started = time.perf_counter()
            first_token_seconds: float | None = None
            pieces: list[str] = []
            reasoning_pieces: list[str] = []
            usage: dict[str, Any] = {}
            resolved_model = ""
            provider = ""
            try:
                with requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                    stream=True,
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                    for raw_line in response.iter_lines(decode_unicode=True):
                        if not raw_line or not raw_line.startswith("data:"):
                            continue
                        value = raw_line[5:].strip()
                        if value == "[DONE]":
                            break
                        chunk = json.loads(value)
                        if chunk.get("error"):
                            raise RuntimeError(str(chunk["error"])[:500])
                        resolved_model = str(chunk.get("model") or resolved_model)
                        provider = str(chunk.get("provider") or provider)
                        if isinstance(chunk.get("usage"), Mapping):
                            usage = dict(chunk["usage"])
                        choices = chunk.get("choices") or []
                        delta = (choices[0].get("delta") or {}) if choices else {}
                        content = delta.get("content")
                        if content:
                            if first_token_seconds is None:
                                first_token_seconds = time.perf_counter() - started
                            pieces.append(str(content))
                        reasoning = delta.get("reasoning")
                        if reasoning:
                            reasoning_pieces.append(str(reasoning))
                text = "".join(pieces).strip()
                elapsed = time.perf_counter() - started
                prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
                cached_tokens = int(details.get("cached_tokens") or 0) if isinstance(details, Mapping) else 0
                actual_cost = usage.get("cost")
                estimated = actual_cost is None
                cost = float(actual_cost) if actual_cost is not None else (
                    prompt_tokens * INPUT_USD_PER_M + completion_tokens * OUTPUT_USD_PER_M
                ) / 1_000_000
                with self._lock:
                    self.budget.cost_usd += cost
                if not text:
                    empty_record = {
                        "request_hash": request_hash,
                        "ok": False,
                        "attempt": attempt,
                        "requested_model": self.model,
                        "resolved_model": resolved_model,
                        "provider": provider,
                        "session_id": session_id,
                        "error": "empty streamed content",
                        "reasoning_chars": len("".join(reasoning_pieces)),
                        "usage": usage,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "cached_tokens": cached_tokens,
                        "cost_usd": cost,
                        "cost_is_estimated": estimated,
                        "total_seconds": elapsed,
                        "created_at": time.time(),
                    }
                    with self._lock:
                        append_jsonl(self.cache_path, empty_record)
                    last_error = (
                        "empty streamed content "
                        f"(reasoning_chars={empty_record['reasoning_chars']}, completion_tokens={completion_tokens})"
                    )
                    if attempt < 3:
                        time.sleep(float(attempt))
                        continue
                    break
                record = {
                    "request_hash": request_hash,
                    "ok": True,
                    "attempt": attempt,
                    "requested_model": self.model,
                    "resolved_model": resolved_model,
                    "provider": provider,
                    "session_id": session_id,
                    "text": text,
                    "usage": usage,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cached_tokens": cached_tokens,
                    "cost_usd": cost,
                    "cost_is_estimated": estimated,
                    "ttft_seconds": first_token_seconds,
                    "total_seconds": elapsed,
                    "created_at": time.time(),
                }
                with self._lock:
                    append_jsonl(self.cache_path, record)
                    self.cache[request_hash] = record
                return {**record, "cache_reused": False}
            except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
                last_error = str(error)
                if attempt < 3:
                    time.sleep(float(attempt))
        failure = {
            "request_hash": request_hash,
            "ok": False,
            "requested_model": self.model,
            "session_id": session_id,
            "error": last_error,
            "created_at": time.time(),
        }
        with self._lock:
            append_jsonl(self.cache_path, failure)
        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def trajectory_id(mode: str, qid: str, batch_size: int, initial_k: int, cap: int) -> str:
    return f"{mode}::{qid}::b{batch_size}::i{initial_k}::c{cap}"


def run_trajectory(
    *,
    data: FrozenData,
    row: Mapping[str, Any],
    mode: str,
    batch_size: int,
    initial_k: int,
    cap: int,
    client: OpenRouterClient,
    max_steps: int | None = None,
) -> dict[str, Any]:
    qid = str(row["qid"])
    seen: set[str] = set()
    state: dict[str, Any] = {}
    steps = []
    current = 0
    step_number = 0
    while current < cap:
        start = current
        if mode == "hard-only" and current == 0:
            end = min(initial_k, cap)
        else:
            end = min(current + batch_size, cap)
        block = page_block(data, row, start, end)
        for page_id in row["ranked"][start:end]:
            seen.add(str(page_id))
        prompt = user_prompt(mode, str(row["query"]), state, block, end)
        system = GENERAL_SYSTEM if mode == "general" else HARD_SYSTEM
        response = client.complete(
            system=system,
            user=prompt,
            session_id=f"adaptive-llm-{PROMPT_VERSION}-{mode}-{qid}",
        )
        parse_error = ""
        try:
            normalized = normalize_result(parse_json_object(str(response["text"])), mode)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            parse_error = str(error)
            normalized = normalize_result({}, mode)
        stop, rejected = accepted_stop(normalized, seen, mode)
        step = {
            "step": step_number,
            "rank_start": start + 1,
            "rank_end": end,
            "new_page_ids": list(row["ranked"][start:end]),
            "parse_error": parse_error,
            "accepted_stop": stop,
            "stop_rejection_reasons": rejected,
            "result": normalized,
            "request": {
                key: response.get(key)
                for key in (
                    "request_hash", "cache_reused", "requested_model", "resolved_model",
                    "provider", "prompt_tokens", "completion_tokens", "cached_tokens",
                    "cost_usd", "cost_is_estimated", "ttft_seconds", "total_seconds",
                )
            },
        }
        steps.append(step)
        current = end
        step_number += 1
        state = next_state(normalized, mode)
        if stop:
            break
        if max_steps is not None and step_number >= max_steps:
            break
    cap_reached = current >= cap and not bool(steps[-1]["accepted_stop"])
    chosen_k = current
    return {
        "trajectory_id": trajectory_id(mode, qid, batch_size, initial_k, cap),
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "qid": qid,
        "source": row["source"],
        "query": row["query"],
        "batch_size": batch_size,
        "initial_k": initial_k,
        "cap": cap,
        "chosen_k": chosen_k,
        "stopped_by_llm": bool(steps[-1]["accepted_stop"]),
        "cap_reached": cap_reached,
        "calls": len(steps),
        "macro_recall_contribution": recall_at(row, chosen_k),
        "recall_at_30": recall_at(row, 30),
        "recall_at_60": recall_at(row, 60),
        "recall_at_100": recall_at(row, 100),
        "recall_at_200": recall_at(row, 200),
        "safe_30_vs_100": math.isclose(recall_at(row, 30), recall_at(row, 100)),
        "safe_30_vs_200": math.isclose(recall_at(row, 30), recall_at(row, 200)),
        "steps": steps,
    }


def historical_hard_mask(data: FrozenData) -> np.ndarray:
    artifact = json.loads(
        (ROOT / "data/work/adaptive_retrieval_20261009/score_confidence_gate.json").read_text(encoding="utf-8")
    )
    probability = np.asarray(artifact["oof_probability"], dtype=float)
    if len(probability) != len(data.rows):
        raise ValueError("Historical HGB artifact is not aligned to frozen rows")
    hard = np.zeros(len(data.rows), dtype=bool)
    hard[np.argsort(-probability, kind="stable")[:134]] = True
    return hard


def choose_one(indices: Sequence[int], rng: random.Random) -> int:
    values = sorted(int(item) for item in indices)
    if not values:
        raise ValueError("Cannot select from an empty diagnostic group")
    return values[rng.randrange(len(values))]


def select_smoke(data: FrozenData) -> dict[str, Any]:
    rng = random.Random(SEED)
    hard = historical_hard_mask(data)
    sources = np.asarray([str(row["source"]) for row in data.rows])
    safe = np.asarray([
        math.isclose(recall_at(row, 30), recall_at(row, 100)) for row in data.rows
    ])
    probe1 = []
    selected_by_source: dict[str, dict[str, str]] = {}
    for source in sorted(set(sources)):
        safe_index = choose_one(np.flatnonzero(hard & safe & (sources == source)), rng)
        unsafe_index = choose_one(np.flatnonzero(hard & ~safe & (sources == source)), rng)
        probe1.extend([safe_index, unsafe_index])
        selected_by_source[source] = {
            "false_hard": str(data.rows[safe_index]["qid"]),
            "true_hard": str(data.rows[unsafe_index]["qid"]),
        }

    safe_candidates = [index for index in probe1 if safe[index]]
    unsafe_candidates = [index for index in probe1 if not safe[index]]
    losses = {index: recall_at(data.rows[index], 100) - recall_at(data.rows[index], 30) for index in unsafe_candidates}
    probe2 = [
        safe_candidates[0],
        min(unsafe_candidates, key=lambda index: losses[index]),
        max(unsafe_candidates, key=lambda index: losses[index]),
    ]

    required = np.asarray([required_k(row, 1.0)[0] for row in data.rows])
    groups = [
        np.flatnonzero(required <= 10),
        np.flatnonzero((required > 10) & (required <= 70)),
        np.flatnonzero(required > 70),
    ]
    probe3 = [choose_one(group, rng) for group in groups]
    return {
        "seed": SEED,
        "probe1_indices": probe1,
        "probe2_indices": probe2,
        "probe3_indices": probe3,
        "probe1_by_source": selected_by_source,
        "probe2_qids": [str(data.rows[index]["qid"]) for index in probe2],
        "probe3_qids": [str(data.rows[index]["qid"]) for index in probe3],
        "probe3_required_k": [int(required[index]) for index in probe3],
    }


def aggregate_trajectories(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not trajectories:
        return {"queries": 0}
    chosen = np.asarray([int(row["chosen_k"]) for row in trajectories])
    recalls = np.asarray([float(row["macro_recall_contribution"]) for row in trajectories])
    steps = [step for row in trajectories for step in row["steps"]]
    requests_by_hash: dict[str, Mapping[str, Any]] = {}
    for step in steps:
        request = step["request"]
        requests_by_hash.setdefault(str(request["request_hash"]), request)
    requests = list(requests_by_hash.values())
    return {
        "queries": len(trajectories),
        "mean_k": float(np.mean(chosen)),
        "median_k": float(np.median(chosen)),
        "p90_k": float(np.percentile(chosen, 90)),
        "macro_recall": float(np.mean(recalls)),
        "llm_stops": int(sum(bool(row["stopped_by_llm"]) for row in trajectories)),
        "parse_errors": int(sum(bool(step["parse_error"]) for step in steps)),
        "invalid_citation_steps": int(sum("invalid_citation" in step["stop_rejection_reasons"] for step in steps)),
        "unique_requests": len(requests),
        "prompt_tokens": int(sum(int(item.get("prompt_tokens") or 0) for item in requests)),
        "completion_tokens": int(sum(int(item.get("completion_tokens") or 0) for item in requests)),
        "cached_tokens": int(sum(int(item.get("cached_tokens") or 0) for item in requests)),
        "cost_usd": float(sum(float(item.get("cost_usd") or 0.0) for item in requests)),
        "mean_ttft_seconds": float(np.mean([item["ttft_seconds"] for item in requests if item.get("ttft_seconds") is not None])) if requests else 0.0,
        "mean_request_seconds": float(np.mean([item["total_seconds"] for item in requests if item.get("total_seconds") is not None])) if requests else 0.0,
    }


def smoke_report(data: FrozenData, selection: Mapping[str, Any], trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    probe1_qids = {str(data.rows[index]["qid"]) for index in selection["probe1_indices"]}
    p1 = [row for row in trajectories if row.get("probe") == "probe1" and row["qid"] in probe1_qids]
    stopped = [row for row in p1 if row["stopped_by_llm"]]
    safe100 = [row for row in stopped if row["safe_30_vs_100"]]
    true_hard = [row for row in p1 if not row["safe_30_vs_100"]]
    unsafe_stops = [row for row in stopped if not row["safe_30_vs_100"]]
    all_steps = [step for row in trajectories for step in row["steps"]]
    unique: dict[str, Mapping[str, Any]] = {}
    for step in all_steps:
        unique.setdefault(str(step["request"]["request_hash"]), step["request"])
    resolved_models = sorted({str(item.get("resolved_model") or "") for item in unique.values()})
    model_mismatch = any(
        value and "deepseek-v4" not in value.lower() for value in resolved_models
    )
    acceptance = {
        "all_responses_parse": all(not step["parse_error"] for step in all_steps),
        "all_citations_valid": all("invalid_citation" not in step["stop_rejection_reasons"] for step in all_steps),
        "within_cost_cap": sum(float(item.get("cost_usd") or 0.0) for item in unique.values()) <= 0.50,
        "unsafe_stops_at_k30_at_most_one": len(unsafe_stops) <= 1,
        "usage_present_every_request": all(
            int(item.get("prompt_tokens") or 0) > 0 and int(item.get("completion_tokens") or 0) > 0
            for item in unique.values()
        ),
        "resolved_model_matches_family": not model_mismatch,
    }
    return {
        "selection": {key: value for key, value in selection.items() if not key.endswith("_indices")},
        "overall": aggregate_trajectories(trajectories),
        "by_probe": {
            probe: aggregate_trajectories([row for row in trajectories if row.get("probe") == probe])
            for probe in ("probe1", "probe2", "probe3")
        },
        "probe1_false_hard": {
            "queries": len(p1),
            "stops_at_30": len(stopped),
            "safe_stops_vs_100": len(safe100),
            "unsafe_stops_vs_100": len(unsafe_stops),
            "safe_stop_precision_vs_100": len(safe100) / len(stopped) if stopped else 1.0,
            "true_hard_queries": len(true_hard),
        },
        "resolved_models": resolved_models,
        "acceptance": acceptance,
        "accepted": all(acceptance.values()),
        "note": "Exploratory smoke test; qrels were used for sampling and post-hoc evaluation only.",
    }


def render_report(path: Path, report: Mapping[str, Any], trajectories: Sequence[Mapping[str, Any]]) -> None:
    overall = report["overall"]
    lines = [
        "# Adaptive LLM stopping smoke test",
        "",
        f"Model: `{DEFAULT_MODEL}`  ",
        f"Prompt: `{PROMPT_VERSION}`  ",
        "Status: **PASS**" if report["accepted"] else "Status: **FAIL / diagnostic only**",
        "",
        "## Overall usage",
        "",
        "| Trajectories | Requests | Prompt tokens | Completion tokens | Cost USD | Mean TTFT | Mean request |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {overall['queries']} | {overall['unique_requests']} | {overall['prompt_tokens']} | "
        f"{overall['completion_tokens']} | ${overall['cost_usd']:.6f} | "
        f"{overall['mean_ttft_seconds']:.2f}s | {overall['mean_request_seconds']:.2f}s |",
        "",
        "## Probe 1: false-hard at K=30",
        "",
        "| QID | Source | Safe 30→100 | LLM STOP | Confidence | Recall@30 | Recall@100 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in [item for item in trajectories if item.get("probe") == "probe1"]:
        confidence = row["steps"][-1]["result"]["confidence"]
        lines.append(
            f"| {row['qid']} | {row['source']} | {row['safe_30_vs_100']} | "
            f"{row['stopped_by_llm']} | {confidence:.2f} | {row['recall_at_30']:.2%} | {row['recall_at_100']:.2%} |"
        )
    lines += [
        "",
        "## All trajectories",
        "",
        "| Probe | Mode | QID | Batch | Chosen K | Recall | Calls | Stop | Cost USD |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trajectories:
        hashes = {}
        for step in row["steps"]:
            hashes.setdefault(step["request"]["request_hash"], step["request"])
        cost = sum(float(item.get("cost_usd") or 0.0) for item in hashes.values())
        lines.append(
            f"| {row.get('probe')} | {row['mode']} | {row['qid']} | {row['batch_size']} | "
            f"{row['chosen_k']} | {row['macro_recall_contribution']:.2%} | {row['calls']} | "
            f"{row['stopped_by_llm']} | ${cost:.6f} |"
        )
    lines += ["", "## Acceptance", ""]
    for key, value in report["acceptance"].items():
        lines.append(f"- {'PASS' if value else 'FAIL'} — `{key}`")
    lines += ["", "This is an exploratory smoke test, not a validated comparison with XGBoost.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_smoke(args: argparse.Namespace, data: FrozenData, client: OpenRouterClient) -> list[dict[str, Any]]:
    selection = select_smoke(data)
    serializable_selection = {
        key: value for key, value in selection.items() if not key.endswith("_indices")
    }
    serializable_selection["probe1_qids"] = [str(data.rows[index]["qid"]) for index in selection["probe1_indices"]]
    write_json(args.output_dir / "smoke_selection.json", serializable_selection)
    trajectories: list[dict[str, Any]] = []

    # Probe 1: one top-30 request for six hard-routed queries.
    for index in selection["probe1_indices"]:
        row = run_trajectory(
            data=data, row=data.rows[index], mode="hard-only", batch_size=10,
            initial_k=30, cap=30, client=client,
        )
        row["probe"] = "probe1"
        trajectories.append(row)

    # Probe 2: reuse the identical top-30 request and continue to K=60.
    for batch_size in (3, 5, 10):
        for index in selection["probe2_indices"]:
            row = run_trajectory(
                data=data, row=data.rows[index], mode="hard-only", batch_size=batch_size,
                initial_k=30, cap=60, client=client,
            )
            row["probe"] = "probe2"
            trajectories.append(row)

    # Probe 3: easy/middle/hard general-adaptive examples.
    for index in selection["probe3_indices"]:
        row = run_trajectory(
            data=data, row=data.rows[index], mode="general", batch_size=10,
            initial_k=0, cap=60, client=client,
        )
        row["probe"] = "probe3"
        trajectories.append(row)
    report = smoke_report(data, selection, trajectories)
    write_jsonl(args.output_dir / "smoke_trajectories.jsonl", trajectories)
    write_json(args.output_dir / "smoke_report.json", report)
    write_json(args.output_dir / "pareto.json", {
        "probe2": [
            {
                "qid": row["qid"], "batch_size": row["batch_size"],
                "chosen_k": row["chosen_k"], "recall": row["macro_recall_contribution"],
                "calls": row["calls"],
            }
            for row in trajectories if row.get("probe") == "probe2"
        ]
    })
    render_report(args.output_dir / "report.md", report, trajectories)
    return trajectories


def selected_indices(data: FrozenData, mode: str, limit_qids: Sequence[str]) -> list[int]:
    requested = set(limit_qids)
    hard = historical_hard_mask(data)
    indices = []
    for index, row in enumerate(data.rows):
        if requested and str(row["qid"]) not in requested:
            continue
        if mode == "hard-only" and not hard[index]:
            continue
        indices.append(index)
    missing = requested - {str(data.rows[index]["qid"]) for index in indices}
    if missing:
        raise ValueError(f"Requested qids unavailable for {mode}: {sorted(missing)}")
    return indices


def materialize(args: argparse.Namespace, data: FrozenData) -> dict[str, Any]:
    selection = select_smoke(data)
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "protocol": data.protocol,
        "smoke_selection": {
            "probe1_qids": [str(data.rows[index]["qid"]) for index in selection["probe1_indices"]],
            "probe2_qids": selection["probe2_qids"],
            "probe3_qids": selection["probe3_qids"],
            "probe3_required_k": selection["probe3_required_k"],
        },
        "guards": {
            "confidence": 0.85,
            "max_calls": args.max_calls,
            "max_cost_usd": args.max_cost_usd,
            "no_qrels_or_answers_in_prompt": True,
        },
    }
    write_json(args.output_dir / "manifest.json", payload)
    (args.output_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts/general_system.txt").write_text(GENERAL_SYSTEM + "\n", encoding="utf-8")
    (args.output_dir / "prompts/hard_only_system.txt").write_text(HARD_SYSTEM + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("general", "hard-only", "both"), default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, choices=(3, 5, 10), default=10)
    parser.add_argument("--initial-k", type=int, default=30)
    parser.add_argument("--cap", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=350)
    parser.add_argument("--max-cost-usd", type=float, default=0.50)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--limit-qids", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--materialize-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.initial_k <= args.cap <= 200:
        raise ValueError("Require 1 <= initial-k <= cap <= 200")
    if args.max_calls <= 0 or args.max_cost_usd <= 0:
        raise ValueError("Budgets must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_frozen_data()
    manifest = materialize(args, data)
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
    if args.smoke:
        trajectories = run_smoke(args, data, client)
    else:
        trajectories = []
        modes = ("general", "hard-only") if args.mode == "both" else (args.mode,)
        for mode in modes:
            for index in selected_indices(data, mode, args.limit_qids):
                trajectories.append(run_trajectory(
                    data=data, row=data.rows[index], mode=mode,
                    batch_size=args.batch_size, initial_k=args.initial_k,
                    cap=args.cap, client=client,
                ))
        write_jsonl(args.output_dir / "trajectories.jsonl", trajectories)
        write_json(args.output_dir / "usage_report.json", aggregate_trajectories(trajectories))
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "trajectories": len(trajectories),
        "new_api_calls": client.budget.calls,
        "new_cost_usd": client.budget.cost_usd,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
