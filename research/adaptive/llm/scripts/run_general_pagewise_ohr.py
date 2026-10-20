"""Minimal page-by-page general Adaptive-LLM experiment on five OHR queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Mapping, Sequence

import requests


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENTS = ROOT / "research/experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from adaptive_k_optimization_core import load_frozen_data, recall_at  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


MODEL = "deepseek/deepseek-v4-flash"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT = ROOT / "research/adaptive/llm/results/general_pagewise_ohr5_20261019"
SEED = 42


SYSTEM_PROMPT = """You control adaptive document retrieval.

A question is followed by retrieved pages, one page at a time. At every turn,
decide whether all pages seen in this conversation contain enough explicit
information to answer the question correctly and completely.

Do not answer the question. Do not summarize the pages.
Choose STOP only when the necessary answer is explicitly supported by the
pages already seen. Otherwise choose CONTINUE.

Return exactly one short JSON object and nothing else:
{"decision":"STOP or CONTINUE","reason":"one short sentence"}"""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False) + "\n")


def setup_logger(output: Path) -> logging.Logger:
    logger = logging.getLogger("general_pagewise_ohr")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output / "console.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def first_gold_rank(row: Mapping[str, Any]) -> int:
    gold = set(row["gold"])
    return next((rank for rank, page_id in enumerate(row["ranked"], 1) if page_id in gold), 201)


def choose_queries(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    rng = random.Random(SEED)
    ohr = [index for index, row in enumerate(rows) if row["source"] == "ohrbench"]
    groups = {
        "top1": [index for index in ohr if first_gold_rank(rows[index]) == 1],
        "top10": [index for index in ohr if 2 <= first_gold_rank(rows[index]) <= 10],
        "top20": [index for index in ohr if 11 <= first_gold_rank(rows[index]) <= 20],
    }
    selected = [rng.choice(sorted(groups["top1"]))]
    selected += rng.sample(sorted(groups["top10"]), 2)
    selected += rng.sample(sorted(groups["top20"]), 2)
    return selected


def parse_decision(text: str) -> tuple[str, str, str]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        payload = json.loads(value)
        decision = str(payload.get("decision") or "").upper()
        reason = str(payload.get("reason") or "").strip()
        if decision not in {"STOP", "CONTINUE"}:
            raise ValueError(f"invalid decision {decision!r}")
        return decision, reason, ""
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        return "CONTINUE", "Invalid JSON; continue conservatively.", str(error)


def request_hash(model: str, messages: Sequence[Mapping[str, str]]) -> str:
    raw = json.dumps({"model": model, "messages": list(messages)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ChatClient:
    def __init__(self, output: Path, model: str, resume: bool, logger: logging.Logger) -> None:
        self.output = output
        self.model = model
        self.logger = logger
        self.key = os.environ.get("OPENROUTER_API_KEY", "")
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is unset")
        self.cache_path = output / "request_cache.jsonl"
        self.cache: dict[str, dict[str, Any]] = {}
        if resume and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("ok"):
                    self.cache[str(item["request_hash"])] = item

    def complete(self, messages: Sequence[Mapping[str, str]], session_id: str) -> dict[str, Any]:
        digest = request_hash(self.model, messages)
        if digest in self.cache:
            return {**self.cache[digest], "cache_reused": True}
        body = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_object"},
            "session_id": session_id,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "X-Title": "AXIOM_DE-RD pagewise Adaptive LLM",
        }
        last_error = ""
        for attempt in range(1, 4):
            started = time.perf_counter()
            ttft = None
            pieces: list[str] = []
            usage: dict[str, Any] = {}
            provider = ""
            resolved_model = ""
            try:
                with requests.post(ENDPOINT, headers=headers, json=body, stream=True, timeout=180) as response:
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
                    for raw in response.iter_lines(decode_unicode=True):
                        if not raw or not raw.startswith("data:"):
                            continue
                        value = raw[5:].strip()
                        if value == "[DONE]":
                            break
                        chunk = json.loads(value)
                        if chunk.get("error"):
                            raise RuntimeError(str(chunk["error"])[:500])
                        provider = str(chunk.get("provider") or provider)
                        resolved_model = str(chunk.get("model") or resolved_model)
                        if isinstance(chunk.get("usage"), Mapping):
                            usage = dict(chunk["usage"])
                        choices = chunk.get("choices") or []
                        content = ((choices[0].get("delta") or {}).get("content") if choices else None)
                        if content:
                            if ttft is None:
                                ttft = time.perf_counter() - started
                            pieces.append(str(content))
                text = "".join(pieces).strip()
                if not text:
                    raise RuntimeError("empty streamed content")
                elapsed = time.perf_counter() - started
                details = usage.get("prompt_tokens_details") or {}
                record = {
                    "request_hash": digest,
                    "ok": True,
                    "attempt": attempt,
                    "model": resolved_model or self.model,
                    "provider": provider,
                    "text": text,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "cached_tokens": int(details.get("cached_tokens") or 0) if isinstance(details, Mapping) else 0,
                    "cost_usd": float(usage.get("cost") or 0.0),
                    "ttft_seconds": ttft,
                    "total_seconds": elapsed,
                    "created_at": time.time(),
                }
                append_jsonl(self.cache_path, record)
                self.cache[digest] = record
                return {**record, "cache_reused": False}
            except (requests.RequestException, RuntimeError, json.JSONDecodeError) as error:
                last_error = str(error)
                self.logger.warning("request retry %d/3: %s", attempt, last_error)
                if attempt < 3:
                    time.sleep(float(attempt))
        raise RuntimeError(f"OpenRouter failed after retries: {last_error}")


def safe_name(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)


def render_chat(path: Path, history: Mapping[str, Any]) -> None:
    lines = [
        f"# Chat history — {history['qid']}", "",
        f"- First gold rank: {history['first_gold_rank']}",
        f"- LLM chosen K: {history['chosen_k']}",
        f"- Recall at chosen K: {history['recall_at_chosen_k']:.2%}",
        f"- Stop type: {history['stop_type']}", "",
        "## System prompt", "", "```text", history["messages"][0]["content"], "```", "",
    ]
    for number, message in enumerate(history["messages"][1:], 1):
        role = str(message["role"]).upper()
        lines += [f"## Turn {number} — {role}", "", "```text", str(message["content"]), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_query(data: Any, row: Mapping[str, Any], cap: int, client: ChatClient, logger: logging.Logger) -> dict[str, Any]:
    qid = str(row["qid"])
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"QUESTION:\n{row['query']}"},
    ]
    steps = []
    chosen_k = cap
    stop_type = "cap_reached"
    for rank in range(1, cap + 1):
        page_index = int(row["ranked_indices"][rank - 1])
        page_id = data.page_ids[page_index]
        page_text = data.page_texts[page_index]
        page_message = (
            f"NEW RETRIEVED PAGE\nrank: {rank}\npage_id: {page_id}\n\n"
            f"{page_text}\n\nDecide STOP or CONTINUE now."
        )
        messages.append({"role": "user", "content": page_message})
        response = client.complete(messages, f"pagewise-ohr-{qid}")
        decision, reason, parse_error = parse_decision(str(response["text"]))
        messages.append({"role": "assistant", "content": str(response["text"])})
        step = {
            "rank": rank,
            "page_id": page_id,
            "is_gold": page_id in set(row["gold"]),
            "decision": decision,
            "reason": reason,
            "parse_error": parse_error,
            "request": {key: response.get(key) for key in (
                "request_hash", "cache_reused", "model", "provider", "prompt_tokens",
                "completion_tokens", "cached_tokens", "cost_usd", "ttft_seconds", "total_seconds",
            )},
        }
        steps.append(step)
        logger.info(
            "%s | rank=%d | gold=%s | %s | %s | %.2fs | $%.6f",
            qid, rank, step["is_gold"], decision, reason,
            float(response.get("total_seconds") or 0.0), float(response.get("cost_usd") or 0.0),
        )
        if decision == "STOP":
            chosen_k = rank
            stop_type = "llm_stop"
            break
    history = {
        "qid": qid,
        "source": row["source"],
        "query": row["query"],
        "first_gold_rank": first_gold_rank(row),
        "chosen_k": chosen_k,
        "stop_type": stop_type,
        "recall_at_chosen_k": recall_at(row, chosen_k),
        "recall_at_cap": recall_at(row, cap),
        "messages": messages,
        "steps": steps,
    }
    name = safe_name(qid)
    write_json(client.output / "histories" / f"{name}.json", history)
    render_chat(client.output / "chats" / f"{name}.md", history)
    return history


def report(output: Path, histories: Sequence[Mapping[str, Any]]) -> None:
    unique_requests = {}
    for history in histories:
        for step in history["steps"]:
            unique_requests.setdefault(step["request"]["request_hash"], step["request"])
    requests = list(unique_requests.values())
    payload = {
        "queries": len(histories),
        "mean_k": sum(int(item["chosen_k"]) for item in histories) / len(histories),
        "macro_recall": sum(float(item["recall_at_chosen_k"]) for item in histories) / len(histories),
        "premature_stops": sum(
            item["stop_type"] == "llm_stop" and item["chosen_k"] < item["first_gold_rank"]
            for item in histories
        ),
        "requests": len(requests),
        "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in requests),
        "completion_tokens": sum(int(item.get("completion_tokens") or 0) for item in requests),
        "cached_tokens": sum(int(item.get("cached_tokens") or 0) for item in requests),
        "cost_usd": sum(float(item.get("cost_usd") or 0.0) for item in requests),
        "mean_request_seconds": sum(float(item.get("total_seconds") or 0.0) for item in requests) / max(len(requests), 1),
    }
    write_json(output / "report.json", payload)
    lines = [
        "# OHR general pagewise Adaptive-LLM", "",
        "| QID | First gold | Chosen K | Recall | Stop | Chat |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in histories:
        filename = safe_name(str(item["qid"])) + ".md"
        lines.append(
            f"| {item['qid']} | {item['first_gold_rank']} | {item['chosen_k']} | "
            f"{item['recall_at_chosen_k']:.2%} | {item['stop_type']} | [log](chats/{filename}) |"
        )
    lines += [
        "", "## Usage", "",
        f"- Requests: {payload['requests']}",
        f"- Cost: ${payload['cost_usd']:.6f}",
        f"- Mean request time: {payload['mean_request_seconds']:.2f}s",
        f"- Premature stops before first gold: {payload['premature_stops']}",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--cap", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--materialize-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.cap <= 200:
        raise ValueError("cap must be between 1 and 200")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir)
    data = load_frozen_data()
    indices = choose_queries(data.rows)
    selection = [{
        "qid": data.rows[index]["qid"],
        "query": data.rows[index]["query"],
        "first_gold_rank": first_gold_rank(data.rows[index]),
    } for index in indices]
    write_json(args.output_dir / "selected_queries.json", selection)
    (args.output_dir / "prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    write_json(args.output_dir / "manifest.json", {
        "model": args.model,
        "cap": args.cap,
        "page_batch_size": 1,
        "conversation": "full message history is resent at every turn",
        "selection": selection,
        "qrels_in_prompt": False,
    })
    logger.info("selected queries: %s", json.dumps(selection, ensure_ascii=False))
    if args.materialize_only:
        return 0
    load_dotenv_file(ROOT)
    client = ChatClient(args.output_dir, args.model, args.resume, logger)
    histories = []
    for number, index in enumerate(indices, 1):
        logger.info("starting query %d/%d: %s", number, len(indices), data.rows[index]["qid"])
        histories.append(run_query(data, data.rows[index], args.cap, client, logger))
    report(args.output_dir, histories)
    logger.info("completed %d queries; report=%s", len(histories), args.output_dir / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
