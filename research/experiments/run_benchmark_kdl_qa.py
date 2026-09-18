"""Answer evaluation for a completed benchmark KDL chunks retrieval run.

This consumes the *context* companion emitted by
``run_benchmark_kdl_second.py``.  It deliberately does not parse, chunk,
embed, or retrieve again: each answer receives precisely the ranked top-K
chunk texts saved by the second-retrieval run.

The defaults mirror ``pipeline.docbench-on-demand-basic.yaml``:
DeepSeek V4 Flash generates, and GPT-4o mini is the independent binary judge.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.generate import MAX_CONTEXT_CHARS, MAX_OUTPUT_TOKENS, ContextChunk, generate
from src.evaluation.llm import complete
from src.utils.env import load_dotenv_file
from src.utils.paths import repo_root


ROOT = repo_root(__file__)
load_dotenv_file(ROOT)


# Kept byte-for-byte with research/data_discovery/run_docbench_e2e.py.  The
# benchmark adapter changes here, but the QA instrument must not change.
DOCBENCH_JUDGE_PROMPT = """Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {{question}}
User Answer: {{sys_ans}}
Reference Answer: {{ref_ans}}
Reference Text: {{ref_text}}

Evaluation Form (score ONLY):
- Correctness:"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _identity(args: argparse.Namespace) -> str:
    payload = {
        "context_run": str(args.context_run.resolve()),
        "generator": args.generator,
        "judge": args.judge,
        "top_k_context": args.top_k_context,
        "max_context_chars": args.max_context_chars,
        "max_unit_chars": args.max_unit_chars,
        "max_output_tokens": args.max_output_tokens,
        "style": "binary_default",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[ContextChunk]]]:
    queries = _read_jsonl(args.dataset_root / "queries.jsonl")
    if not queries:
        raise ValueError("queries.jsonl is empty")
    contexts: dict[str, list[ContextChunk]] = {}
    for row in _read_jsonl(args.context_run):
        qid = str(row.get("query_id") or "")
        chunks = sorted(row.get("chunks") or [], key=lambda item: int(item.get("rank") or 0))
        if not qid:
            raise ValueError("Context run has a row without query_id")
        if not chunks:
            raise ValueError(f"{qid}: no saved chunks")
        converted: list[ContextChunk] = []
        for chunk in chunks[: args.top_k_context]:
            text = str(chunk.get("text") or "").strip()[: args.max_unit_chars]
            if not text:
                raise ValueError(
                    f"{qid}: chunk {chunk.get('chunk_id')!r} has no text. "
                    "Use the *_context.jsonl artifact emitted by the updated second runner."
                )
            converted.append(ContextChunk(
                chunk_id=str(chunk.get("chunk_id") or ""),
                doc_id=str(chunk.get("page_id") or ""),
                text=text,
                score=1.0 / max(1, int(chunk.get("rank") or 1)),
            ))
        contexts[qid] = converted
    expected = {str(row.get("query_id") or "") for row in queries}
    missing = expected - set(contexts)
    extra = set(contexts) - expected
    if missing or extra:
        raise ValueError(
            f"Context/query mismatch: missing={len(missing)} extra={len(extra)}; "
            f"examples missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}"
        )
    for query in queries:
        if not isinstance(query.get("answers"), list) or not query["answers"]:
            raise ValueError(f"{query['query_id']}: missing reference answer")
    return queries, contexts


def _load_checkpoint(path: Path, identity: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != identity:
        raise ValueError(f"Checkpoint {path} belongs to a different QA configuration")
    return {
        str(row["query_id"]): row
        for row in payload.get("rows") or []
        if not row.get("error") and not row.get("judge_error")
    }


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace, wall_seconds: float) -> dict[str, Any]:
    completed = [row for row in rows if not row.get("error") and not row.get("judge_error")]
    total = len(rows) or 1
    correct = sum(int(row.get("score") or 0) for row in completed)
    abstained = sum(bool(row.get("abstained")) for row in rows)
    errors = total - len(completed)
    return {
        "contract_version": "benchmark-kdl-qa-v1",
        "pipeline": "saved KDL top-K chunks -> DeepSeek V4 Flash -> GPT-4o mini binary judge",
        "generator": args.generator,
        "judge": args.judge,
        "style": "binary_default",
        "questions": total,
        "completed": len(completed),
        "errors": errors,
        "correct": correct,
        "incorrect": len(completed) - correct,
        "correct_only": correct / len(completed) if completed else None,
        # Binary baseline: no partial-credit label exists, so the two values
        # are intentionally identical rather than pretending partial scores.
        "correct_plus_partial": correct / len(completed) if completed else None,
        "abstain_rate": abstained / total,
        "mean_chunks_used": sum(int(row.get("chunks_used") or 0) for row in rows) / total,
        "mean_context_chars": sum(int(row.get("chars_used") or 0) for row in rows) / total,
        "generation_seconds_sum": round(sum(float(row.get("generation_seconds") or 0) for row in rows), 3),
        "judge_seconds_sum": round(sum(float(row.get("judge_seconds") or 0) for row in rows), 3),
        "wall_seconds": round(wall_seconds, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--context-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--judge", default="openai/gpt-4o-mini")
    parser.add_argument("--top-k-context", type=int, default=10)
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument("--max-unit-chars", type=int, default=8000)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.generator == args.judge:
        raise ValueError("Generator and judge must be different models")
    if args.top_k_context < 1:
        raise ValueError("--top-k-context must be positive")

    questions, contexts = _inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation = {
        "queries": len(questions),
        "context_queries": len(contexts),
        "top_k_context": args.top_k_context,
        "saved_chunks": sum(len(value) for value in contexts.values()),
        "all_context_has_text": True,
    }
    _write_json(args.output_dir / "validation.json", validation)
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0

    identity = _identity(args)
    checkpoint_path = args.output_dir / "checkpoint.json"
    rows = _load_checkpoint(checkpoint_path, identity)
    pending = [question for question in questions if str(question["query_id"]) not in rows]
    started = time.perf_counter()

    def evaluate(question: dict[str, Any]) -> dict[str, Any]:
        qid = str(question["query_id"])
        generation_started = time.perf_counter()
        generation = generate(
            qid, str(question.get("query") or ""), contexts[qid],
            model=args.generator, max_chars=args.max_context_chars,
            max_output_tokens=args.max_output_tokens, render_prompt=style.render_prompt,
        )
        generation_seconds = time.perf_counter() - generation_started
        judge_started = time.perf_counter()
        result: dict[str, Any] = {
            "query_id": qid,
            "query": question.get("query") or "",
            "answers": question["answers"],
            "sys_ans": generation.answer,
            "abstained": generation.abstained,
            "chunks_used": generation.chunks_used,
            "chars_used": generation.chars_used,
            "context_page_ids": generation.context_doc_ids,
            "generation_seconds": round(generation_seconds, 3),
            "judge_seconds": 0.0,
            "status": "error" if generation.error else "ok",
            "score": None,
            "judge_raw": "",
            "error": generation.error,
            "judge_error": None,
            "qa_config_hash": identity,
        }
        if generation.error:
            return result
        if generation.abstained or not generation.answer.strip():
            result.update({"score": 0, "judge_raw": "abstained", "error": None})
            return result
        judge_started = time.perf_counter()
        prompt = (
            DOCBENCH_JUDGE_PROMPT.replace("{{question}}", str(question.get("query") or ""))
            .replace("{{sys_ans}}", generation.answer)
            # The merged benchmark stores alternative acceptable answers as a
            # list (for example ``Page 2`` and ``2``).  Keep the on-demand
            # judge prompt/parser unchanged, and present all aliases as one
            # reference answer.
            .replace("{{ref_ans}}", " / ".join(str(value) for value in question["answers"]))
            .replace("{{ref_text}}", "")
        )
        try:
            judge_raw = complete(args.judge, prompt, temperature=0.0, max_output_tokens=16)
            result.update({"score": _parse_score(judge_raw), "judge_raw": judge_raw, "error": None})
        except Exception as error:  # noqa: BLE001 - persist and resume failed rows
            result.update({"status": "error", "judge_error": repr(error), "error": None})
        result["judge_seconds"] = round(time.perf_counter() - judge_started, 3)
        return result

    def save() -> None:
        ordered = [rows[str(question["query_id"])] for question in questions if str(question["query_id"]) in rows]
        _write_json(checkpoint_path, {"identity": identity, "rows": ordered})

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(evaluate, question): str(question["query_id"]) for question in pending}
            for count, future in enumerate(as_completed(futures), start=1):
                rows[futures[future]] = future.result()
                save()
                print(f"QA progress: {count}/{len(pending)}", flush=True)
    except BaseException:
        save()
        raise

    ordered = [rows[str(question["query_id"])] for question in questions]
    _write_jsonl(args.output_dir / "per_query.jsonl", ordered)
    report = _summary(ordered, args, time.perf_counter() - started)
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _parse_score(text: str) -> int:
    prefix = str(text or "")[:200]
    import re
    match = re.search(r"correctness\s*:\s*([01])\b", prefix, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([01])\b", prefix)
    if match:
        return int(match.group(1))
    raise ValueError(f"judge response has no 0/1 score: {prefix!r}")


if __name__ == "__main__":
    raise SystemExit(main())
