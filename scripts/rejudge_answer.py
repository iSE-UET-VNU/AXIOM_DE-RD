#!/usr/bin/env python3
"""Re-answer and re-judge selected DocBench questions interactively.

Example:
  .venv/bin/python scripts/rejudge_answer.py \
    --docbench-root 'data/raw/0. BENCHMARK-.../0. BENCHMARK' \
    --qid mpdocvqa::748 --results data/benchmark/run/qa/lake_baseline_legacy.jsonl

The script does not run retrieval or generation.  It only loads the question
and gold answer, accepts a replacement answer, and sends it through the
ViDoRe judge prompt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.run_docbench_e2e import _load_docbench  # noqa: E402
from src.evaluation.benchmarks.vidore_v3_judge import JUDGE_PROMPT  # noqa: E402
from src.evaluation.llm import complete  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


LABELS = {"Correct", "Partially Correct", "Incorrect"}


def _read_rows(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        items = value if isinstance(value, list) else [value]
    else:
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in items:
        if isinstance(row, dict) and row.get("qid"):
            rows[str(row["qid"])] = row
    return rows


def _parse_judge(raw: str) -> dict[str, str]:
    text = raw.strip().removeprefix("```").removesuffix("```").strip()
    payload = json.loads(text)
    judgment = str(payload.get("judgment", "")).strip()
    if judgment not in LABELS:
        raise ValueError(f"Invalid judgment {judgment!r}; expected {sorted(LABELS)}")
    return {"explanation": str(payload.get("explanation", "")), "judgment": judgment}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docbench-root", type=Path, required=True)
    parser.add_argument("--qid", required=True, help="For example: mpdocvqa::748")
    parser.add_argument("--answer", help="Replacement answer; omit to enter it interactively")
    parser.add_argument("--results", type=Path, help="Existing QA JSONL/JSON, shown for reference")
    parser.add_argument("--judge-model", default="gpt-5.2")
    parser.add_argument("--output", type=Path, help="Optional JSONL file to append the re-judgment")
    args = parser.parse_args()

    load_dotenv_file(ROOT)
    _documents, questions = _load_docbench(args.docbench_root)
    question = next((item for item in questions if str(item["qid"]) == args.qid), None)
    if question is None:
        available = [str(item["qid"]) for item in questions[:10]]
        raise SystemExit(f"Question {args.qid!r} not found. First available ids: {available}")

    old_row = _read_rows(args.results).get(args.qid, {})
    print(f"qid:         {args.qid}")
    print(f"question:    {question['question']}")
    print(f"gold answer: {question['answer']}")
    if old_row:
        print(f"old answer:  {old_row.get('answer', old_row.get('sys_ans', ''))}")
        print(f"old judgment: {old_row.get('judgment', old_row.get('score', ''))}")

    answer = args.answer
    if answer is None:
        answer = input("\nNew test answer: ").strip()
    if not answer:
        raise SystemExit("Replacement answer is empty")

    prompt = JUDGE_PROMPT.format(
        query=question["question"],
        true_answer=question["answer"],
        test_answer=answer,
    )
    raw = complete(args.judge_model, prompt, temperature=0.0, max_output_tokens=256)
    verdict = _parse_judge(raw)
    result = {
        "qid": args.qid,
        "query": question["question"],
        "gold_answer": question["answer"],
        "test_answer": answer,
        "judge_model": args.judge_model,
        **verdict,
        "judge_raw": raw,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
