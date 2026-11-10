import csv
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import numpy as np

from .bench import BENCH, group_of, load_jsonl, queries as load_queries, work, write_jsonl
from .chunks import page_chunks
from .openrouter import chat

GENERATOR = "deepseek/deepseek-v4-flash"
JUDGE = "openai/gpt-4o-mini"
TOP_K = 10
MAX_TOKENS = 512
MAX_TOKENS_RETRY = 2048
WORKERS = 8
TEMPERATURE = 0.0
SEED = 0

ANSWER_PROMPT = (
    "You are an expert at answering query based on documents.\n"
    "Here is a list of relevant documents:\n"
    "{documents}\n"
    "\n"
    "Based on the above documents, answer the following query:\n"
    "{query}\n"
    "\n"
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
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_JUDGMENT = re.compile(r'"judgment"\s*:\s*"([^"]+)"')


def format_gold(answers):
    answers = list(dict.fromkeys(str(a).strip() for a in answers if str(a).strip()))
    if len(answers) <= 1:
        return answers[0] if answers else ""
    return "Any one of the following is fully correct: " + " | ".join(answers)


def parse_judgment(reply):
    text = _FENCE.sub("", (reply or "").strip())
    try:
        label = str(json.loads(text)["judgment"]).strip()
    except (ValueError, KeyError, TypeError):
        match = _JUDGMENT.search(text)
        if not match:
            raise ValueError(f"no judgment in {reply!r}")
        label = match.group(1).strip()
    if label not in LABELS:
        raise ValueError(f"judgment {label!r} not in {LABELS}")
    return label


def cached(folder, key, compute):
    path = folder / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    value = compute()
    path.write_text(json.dumps(value, ensure_ascii=False))
    return value


def context(ranked_units, kdl):
    chunks, used = [], []
    for unit in ranked_units:
        for chunk in page_chunks(kdl.get(unit, "")):
            chunks.append(chunk)
            if unit not in used:
                used.append(unit)
            if len(chunks) >= TOP_K:
                return chunks, used
    return chunks, used


def settings_key(model):
    return f"{model}|t={TEMPERATURE}|seed={SEED}|max_tokens={MAX_TOKENS}\n"


def generate(prompt):
    started = perf_counter()
    budget = MAX_TOKENS
    try:
        answer = chat(GENERATOR, prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=SEED).strip()
    except RuntimeError:
        budget = MAX_TOKENS_RETRY
        answer = chat(GENERATOR, prompt, max_tokens=MAX_TOKENS_RETRY, temperature=TEMPERATURE, seed=SEED).strip()
    return {"answer": answer, "gen_seconds": perf_counter() - started, "max_tokens_used": budget}


def judge(query, answers, answer):
    prompt = JUDGE_PROMPT.format(query=query, true_answer=format_gold(answers), test_answer=answer.strip())
    return cached(work("qa_cache", "judge"), settings_key(JUDGE) + prompt,
                  lambda: {"judgment": parse_judgment(
                      chat(JUDGE, prompt, max_tokens=256, temperature=TEMPERATURE, seed=SEED))})["judgment"]


def run(tag, arm, bench=BENCH):
    res = work("results", tag, bench=bench)
    recs = load_jsonl(res / "per_query.jsonl")
    queries = load_queries(bench)
    kdl = {r["page_id"]: r["text"] for r in load_jsonl(work("kdl", bench=bench) / "kdl_pages.jsonl")}

    def one(rec):
        ranked = [h["unit"] for h in rec["arms"][arm]["top"]]
        chunks, used = context(ranked, kdl)
        prompt = ANSWER_PROMPT.format(documents="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks)), query=rec["query"])
        try:
            gen = cached(work("qa_cache", "answers"), settings_key(GENERATOR) + prompt, lambda: generate(prompt))
            label = judge(rec["query"], rec["answers"] or [""], gen["answer"])
        except Exception as error:
            return {"qid": rec["query_id"], "answer": "", "label": None, "error": str(error)[:300],
                    "context": chunks, "context_pages": used}
        return {"qid": rec["query_id"], "answer": gen["answer"], "label": label, "error": None,
                "gen_seconds": gen["gen_seconds"], "context": chunks, "context_pages": used,
                "correct": label == "Correct", "credited": label in ("Correct", "Partially Correct")}

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = {r["qid"]: r for r in (f.result() for f in as_completed([pool.submit(one, rec) for rec in recs]))}
    ok = [r for r in rows.values() if r["label"]]
    by = defaultdict(list)
    for rec in recs:
        by[group_of(rec["query_id"], queries[rec["query_id"]])].append(rows[rec["query_id"]])
    summary = {"arm": arm, "generator": GENERATOR, "judge": JUDGE, "top_k": TOP_K, "context_text": "kdl",
               "temperature": TEMPERATURE, "seed": SEED, "max_tokens": MAX_TOKENS,
               "max_tokens_retry_on_empty": MAX_TOKENS_RETRY,
               "chunker": "fixed_overlap n_words=512 overlap=128",
               "n": len(recs), "judged": len(ok), "failed": len(recs) - len(ok),
               "correct_only": round(100 * sum(r["correct"] for r in ok) / len(recs), 2),
               "correct_plus_partial": round(100 * sum(r["credited"] for r in ok) / len(recs), 2),
               "infer_seconds_per_query": round(float(np.mean([r["gen_seconds"] for r in ok])), 2) if ok else None,
               "per_group": {g: {"n": len(v), "correct_only": round(100 * np.mean([bool(x.get("correct")) for x in v]), 1),
                                 "correct_plus_partial": round(100 * np.mean([bool(x.get("credited")) for x in v]), 1)}
                             for g, v in sorted(by.items())}}
    ordered = [rows[r["query_id"]] for r in recs]
    (res / f"qa_{arm}.json").write_text(json.dumps(ordered, ensure_ascii=False))
    (res / f"qa_{arm}_summary.json").write_text(json.dumps(summary, indent=1))
    by_qid = {r["query_id"]: r for r in recs}
    readable = []
    for row in ordered:
        rec = by_qid[row["qid"]]
        readable.append({"query_id": row["qid"], "group": group_of(row["qid"], queries[row["qid"]]),
                         "query": rec["query"], "gold_answers": rec["answers"],
                         "generated_answer": row["answer"], "judgment": row["label"],
                         "correct": bool(row.get("correct")), "credited": bool(row.get("credited")),
                         "n_chunks": len(row["context"]), "context_pages": row.get("context_pages", []),
                         "gen_seconds": round(row.get("gen_seconds", 0.0), 3), "error": row["error"],
                         "context": row["context"]})
    write_jsonl(res / f"qa_{arm}.jsonl", readable)
    with (res / f"qa_{arm}.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(["query_id", "group", "judgment", "correct", "credited", "n_chunks",
                         "gen_seconds", "query", "gold_answers", "generated_answer", "context_pages", "error"])
        for r in readable:
            writer.writerow([r["query_id"], r["group"], r["judgment"], r["correct"], r["credited"],
                             r["n_chunks"], r["gen_seconds"], r["query"], " | ".join(r["gold_answers"]),
                             r["generated_answer"], " ".join(r["context_pages"]), r["error"] or ""])
    return summary
