import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..bench import BENCH, group_of, load_jsonl, queries as load_queries, work, write_jsonl
from ..qa import (ANSWER_PROMPT, GENERATOR, JUDGE, JUDGE_PROMPT, MAX_TOKENS, MAX_TOKENS_RETRY, SEED, TEMPERATURE,
                  WORKERS, format_gold, generate, judge, settings_key)


def cache_path(folder, key):
    return folder / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def cached_or_refuse(folder, key, compute, allow_api, what):
    path = cache_path(folder, key)
    if path.exists():
        return json.loads(path.read_text())
    if not allow_api:
        raise SystemExit(f"{what} is not cached (would call a live API); rerun with --allow-api to permit it")
    value = compute()
    path.write_text(json.dumps(value, ensure_ascii=False))
    return value


def judge_guarded(query, answers, answer, allow_api, bench):
    folder = work("qa_cache", "judge", bench=bench)
    prompt = JUDGE_PROMPT.format(query=query, true_answer=format_gold(answers), test_answer=answer.strip())
    key = settings_key(JUDGE) + prompt
    if not cache_path(folder, key).exists() and not allow_api:
        raise SystemExit("the judge call for this answer is not cached; rerun with --allow-api to permit it")
    return judge(query, answers, answer)


def run(tag, bench=BENCH, allow_api=False):
    folder = work("results", tag, bench=bench)
    records = load_jsonl(folder / "per_query.jsonl")
    queries = load_queries(bench)

    def one(record):
        chunks = [c["text"] for c in record["chunks"]]
        prompt = ANSWER_PROMPT.format(
            documents="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks)), query=record["query"])
        try:
            answer = cached_or_refuse(work("qa_cache", "answers", bench=bench), settings_key(GENERATOR) + prompt,
                                      lambda: generate(prompt), allow_api, "the answer for this query")
            label = judge_guarded(record["query"], record["answers"] or [""], answer["answer"], allow_api, bench)
        except Exception as error:
            return {"qid": record["query_id"], "answer": "", "label": None, "error": str(error)[:300],
                    "n_chunks": len(chunks)}
        return {"qid": record["query_id"], "answer": answer["answer"], "label": label, "error": None,
                "gen_seconds": answer["gen_seconds"], "n_chunks": len(chunks),
                "correct": label == "Correct", "credited": label in ("Correct", "Partially Correct")}

    print(f"[qa] evaluating {len(records)} queries with {WORKERS} workers...")
    rows = {}
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(one, r) for r in records]
        done = 0
        for f in as_completed(futures):
            r = f.result()
            rows[r["qid"]] = r
            done += 1
            if done % 10 == 0 or done == len(records):
                print(f"\r[qa] evaluated {done}/{len(records)} queries...", end="", flush=True)
    print("\r" + " " * 50 + "\r", end="", flush=True)

    ok = [r for r in rows.values() if r["label"]]
    by = defaultdict(list)
    for record in records:
        by[group_of(record["query_id"], queries[record["query_id"]])].append(rows[record["query_id"]])
    summary = {"generator": GENERATOR, "judge": JUDGE, "temperature": TEMPERATURE, "seed": SEED,
               "max_tokens": MAX_TOKENS, "max_tokens_retry_on_empty": MAX_TOKENS_RETRY,
               "n": len(records), "judged": len(ok), "failed": len(records) - len(ok),
               "correct_only": round(100 * sum(r["correct"] for r in ok) / len(records), 2),
               "correct_plus_partial": round(100 * sum(r["credited"] for r in ok) / len(records), 2),
               "infer_seconds_per_query": round(float(np.mean([r["gen_seconds"] for r in ok])), 2) if ok else None,
               "per_group": {g: {"n": len(v),
                                 "correct_only": round(100 * float(np.mean([bool(x.get("correct")) for x in v])), 1),
                                 "correct_plus_partial": round(100 * float(np.mean([bool(x.get("credited")) for x in v])), 1)}
                             for g, v in sorted(by.items())}}

    ordered = [rows[r["query_id"]] for r in records]
    by_qid = {r["query_id"]: r for r in records}
    readable = [{"query_id": r["qid"], "group": group_of(r["qid"], queries[r["qid"]]),
                 "branch": by_qid[r["qid"]]["branch"], "k": by_qid[r["qid"]]["k"],
                 "query": by_qid[r["qid"]]["query"], "gold_answers": by_qid[r["qid"]]["answers"],
                 "generated_answer": r["answer"], "judgment": r["label"],
                 "correct": bool(r.get("correct")), "credited": bool(r.get("credited")),
                 "n_chunks": r["n_chunks"], "gen_seconds": round(r.get("gen_seconds", 0.0), 3),
                 "error": r["error"]} for r in ordered]
    (folder / "qa_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_jsonl(folder / "qa.jsonl", readable)
    with (folder / "qa.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(["query_id", "group", "branch", "k", "judgment", "correct", "credited", "n_chunks",
                         "gen_seconds", "query", "gold_answers", "generated_answer", "error"])
        for r in readable:
            writer.writerow([r["query_id"], r["group"], r["branch"], r["k"], r["judgment"], r["correct"],
                             r["credited"], r["n_chunks"], r["gen_seconds"], r["query"],
                             " | ".join(r["gold_answers"]), r["generated_answer"], r["error"] or ""])
    print(json.dumps({k: summary[k] for k in ("correct_only", "correct_plus_partial", "judged", "failed")}))
    return summary
