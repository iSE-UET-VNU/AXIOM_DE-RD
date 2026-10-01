import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from ..bench import BENCH, group_of, load_jsonl, queries as load_queries, work, write_jsonl
from ..qa import (ANSWER_PROMPT, GENERATOR, JUDGE, MAX_TOKENS, MAX_TOKENS_RETRY, SEED, TEMPERATURE, WORKERS,
                  cached, generate, judge, settings_key)


def run(tag, bench=BENCH):
    folder = work("results", tag, bench=bench)
    records = load_jsonl(folder / "per_query.jsonl")
    queries = load_queries(bench)

    def one(record):
        chunks = [c["text"] for c in record["chunks"]]
        prompt = ANSWER_PROMPT.format(
            documents="\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks)), query=record["query"])
        try:
            answer = cached(work("qa_cache", "answers", bench=bench), settings_key(GENERATOR) + prompt,
                            lambda: generate(prompt))
            label = judge(record["query"], record["answers"] or [""], answer["answer"])
        except Exception as error:
            return {"qid": record["query_id"], "answer": "", "label": None, "error": str(error)[:300],
                    "n_chunks": len(chunks)}
        return {"qid": record["query_id"], "answer": answer["answer"], "label": label, "error": None,
                "gen_seconds": answer["gen_seconds"], "n_chunks": len(chunks),
                "correct": label == "Correct", "credited": label in ("Correct", "Partially Correct")}

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = {r["qid"]: r for r in (f.result() for f in as_completed([pool.submit(one, r) for r in records]))}

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
