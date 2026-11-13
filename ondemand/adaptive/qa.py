import csv
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..bench import BENCH, group_of, load_jsonl, queries as load_queries, work, write_jsonl
from ..qa import (ANSWER_PROMPT, GENERATOR, GENERATOR_MAX_TOKENS, GENERATOR_PROVIDER, GENERATOR_REASONING, JUDGE,
                  JUDGE_PROMPT, SEED, TEMPERATURE, WORKERS, format_gold, generate, generator_key, judge_detail,
                  settings_key)


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
    folder = work("qa_cache", "judge_v2", bench=bench)
    prompt = JUDGE_PROMPT.format(query=query, true_answer=format_gold(answers), test_answer=answer.strip())
    key = settings_key(JUDGE) + prompt
    if not cache_path(folder, key).exists() and not allow_api:
        raise SystemExit("the judge call for this answer is not cached; rerun with --allow-api to permit it")
    return judge_detail(query, answers, answer)


def outputs_name(reader, judge):
    return f"{reader}__{judge}".replace(":", "-").replace("/", "-")


def run(tag, bench=BENCH, allow_api=False, reader="qwen3_vl_32b", judge="gpt4o_mini_structured", modes=None,
        out_dir=None):
    from . import qa_judge, qa_reader  # noqa: F401
    from .registry import build
    folder = Path(out_dir) if out_dir else work("results", tag, bench=bench)
    records = load_jsonl(folder / "per_query.jsonl")
    queries = load_queries(bench)
    read_online = modes.online("qa_reader") if modes else allow_api
    judge_online = modes.online("qa_judge") if modes else allow_api
    reader_obj, judge_obj = build("qa_reader", reader, bench=bench), build("qa_judge", judge, bench=bench)
    name = outputs_name(reader, judge)

    def one(record):
        chunks = [c["text"] for c in record["chunks"]]
        try:
            answer = reader_obj.read(record["query"], chunks, read_online)
            verdict = judge_obj.judge(record["query"], record["answers"] or [""], answer["answer"], judge_online)
            label = verdict["judgment"]
        except (Exception, SystemExit) as error:
            return {"qid": record["query_id"], "answer": "", "label": None, "error": str(error)[:300],
                    "n_chunks": len(chunks)}
        return {"qid": record["query_id"], "answer": answer["answer"], "label": label, "error": None,
                "explanation": verdict.get("explanation"), "verdict": verdict,
                "gen_seconds": answer.get("gen_seconds", 0.0), "n_chunks": len(chunks),
                "correct": label == "Correct", "credited": label in ("Correct", "Partially Correct")}

    print(f"[qa] {len(records)} queries, reader {reader} ({'online' if read_online else 'offline'}), "
          f"judge {judge} ({'online' if judge_online else 'offline'}), {WORKERS} workers")
    rows = {}
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(one, r) for r in records]
        for done, f in enumerate(as_completed(futures), 1):
            r = f.result()
            rows[r["qid"]] = r
            if done % 10 == 0 or done == len(records):
                print(f"\r[qa] evaluated {done}/{len(records)} queries...", end="", flush=True)
    print("\r" + " " * 50 + "\r", end="", flush=True)

    ok = [r for r in rows.values() if r["label"]]
    by = defaultdict(list)
    for record in records:
        by[group_of(record["query_id"], queries[record["query_id"]])].append(rows[record["query_id"]])
    summary = {"reader": reader_obj.settings(), "judge": judge_obj.settings(),
               "modes": {"qa_reader": "online" if read_online else "offline",
                         "qa_judge": "online" if judge_online else "offline"},
               "n": len(records), "judged": len(ok), "failed": len(records) - len(ok),
               "correct_only": round(100 * sum(r["correct"] for r in ok) / len(records), 2) if records else None,
               "correct_plus_partial": round(100 * sum(r["credited"] for r in ok) / len(records), 2) if records else None,
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
                 "generated_answer": r["answer"], "judgment": r["label"], "judge_explanation": r.get("explanation"),
                 "verdict": r.get("verdict"), "correct": bool(r.get("correct")), "credited": bool(r.get("credited")),
                 "n_chunks": r["n_chunks"], "gen_seconds": round(r.get("gen_seconds", 0.0), 3),
                 "error": r["error"]} for r in ordered]
    (folder / f"qa_summary_{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_jsonl(folder / f"qa_{name}.jsonl", readable)
    with (folder / f"qa_{name}.csv").open("w", newline="", encoding="utf-8") as sink:
        writer = csv.writer(sink)
        writer.writerow(["query_id", "group", "branch", "k", "judgment", "correct", "credited", "n_chunks",
                         "gen_seconds", "query", "gold_answers", "generated_answer", "error"])
        for r in readable:
            writer.writerow([r["query_id"], r["group"], r["branch"], r["k"], r["judgment"], r["correct"],
                             r["credited"], r["n_chunks"], r["gen_seconds"], r["query"],
                             " | ".join(r["gold_answers"]), r["generated_answer"], r["error"] or ""])
    print(json.dumps({k: summary[k] for k in ("correct_only", "correct_plus_partial", "judged", "failed")}))
    return summary
