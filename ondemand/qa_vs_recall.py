import argparse
import json
from collections import Counter

from .bench import BENCH, FLAWED_QIDS, gold as load_gold, load_jsonl, work

RUNS = {"adaptive_regression": "qa.jsonl", "legacy_fullcorpus": "qa_legacy.jsonl", "plain_bm25_colvec": "qa_A.jsonl",
        "oracle_gold": "qa_oracle.jsonl", "oracle_image": "qa.jsonl"}
GROUPS = ("all", "vidore_industrial", "vidore_physics", "ohrbench")
VERDICTS = ("Correct", "Partially Correct", "Incorrect")
BINS = ("none", "some", "all")


def context_pages(row, per_query):
    if "context_pages" in row:
        return set(row["context_pages"])
    if "pages" in row:
        return set(row["pages"])
    return {c["page_id"] for c in per_query[row["query_id"]]["chunks"][:row["n_chunks"]]}


def bin_of(frac):
    return "none" if frac == 0 else "all" if frac == 1 else "some"


def table(run, name, bench=BENCH):
    gold = {q: {u for u, g in rel.items() if g > 0} for q, rel in load_gold(bench).items()}
    folder = work("results", bench=bench) / run
    rows = [r for r in load_jsonl(folder / name) if r["query_id"] not in FLAWED_QIDS]
    per_query = {}
    if "context_pages" not in rows[0] and "pages" not in rows[0]:
        per_query = {r["query_id"]: r for r in load_jsonl(folder / "per_query.jsonl")}
    print(json.dumps({"run": run, "n": len(rows), "verdicts": dict(Counter(r["judgment"] for r in rows))}))
    for group in GROUPS:
        sub = [r for r in rows if group == "all" or r["group"].startswith(group)]
        cells = {b: Counter() for b in BINS}
        many, pages_used = Counter(), []
        for r in sub:
            g = gold[r["query_id"]]
            ctx = context_pages(r, per_query)
            pages_used.append(len(ctx))
            cells[bin_of(len(g & ctx) / len(g))][r["judgment"]] += 1
            if len(g) > 10:
                many[r["judgment"]] += 1
        out = {"group": group, "n": len(sub), "context_pages_mean": round(sum(pages_used) / len(sub), 1),
               "correct": round(100 * sum(r["judgment"] == "Correct" for r in sub) / len(sub), 1),
               "correct_or_partial": round(100 * sum(r["judgment"] in VERDICTS[:2] for r in sub) / len(sub), 1)}
        for b, c in cells.items():
            out["gold_" + b] = {"n": sum(c.values()), **{v: c[v] for v in VERDICTS}}
        out["over_10_gold"] = dict(many)
        print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.qa_vs_recall")
    ap.add_argument("runs", nargs="*", default=list(RUNS), help="result folders; default: " + ", ".join(RUNS))
    args = ap.parse_args()
    for run in args.runs:
        table(run, RUNS[run])


if __name__ == "__main__":
    main()
