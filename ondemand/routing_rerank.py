import argparse
import copy
import json
import re

import numpy as np

from .arms import ColVec
from .bench import BENCH, gold as load_gold, load_jsonl, work, write_jsonl
from .routing_api import call, load_defs, parallel, repo_dir, slug
from .routing_labels import recall10
from .text import real_text

MODEL = "deepseek/deepseek-v4-flash"
FLIPS = 10000
CHAIN = re.compile(r"\[\d+\](?:\s*>\s*\[\d+\])+")


def rankgpt():
    defs = load_defs([(repo_dir("RankGPT") / "rank_gpt.py").read_text()],
                     {"get_prefix_prompt", "get_post_prompt", "create_permutation_instruction",
                      "clean_response", "remove_duplicate", "receive_permutation"})
    defs["copy"] = copy
    return defs


def ranking_text(raw):
    chains = [m.group(0) for m in CHAIN.finditer(raw)]
    if not chains:
        return "", "no_chain"
    chain = max(chains, key=len)
    return chain, "extra_digits" if any(c.isdigit() for c in raw.replace(chain, "", 1)) else "ok"


def sign_flip(diff):
    rng = np.random.default_rng(0)
    observed = abs(diff.mean())
    flips = np.abs((rng.choice([-1, 1], size=(FLIPS, len(diff))) * diff).mean(axis=1))
    return round(float((flips >= observed).mean()), 4)


def request(model, messages, reasoning):
    return {"model": model, "messages": messages, "temperature": 0,
            "reasoning": {"effort": "none"} if reasoning == "none" else {"exclude": True}}


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_rerank")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--depth", type=int, default=20)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--reasoning", choices=("none", "exclude"), default="none")
    ap.add_argument("--gate", default="gate_ppocrv5_backfill.json")
    ap.add_argument("--pages", default="pages_ppocrv5.jsonl")
    ap.add_argument("--labels", default="labels_r10_k100.jsonl")
    ap.add_argument("--allow-api", action="store_true")
    args = ap.parse_args()
    defs = rankgpt()
    gold = load_gold(BENCH)
    labels = [r for r in load_jsonl(work("routing", bench=BENCH) / args.labels) if r["fold"] in args.folds]
    gate = json.loads((work("light_prep", bench=BENCH) / args.gate).read_text(encoding="utf-8"))
    text = {r["page_id"]: real_text(r["text"]) for r in load_jsonl(work("light_prep", bench=BENCH) / args.pages)}
    colvec = ColVec(BENCH)
    folder = work("routing", "cache", "rankgpt", bench=BENCH)
    bases = {"light": {}, "colvec": {}}
    for r in labels:
        light = gate["light_run_top100"][r["query_id"]][:r["k"]]
        bases["light"][r["query_id"]] = light
        bases["colvec"][r["query_id"]] = sorted(light, key=lambda u: -colvec.score(r["query_id"], u))

    for arm, base in bases.items():
        def one(row):
            item = {"query": row["query"], "hits": [{"page_id": u, "content": text[u]} for u in base[row["query_id"]]]}
            messages = defs["create_permutation_instruction"](item=item, rank_start=0, rank_end=args.depth,
                                                              model_name=args.model)
            raw = call(folder, request(args.model, messages, args.reasoning), args.allow_api)
            chain, status = ranking_text(raw)
            ids = [int(x) for x in re.findall(r"\d+", chain)]
            item = defs["receive_permutation"](item, chain, rank_start=0, rank_end=args.depth)
            return {"query_id": row["query_id"], "group": row["group"], "fold": row["fold"], "arm": arm,
                    "ranking": [h["page_id"] for h in item["hits"]], "raw": raw, "status": status,
                    "out_of_range": sum(not 1 <= i <= args.depth for i in ids),
                    "unranked": args.depth - len({i for i in ids if 1 <= i <= args.depth})}

        rows = parallel(one, labels)
        before = recall10(base, gold)
        after = recall10({r["query_id"]: r["ranking"] for r in rows}, gold)
        for r in rows:
            r["base_recall@10"], r["rerank_recall@10"] = before[r["query_id"]], after[r["query_id"]]
        write_jsonl(work("routing", "rerank", bench=BENCH) / f"rankgpt_{arm}_d{args.depth}_{slug(args.model)}.jsonl", rows)
        for group in ["all"] + sorted({r["group"] for r in rows}):
            sub = [r for r in rows if group in ("all", r["group"])]
            diff = np.array([r["rerank_recall@10"] - r["base_recall@10"] for r in sub])
            print(json.dumps({"arm": arm, "group": group, "n": len(sub),
                              "base": round(float(np.mean([r["base_recall@10"] for r in sub])), 2),
                              "rerank": round(float(np.mean([r["rerank_recall@10"] for r in sub])), 2),
                              "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": sign_flip(diff),
                              "status": {s: sum(r["status"] == s for r in sub) for s in ("ok", "extra_digits", "no_chain")},
                              "out_of_range": sum(r["out_of_range"] for r in sub),
                              "mean_unranked": round(float(np.mean([r["unranked"] for r in sub])), 2)}))


if __name__ == "__main__":
    main()
