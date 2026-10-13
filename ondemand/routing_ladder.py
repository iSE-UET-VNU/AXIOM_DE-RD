import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytrec_eval

from .arms import ColVec
from .bench import BENCH, FLAWED_QIDS, gold as load_gold, load_jsonl, queries as load_queries, work, write_jsonl
from .evaluate import as_run

ARMS = {"light": 0, "visual20": 20, "visual50": 50, "visual100": 100}
ALIAS = {"visual": "visual100", "visual@20": "visual20", "visual@50": "visual50", "visual@100": "visual100"}
UNIT = 10
SEEDS = 1000
GROUPS = ("all", "vidore", "vidore_industrial", "vidore_physics", "ohrbench:scan", "ohrbench:handwritten")
LABELS = "labels_ladder_k100.jsonl"
RR_ARMS = ["LIGHT", "VISUAL20", "VISUAL50", "VISUAL100"]
RR_CHEAP = ('ARM_NAMES.index("BM25")', 'ARM_NAMES.index("LIGHT")')


def evaluate(runs, gold, measures=("recall_10", "recip_rank")):
    per_query = pytrec_eval.RelevanceEvaluator({q: gold[q] for q in runs}, set(measures)).evaluate(
        {q: as_run(r) for q, r in runs.items()})
    return {q: {m: 100 * v[m] for m in measures} for q, v in per_query.items()}


def at_depth(light, colvec, d):
    return {q: sorted(r[:d], key=lambda u: -colvec.score(q, u)) + r[d:] for q, r in light.items()}


def members(rows, group):
    if group == "all":
        return rows
    if group == "vidore":
        return [r for r in rows if r["group"].startswith("vidore")]
    return [r for r in rows if r["group"] == group]


def depth_curve(gate, gold):
    colvec = ColVec(BENCH)
    light = {q: r for q, r in gate["light_run_top100"].items() if q in gold and q not in FLAWED_QIDS}
    return {d: {q: v["recall_10"] for q, v in evaluate(at_depth(light, colvec, d), gold, ("recall_10",)).items()}
            for d in range(UNIT, 101)}


def build(args):
    gold, queries = load_gold(BENCH), load_queries(BENCH)
    gate = json.loads((work("light_prep", bench=BENCH) / args.gate).read_text(encoding="utf-8"))
    colvec = ColVec(BENCH)
    light = {q: r[:args.k] for q, r in gate["light_run_top100"].items() if q in gold and q not in FLAWED_QIDS}
    short = [q for q, r in light.items() if len(r) < args.k]
    if short:
        raise SystemExit(f"{len(short)} queries have fewer than {args.k} light pages, e.g. {short[0]}")
    scores = {arm: evaluate(at_depth(light, colvec, d) if d else light, gold) for arm, d in ARMS.items()}
    base = {r["query_id"]: r for r in load_jsonl(work("routing", bench=BENCH) / args.folds)}
    if set(base) - FLAWED_QIDS != set(light):
        raise SystemExit("query set differs from the fold file")
    rows = []
    for q in sorted(light):
        recall = {arm: scores[arm][q]["recall_10"] for arm in ARMS}
        best = max(recall.values())
        label = next(arm for arm in ARMS if recall[arm] == best)
        rows.append({"query_id": q, "query": queries[q]["query"], "group": base[q]["group"], "fold": base[q]["fold"],
                     "gate": args.gate, "k": args.k, "label": label, "label_pages": ARMS[label],
                     **{f"{arm}_recall@10": recall[arm] for arm in ARMS},
                     **{f"{arm}_mrr": scores[arm][q]["recip_rank"] / 100 for arm in ARMS},
                     "light_recall@10": recall["light"], "visual_recall@10": recall["visual100"],
                     "visual_helps": recall["visual100"] > recall["light"]})
    write_jsonl(work("routing", bench=BENCH) / LABELS, rows)
    curve = depth_curve(gate, gold)
    for group in GROUPS:
        sub = members(rows, group)
        n = len(sub)
        if not n:
            continue
        mean = {arm: round(sum(r[f"{arm}_recall@10"] for r in sub) / n, 2) for arm in ARMS}
        oracle = round(sum(max(r[f"{a}_recall@10"] for a in ARMS) for r in sub) / n, 2)
        counts = {arm: sum(r["label"] == arm for r in sub) for arm in ARMS}
        label_pages = sum(r["label_pages"] for r in sub) / n
        ids = [r["query_id"] for r in sub]
        fixed = {d: round(sum(curve[d][q] for q in ids) / n, 2) for d in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)}
        at_cost = {b: knapsack(sub, b * n)["value"] for b in (10, 20, 30, 50)}
        print(json.dumps({"group": group, "n": n, "arms": mean, "oracle": oracle, "label_counts": counts,
                          "oracle_mean_pages": round(label_pages, 1), "fixed_depth": fixed,
                          "oracle_at_mean_pages": at_cost,
                          "headroom_at_50": round(at_cost[50] - fixed[50], 2)}))


def knapsack(rows, budget_pages):
    budget = int(budget_pages // UNIT)
    dp = np.full(budget + 1, -np.inf)
    dp[0] = 0.0
    for r in rows:
        nxt = np.full_like(dp, -np.inf)
        for arm, pages in ARMS.items():
            c = pages // UNIT
            if c <= budget:
                nxt[c:] = np.maximum(nxt[c:], dp[:budget + 1 - c] + r[f"{arm}_recall@10"])
        dp = nxt
    return {"value": round(float(dp.max()) / len(rows), 2)}


def fixed_at(curve, ids, pages):
    if pages <= UNIT:
        return sum(curve[UNIT][q] for q in ids) / len(ids)
    low, high = int(np.floor(pages)), int(np.ceil(pages))
    a = sum(curve[low][q] for q in ids) / len(ids)
    b = sum(curve[high][q] for q in ids) / len(ids)
    return a + (pages - low) * (b - a)


def score_group(sub, arm, curve, rng_seeds=SEEDS):
    n = len(sub)
    value = np.array([r[f"{arm[r['query_id']]}_recall@10"] for r in sub])
    pages = np.array([ARMS[arm[r["query_id"]]] for r in sub])
    groups = {}
    for i, r in enumerate(sub):
        groups.setdefault(r["group"], []).append(i)
    table = np.array([[r[f"{a}_recall@10"] for a in ARMS] for r in sub])
    picks = np.array([list(ARMS).index(arm[r["query_id"]]) for r in sub])
    draws = []
    for s in range(rng_seeds):
        rng = np.random.default_rng(s)
        shuffled = picks.copy()
        for idx in groups.values():
            shuffled[idx] = rng.permutation(picks[idx])
        draws.append(table[np.arange(n), shuffled].mean())
    draws = np.array(draws)
    fixed = fixed_at(curve, [r["query_id"] for r in sub], pages.mean())
    return {"n": n, "recall@10": round(float(value.mean()), 2), "mean_pages": round(float(pages.mean()), 1),
            "visual_rate": round(float((pages > 0).mean()), 4),
            "counts": {a: int((picks == i).sum()) for i, a in enumerate(ARMS)},
            "random_mean": round(float(draws.mean()), 2), "random_p95": round(float(np.percentile(draws, 95)), 2),
            "fixed_depth_at_pages": round(fixed, 2), "oracle_at_pages": knapsack(sub, int(pages.sum()))["value"],
            "beats_random_p95": bool(value.mean() > np.percentile(draws, 95)),
            "beats_fixed_depth": bool(value.mean() > fixed)}


def score(args):
    folder = work("routing", bench=BENCH)
    rows = load_jsonl(folder / LABELS)
    gate = json.loads((work("light_prep", bench=BENCH) / rows[0]["gate"]).read_text(encoding="utf-8"))
    curve = depth_curve(gate, load_gold(BENCH))
    files = [Path(p) for p in args.decisions] or sorted((folder / "decisions" / Path(LABELS).stem.removeprefix("labels_")).glob("*.jsonl"))
    results = []
    for path in files:
        arm = {}
        for d in load_jsonl(path):
            branch = str(d["branch"]).lower()
            arm[d["query_id"]] = ALIAS.get(branch, branch)
        missing = [r["query_id"] for r in rows if r["query_id"] not in arm]
        unknown = sorted({a for a in arm.values() if a not in ARMS})
        if missing or unknown:
            raise SystemExit(f"{path.name}: {len(missing)} queries without a decision, unknown arms {unknown}")
        for group in GROUPS:
            sub = members(rows, group)
            if sub:
                results.append({"method": path.stem, "group": group, **score_group(sub, arm, curve)})
                print(json.dumps(results[-1]))
    (folder / "scores_ladder_k100.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


def their_defs(path, names, replace, **env):
    source = path.read_text(encoding="utf-8")
    if source.count(replace[0]) != 1:
        raise SystemExit(f"{path.name}: expected exactly one {replace[0]}")
    namespace, found = dict(env), set()
    for node in ast.parse(source.replace(*replace)).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found.add(node.name)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            found.update(t.id for t in node.targets if getattr(t, "id", None) in names)
        else:
            continue
        exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
    if set(names) - found:
        raise SystemExit(f"not found in {path.name}: {sorted(set(names) - found)}")
    return namespace


def arabzadeh(args):
    from .routing_api import repo_dir
    defs = their_defs(repo_dir("retrieval-router") / "evaluation/run_adaptive_significance.py",
                      {"baseline_selections", "LAMBDAS"}, RR_CHEAP, np=np, pd=pd, ARM_NAMES=RR_ARMS)
    folder = work("routing", bench=BENCH)
    out = folder / "decisions" / Path(LABELS).stem.removeprefix("labels_")
    files = [Path(p) for p in args.probs] or sorted((folder / "ladder_probs").glob("*.jsonl"))
    if not files:
        raise SystemExit("no probability files; copy routing_out/probs/*.jsonl into routing/ladder_probs/")
    for path in files:
        rows = load_jsonl(path)
        frame = pd.DataFrame([{f"pred_prob_{a}": r["probs"][a] for a in RR_ARMS} for r in rows])
        for lam, picks in defs["baseline_selections"](frame).items():
            name = f"{path.stem}-tau-l{int(round(lam * 100)):02d}"
            write_jsonl(out / f"{name}.jsonl", [{"query_id": r["query_id"], "branch": RR_ARMS[i].lower(), "fold": r["fold"],
                                                  "matched_lambda": lam, "target_budget": round(1 - lam, 2)}
                                                 for r, i in zip(rows, picks)])
            print(json.dumps({"out": name, "budget": round(float((picks != 0).mean()), 4),
                              "counts": {a: int((picks == j).sum()) for j, a in enumerate(RR_ARMS)}}))


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_ladder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("labels")
    p.add_argument("--gate", default="gate_ppocrv5_backfill.json")
    p.add_argument("--k", type=int, default=100)
    p.add_argument("--folds", default="labels_r10_k100.jsonl")
    p = sub.add_parser("score")
    p.add_argument("decisions", nargs="*")
    p = sub.add_parser("arabzadeh")
    p.add_argument("probs", nargs="*")
    args = ap.parse_args()
    {"labels": build, "score": score, "arabzadeh": arabzadeh}[args.cmd](args)


if __name__ == "__main__":
    main()
