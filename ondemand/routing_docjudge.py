import argparse
import base64
import io
import json
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from .adaptive.services.base import Pages
from .arms import ColVec
from .bench import BENCH, gold as load_gold, load_jsonl, work, write_jsonl
from .routing_api import call, parallel, slug
from .routing_labels import recall10
from .routing_rerank import rankgpt, ranking_text, request, sign_flip

RATES = (0, 0.1, 0.2, 0.3, 0.5, 1.0)


def doc(u):
    return u.partition("#page=")[0]


def commit(order, chosen):
    return [u for u in order if doc(u) == chosen] + [u for u in order if doc(u) != chosen]


def prepare(labels, gate, gold, colvec):
    info = {}
    for r in labels:
        q = r["query_id"]
        light = gate["light_run_top100"][q][:r["k"]]
        order = sorted(light, key=lambda u: -colvec.score(q, u))
        maxp, seen, within = {}, Counter(), []
        for u in order:
            maxp.setdefault(doc(u), colvec.score(q, u))
            seen[doc(u)] += 1
            within.append(seen[doc(u)])
        rel = {u for u, v in gold[q].items() if v > 0}
        info[q] = {"row": r, "light": light, "order": order, "maxp": maxp, "within": within, "rel": rel,
                   "main": Counter(doc(u) for u in sorted(rel)).most_common(1)[0][0],
                   "top2": sorted(maxp, key=lambda d: -maxp[d])[:2]}
    return info


def doc_probs(i, beta):
    docs = list(i["maxp"])
    x = beta * np.array([i["maxp"][d] for d in docs])
    p = np.exp(x - x.max())
    return dict(zip(docs, p / p.sum()))


def fit(info, train):
    usable = [q for q in train if info[q]["main"] in info[q]["maxp"]]
    nll = lambda log_beta: -sum(np.log(doc_probs(info[q], np.exp(log_beta))[info[q]["main"]] + 1e-12) for q in usable)
    beta = float(np.exp(minimize_scalar(nll, bounds=(-10, 5), method="bounded").x))
    pairs = [(w, u in info[q]["rel"]) for q in train for u, w in zip(info[q]["order"], info[q]["within"])
             if doc(u) == info[q]["main"]]
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(*zip(*pairs))
    return beta, iso


def two_level(i, beta, iso):
    pd = doc_probs(i, beta)
    score = {u: pd[doc(u)] * p for u, p in zip(i["order"], iso.predict(i["within"]))}
    return sorted(i["order"], key=lambda u: -score[u]), max(pd.values())


def image_url(pages, page_id, dpi):
    buf = io.BytesIO()
    pages.image(page_id, dpi).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def judge_messages(defs, i, model, pages, dpi, per_doc):
    item = {"query": i["row"]["query"], "hits": [{"content": d} for d in i["top2"]]}
    messages = defs["create_permutation_instruction"](item=item, rank_start=0, rank_end=2, model_name=model)
    for n, d in enumerate(i["top2"], 1):
        shown = [u for u in i["order"] if doc(u) == d][:per_doc]
        messages[1 + 2 * n]["content"] = [{"type": "text", "text": f"[{n}]"}] + [
            {"type": "image_url", "image_url": {"url": image_url(pages, u, dpi)}} for u in shown]
    return messages


def judge(args, info, escalated, two_level_runs, gold):
    defs, pages = rankgpt(), Pages(BENCH)
    folder = work("routing", "cache", "docjudge", bench=BENCH)
    todo = [q for q in sorted(escalated) if info[q]["row"]["fold"] in args.folds and len(info[q]["top2"]) == 2]

    def one(q):
        i = info[q]
        raw = call(folder, request(args.judge_model, judge_messages(defs, i, args.judge_model, pages, args.dpi,
                                                                    args.per_doc), args.reasoning), args.allow_api)
        chain, status = ranking_text(raw)
        ids = [int(x) for x in re.findall(r"\d+", chain) if x in ("1", "2")]
        pick = i["top2"][ids[0] - 1] if ids else i["top2"][0]
        return {"query_id": q, "group": i["row"]["group"], "fold": i["row"]["fold"], "top2": i["top2"],
                "main": i["main"], "pick": pick, "raw": raw, "status": status if ids else "no_pick"}

    rows = parallel(one, todo)
    write_jsonl(work("routing", "docjudge", bench=BENCH) / f"judge_{slug(args.judge_model)}_r{args.rate}.jsonl", rows)
    pick = {r["query_id"]: r["pick"] for r in rows}
    tested = [q for q in info if info[q]["row"]["fold"] in args.folds]
    judged = {q: commit(info[q]["order"], pick[q]) if q in pick else two_level_runs[q] for q in tested}
    before, after = recall10({q: two_level_runs[q] for q in tested}, gold), recall10(judged, gold)
    print(json.dumps({"judged": len(rows), "judge_right": sum(r["pick"] == r["main"] for r in rows),
                      "keep_top_right": sum(r["top2"][0] == r["main"] for r in rows),
                      "main_in_top2": sum(r["main"] in r["top2"] for r in rows),
                      "status": dict(Counter(r["status"] for r in rows))}))
    for group in ["all"] + sorted({info[q]["row"]["group"] for q in tested}):
        sub = [q for q in tested if group in ("all", info[q]["row"]["group"])]
        diff = np.array([after[q] - before[q] for q in sub])
        print(json.dumps({"group": group, "n": len(sub), "two_level": round(float(np.mean([before[q] for q in sub])), 2),
                          "with_judge": round(float(np.mean([after[q] for q in sub])), 2),
                          "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": sign_flip(diff)}))


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_docjudge")
    ap.add_argument("--gate", default="gate_ppocrv5_backfill.json")
    ap.add_argument("--labels", default="labels_r10_k100.jsonl")
    ap.add_argument("--judge-model")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--per-doc", type=int, default=3)
    ap.add_argument("--reasoning", choices=("none", "exclude"), default="none")
    ap.add_argument("--allow-api", action="store_true")
    args = ap.parse_args()
    gold = load_gold(BENCH)
    labels = load_jsonl(work("routing", bench=BENCH) / args.labels)
    gate = json.loads((work("light_prep", bench=BENCH) / args.gate).read_text(encoding="utf-8"))
    info = prepare(labels, gate, gold, ColVec(BENCH))

    runs, ptop = defaultdict(dict), {}
    for fold in sorted({i["row"]["fold"] for i in info.values()}):
        train = [q for q, i in info.items() if i["row"]["fold"] != fold]
        beta, iso = fit(info, train)
        print(json.dumps({"test_fold": fold, "beta": round(beta, 4)}))
        for q, i in info.items():
            if i["row"]["fold"] != fold:
                continue
            ranked, ptop[q] = two_level(i, beta, iso)
            runs["light"][q], runs["colvec"][q], runs["two_level"][q] = i["light"], i["order"], ranked
            runs["oracle_judge"][q] = commit(i["order"], i["main"]) if i["main"] in i["top2"] else ranked
    rec = {name: recall10(run, gold) for name, run in runs.items()}
    queries = sorted(info)

    for group in ["all"] + sorted({info[q]["row"]["group"] for q in queries}):
        sub = [q for q in queries if group in ("all", info[q]["row"]["group"])]
        diff = np.array([rec["two_level"][q] - rec["colvec"][q] for q in sub])
        row = {"group": group, "n": len(sub)}
        row.update({name: round(float(np.mean([rec[name][q] for q in sub])), 2) for name in ("light", "colvec", "two_level")})
        row.update({"vs_colvec": {"wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": sign_flip(diff)},
                    "unique_wins": sum(rec["two_level"][q] > max(rec["light"][q], rec["colvec"][q]) for q in sub),
                    "oracle_light_colvec": round(float(np.mean([max(rec["light"][q], rec["colvec"][q]) for q in sub])), 2),
                    "oracle_with_two_level": round(float(np.mean([max(rec["light"][q], rec["colvec"][q], rec["two_level"][q])
                                                                  for q in sub])), 2)})
        print(json.dumps(row))

    by_confidence = sorted(queries, key=lambda q: ptop[q])
    for rate in RATES:
        escalated = set(by_confidence[:round(rate * len(queries))])
        row = {"oracle_judge_rate": rate}
        for group in ("all", "vidore_industrial", "vidore_physics"):
            sub = [q for q in queries if group in ("all", info[q]["row"]["group"])]
            row[group] = round(float(np.mean([rec["oracle_judge" if q in escalated else "two_level"][q] for q in sub])), 2)
            row[group + "_escalated"] = sum(q in escalated for q in sub)
            row[group + "_top1_wrong"] = sum(q in escalated and info[q]["top2"][0] != info[q]["main"] for q in sub)
        print(json.dumps(row))

    if args.judge_model:
        judge(args, info, set(by_confidence[:round(args.rate * len(queries))]), runs["two_level"], gold)


if __name__ == "__main__":
    main()
