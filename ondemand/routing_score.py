import argparse
import json
from pathlib import Path

import numpy as np

from .bench import BENCH, load_jsonl, work

SEEDS = 1000


def at_rate(light, visual, pick, seeds=SEEDS):
    n, m = len(light), int(pick.sum())
    gain = visual - light
    draws = np.array([(light.sum() + gain[np.random.default_rng(s).choice(n, m, replace=False)].sum()) / n
                      for s in range(seeds)])
    value = float(np.where(pick, visual, light).mean())
    return {"value": round(value, 2),
            "random_mean": round(float(draws.mean()), 2),
            "random_p95": round(float(np.percentile(draws, 95)), 2),
            "beats_random_p95": bool(value > np.percentile(draws, 95)),
            "oracle_at_rate": round(float((light.sum() + np.sort(gain)[::-1][:m].sum()) / n), 2)}


def reference(labels):
    light = np.array([r["light_recall@10"] for r in labels])
    visual = np.array([r["visual_recall@10"] for r in labels])
    helps = np.array([r["visual_helps"] for r in labels])
    return {"method": "reference", "n": len(labels), "always_light": round(float(light.mean()), 2),
            "always_visual": round(float(visual.mean()), 2),
            "oracle": round(float(np.maximum(light, visual).mean()), 2),
            "oracle_rate": round(float(helps.mean()), 4)}


def score(decisions, labels):
    by_qid = {r["query_id"]: r for r in labels}
    missing = [r["query_id"] for r in labels if r["query_id"] not in decisions]
    if missing:
        raise SystemExit(f"{len(missing)} queries have no decision, e.g. {missing[0]}")
    qids = [r["query_id"] for r in labels]
    pick = np.array([decisions[q] == "visual" for q in qids])
    helps = np.array([by_qid[q]["visual_helps"] for q in qids])
    return {"n": len(qids), "escalated": int(pick.sum()), "rate": round(float(pick.mean()), 4),
            "helps_caught": int((pick & helps).sum()), "helps_total": int(helps.sum()),
            "recall@10": at_rate(np.array([by_qid[q]["light_recall@10"] for q in qids]),
                                 np.array([by_qid[q]["visual_recall@10"] for q in qids]), pick)}


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_score")
    ap.add_argument("decisions", nargs="*", help="decision jsonl files; default: every file in routing/decisions/<label set>/")
    ap.add_argument("--labels", default="labels_r10_k100.jsonl")
    ap.add_argument("--out", default=None, help="default: scores_<label set>.jsonl")
    args = ap.parse_args()
    folder = work("routing", bench=BENCH)
    labels = load_jsonl(folder / args.labels)
    name = Path(args.labels).stem.removeprefix("labels_")
    files = [Path(p) for p in args.decisions] or sorted((folder / "decisions" / name).glob("*.jsonl"))
    results = [reference(labels)]
    print(json.dumps(results[-1]))
    for path in files:
        decisions = {r["query_id"]: r["branch"] for r in load_jsonl(path)}
        results.append({"method": path.stem, **score(decisions, labels)})
        print(json.dumps(results[-1]))
    (folder / (args.out or f"scores_{name}.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


if __name__ == "__main__":
    main()
