import argparse
import json

import numpy as np

from .bench import BENCH, load_jsonl, work


def curve(rows, rates, seeds):
    light = np.array([r["light_ndcg@10"] for r in rows])
    visual = np.array([r["visual_ndcg@10"] for r in rows])
    gain = visual - light
    order = np.argsort(-gain)
    n = len(rows)
    out = []
    for rate in rates:
        m = round(rate * n)
        draws = []
        for seed in seeds:
            pick = np.random.default_rng(seed).choice(n, m, replace=False)
            draws.append((light.sum() + gain[pick].sum()) / n)
        draws = np.array(draws)
        out.append({"rate": rate, "escalated": m,
                    "random_mean": round(float(draws.mean()), 2),
                    "random_p95": round(float(np.percentile(draws, 95)), 2),
                    "oracle": round(float((light.sum() + gain[order[:m]].sum()) / n), 2)})
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m ondemand.routing_random")
    ap.add_argument("--labels", default="labels_light_visual.jsonl")
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--out", default="random_light_visual.json")
    args = ap.parse_args()
    rows = load_jsonl(work("routing", bench=BENCH) / args.labels)
    rates = [round(i * args.step, 4) for i in range(round(1 / args.step) + 1)]
    result = curve(rows, rates, range(args.seeds))
    (work("routing", bench=BENCH) / args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    for r in result:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
