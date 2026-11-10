"""Paired evaluation of MetaRank-LTR adaptive K against existing K policies."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
EVIE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/routing_axes/k3/k3_out_shard0/evie_scores_2048.npy"
COLVEC = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/colvec"
NEW = ROOT / "data/work/metarank_adaptive_k_nested_20261027/budget_54/per_query.jsonl"
XGB54_COHERE = ROOT / "data/work/reranker_adaptive_k_20261026/budget_54_cohere4pro/oof_predictions.jsonl"
XGB54_DISCOVERY = ROOT / "data/work/reranker_adaptive_k_20261026/budget_54_original/oof_predictions.jsonl"
XGB58 = ROOT / "data/work/reranker_adaptive_k_20261026/budget_58/oof_predictions.jsonl"
OUT = ROOT / "data/work/metarank_adaptive_k_nested_20261027/budget_54/comparison_to_xgb_policies"


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def evaluate(rows: list[dict[str, Any]], arm: str, mode: str, gold: dict[str, set[str]], evie: np.ndarray,
             qpos: dict[str, int], ppos: dict[str, int]) -> dict[str, dict[str, float]]:
    out = {}
    for row in rows:
        if mode == "new":
            ranked, k = row["ranked_pages"], int(row["predicted_k_oof"])
        else:
            if row["arm"] != arm:
                continue
            ranked, k = row["ranking_top100"], int(row["prediction_k_oof"])
        qid = row["qid"]
        prefix = ranked[:k]
        g = gold[qid]
        hit = len(set(prefix) & g)
        rec = hit / max(1, len(g))
        prec = hit / max(1, k)
        top = sorted(prefix, key=lambda page: -float(evie[qpos[qid], ppos[page]]))[:10]
        dcg = sum(1 / math.log2(i + 2) for i, page in enumerate(top) if page in g)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(g), len(top))))
        out[qid] = {"k": float(k), "prefix_recall": rec, "prefix_precision": prec,
                    "evie_recall_at_10": len(set(top) & g) / max(1, len(g)),
                    "evie_ndcg_at_10": dcg / idcg if idcg else 0.0}
    return out


def bootstrap_delta(a: dict[str, dict[str, float]], b: dict[str, dict[str, float]], metric: str, seed: int) -> dict[str, float]:
    qids = sorted(set(a) & set(b))
    delta = np.asarray([a[q][metric] - b[q][metric] for q in qids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(10000, len(qids)))
    draws = delta[idx].mean(axis=1)
    scale = 100 if metric != "k" else 1
    lo, hi = np.quantile(draws, [.025, .975])
    return {"delta": float(delta.mean() * scale), "ci95_low": float(lo * scale), "ci95_high": float(hi * scale), "n": len(qids)}


def summarize(rows: dict[str, dict[str, float]]) -> dict[str, float]:
    return {m: float(np.mean([r[m] for r in rows.values()])) for m in ("k", "prefix_recall", "prefix_precision", "evie_recall_at_10", "evie_ndcg_at_10")}


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    qids = sorted(gate["light_run_top100"])
    gold = {q: set(gate["gold_rank"][q]) for q in qids}
    evie = np.load(EVIE, mmap_mode="r")
    qpos = {q: i for i, q in enumerate(json.loads((COLVEC / "colvec_qids.json").read_text(encoding="utf-8")))}
    ppos = {p: i for i, p in enumerate(json.loads((COLVEC / "colvec_keys.json").read_text(encoding="utf-8")))}
    new_rows = [r for r in read(NEW) if r["arm"] == "discovery_plus_cohere_page_shape"]
    raw54_cohere, raw54_discovery, raw58 = read(XGB54_COHERE), read(XGB54_DISCOVERY), read(XGB58)
    results = {
        "MetaRank_LTR_plus_adaptiveK_budget54": evaluate(new_rows, "", "new", gold, evie, qpos, ppos),
        "XGB_Cohere_scores_adaptiveK_budget54": evaluate(raw54_cohere, "cohere4pro", "xgb", gold, evie, qpos, ppos),
        "XGB_discovery_adaptiveK_budget54": evaluate(raw54_discovery, "original", "xgb", gold, evie, qpos, ppos),
        "XGB_discovery_adaptiveK_budget58": evaluate(raw58, "original", "xgb", gold, evie, qpos, ppos),
    }
    report = {"queries": len(qids), "protocol": "paired query-level OOF comparisons; cached EVIE reranks only the selected prefix; no new API or model inference",
              "arms": {k: summarize(v) for k, v in results.items()}, "paired_bootstrap": {}}
    leader = results["MetaRank_LTR_plus_adaptiveK_budget54"]
    for name, rows in results.items():
        if name == "MetaRank_LTR_plus_adaptiveK_budget54":
            continue
        report["paired_bootstrap"][name] = {metric: bootstrap_delta(leader, rows, metric, 20261028 + i)
            for i, metric in enumerate(("k", "prefix_recall", "evie_recall_at_10", "evie_ndcg_at_10"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "per_query.jsonl").write_text("".join(json.dumps({"qid": q, **{name: arms[q] for name, arms in results.items()}}, ensure_ascii=False) + "\n" for q in qids), encoding="utf-8")
    lines = ["# MetaRank adaptive comparison", "", "Paired query-level five-fold OOF; same 220 qids, cached candidate scores and EVIE; no new inference.", "",
             "| Arm | Mean K | Prefix recall | EVIE recall@10 | EVIE nDCG@10 |", "|---|---:|---:|---:|---:|"]
    for name, row in report["arms"].items():
        lines.append(f"| {name} | {row['k']:.2f} | {row['prefix_recall']*100:.2f}% | {row['evie_recall_at_10']*100:.2f}% | {row['evie_ndcg_at_10']*100:.2f} |")
    lines += ["", "## Paired delta: MetaRank LTR + adaptive K minus baseline", "", "| Baseline | Mean K delta (95% CI) | Prefix recall delta (95% CI) | EVIE R@10 delta (95% CI) | EVIE nDCG delta (95% CI) |", "|---|---:|---:|---:|---:|"]
    for name, metrics in report["paired_bootstrap"].items():
        k, r, e, n = (metrics[m] for m in ("k", "prefix_recall", "evie_recall_at_10", "evie_ndcg_at_10"))
        lines.append(f"| {name} | {k['delta']:+.2f} [{k['ci95_low']:+.2f}, {k['ci95_high']:+.2f}] | {r['delta']:+.2f} pp [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}] | {e['delta']:+.2f} pp [{e['ci95_low']:+.2f}, {e['ci95_high']:+.2f}] | {n['delta']:+.2f} [{n['ci95_low']:+.2f}, {n['ci95_high']:+.2f}] |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["arms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
