"""Nested OOF XGBoost K policy on frozen top-100 reranker score curves.

Uses the same 21 score features, XGB hyperparameters, folds and inner
calibration as adaptive branch `ondemand/adaptive_retrieval/model.py`.
Reranker qrels only construct training labels and offline metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
RERANK = ROOT / "data/work/frozen_top100_reranker_evie_20261026/rerank"
ORIGINAL = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100/retrieval_scores.jsonl"
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
EVIE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/routing_axes/k3/k3_out_shard0/evie_scores_2048.npy"
COLVEC = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/colvec"
OUT = ROOT / "data/work/reranker_adaptive_k_20261026"
CAP = 100
FEATURE_NAMES = [*(f"score_mean_top_{k}" for k in (1, 5, 10, 20, 50, 100)),
                 "score_mean_tail", "score_std", "score_range", *(f"gap_at_{k}" for k in (1, 5, 10, 20, 50)),
                 "max_gap", "max_gap_rank", *(f"slope_top_{k}" for k in (10, 20, 50, 100)), "softmax_entropy"]
PARAMS = {"n_estimators": 193, "max_depth": 2, "learning_rate": .03, "min_child_weight": 5,
          "subsample": .85, "colsample_bytree": .85, "reg_lambda": 5., "objective": "reg:squarederror",
          "eval_metric": "rmse", "tree_method": "hist", "n_jobs": 4, "verbosity": 0}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def features(scores: list[float]) -> dict[str, float]:
    v = np.asarray(scores, dtype=np.float64)
    if v.shape != (CAP,) or not np.isfinite(v).all() or np.any(v[1:] > v[:-1] + 1e-6):
        raise ValueError("expected finite descending 100-score curve")
    result = {f"score_mean_top_{k}": float(v[:k].mean()) for k in (1, 5, 10, 20, 50, 100)}
    result.update(score_mean_tail=float(v[-25:].mean()), score_std=float(v.std()), score_range=float(v.max()-v.min()))
    result.update({f"gap_at_{k}": float(v[k-1]-v[k]) for k in (1, 5, 10, 20, 50)})
    gaps = v[:-1]-v[1:]
    ix = int(np.argmax(gaps))
    result.update(max_gap=float(gaps[ix]), max_gap_rank=float((ix+1)/CAP))
    ranks = np.arange(1, CAP+1, dtype=np.float64)
    result.update({f"slope_top_{k}": float(np.polyfit(ranks[:k], v[:k], 1)[0]) for k in (10, 20, 50, 100)})
    p = np.exp(v-v.max()); p /= p.sum()
    result["softmax_entropy"] = float(-(p*np.log(p+1e-15)).sum())
    return result


def load_arm(arm: str, gate: dict, qids: list[str]) -> tuple[np.ndarray, list[list[str]], list[list[float]], list[dict]]:
    original = {row["qid"]: row for row in read_jsonl(ORIGINAL)} if arm == "original" else None
    x_rows, ranked_rows, score_rows, gold_maps = [], [], [], []
    for qid in qids:
        candidates = gate["light_run_top100"][qid]
        if original is not None:
            record = original[qid]
            assert record["ranked_pages"] == candidates
            ranked = candidates
            scores = record["scores"]
        else:
            path = RERANK / arm / (hashlib.sha256(qid.encode()).hexdigest()+".json")
            record = json.loads(path.read_text(encoding="utf-8"))
            assert record["candidate_page_ids"] == candidates and len(record["scores"]) == CAP
            order = sorted(range(CAP), key=lambda i: (-float(record["scores"][i]), i))
            ranked = [candidates[i] for i in order]
            scores = [float(record["scores"][i]) for i in order]
        x_rows.append([features(scores)[name] for name in FEATURE_NAMES])
        ranked_rows.append(ranked)
        score_rows.append(scores)
        gold = set(gate["gold_rank"][qid])
        gold_maps.append({page: ranked.index(page)+1 if page in ranked else None for page in gold})
    return np.asarray(x_rows, dtype=np.float32), ranked_rows, score_rows, gold_maps


def model(seed: int) -> XGBRegressor:
    return XGBRegressor(**PARAMS, random_state=seed)


def integer_k(pred: np.ndarray, scale: float, margin: float) -> np.ndarray:
    return np.clip(np.rint(pred*scale+margin), 1, CAP).astype(int)


def recall(k: np.ndarray, gold: list[dict]) -> float:
    return float(np.mean([sum(rank is not None and rank <= int(ki) for rank in row.values())/len(row)
                          for ki, row in zip(k, gold)]))


def calibration(pred: np.ndarray, gold: list[dict], budget: float) -> tuple[float, float]:
    best = None
    for scale in np.round(np.arange(.6, 1.5001, .05), 2):
        for margin in np.arange(-20, 60.01, 5):
            k = integer_k(pred, float(scale), float(margin))
            mk = float(k.mean())
            if mk >= budget:
                continue
            value = (recall(k, gold), -mk, -float(scale))
            if best is None or value > best[0]:
                best = (value, float(scale), float(margin))
    if best is None:
        raise RuntimeError(f"no calibration under mean-K budget {budget}")
    return best[1], best[2]


def metrics(k: np.ndarray, ranked: list[list[str]], gold: list[dict], evie: np.ndarray,
            qids: list[str], q_index: dict, p_index: dict) -> dict:
    prefix_recall, precision, f1, ndcg_k, evie_recall, evie_ndcg = ([] for _ in range(6))
    for ki, pages, row, qid in zip(k, ranked, gold, qids):
        relevant = set(row)
        prefix = pages[:int(ki)]
        hits = len(relevant.intersection(prefix))
        r = hits/len(relevant)
        p = hits/int(ki)
        prefix_recall.append(r); precision.append(p); f1.append(2*p*r/(p+r) if p+r else 0.)
        def ndcg(selected: list[str]) -> float:
            dcg = sum(1/math.log2(i+2) for i, page in enumerate(selected) if page in relevant)
            ideal = sum(1/math.log2(i+2) for i in range(min(len(relevant), len(selected))))
            return dcg/ideal if ideal else 0.
        ndcg_k.append(ndcg(prefix))
        top = sorted(prefix, key=lambda page: -float(evie[q_index[qid], p_index[page]]))[:10]
        evie_recall.append(len(relevant.intersection(top))/len(relevant))
        evie_ndcg.append(ndcg(top))
    return {"mean_k": float(np.mean(k)), "median_k": float(np.median(k)), "p90_k": float(np.percentile(k, 90)),
            "prefix_recall": float(np.mean(prefix_recall)), "prefix_precision": float(np.mean(precision)),
            "prefix_f1": float(np.mean(f1)), "prefix_ndcg": float(np.mean(ndcg_k)),
            "evie_recall_at_10": float(np.mean(evie_recall)), "evie_ndcg_at_10": float(np.mean(evie_ndcg))}


def run(arm: str, budget: float, gate: dict, qids: list[str], sources: np.ndarray,
        evie: np.ndarray, q_index: dict, p_index: dict) -> tuple[dict, list[dict]]:
    x, ranked, curves, gold = load_arm(arm, gate, qids)
    y = np.asarray([max((rank for rank in row.values() if rank is not None), default=CAP)
                    if all(rank is not None for rank in row.values()) else CAP for row in gold], dtype=np.float32)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    chosen = np.zeros(len(qids), dtype=int)
    raw = np.full(len(qids), np.nan)
    folds = []
    for fold, (train, test) in enumerate(outer.split(x, sources), 1):
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=100+fold)
        inner_pred = np.full(len(train), np.nan)
        for inner_fold, (tr, val) in enumerate(inner.split(x[train], sources[train])):
            fitted = model(100+fold+inner_fold).fit(x[train[tr]], y[train[tr]])
            inner_pred[val] = fitted.predict(x[train[val]])
        scale, margin = calibration(inner_pred, [gold[i] for i in train], budget)
        fitted = model(1000+fold).fit(x[train], y[train])
        raw[test] = fitted.predict(x[test])
        chosen[test] = integer_k(raw[test], scale, margin)
        fm = metrics(chosen[test], [ranked[i] for i in test], [gold[i] for i in test], evie,
                     [qids[i] for i in test], q_index, p_index)
        folds.append({"fold": fold, "scale": scale, "margin": margin, **fm})
        print(f"[{arm}] fold {fold}/5: mean K={fm['mean_k']:.2f}, prefix recall={fm['prefix_recall']*100:.2f}%", flush=True)
    all_metrics = metrics(chosen, ranked, gold, evie, qids, q_index, p_index)
    by_source = {source: metrics(chosen[sources == source], [ranked[i] for i in np.flatnonzero(sources == source)],
                                 [gold[i] for i in np.flatnonzero(sources == source)], evie,
                                 [qids[i] for i in np.flatnonzero(sources == source)], q_index, p_index)
                 for source in sorted(set(sources))}
    per_query = [{"arm": arm, "qid": qid, "source": source, "target_k": int(y[i]),
                  "prediction_raw_oof": float(raw[i]), "prediction_k_oof": int(chosen[i]),
                  "prefix_recall": sum(page in set(gold[i]) for page in ranked[i][:chosen[i]])/len(gold[i]),
                  "ranking_top100": ranked[i], "scores_top100": curves[i]} for i, (qid, source) in enumerate(zip(qids, sources))]
    return {"all": all_metrics, "by_source": by_source, "folds": folds}, per_query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["original", "cohere4pro", "qwen3_8b"])
    parser.add_argument("--budget", type=float, default=60.)
    args = parser.parse_args()
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    qids = sorted(gate["light_run_top100"])
    sources = np.asarray([qid.split("::", 1)[0] for qid in qids])
    assert len(qids) == 220 and all(len(gate["light_run_top100"][q]) == CAP for q in qids)
    evie = np.load(EVIE, mmap_mode="r")
    q_index = {q:i for i,q in enumerate(json.loads((COLVEC/"colvec_qids.json").read_text(encoding="utf-8")))}
    p_index = {p:i for i,p in enumerate(json.loads((COLVEC/"colvec_keys.json").read_text(encoding="utf-8")))}
    assert evie.shape == (220, 5091)
    report = {"protocol": {"query_count": 220, "candidate_count": 100, "model": "XGBRegressor",
                           "features": FEATURE_NAMES, "parameters": PARAMS, "outer_folds": 5, "inner_folds": 4,
                           "calibration_budget_mean_k_lt": args.budget,
                           "label": "first rank covering all qrel pages; 100 if unreachable",
                           "qrels": "labels and evaluation only, never model inputs"}, "arms": {}}
    rows = []
    for arm in args.arms:
        report["arms"][arm], per_query = run(arm, args.budget, gate, qids, sources, evie, q_index, p_index)
        rows.extend(per_query)
    output = OUT if args.budget == 60 else OUT / f"budget_{args.budget:g}"
    output.mkdir(parents=True, exist_ok=True)
    (output/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output/"oof_predictions.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows)+"\n", encoding="utf-8")
    print(json.dumps({arm: value["all"] for arm, value in report["arms"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
