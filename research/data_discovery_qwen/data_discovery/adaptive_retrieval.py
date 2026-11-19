"""Stage 3 - adaptive retrieval: choose K (how many of the top-100 pages to pass on) per question.

Protocol (same as research/experiments/evaluate_reranker_adaptive_k_20261026.py and the adaptive branch):
- 21 features of the descending score curve of the light-retrieval list (padded with its last score to 100 when the
  top-5 files hold fewer pages);
- label = rank of the last gold page in the list (100 if a gold page is missing from the list);
- XGBRegressor (PARAMS); nested OOF: 5 outer x 4 inner source-stratified folds (StratifiedKFold seeds 42 / 100+fold);
- inner calibration K = clip(round(pred * scale + margin), 1, 100), scale in 0.60..1.50, margin in -20..60, chosen to
  maximise prefix recall with mean K < BUDGET (60); K is then clipped to the list length.
- Reported metrics use the OOF K (each question's K comes from a model that never saw its qrels).
- Deployment model: fit on all 220 questions, calibration chosen on the outer OOF predictions (saved to
  outputs/adaptive_model.json + adaptive_calibration.json). Its in-sample K is written as k_final_model, for new data only.
Qrels: labels and evaluation only, never inputs.
"""
from __future__ import annotations

import math
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRegressor

from . import common as C

CAP, BUDGET = 100, 60.0
FEATURES = [*(f"score_mean_top_{k}" for k in (1, 5, 10, 20, 50, 100)), "score_mean_tail", "score_std", "score_range",
            *(f"gap_at_{k}" for k in (1, 5, 10, 20, 50)), "max_gap", "max_gap_rank",
            *(f"slope_top_{k}" for k in (10, 20, 50, 100)), "softmax_entropy"]
PARAMS = {"n_estimators": 193, "max_depth": 2, "learning_rate": .03, "min_child_weight": 5, "subsample": .85,
          "colsample_bytree": .85, "reg_lambda": 5., "objective": "reg:squarederror", "eval_metric": "rmse",
          "tree_method": "hist", "n_jobs": 4, "verbosity": 0}


def features(scores: list[float]) -> list[float]:
    v = np.asarray(list(scores) + [scores[-1]] * (CAP - len(scores)), dtype=np.float64)
    r = {f"score_mean_top_{k}": float(v[:k].mean()) for k in (1, 5, 10, 20, 50, 100)}
    r.update(score_mean_tail=float(v[-25:].mean()), score_std=float(v.std()), score_range=float(v.max() - v.min()))
    r.update({f"gap_at_{k}": float(v[k - 1] - v[k]) for k in (1, 5, 10, 20, 50)})
    gaps = v[:-1] - v[1:]; ix = int(np.argmax(gaps))
    r.update(max_gap=float(gaps[ix]), max_gap_rank=float((ix + 1) / CAP))
    ranks = np.arange(1, CAP + 1, dtype=np.float64)
    r.update({f"slope_top_{k}": float(np.polyfit(ranks[:k], v[:k], 1)[0]) for k in (10, 20, 50, 100)})
    p = np.exp(v - v.max()); p /= p.sum()
    r["softmax_entropy"] = float(-(p * np.log(p + 1e-15)).sum())
    return [r[f] for f in FEATURES]


def model(seed: int) -> XGBRegressor:
    return XGBRegressor(**PARAMS, random_state=seed)


def integer_k(pred, scale, margin):
    return np.clip(np.rint(pred * scale + margin), 1, CAP).astype(int)


def recall(k, gold):
    return float(np.mean([sum(r is not None and r <= int(ki) for r in g.values()) / len(g) for ki, g in zip(k, gold)]))


def calibration(pred, gold, budget=BUDGET):
    best = None
    for scale in np.round(np.arange(.6, 1.5001, .05), 2):
        for margin in np.arange(-20, 60.01, 5):
            k = integer_k(pred, float(scale), float(margin)); mk = float(k.mean())
            if mk >= budget:
                continue
            value = (recall(k, gold), -mk, -float(scale))
            if best is None or value > best[0]:
                best = (value, float(scale), float(margin))
    return best[1], best[2]


def prefix_metrics(k, ranked, gold):
    rec, prec, f1, nd = [], [], [], []
    for ki, pages, g in zip(k, ranked, gold):
        rel = set(g); pref = pages[:int(ki)]; hits = len(rel & set(pref))
        r, p = hits / len(rel), hits / int(ki)
        rec.append(r); prec.append(p); f1.append(2 * p * r / (p + r) if p + r else 0.)
        dcg = sum(1 / math.log2(i + 2) for i, x in enumerate(pref) if x in rel)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), len(pref))))
        nd.append(dcg / ideal if ideal else 0.)
    return {"mean_k": round(float(np.mean(k)), 2), "median_k": float(np.median(k)), "p90_k": round(float(np.percentile(k, 90)), 1),
            "prefix_recall": round(100 * float(np.mean(rec)), 2), "prefix_precision": round(100 * float(np.mean(prec)), 2),
            "prefix_f1": round(100 * float(np.mean(f1)), 2), "prefix_ndcg": round(100 * float(np.mean(nd)), 2)}


def evaluate_k(k_by_qid: dict, light_rows: list[dict], qrels: dict, keep: set) -> dict:
    """Prefix metrics of given K values on a subset of questions."""
    out = {}
    for src in ("all",) + C.SOURCES:
        rows = [r for r in light_rows if r["qid"] in keep and (src == "all" or r["qid"].startswith(src))]
        k = np.asarray([min(k_by_qid[r["qid"]], len(r["ranked_pages"])) for r in rows])
        out[src] = prefix_metrics(k, [r["ranked_pages"] for r in rows], [qrels[r["qid"]] for r in rows])
    return out


def run(light_rows: list[dict], qrels: dict, budget: float = BUDGET):
    qids = [r["qid"] for r in light_rows]
    sources = np.asarray([q.split("::")[0] for q in qids])
    t0 = time.perf_counter()
    x = np.asarray([features(r["scores"]) for r in light_rows], np.float32)
    t_feat = time.perf_counter() - t0
    ranked = [r["ranked_pages"] for r in light_rows]; n = np.asarray([len(p) for p in ranked])
    gold = [{p: rk.index(p) + 1 if p in rk else None for p in qrels[q]} for q, rk in zip(qids, ranked)]
    y = np.asarray([max(v for v in g.values()) if all(v is not None for v in g.values()) else CAP for g in gold], np.float32)

    t1 = time.perf_counter()
    k, raw, folds = np.zeros(len(qids), int), np.zeros(len(qids)), []
    for fold, (tr, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, sources), 1):
        ip = np.full(len(tr), np.nan)
        for j, (a, b) in enumerate(StratifiedKFold(n_splits=4, shuffle=True, random_state=100 + fold).split(x[tr], sources[tr])):
            ip[b] = model(100 + fold + j).fit(x[tr[a]], y[tr[a]]).predict(x[tr[b]])
        scale, margin = calibration(ip, [gold[i] for i in tr], budget)
        raw[te] = model(1000 + fold).fit(x[tr], y[tr]).predict(x[te])
        k[te] = np.minimum(integer_k(raw[te], scale, margin), n[te])
        folds.append({"fold": fold, "scale": scale, "margin": margin})
    t_oof = time.perf_counter() - t1

    t2 = time.perf_counter()
    final = model(2026).fit(x, y)
    scale, margin = calibration(raw, gold, budget)
    t_fit = time.perf_counter() - t2
    t3 = time.perf_counter()
    k_final = np.minimum(integer_k(final.predict(x), scale, margin), n)
    t_pred = time.perf_counter() - t3

    metrics = {"adaptive K (nested OOF)": {}, "fixed K=50": {}, "fixed K=100": {}}
    for src in ("all",) + C.SOURCES:
        idx = np.flatnonzero(np.ones(len(qids), bool) if src == "all" else sources == src)
        sub = ([ranked[i] for i in idx], [gold[i] for i in idx])
        metrics["adaptive K (nested OOF)"][src] = prefix_metrics(k[idx], *sub)
        metrics["fixed K=50"][src] = prefix_metrics(np.minimum(50, n[idx]), *sub)
        metrics["fixed K=100"][src] = prefix_metrics(np.minimum(100, n[idx]), *sub)
    rows = [{"qid": q, "k": int(k[i]), "k_final_model": int(k_final[i]), "target_k": int(y[i]),
             "n_candidates": int(n[i]), "oof_prediction_raw": round(float(raw[i]), 3)} for i, q in enumerate(qids)]
    timing = {"features_ms_per_query": round(1000 * t_feat / len(qids), 3), "nested_oof_train_s": round(t_oof, 3),
              "final_model_fit_s": round(t_fit, 3), "predict_ms_per_query": round(1000 * t_pred / len(qids), 3)}
    deploy = {"features": FEATURES, "params": PARAMS, "scale": scale, "margin": margin, "cap": CAP, "budget_mean_k_lt": budget,
              "oof_fold_calibration": folds, "note": "K = min(clip(round(pred*scale+margin), 1, 100), list length)"}
    return rows, metrics, timing, final, deploy
