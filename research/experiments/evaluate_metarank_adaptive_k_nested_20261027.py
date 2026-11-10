"""Nested evaluation of MetaRank-style reranking followed by adaptive K.

The LTR ranker is cross-fitted inside each outer training fold to build
training score-curve features for the K regressor. The outer test queries are
ranked by an LTR model trained only on outer-training qrels. Inner OOF K
predictions choose the K calibration under a mean-K budget.
"""
from __future__ import annotations

import json
import math
import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker, XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.experiments.evaluate_reranker_adaptive_k_20261026 import features as score_curve_features

GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
EVIE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/routing_axes/k3/k3_out_shard0/evie_scores_2048.npy"
COLVEC = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/colvec"
FEATURE_DIR = ROOT / "data/work/metarank_feature_ablation_20261027"
BASE_K_54 = ROOT / "data/work/reranker_adaptive_k_20261026/budget_54/oof_predictions.jsonl"
BASE_K_58 = ROOT / "data/work/reranker_adaptive_k_20261026/budget_58/oof_predictions.jsonl"
OUT = ROOT / "data/work/metarank_adaptive_k_nested_20261027"
CAP = 100
FOLDS = 5
CALIBRATION_BUDGET = 58.0
LTR_PARAMS = dict(objective="rank:pairwise", eval_metric="ndcg@20", n_estimators=180,
                  max_depth=3, learning_rate=0.05, min_child_weight=5,
                  subsample=0.85, colsample_bytree=0.9, reg_lambda=5.0,
                  tree_method="hist", n_jobs=4, verbosity=0)
K_PARAMS = dict(n_estimators=193, max_depth=2, learning_rate=.03, min_child_weight=5,
                subsample=.85, colsample_bytree=.85, reg_lambda=5.,
                objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
                n_jobs=4, verbosity=0)
K_FEATURE_NAMES = [*(f"score_mean_top_{k}" for k in (1, 5, 10, 20, 50, 100)),
                   "score_mean_tail", "score_std", "score_range", *(f"gap_at_{k}" for k in (1, 5, 10, 20, 50)),
                   "max_gap", "max_gap_rank", *(f"slope_top_{k}" for k in (10, 20, 50, 100)), "softmax_entropy"]
ARMS = ["discovery_plus_cohere_page_shape", "score_rank", "score_rank_plus_page_shape", "all_engineered"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def candidate_groups(q_indices: list[int]) -> np.ndarray:
    return np.concatenate([np.arange(i * CAP, (i + 1) * CAP) for i in q_indices])


def rank_from_scores(qid: str, scores: np.ndarray, candidates: list[str]) -> list[str]:
    order = sorted(range(CAP), key=lambda i: (-float(scores[i]), i))
    return [candidates[i] for i in order]


def k_row(scores_sorted: list[float]) -> list[float]:
    f = score_curve_features(scores_sorted)
    return [f[n] for n in K_FEATURE_NAMES]


def target_and_positions(ranked: list[str], gold: set[str]) -> tuple[int, dict[str, int | None]]:
    pos = {p: i + 1 for i, p in enumerate(ranked)}
    gr = {p: pos.get(p) for p in gold}
    target = max((v for v in gr.values() if v is not None), default=CAP)
    if any(v is None for v in gr.values()):
        target = CAP
    return int(target), gr


def integer_k(raw: np.ndarray, scale: float, margin: float) -> np.ndarray:
    return np.clip(np.rint(raw * scale + margin), 1, CAP).astype(int)


def prefix_recall(ks: np.ndarray, rankings: list[list[str]], golds: list[set[str]]) -> float:
    return float(np.mean([len(set(pages[:int(k)]) & gold) / max(1, len(gold))
                          for k, pages, gold in zip(ks, rankings, golds)]))


def choose_calibration(raw: np.ndarray, rankings: list[list[str]], golds: list[set[str]]) -> tuple[float, float, dict[str, float]]:
    best = None
    for scale in np.round(np.arange(.6, 1.5001, .05), 2):
        for margin in np.arange(-20, 60.01, 5):
            ks = integer_k(raw, float(scale), float(margin))
            mean_k = float(ks.mean())
            if mean_k >= CALIBRATION_BUDGET:
                continue
            recall = prefix_recall(ks, rankings, golds)
            key = (recall, -mean_k, -float(scale))
            if best is None or key > best[0]:
                best = (key, float(scale), float(margin), float(recall), mean_k)
    if best is None:
        raise RuntimeError("No K calibration found under mean-K budget")
    return best[1], best[2], {"inner_oof_recall": best[3], "inner_oof_mean_k": best[4]}


def metrics(ks: np.ndarray, rankings: list[list[str]], golds: list[set[str]], qids: list[str],
            evie: np.ndarray, qpos: dict[str, int], ppos: dict[str, int]) -> dict[str, float]:
    recall, precision, f1, ndcg, evie_recall, evie_ndcg = [], [], [], [], [], []
    for k, ranked, gold, qid in zip(ks, rankings, golds, qids):
        chosen = ranked[:int(k)]
        hits = len(set(chosen) & gold)
        r, p = hits / max(1, len(gold)), hits / max(1, int(k))
        recall.append(r); precision.append(p); f1.append(2*r*p/(r+p) if r+p else 0)
        dcg = sum(1/math.log2(i+2) for i, page in enumerate(chosen) if page in gold)
        idcg = sum(1/math.log2(i+2) for i in range(min(len(gold), int(k))))
        ndcg.append(dcg/idcg if idcg else 0)
        accurate_top = sorted(chosen, key=lambda page: -float(evie[qpos[qid], ppos[page]]))[:10]
        evie_recall.append(len(set(accurate_top) & gold) / max(1, len(gold)))
        evdcg = sum(1/math.log2(i+2) for i, page in enumerate(accurate_top) if page in gold)
        evideal = sum(1/math.log2(i+2) for i in range(min(len(gold), len(accurate_top))))
        evie_ndcg.append(evdcg/evideal if evideal else 0)
    return {"mean_k": float(np.mean(ks)), "median_k": float(np.median(ks)), "p90_k": float(np.percentile(ks, 90)),
            "recall": float(np.mean(recall)), "precision": float(np.mean(precision)),
            "f1": float(np.mean(f1)), "ndcg": float(np.mean(ndcg)),
            "evie_recall_at_10": float(np.mean(evie_recall)), "evie_ndcg_at_10": float(np.mean(evie_ndcg))}


def run_arm(name: str, x_rank: np.ndarray, qids: list[str], candidates: dict[str, list[str]],
            gold: dict[str, set[str]], sources: np.ndarray, outer_splits,
            evie: np.ndarray, qpos: dict[str, int], ppos: dict[str, int]) -> tuple[dict, list[dict]]:
    all_ks = np.zeros(len(qids), dtype=np.int32)
    all_rankings: list[list[str] | None] = [None] * len(qids)
    all_targets = np.zeros(len(qids), dtype=np.int32)
    fold_rows = []
    for fold, (outer_tr, outer_te) in enumerate(outer_splits, 1):
        x_train_k = np.full((len(outer_tr), len(K_FEATURE_NAMES)), np.nan, dtype=np.float32)
        y_train_k = np.full(len(outer_tr), CAP, dtype=np.float32)
        ranked_train: list[list[str] | None] = [None] * len(outer_tr)
        # Train-fold score features are produced strictly OOF within outer train.
        inner_rank = StratifiedKFold(n_splits=4, shuffle=True, random_state=200 + fold)
        for ltr_fold, (loc_tr, loc_val) in enumerate(inner_rank.split(np.zeros(len(outer_tr)), sources[outer_tr]), 1):
            qtr = outer_tr[loc_tr].tolist(); qval = outer_tr[loc_val].tolist()
            tr_rows = candidate_groups(qtr); val_rows = candidate_groups(qval)
            ranker = XGBRanker(**LTR_PARAMS, random_state=3000 + 100 * fold + ltr_fold)
            ranker.fit(x_rank[tr_rows], np.asarray([int(p in gold[qids[i]]) for i in qtr for p in candidates[qids[i]]], dtype=np.int32),
                       group=[CAP] * len(qtr), verbose=False)
            pred = ranker.predict(x_rank[val_rows])
            for j, qi in enumerate(loc_val):
                qidx = int(outer_tr[qi]); qid = qids[qidx]
                s = pred[j*CAP:(j+1)*CAP]
                ranked = rank_from_scores(qid, s, candidates[qid])
                target, _ = target_and_positions(ranked, gold[qid])
                ranked_train[qi] = ranked; y_train_k[qi] = target
                x_train_k[qi] = np.asarray(k_row(sorted(map(float, s), reverse=True)), dtype=np.float32)

        # Outer LTR scores and ranking for validation queries.
        tr_rows = candidate_groups(outer_tr.tolist()); te_rows = candidate_groups(outer_te.tolist())
        ranker = XGBRanker(**LTR_PARAMS, random_state=4000 + fold)
        ranker.fit(x_rank[tr_rows], np.asarray([int(p in gold[qids[int(i)]]) for i in outer_tr for p in candidates[qids[int(i)]]], dtype=np.int32),
                   group=[CAP] * len(outer_tr), verbose=False)
        pred_test = ranker.predict(x_rank[te_rows])
        test_rankings, test_golds = [], []
        x_test_k = np.zeros((len(outer_te), len(K_FEATURE_NAMES)), dtype=np.float32)
        for j, qix in enumerate(outer_te):
            qid = qids[int(qix)]; s = pred_test[j*CAP:(j+1)*CAP]
            ranked = rank_from_scores(qid, s, candidates[qid])
            target, _ = target_and_positions(ranked, gold[qid])
            all_targets[int(qix)] = target
            all_rankings[int(qix)] = ranked
            test_rankings.append(ranked); test_golds.append(gold[qid])
            x_test_k[j] = np.asarray(k_row(sorted(map(float, s), reverse=True)), dtype=np.float32)

        # Inner OOF predictions calibrate only on outer-training queries.
        k_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=500 + fold)
        raw_inner = np.full(len(outer_tr), np.nan, dtype=np.float32)
        for kfold, (ktr, kval) in enumerate(k_inner.split(x_train_k, sources[outer_tr]), 1):
            reg = XGBRegressor(**K_PARAMS, random_state=6000 + fold*10 + kfold)
            reg.fit(x_train_k[ktr], y_train_k[ktr])
            raw_inner[kval] = reg.predict(x_train_k[kval])
        scale, margin, calib_info = choose_calibration(raw_inner,
            [r for r in ranked_train if r is not None], [gold[qids[int(i)]] for i in outer_tr])
        reg = XGBRegressor(**K_PARAMS, random_state=7000 + fold)
        reg.fit(x_train_k, y_train_k)
        raw_test = reg.predict(x_test_k)
        ks = integer_k(raw_test, scale, margin)
        all_ks[outer_te] = ks
        fm = metrics(ks, test_rankings, test_golds, [qids[int(i)] for i in outer_te], evie, qpos, ppos)
        fold_rows.append({"fold": fold, "scale": scale, "margin": margin, **calib_info, **fm})
        print(f"[{name}] fold {fold}/5 meanK={fm['mean_k']:.2f} recall={fm['recall']*100:.2f}%", flush=True)

    ranked_final = [r for r in all_rankings if r is not None]
    gold_final = [gold[q] for q in qids]
    all_metrics = metrics(all_ks, ranked_final, gold_final, qids, evie, qpos, ppos)
    by_source = {s: metrics(all_ks[sources == s], [ranked_final[i] for i in np.flatnonzero(sources == s)],
                            [gold_final[i] for i in np.flatnonzero(sources == s)],
                            [qids[i] for i in np.flatnonzero(sources == s)], evie, qpos, ppos) for s in sorted(set(sources))}
    rows = [{"arm": name, "qid": q, "source": q.split("::", 1)[0], "target_k_after_ltr": int(all_targets[i]),
             "predicted_k_oof": int(all_ks[i]), "prefix_recall": len(set(ranked_final[i][:all_ks[i]]) & gold[q]) / max(1,len(gold[q])),
             "ranked_pages": ranked_final[i]} for i,q in enumerate(qids)]
    return {"all": all_metrics, "by_source": by_source, "folds": fold_rows}, rows


def main() -> None:
    global CALIBRATION_BUDGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=58.0, help="Inner calibration mean-K ceiling (strictly less than this)")
    args = parser.parse_args()
    CALIBRATION_BUDGET = args.budget
    start = time.perf_counter()
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    qids = sorted(gate["light_run_top100"])
    candidates = {q: gate["light_run_top100"][q] for q in qids}
    gold = {q: set(gate["gold_rank"][q]) for q in qids}
    sources = np.asarray([q.split("::", 1)[0] for q in qids])
    evie = np.load(EVIE, mmap_mode="r")
    qpos = {q: i for i, q in enumerate(json.loads((COLVEC / "colvec_qids.json").read_text(encoding="utf-8")))}
    ppos = {p: i for i, p in enumerate(json.loads((COLVEC / "colvec_keys.json").read_text(encoding="utf-8")))}
    if evie.shape != (220, 5091) or set(qids) != set(qpos):
        raise ValueError("Cached EVIE matrix does not match the benchmark")
    metadata = json.loads((FEATURE_DIR / "feature_matrix_metadata.json").read_text(encoding="utf-8"))
    if metadata["qids"] != qids:
        raise ValueError("Feature matrix qid alignment mismatch")
    outer = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42).split(np.zeros(len(qids)), sources))
    reports, all_rows = {}, []
    for arm in ARMS:
        x = np.load(FEATURE_DIR / f"{arm}_feature_matrix.npy", mmap_mode="r")
        reports[arm], rows = run_arm(arm, x, qids, candidates, gold, sources, outer, evie, qpos, ppos)
        all_rows.extend(rows)
    baseline_sets = {}
    for budget, path in ((54, BASE_K_54), (58, BASE_K_58)):
        base = {r["qid"]: r for r in read_jsonl(path) if r["arm"] == "original"}
        baseline_rows = []
        for q in qids:
            k = int(base[q]["prediction_k_oof"])
            ranked = candidates[q]
            hit = len(set(ranked[:k]) & gold[q]); rec = hit / max(1, len(gold[q])); prec = hit / k
            baseline_rows.append({"qid": q, "k": k, "recall": rec, "precision": prec, "f1": 2*rec*prec/(rec+prec) if rec+prec else 0})
        baseline_sets[budget] = baseline_rows
        baseline_stats = metrics(np.asarray([r["k"] for r in baseline_rows], dtype=np.int32),
            [candidates[q] for q in qids], [gold[q] for q in qids], qids, evie, qpos, ppos)
        reports[f"xgb_only_discovery_budget{budget}"] = {"all": baseline_stats}
    boot = {}
    rng = np.random.default_rng(20261028)
    base_maps = {budget: {r["qid"]: r for r in rows} for budget, rows in baseline_sets.items()}
    for arm in ARMS:
        current = {r["qid"]: r for r in all_rows if r["arm"] == arm}
        boot[arm] = {}
        for budget, base_map in base_maps.items():
            d_recall = np.asarray([current[q]["prefix_recall"] - base_map[q]["recall"] for q in qids])
            d_k = np.asarray([current[q]["predicted_k_oof"] - base_map[q]["k"] for q in qids], dtype=float)
            ix = rng.integers(0, len(qids), size=(5000, len(qids)))
            recall_draws = d_recall[ix].mean(axis=1) * 100
            k_draws = d_k[ix].mean(axis=1)
            boot[arm][str(budget)] = {
                "recall_delta_pp": float(d_recall.mean() * 100),
                "recall_delta_ci95_pp": [float(x) for x in np.quantile(recall_draws, [.025, .975])],
                "mean_k_delta": float(d_k.mean()),
                "mean_k_delta_ci95": [float(x) for x in np.quantile(k_draws, [.025, .975])],
            }
    out = {"date":"2026-10-27", "status":"Nested exploratory query OOF; LTR train score curves are inner OOF within each outer training fold, outer test ranking is fit only on outer train qrels.",
           "protocol":{"queries":220,"candidate_depth":100,"outer_folds":5,"inner_ranker_folds":4,"inner_k_calibration_folds":4,
                       "mean_k_calibration_budget_lt":CALIBRATION_BUDGET,"k_label":"earliest predicted rank covering every gold page; 100 if unreachable",
                       "ltr_params":LTR_PARAMS,"k_regressor_params":K_PARAMS,"source_id_feature":False},
           "arms":reports,"paired_bootstrap_vs_budget58_discovery":boot,"runtime_seconds":time.perf_counter()-start}
    output_dir = OUT / f"budget_{args.budget:g}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "report.json", out)
    (output_dir / "per_query.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in all_rows), encoding="utf-8")
    lines=["# MetaRank-style LTR + nested adaptive K", "", f"Nested source-stratified 5-fold query OOF. K calibration budget is mean K <{CALIBRATION_BUDGET:g}.", "",
           "| Arm | Mean K | Prefix recall | Precision | F1 | Prefix nDCG | EVIE recall@10 | EVIE nDCG@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, report in reports.items():
        a=report["all"]
        lines.append(f"| {arm} | {a['mean_k']:.2f} | {a['recall']*100:.2f}% | {a['precision']*100:.2f}% | {a['f1']*100:.2f}% | {a.get('ndcg',0)*100:.2f} | {a.get('evie_recall_at_10',0)*100:.2f}% | {a.get('evie_ndcg_at_10',0)*100:.2f} |")
    lines += ["", "## Paired comparison to discovery adaptive K baselines", "", "| Arm | Baseline calibration budget | Δ Recall | 95% CI | Δ Mean K | 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for arm, by_budget in boot.items():
        for budget, d in by_budget.items():
            lines.append(f"| {arm} | {budget} | {d['recall_delta_pp']:+.2f} pp | [{d['recall_delta_ci95_pp'][0]:+.2f}, {d['recall_delta_ci95_pp'][1]:+.2f}] | {d['mean_k_delta']:+.2f} | [{d['mean_k_delta_ci95'][0]:+.2f}, {d['mean_k_delta_ci95'][1]:+.2f}] |")
    lines += ["", "This stage uses only cached per-page features; no embedding or reranker API calls. Gold pages are labels and evaluation only. The outer test qrels do not enter the ranker, K regressor, or calibration for that fold.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({arm: value["all"] for arm,value in reports.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
