"""Evaluate discovery + page-shape MetaRank features with no Cohere signal."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
DATA = ROOT / "data/work/metarank_feature_ablation_20261027"
OUT = DATA / "no_cohere_discovery_page_shape"
CAP = 100
PARAMS = dict(objective="rank:pairwise", eval_metric="ndcg@20", n_estimators=180,
              max_depth=3, learning_rate=0.05, min_child_weight=5,
              subsample=0.85, colsample_bytree=0.9, reg_lambda=5.0,
              tree_method="hist", n_jobs=4, verbosity=0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def ndcg(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked[:k]) if p in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    qids = sorted(gate["light_run_top100"])
    sources = np.asarray([q.split("::", 1)[0] for q in qids])
    candidates = {q: gate["light_run_top100"][q] for q in qids}
    gold = {q: set(gate["gold_rank"][q]) for q in qids}
    meta = json.loads((DATA / "feature_matrix_metadata.json").read_text(encoding="utf-8"))
    matrix = np.load(DATA / "all_engineered_feature_matrix.npy", mmap_mode="r")
    if meta["qids"] != qids:
        raise ValueError("Feature qid alignment mismatch")
    wanted = ["discovery_score_minmax", "discovery_score_z", "discovery_rank_pct",
              "page_log_token_count", "page_unique_token_ratio", "page_numeric_density",
              "page_table_marker_count", "page_ocr_marker_count"]
    cols = [meta["feature_names"].index(name) for name in wanted]
    x = np.asarray(matrix[:, cols], dtype=np.float32)
    labels = np.asarray([int(p in gold[q]) for q in qids for p in candidates[q]], dtype=np.int32)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    runs: dict[str, dict[str, list[str]]] = {"discovery": {q: candidates[q] for q in qids}, "metarank_discovery_page_shape": {}}
    folds = []
    for fold, (tr, te) in enumerate(splitter.split(np.zeros(len(qids)), sources), 1):
        tr_rows = np.concatenate([np.arange(i*CAP,(i+1)*CAP) for i in tr])
        te_rows = np.concatenate([np.arange(i*CAP,(i+1)*CAP) for i in te])
        model = XGBRanker(**PARAMS, random_state=42+fold)
        model.fit(x[tr_rows], labels[tr_rows], group=[CAP]*len(tr), verbose=False)
        pred = model.predict(x[te_rows])
        for j, qi in enumerate(te):
            q = qids[int(qi)]; scores = pred[j*CAP:(j+1)*CAP]
            order = sorted(range(CAP), key=lambda i: (-float(scores[i]), i))
            runs["metarank_discovery_page_shape"][q] = [candidates[q][i] for i in order]
        folds.append({"fold": fold, "train_queries": len(tr), "test_queries": len(te)})

    per_query, summaries = [], {}
    for arm, ranking in runs.items():
        rows = []
        for q in qids:
            row = {"qid": q, "source": q.split("::",1)[0], "arm": arm}
            for k in (20,50):
                hit = len(set(ranking[q][:k]) & gold[q]); r=hit/len(gold[q]); p=hit/k
                row.update({f"recall@{k}":r,f"precision@{k}":p,f"f1@{k}":2*r*p/(r+p) if r+p else 0,
                            f"ndcg@{k}":ndcg(ranking[q],gold[q],k)})
            rows.append(row); per_query.append(row)
        summaries[arm] = {m:float(np.mean([row[m] for row in rows])) for m in rows[0] if m not in ("qid","source","arm")}
    base = {r["qid"]:r for r in per_query if r["arm"]=="discovery"}
    cur = {r["qid"]:r for r in per_query if r["arm"]=="metarank_discovery_page_shape"}
    rng=np.random.default_rng(20261028); boot={}
    for k in (20,50):
        d=np.asarray([cur[q][f"recall@{k}"]-base[q][f"recall@{k}"] for q in qids])
        idx=rng.integers(0,len(qids),size=(5000,len(qids))); draw=d[idx].mean(axis=1)*100
        lo,hi=np.quantile(draw,[.025,.975]); boot[str(k)]={"delta_pp":float(d.mean()*100),"ci95_low_pp":float(lo),"ci95_high_pp":float(hi)}
    OUT.mkdir(parents=True,exist_ok=True)
    report={"protocol":{"queries":len(qids),"candidate_depth":CAP,"folds":5,"features":wanted,"cohere_features":False,"source_id_feature":False,"xgboost_params":PARAMS},"arms":summaries,"paired_bootstrap_vs_discovery":boot,"folds":folds}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (OUT/"per_query.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in per_query),encoding="utf-8")
    lines=["# MetaRank without Cohere features", "", "Source-stratified five-fold query OOF, frozen top-100 candidate set. Features are discovery scores/ranks plus page-shape statistics; no Cohere score is used.", "", "| Arm | Recall@20 | Recall@50 | nDCG@20 | nDCG@50 |", "|---|---:|---:|---:|---:|"]
    for arm,m in summaries.items(): lines.append(f"| {arm} | {m['recall@20']*100:.2f}% | {m['recall@50']*100:.2f}% | {m['ndcg@20']*100:.2f} | {m['ndcg@50']*100:.2f} |")
    lines += ["", "Paired recall change vs discovery:", ""]
    for k,v in boot.items(): lines.append(f"- Recall@{k}: {v['delta_pp']:+.2f} pp (95% CI [{v['ci95_low_pp']:+.2f}, {v['ci95_high_pp']:+.2f}]).")
    (OUT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"arms":summaries,"paired_bootstrap_vs_discovery":boot},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
