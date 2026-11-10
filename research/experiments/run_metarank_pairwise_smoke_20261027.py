"""Quick query-grouped pairwise LTR smoke test using Metarank's XGBoost objective.

This runs the same XGBoost `rank:pairwise` objective used by Metarank's
LambdaMART backend, with already cached score/rank features. It deliberately
does not start Metarank's Java service; see the generated report for that
runtime limitation. Outer folds are split by query and stratified by source.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
ORIGINAL = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100/retrieval_scores.jsonl"
RERANK = ROOT / "data/work/frozen_top100_reranker_evie_20261026/rerank"
EVIE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/routing_axes/k3/k3_out_shard0/evie_scores_2048.npy"
COLVEC = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/colvec"
OUT = ROOT / "data/work/metarank_pairwise_smoke_20261027"
CAP = 100
TOPK = 50
ARMS = ["cohere4pro", "qwen3_8b", "bge_m3_local"]
MODEL_PARAMS = dict(
    objective="rank:pairwise", eval_metric="ndcg@10", n_estimators=120,
    max_depth=3, learning_rate=0.05, min_child_weight=5,
    subsample=0.85, colsample_bytree=0.9, reg_lambda=5.0,
    tree_method="hist", n_jobs=4, verbosity=0,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_scores(path: Path) -> tuple[list[str], list[float]]:
    row = json.loads(path.read_text(encoding="utf-8"))
    return row["candidate_page_ids"], [float(v) for v in row["scores"]]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def metrics(ranked: list[str], gold: set[str], evie_row: np.ndarray, p_index: dict[str, int]) -> dict:
    prefix = ranked[:TOPK]
    hits50 = len(set(prefix) & gold)
    recall50 = hits50 / len(gold) if gold else 0.0
    precision50 = hits50 / TOPK
    f1 = 2 * recall50 * precision50 / (recall50 + precision50) if recall50 + precision50 else 0.0
    top10 = sorted(prefix, key=lambda page: -float(evie_row[p_index[page]]))[:10]

    def ndcg(pages: list[str]) -> float:
        dcg = sum(1 / math.log2(rank + 2) for rank, page in enumerate(pages) if page in gold)
        ideal_n = min(len(gold), len(pages))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(ideal_n))
        return dcg / ideal if ideal else 0.0

    return {
        "prefix_recall_at_50": recall50,
        "prefix_precision_at_50": precision50,
        "prefix_f1_at_50": f1,
        "prefix_ndcg_at_50": ndcg(prefix),
        "evie_recall_at_10": sum(page in gold for page in top10) / len(gold) if gold else 0.0,
        "evie_ndcg_at_10": ndcg(top10),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    qids = sorted(gate["light_run_top100"])
    assert len(qids) == 220 and all(len(gate["light_run_top100"][q]) == CAP for q in qids)
    sources = np.asarray([q.split("::", 1)[0] for q in qids])
    original_by_qid = {row["qid"]: row for row in read_jsonl(ORIGINAL)}
    assert set(original_by_qid) == set(qids)
    page_index = {p: i for i, p in enumerate(json.loads((COLVEC / "colvec_keys.json").read_text(encoding="utf-8")))}
    query_index = {q: i for i, q in enumerate(json.loads((COLVEC / "colvec_qids.json").read_text(encoding="utf-8")))}
    evie = np.load(EVIE, mmap_mode="r")
    if evie.shape != (len(query_index), len(page_index)):
        raise ValueError(f"EVIE matrix shape mismatch: {evie.shape}")

    # Each modality provides per-page score and relative rank. Candidate set is fixed.
    modalities = ["discovery", *ARMS]
    score_by_mod: dict[str, dict[str, dict[str, float]]] = {m: {} for m in modalities}
    rank_by_mod: dict[str, dict[str, dict[str, int]]] = {m: {} for m in modalities}
    for qid in qids:
        base = original_by_qid[qid]
        pages = base["ranked_pages"]
        if pages != gate["light_run_top100"][qid] or len(base["scores"]) != CAP:
            raise ValueError(f"Frozen discovery alignment mismatch: {qid}")
        score_by_mod["discovery"][qid] = dict(zip(pages, map(float, base["scores"])))
        rank_by_mod["discovery"][qid] = {p: i + 1 for i, p in enumerate(pages)}
        for arm in ARMS:
            file = RERANK / arm / (hashlib.sha256(qid.encode()).hexdigest() + ".json")
            candidates, scores = read_scores(file)
            if candidates != pages or len(scores) != CAP:
                raise ValueError(f"Reranker alignment mismatch: {arm} {qid}")
            order = sorted(range(CAP), key=lambda i: (-scores[i], i))
            ranked = [pages[i] for i in order]
            ordered_scores = [scores[i] for i in order]
            score_by_mod[arm][qid] = dict(zip(ranked, ordered_scores))
            rank_by_mod[arm][qid] = {p: i + 1 for i, p in enumerate(ranked)}

    gold_by_qid = {q: set(gate["gold_rank"][q]) for q in qids}
    if any(not g for g in gold_by_qid.values()):
        raise ValueError("Gold page map contains an empty query")

    # Each feature set asks whether adding an already-computed reranker signal helps.
    feature_sets = {
        "discovery_score_rank": ["discovery"],
        "discovery_plus_cohere": ["discovery", "cohere4pro"],
        "discovery_plus_qwen": ["discovery", "qwen3_8b"],
        "discovery_plus_bge": ["discovery", "bge_m3_local"],
        "all_score_rank_signals": modalities,
    }
    feature_names = {
        name: [f"{mod}_{suffix}" for mod in mods for suffix in ("score_minmax", "rank_percentile")]
        for name, mods in feature_sets.items()
    }
    x_by_set: dict[str, np.ndarray] = {}
    rows_per_query = CAP
    for set_name, mods in feature_sets.items():
        rows = []
        for qid in qids:
            pages = gate["light_run_top100"][qid]
            norm_scores = {m: minmax(np.asarray([score_by_mod[m][qid][p] for p in pages], dtype=np.float32)) for m in mods}
            for page in pages:
                values = []
                for m in mods:
                    values.extend([float(norm_scores[m][pages.index(page)]), 1.0 - (rank_by_mod[m][qid][page] - 1) / (CAP - 1)])
                rows.append(values)
        x_by_set[set_name] = np.asarray(rows, dtype=np.float32)

    # Emit a Metarank-native event/config pair using the same precomputed
    # per-page feature values and explicit qrels labels. This makes the data
    # side directly reusable once its Java 21 runtime is available.
    metarank_features = feature_names["all_score_rank_signals"]
    event_path = OUT / "metarank_events.jsonl"
    rng = np.random.default_rng(42)
    shuffled_qids = list(np.asarray(qids, dtype=object)[rng.permutation(len(qids))])
    with event_path.open("w", encoding="utf-8") as stream:
        for event_no, qid in enumerate(shuffled_qids, start=1):
            pages = gate["light_run_top100"][qid]
            item_rows = []
            for page in pages:
                vals = {}
                for mod in modalities:
                    page_rank = rank_by_mod[mod][qid][page]
                    score_values = np.asarray([score_by_mod[mod][qid][p] for p in pages], dtype=np.float32)
                    vals[f"{mod}_score_minmax"] = float(minmax(score_values)[pages.index(page)])
                    vals[f"{mod}_rank_percentile"] = float(1.0 - (page_rank - 1) / (CAP - 1))
                item_rows.append({
                    "id": page,
                    "label": int(page in gold_by_qid[qid]),
                    "fields": [{"name": name, "value": vals[name]} for name in metarank_features],
                })
            event = {"event": "ranking", "id": f"ltr-{event_no:04d}",
                     "timestamp": str(1_798_000_000_000 + event_no), "items": item_rows}
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    config_lines = [
        "state:", "  type: memory", "models:", "  ltr:", "    type: lambdamart",
        "    backend:", "      type: xgboost", "      iterations: 120", "      learningRate: 0.05",
        "      maxDepth: 3", "      sampling: 0.9", "      seed: 42", "      ndcgCutoff: 10",
        "    weights:", "      click: 1", "    features:",
    ]
    config_lines += [f"      - {name}" for name in metarank_features]
    config_lines += ["    split: random=80%", "    eval: [ndcg@10]", "features:"]
    for name in metarank_features:
        config_lines += [f"  - name: {name}", "    type: number", "    scope: item", f"    field: ranking.{name}"]
    (OUT / "metarank_config.yml").write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    # Fixed reranker baselines are useful to see if the learned combination changes ordering.
    fixed_rankings: dict[str, dict[str, list[str]]] = {"discovery_fixed50": {}}
    for qid in qids:
        fixed_rankings["discovery_fixed50"][qid] = gate["light_run_top100"][qid]
    for arm in ARMS:
        fixed_rankings[f"{arm}_fixed50"] = {q: sorted(gate["light_run_top100"][q], key=lambda p: rank_by_mod[arm][q][p]) for q in qids}

    results: dict[str, dict[str, list[str]]] = {**fixed_rankings}
    fold_info: dict[str, list[dict]] = {name: [] for name in feature_sets}
    y = np.asarray([int(p in gold_by_qid[q]) for q in qids for p in gate["light_run_top100"][q]], dtype=np.float32)
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for set_name, X in x_by_set.items():
        oof_rankings: dict[str, list[str]] = {}
        for fold, (train_q, test_q) in enumerate(split.split(np.zeros(len(qids)), sources), start=1):
            train_ix = np.concatenate([np.arange(i * CAP, (i + 1) * CAP) for i in train_q])
            test_ix = np.concatenate([np.arange(i * CAP, (i + 1) * CAP) for i in test_q])
            train_y = y[train_ix]
            train_groups = np.full(len(train_q), CAP, dtype=np.uint32)
            ranker = XGBRanker(**MODEL_PARAMS, random_state=100 + fold)
            started = time.perf_counter()
            ranker.fit(X[train_ix], train_y, group=train_groups, verbose=False)
            fit_seconds = time.perf_counter() - started
            pred = ranker.predict(X[test_ix])
            for local, qidx in enumerate(test_q):
                qid = qids[qidx]
                candidate_pages = gate["light_run_top100"][qid]
                offset = local * CAP
                score_slice = pred[offset:offset + CAP]
                order = sorted(range(CAP), key=lambda j: (-float(score_slice[j]), j))
                oof_rankings[qid] = [candidate_pages[j] for j in order]
            fold_info[set_name].append({"fold": fold, "n_train_queries": len(train_q), "n_test_queries": len(test_q), "fit_seconds": fit_seconds})
            print(f"[{set_name}] fold {fold}/5 fit={fit_seconds:.2f}s", flush=True)
        results[f"metarank_ltr_{set_name}"] = oof_rankings

    # Evaluate each arm globally and by source using cached EVIE scores.
    report_arms = {}
    per_query = []
    for arm, rankings in results.items():
        rows = []
        for qid in qids:
            m = metrics(rankings[qid], gold_by_qid[qid], evie[query_index[qid]], page_index)
            rows.append(m)
            per_query.append({"arm": arm, "qid": qid, "source": qid.split("::", 1)[0], **m})
        report_arms[arm] = {"all": {k: float(np.mean([r[k] for r in rows])) for k in rows[0]},
                            "by_source": {s: {k: float(np.mean([per_query_row[k] for per_query_row in rows_for_source]))
                                               for k in rows_for_source[0]}
                                          for s in sorted(set(sources))
                                          for rows_for_source in [[r for q, r in zip(qids, rows) if q.split("::", 1)[0] == s]]}}

    report = {
        "date": "2026-10-27",
        "status": "Exploratory pairwise LTR backend smoke test; XGBoost rank:pairwise, the objective used by Metarank LambdaMART backend. Metarank Java service itself was not executed because Docker daemon is unavailable and only Java 17 is installed while this checkout requires Java 21.",
        "protocol": {"queries": len(qids), "sources": {s: int(np.sum(sources == s)) for s in sorted(set(sources))},
                     "candidate_pages_per_query": CAP, "selected_prefix": TOPK, "folds": 5,
                     "split": "query-level, stratified by source; qrels used only as training labels in train folds and evaluation labels in held-out folds",
                     "features": {key: feature_names[key] for key in feature_sets},
                     "xgboost_params": MODEL_PARAMS,
                     "reranker_feature_scores_from": str(RERANK.relative_to(ROOT))},
        "arms": report_arms,
        "folds": fold_info,
        "total_fit_seconds": float(sum(v["fit_seconds"] for folds in fold_info.values() for v in folds)),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for row in per_query:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    md = ["# Metarank pairwise LTR smoke test", "", "Date: 2026-10-27. Exploratory, same-benchmark 5-fold OOF.", "",
          "This is the XGBoost `rank:pairwise` LambdaMART backend objective used by Metarank, tested directly through `XGBRanker`. It is not a run of Metarank's Java service: Docker daemon is unavailable and the checked-out source targets Java 21, while this machine has Java 17. A Metarank-native event file and config are included for a direct run later.", "",
          "Every query has the same 100 candidate pages. The LTR model reorders them; metrics show top-50 candidate coverage, then cached EVIE top-10 recall/nDCG. Qrels train only the 4 training folds per outer split. No source ID is a feature.", "",
          "## Results", "", "| Method | Prefix Recall@50 | Prefix Precision@50 | Prefix F1@50 | EVIE Recall@10 | EVIE nDCG@10 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, values in report_arms.items():
        a = values["all"]
        md.append(f"| {arm} | {a['prefix_recall_at_50']*100:.2f}% | {a['prefix_precision_at_50']*100:.2f}% | {a['prefix_f1_at_50']*100:.2f}% | {a['evie_recall_at_10']*100:.2f}% | {a['evie_ndcg_at_10']*100:.2f} |")
    md += ["", "## By source", ""]
    for arm, values in report_arms.items():
        md.append(f"### {arm}")
        md.append("")
        md.append("| Source | n | Prefix Recall@50 | EVIE Recall@10 | EVIE nDCG@10 |")
        md.append("|---|---:|---:|---:|---:|")
        for source, a in values["by_source"].items():
            n = int(np.sum(sources == source))
            md.append(f"| {source} | {n} | {a['prefix_recall_at_50']*100:.2f}% | {a['evie_recall_at_10']*100:.2f}% | {a['evie_ndcg_at_10']*100:.2f} |")
        md.append("")
    md += ["## Timing", "", f"Total XGBoost fitting time across five-fold feature ablations: {report['total_fit_seconds']:.2f} seconds (excluding data loading and evaluation).", "",
           "## Feature sets and caveats", "", "Features are per-page score min-max normalized within the query plus rank percentile: original discovery alone; original plus one reranker score stream; and all score streams together. Candidate generation stays frozen at top-100. Since the score sources were produced by existing rerankers, this tests pairwise learning to combine their signals, not a new text encoder.", "",
           "The 220-query dataset is small. Treat results as an initial signal; feature-set selection on these same OOF outputs is exploratory. Full Metarank deployment and a second held-out benchmark are still needed before claiming an improvement.", "",
           "Metarank input materialization: `metarank_events.jsonl` (one labeled ranking group/query) and `metarank_config.yml` (numeric score/rank features, XGBoost LambdaMART, random 80/20 split). The configuration is generated from the checked-out Metarank schema but could not be runtime-validated on this host.", "",
           "Files: `report.json`, `per_query.jsonl`."]
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
