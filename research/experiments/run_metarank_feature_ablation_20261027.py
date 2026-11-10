"""Feature ablation for Metarank-style LambdaMART on frozen top-100 candidates.

All reranker and embedding inference is reused from local caches. The only
supervision is binary gold-page relevance, grouped by query and evaluated with
source-stratified 5-fold query OOF. No source identifier or gold-derived
feature is supplied to the ranker.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBRanker

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GATE = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2/ef7f11811803/light_prep/gate_ppocrv5_ungated.json"
PAGES = GATE.parent / "pages_ppocrv5.jsonl"
VECTOR_DIR = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803/light_v2"
DISCOVERY = ROOT / "data/work/adaptive_retrieval_20261020_router_te3s_top100/retrieval_scores.jsonl"
RERANK_DIR = ROOT / "data/work/frozen_top100_reranker_evie_20261026/rerank"
OUT = ROOT / "data/work/metarank_feature_ablation_20261027"
CAP = 100
FOLDS = 5
SEED = 42
MODS = ["discovery", "cohere4pro", "qwen3_8b", "bge_m3_local"]
PARAMS = dict(objective="rank:pairwise", eval_metric="ndcg@20", n_estimators=180,
              max_depth=3, learning_rate=0.05, min_child_weight=5,
              subsample=0.85, colsample_bytree=0.9, reg_lambda=5.0,
              tree_method="hist", n_jobs=4, verbosity=0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def norm_rank(values: list[float], descending: bool = True) -> np.ndarray:
    order = np.argsort(-np.asarray(values) if descending else np.asarray(values), kind="stable")
    ranks = np.empty(len(order), dtype=np.float32)
    ranks[order] = np.arange(len(order), dtype=np.float32)
    return 1.0 - ranks / max(1, len(order) - 1)


def norm_score(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    return (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)


def dcg(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i, page in enumerate(ranked[:k]) if page in gold)


def metric_row(ranked: list[str], gold: set[str], k: int) -> dict[str, float]:
    hit = len(set(ranked[:k]) & gold)
    recall = hit / max(1, len(gold))
    precision = hit / k
    f1 = 2 * recall * precision / max(1e-12, recall + precision)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
    return {"recall": recall, "precision": precision, "f1": f1,
            "ndcg": dcg(ranked, gold, k) / ideal if ideal else 0.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    qids = sorted(gate["light_run_top100"])
    if len(qids) != 220 or any(len(gate["light_run_top100"][q]) != CAP for q in qids):
        raise ValueError("Expected 220 queries with exactly 100 frozen discovery candidates each")
    query_rows = {r["query_id"]: r for r in read_jsonl(ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl")}
    if set(query_rows) != set(qids):
        raise ValueError("Query ID set mismatch")
    source = np.asarray([q.split("::", 1)[0] for q in qids])
    gold = {q: set(gate["gold_rank"][q]) for q in qids}
    pages = {r["page_id"]: r for r in read_jsonl(PAGES)}
    if len(pages) != 5091:
        raise ValueError(f"Expected 5,091 PP-OCRv5 pages, got {len(pages)}")
    discovery = {r["qid"]: r for r in read_jsonl(DISCOVERY)}
    if set(discovery) != set(qids):
        raise ValueError("Discovery score qids do not match frozen candidate qids")

    # Cached full-corpus dense vectors add independent candidate-page evidence.
    dense_by_mod: dict[str, dict[str, dict[str, float]]] = {}
    for short, prefix in (("te3", "dense_te3s"), ("qwen8b", "dense_qwen8b_ppocr")):
        meta = json.loads((VECTOR_DIR / f"{prefix}_meta.json").read_text(encoding="utf-8"))
        qmat = np.load(VECTOR_DIR / f"{prefix}_queries.npy", mmap_mode="r")
        pmat = np.load(VECTOR_DIR / f"{prefix}_pages.npy", mmap_mode="r")
        qpos, ppos = {q: i for i, q in enumerate(meta["qids"])}, {p: i for i, p in enumerate(meta["units"])}
        if set(ppos) != set(pages) or any(q not in qpos for q in qids):
            raise ValueError(f"{short} embedding cache is not aligned to current benchmark")
        pv = np.array(pmat, dtype=np.float32, copy=True)
        pv /= np.maximum(np.linalg.norm(pv, axis=1, keepdims=True), 1e-12)
        dense_by_mod[short] = {}
        for q in qids:
            qv = np.array(qmat[qpos[q]], dtype=np.float32, copy=True)
            qv /= max(float(np.linalg.norm(qv)), 1e-12)
            cand = gate["light_run_top100"][q]
            dense_by_mod[short][q] = {p: float(pv[ppos[p]] @ qv) for p in cand}

    # Parse cached discovery/reranker outputs into aligned per-query scores and ranks.
    score: dict[str, dict[str, dict[str, float]]] = {m: {} for m in MODS}
    rank: dict[str, dict[str, dict[str, int]]] = {m: {} for m in MODS}
    fixed_runs: dict[str, dict[str, list[str]]] = {"discovery": {}, "cohere4pro": {}, "qwen3_8b": {}, "bge_m3_local": {}}
    for q in qids:
        cand = gate["light_run_top100"][q]
        dr = discovery[q]
        if dr["ranked_pages"] != cand:
            raise ValueError(f"Discovery candidate order mismatch: {q}")
        score["discovery"][q] = dict(zip(cand, map(float, dr["scores"])))
        rank["discovery"][q] = {p: i + 1 for i, p in enumerate(cand)}
        fixed_runs["discovery"][q] = cand
        for arm in MODS[1:]:
            f = RERANK_DIR / arm / (hashlib.sha256(q.encode()).hexdigest() + ".json")
            payload = json.loads(f.read_text(encoding="utf-8"))
            if payload["candidate_page_ids"] != cand or len(payload["scores"]) != CAP:
                raise ValueError(f"Reranker candidate alignment mismatch: {arm} {q}")
            vals = list(map(float, payload["scores"]))
            order = sorted(range(CAP), key=lambda i: (-vals[i], i))
            ranked = [cand[i] for i in order]
            fixed_runs[arm][q] = ranked
            score[arm][q] = {cand[i]: vals[i] for i in range(CAP)}
            rank[arm][q] = {p: i + 1 for i, p in enumerate(ranked)}

    # Full-corpus BM25 scores are added as a cheap lexical feature, never used
    # to expand the candidate pool. Build from the same cached PP-OCRv5 text.
    from src.retrieval.sparse import BM25Index
    bm25 = BM25Index(analyzer_name="plain").build(
        {"chunk_id": p, "doc_id": p.split("#page=", 1)[0], "text": row.get("text") or ""}
        for p, row in pages.items())
    lexical: dict[str, dict[str, dict[str, float]]] = {"bm25": {}, "overlap": {}, "page": {}}
    token_sets = {p: set(bm25.analyzer(str(row.get("text") or ""))) for p, row in pages.items()}
    token_count = {p: len(bm25.analyzer(str(row.get("text") or ""))) for p, row in pages.items()}
    numeric_re = re.compile(r"\d+(?:[.,]\d+)?(?:\s?(?:%|°|kg|mg|mm|cm|km|m|s|ms|hz|khz|mhz|ghz|v|a|w|kw))?", re.I)
    phrase_q = {q: str(query_rows[q]["query"]).lower().strip() for q in qids}
    for q in qids:
        query = str(query_rows[q]["query"])
        qterms = list(dict.fromkeys(bm25.analyzer(query)))
        hits = BM25Index.search(bm25, query, 5091)
        bm = {bm25.chunk_ids[i]: float(v) for i, v in hits}
        bm_vals = [bm.get(p, 0.0) for p in gate["light_run_top100"][q]]
        lexical["bm25"][q] = {p: bm.get(p, 0.0) for p in gate["light_run_top100"][q]}
        lexical["overlap"][q] = {}
        lexical["page"][q] = {}
        qnums = set(numeric_re.findall(query.lower()))
        qset = set(qterms)
        for p in gate["light_run_top100"][q]:
            tset = token_sets[p]
            matched = qset & tset
            text = str(pages[p].get("text") or "").lower()
            page_nums = set(numeric_re.findall(text))
            lexical["overlap"][q][p] = {
                "query_coverage": len(matched) / max(1, len(qset)),
                "query_precision": len(matched) / max(1, len(tset)),
                "matched_terms": float(len(matched)),
                "phrase_match": float(bool(phrase_q[q]) and phrase_q[q] in text),
                "numeric_overlap": len(qnums & page_nums) / max(1, len(qnums)),
            }
            lexical["page"][q][p] = {
                "log_token_count": math.log1p(token_count[p]),
                "unique_token_ratio": len(tset) / max(1, token_count[p]),
                "numeric_density": len(page_nums) / max(1, token_count[p]),
                "table_marker_count": float(text.count("[table") + text.count("|")),
                "ocr_marker_count": float(text.count("[image:") + text.count("[table:")),
            }

    families: dict[str, list[str]] = {}
    feat_by_q: dict[str, dict[str, dict[str, float]]] = {q: {} for q in qids}
    family_map: dict[str, list[str]] = {}
    for q in qids:
        cand = gate["light_run_top100"][q]
        base_names = []
        for m in MODS:
            vals = [score[m][q][p] for p in cand]
            ranks = [rank[m][q][p] for p in cand]
            sn, rp = norm_score(vals), norm_rank(vals)
            for i, p in enumerate(cand):
                feat_by_q[q].setdefault(p, {})[f"{m}_score_minmax"] = float(sn[i])
                feat_by_q[q][p][f"{m}_score_z"] = float((vals[i] - np.mean(vals)) / (np.std(vals) + 1e-8))
                feat_by_q[q][p][f"{m}_rank_pct"] = float(1 - (ranks[i] - 1) / (CAP - 1))
            base_names += [f"{m}_score_minmax", f"{m}_score_z", f"{m}_rank_pct"]
        family_map["score_rank"] = base_names

        for m in ("te3", "qwen8b"):
            vals = [dense_by_mod[m][q][p] for p in cand]
            sn, rp = norm_score(vals), norm_rank(vals)
            for i, p in enumerate(cand):
                feat_by_q[q][p][f"{m}_cosine"] = vals[i]
                feat_by_q[q][p][f"{m}_score_minmax"] = float(sn[i])
                feat_by_q[q][p][f"{m}_rank_pct"] = float(rp[i])
        family_map["dense_similarity"] = ["te3_cosine", "te3_score_minmax", "te3_rank_pct", "qwen8b_cosine", "qwen8b_score_minmax", "qwen8b_rank_pct"]

        bmvals = [lexical["bm25"][q][p] for p in cand]
        bm_norm = norm_score(bmvals)
        for i, p in enumerate(cand):
            feat_by_q[q][p]["bm25_raw"] = bmvals[i]
            feat_by_q[q][p]["bm25_score_minmax"] = float(bm_norm[i])
            feat_by_q[q][p]["bm25_rank_pct"] = float(norm_rank(bmvals)[i])
            for name, val in lexical["overlap"][q][p].items():
                feat_by_q[q][p][f"lex_{name}"] = float(val)
            for name, val in lexical["page"][q][p].items():
                feat_by_q[q][p][f"page_{name}"] = float(val)
        family_map["lexical_match"] = ["bm25_raw", "bm25_score_minmax", "bm25_rank_pct", "lex_query_coverage", "lex_query_precision", "lex_matched_terms", "lex_phrase_match", "lex_numeric_overlap"]
        family_map["page_shape"] = ["page_log_token_count", "page_unique_token_ratio", "page_numeric_density", "page_table_marker_count", "page_ocr_marker_count"]

        rank_mats = np.asarray([[rank[m][q][p] for m in MODS[1:]] for p in cand], dtype=np.float32)
        norm_mats = np.asarray([[feat_by_q[q][p][f"{m}_score_minmax"] for m in MODS[1:]] for p in cand], dtype=np.float32)
        for i, p in enumerate(cand):
            feat_by_q[q][p]["consensus_mean_rank"] = float(np.mean(rank_mats[i]))
            feat_by_q[q][p]["consensus_rank_std"] = float(np.std(rank_mats[i]))
            feat_by_q[q][p]["consensus_best_rank"] = float(np.min(rank_mats[i]))
            feat_by_q[q][p]["consensus_top20_votes"] = float(np.sum(rank_mats[i] <= 20))
            feat_by_q[q][p]["consensus_score_mean"] = float(np.mean(norm_mats[i]))
            feat_by_q[q][p]["consensus_score_std"] = float(np.std(norm_mats[i]))
            feat_by_q[q][p]["consensus_top50_votes"] = float(np.sum(rank_mats[i] <= 50))
            feat_by_q[q][p]["te3_x_rerank_consensus"] = float(feat_by_q[q][p]["te3_score_minmax"] * feat_by_q[q][p]["consensus_score_mean"])
        family_map["consensus"] = ["consensus_mean_rank", "consensus_rank_std", "consensus_best_rank", "consensus_top20_votes", "consensus_top50_votes", "consensus_score_mean", "consensus_score_std", "te3_x_rerank_consensus"]

    set_families = {
        "discovery_only": ["discovery_score_minmax", "discovery_score_z", "discovery_rank_pct"],
        "discovery_plus_cohere": family_map["score_rank"][:6],
        "discovery_plus_qwen": family_map["score_rank"][:3] + family_map["score_rank"][6:9],
        "discovery_plus_bge": family_map["score_rank"][:3] + family_map["score_rank"][9:12],
        "discovery_plus_cohere_page_shape": family_map["score_rank"][:6] + family_map["page_shape"],
        "score_rank": family_map["score_rank"],
        "score_rank_plus_dense": family_map["score_rank"] + family_map["dense_similarity"],
        "score_rank_plus_lexical": family_map["score_rank"] + family_map["lexical_match"],
        "score_rank_plus_page_shape": family_map["score_rank"] + family_map["page_shape"],
        "score_rank_plus_consensus": family_map["score_rank"] + family_map["consensus"],
        "all_engineered": family_map["score_rank"] + family_map["dense_similarity"] + family_map["lexical_match"] + family_map["page_shape"] + family_map["consensus"],
    }
    labels = np.asarray([int(p in gold[q]) for q in qids for p in gate["light_run_top100"][q]], dtype=np.int32)
    split = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    predictions: dict[str, dict[str, list[str]]] = {name: {} for name in set_families}
    oof_model_scores: dict[str, dict[str, list[float]]] = {name: {} for name in set_families}
    folds: dict[str, list[dict[str, Any]]] = {name: [] for name in set_families}
    importance: dict[str, list[dict[str, float]]] = {name: [] for name in set_families}
    feature_matrices = {}
    for name, names in set_families.items():
        feature_matrices[name] = np.asarray([
            [feat_by_q[q][p][n] for n in names]
            for q in qids for p in gate["light_run_top100"][q]
        ], dtype=np.float32)
        if not np.isfinite(feature_matrices[name]).all():
            raise ValueError(f"Non-finite features in {name}")
        for fold, (tr_q, te_q) in enumerate(split.split(np.zeros(len(qids)), source), 1):
            tr_ix = np.concatenate([np.arange(i * CAP, (i + 1) * CAP) for i in tr_q])
            te_ix = np.concatenate([np.arange(i * CAP, (i + 1) * CAP) for i in te_q])
            train_groups = [CAP] * len(tr_q)
            model = XGBRanker(**PARAMS, random_state=SEED + fold)
            t0 = time.perf_counter()
            model.fit(feature_matrices[name][tr_ix], labels[tr_ix], group=train_groups, verbose=False)
            fit_time = time.perf_counter() - t0
            pred = model.predict(feature_matrices[name][te_ix])
            for j, qix in enumerate(te_q):
                q = qids[int(qix)]
                cand = gate["light_run_top100"][q]
                scores = pred[j * CAP:(j + 1) * CAP]
                order = sorted(range(CAP), key=lambda x: (-float(scores[x]), x))
                predictions[name][q] = [cand[i] for i in order]
                oof_model_scores[name][q] = [float(v) for v in scores]
            folds[name].append({"fold": fold, "train_queries": len(tr_q), "test_queries": len(te_q), "fit_seconds": fit_time})
            imp = model.feature_importances_
            importance[name].append({n: float(v) for n, v in zip(names, imp)})
            print(f"{name}: fold {fold}/{FOLDS}, fit {fit_time:.2f}s", flush=True)

    all_runs: dict[str, dict[str, list[str]]] = {**fixed_runs}
    all_runs.update({f"metarank_ltr_{k}": v for k, v in predictions.items()})
    per_query = []
    summaries = {}
    for arm, runs in all_runs.items():
        ar = []
        for q in qids:
            row = {"qid": q, "source": q.split("::", 1)[0], "arm": arm}
            for k in (20, 50):
                row.update({f"{key}@{k}": val for key, val in metric_row(runs[q], gold[q], k).items()})
            ar.append(row); per_query.append(row)
        summaries[arm] = {
            "all": {metric: float(np.mean([r[metric] for r in ar])) for metric in ar[0] if metric not in ("qid", "source", "arm")},
            "by_source": {s: {metric: float(np.mean([r[metric] for r in ar if r["source"] == s])) for metric in ar[0] if metric not in ("qid", "source", "arm")} for s in sorted(set(source))},
        }

    # Paired query bootstrap against frozen discovery, reported for every learned arm.
    rng = np.random.default_rng(20261027)
    boot = {}
    baseline_rows = {r["qid"]: r for r in per_query if r["arm"] == "discovery"}
    for arm in all_runs:
        if not arm.startswith("metarank_ltr_"):
            continue
        rows = {r["qid"]: r for r in per_query if r["arm"] == arm}
        boot[arm] = {}
        for k in (20, 50):
            deltas = np.asarray([rows[q][f"recall@{k}"] - baseline_rows[q][f"recall@{k}"] for q in qids])
            draws = np.asarray([deltas[rng.integers(0, len(deltas), len(deltas))].mean() for _ in range(5000)])
            lo, hi = np.quantile(draws, [0.025, 0.975])
            boot[arm][str(k)] = {"delta_pp": float(deltas.mean() * 100), "ci95_low_pp": float(lo * 100), "ci95_high_pp": float(hi * 100)}

    report = {
        "date": "2026-10-27", "status": "Exploratory source-stratified 5-fold query OOF; direct XGBoost rank:pairwise equivalent backend objective, not a Metarank Java service run.",
        "protocol": {"queries": len(qids), "pages": len(pages), "candidates_per_query": CAP, "folds": FOLDS,
                     "split": "source-stratified query-level; query qrels only label train groups and held-out metrics",
                     "no_source_id_features": True, "candidate_set_frozen": True, "feature_sets": set_families,
                     "feature_source": "PP-OCRv5 pages and cached te3s/Qwen8B embeddings; cached Cohere4Pro/Qwen3-8B/BGE-M3 rerank scores; corpus BM25 from plain analyzer",
                     "xgboost_params": PARAMS},
        "arms": summaries, "paired_bootstrap_vs_discovery": boot, "folds": folds, "feature_importance_by_fold": importance,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "report.json", report)
    (OUT / "per_query.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_query), encoding="utf-8")
    (OUT / "oof_rankings.jsonl").write_text("".join(json.dumps({"qid": q, **{k: v[q] for k, v in predictions.items()}}, ensure_ascii=False) + "\n" for q in qids), encoding="utf-8")
    (OUT / "oof_model_scores.jsonl").write_text("".join(json.dumps({"qid": q, **{k: v[q] for k, v in oof_model_scores.items()}}, ensure_ascii=False) + "\n" for q in qids), encoding="utf-8")
    np.save(OUT / "all_engineered_feature_matrix.npy", feature_matrices["all_engineered"])
    np.save(OUT / "score_rank_feature_matrix.npy", feature_matrices["score_rank"])
    for feature_set, matrix in feature_matrices.items():
        np.save(OUT / f"{feature_set}_feature_matrix.npy", matrix)
    write_json(OUT / "feature_matrix_metadata.json", {
        "qids": qids,
        "candidate_page_ids_by_qid": {q: gate["light_run_top100"][q] for q in qids},
        "feature_names": set_families["all_engineered"],
        "row_order": "qid order in qids, then candidate order in candidate_page_ids_by_qid",
    })
    md = ["# MetaRank feature ablation", "", "Exploratory source-stratified 5-fold query OOF on 220 queries, frozen top-100 discovery candidates.", "",
          "The model uses XGBoost `rank:pairwise`, matching the LambdaMART backend objective. This is a local backend experiment, not a Metarank Java service deployment.", "",
          "## Prefix metrics", "", "| Arm | Recall@20 | Precision@20 | F1@20 | nDCG@20 | Recall@50 | Precision@50 | F1@50 | nDCG@50 | Δ Recall@20 vs discovery (95% CI) | Δ Recall@50 vs discovery (95% CI) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, val in summaries.items():
        a = val["all"]
        d20 = boot.get(arm, {}).get("20")
        d50 = boot.get(arm, {}).get("50")
        ci20 = f"{d20['delta_pp']:+.2f} [{d20['ci95_low_pp']:+.2f}, {d20['ci95_high_pp']:+.2f}]" if d20 else "—"
        ci50 = f"{d50['delta_pp']:+.2f} [{d50['ci95_low_pp']:+.2f}, {d50['ci95_high_pp']:+.2f}]" if d50 else "—"
        md.append(f"| {arm} | {a['recall@20']*100:.2f}% | {a['precision@20']*100:.2f}% | {a['f1@20']*100:.2f}% | {a['ndcg@20']*100:.2f} | {a['recall@50']*100:.2f}% | {a['precision@50']*100:.2f}% | {a['f1@50']*100:.2f}% | {a['ndcg@50']*100:.2f} | {ci20} | {ci50} |")
    md += ["", "## Feature definitions", "", "- `score_rank`: candidate-local score min-max, z-score and percentile rank from discovery and each cached reranker.", "- `dense_similarity`: cached TE3 and Qwen8B query-page cosine, candidate-local normalized score and rank.", "- `lexical_match`: BM25 score/rank and query-term coverage, precision, exact phrase, numeric overlap.", "- `page_shape`: text length, unique-token ratio, numeric density, table/image markers.", "- `consensus`: reranker rank mean/std/best, top-20/top-50 votes, score agreement and a dense-consensus interaction.", "", "No source labels, qrels-derived aggregates, or gold page identities are input features. Reranker cache scores and candidate sets are aligned and frozen.", "",
          "## By source", ""]
    for arm, val in summaries.items():
        md += [f"### {arm}", "", "| Source | n | Recall@20 | Recall@50 | nDCG@20 | nDCG@50 |", "|---|---:|---:|---:|---:|---:|"]
        for s, row in val["by_source"].items():
            md.append(f"| {s} | {int(np.sum(source == s))} | {row['recall@20']*100:.2f}% | {row['recall@50']*100:.2f}% | {row['ndcg@20']*100:.2f} | {row['ndcg@50']*100:.2f} |")
        md.append("")
    md += ["## Caveats", "", "This is same-benchmark exploratory model selection; OOF protects each query from its own qrels but does not provide an independent dataset. Bootstrap intervals quantify paired query variation, not external generalization. A learned arm must exceed discovery by +5 percentage points at top-20 or top-50 before being considered for adaptive-K evaluation. The cached reranker arms are included as direct baselines.", ""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
