import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..bench import BENCH, FLAWED_QIDS, ROOT, gold as load_gold, queries as load_queries, work, write_jsonl
from .modes import SEED, Modes, seed_everything
from .query_prep import NoPrep
from .selector import K_ARTIFACT, retriever_signature

CAP, MAX_MEAN_K = 100, 60.0


def ranked_rows(light, qids, queries, gold):
    prep, rows = NoPrep(), []
    from ..adaptive_retrieval.dataset import _target_from_gold_ranks
    from ..adaptive_retrieval.features import extract_score_features
    for qid in qids:
        subdata, _ = light.retrieve(qid, prep.prepare(queries[qid]["query"]))
        top = subdata.pages[:CAP]
        rank = {u: i + 1 for i, u in enumerate(top)}
        gold_ranks = {u: rank.get(u) for u in gold[qid]}
        target, reachable = _target_from_gold_ranks(gold_ranks, CAP)
        rows.append({"qid": qid, "source": queries[qid].get("source") or qid.split("::", 1)[0], "target_k": target,
                     "reachable_at_cap": reachable, "gold_count": len(gold_ranks), "gold_ranks": gold_ranks,
                     "features": extract_score_features([subdata.scores[u] for u in top]),
                     "scores": [subdata.scores[u] for u in top], "pages": top})
    return rows


def train(rows, max_mean_k=MAX_MEAN_K):
    from sklearn.model_selection import StratifiedKFold
    from ..adaptive_retrieval.model import _choose_calibration, _inner_oof, _integer_k, _matrix, _new_model, \
        _row_metrics
    x, y, sources, gold_ranks = _matrix(rows)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    chosen = np.zeros(len(rows), dtype=np.int32)
    folds = []
    for fold, (tr, te) in enumerate(outer.split(x, sources), start=1):
        cal = _choose_calibration(_inner_oof(x[tr], y[tr], sources[tr], 100 + fold), [gold_ranks[i] for i in tr],
                                  CAP, max_mean_k)
        model = _new_model(1000 + fold)
        model.fit(x[tr], y[tr])
        chosen[te] = _integer_k(model.predict(x[te]).astype(np.float64), cal["scale"], cal["margin"], CAP)
        folds.append({"fold": fold, "scale": cal["scale"], "margin": cal["margin"],
                      "constraint_met": cal["constraint_met"], "test": [rows[i]["qid"] for i in te]})
    deploy = _choose_calibration(_inner_oof(x, y, sources, 777), gold_ranks, CAP, max_mean_k)
    final = _new_model(4242)
    final.fit(x, y)
    metrics = _row_metrics(chosen, gold_ranks)
    per_source = {s: _row_metrics(chosen[sources == s], [gold_ranks[i] for i in np.flatnonzero(sources == s)])
                  for s in sorted(set(sources))}
    return final, deploy, chosen, folds, metrics, per_source


def main():
    from .pipeline import build_light
    from ..adaptive_retrieval.features import FEATURE_NAMES
    from ..adaptive_retrieval.model import XGB_PARAMETERS
    ap = argparse.ArgumentParser(prog="python -m ondemand.adaptive.retrain_k")
    ap.add_argument("--embedder", default="qwen3_8b")
    ap.add_argument("--signals", default="lang")
    ap.add_argument("--file-mode", default="blended_pool")
    ap.add_argument("--w-dense", type=float, default=0.70)
    ap.add_argument("--parent", type=float, default=0.15)
    ap.add_argument("--pages", default="pages_ppocrv5.jsonl")
    ap.add_argument("--max-mean-k", type=float, default=MAX_MEAN_K)
    ap.add_argument("--out", help="artifact folder (default: <work>/adaptive_k/qwen8b_lang_blended_214)")
    args = ap.parse_args()

    seed_everything(SEED)
    out = (Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out) if args.out \
        else work("adaptive_k", K_ARTIFACT, bench=BENCH)
    if (out / "xgboost_adaptive_k.json").exists():
        raise SystemExit(f"{out} already holds a model; pick a new --out")
    out.mkdir(parents=True, exist_ok=True)
    stages = {"light": "hier", "embedder": args.embedder, "signals": args.signals, "file_mode": args.file_mode,
              "w_dense": args.w_dense, "parent": args.parent}
    gold, queries = load_gold(BENCH), load_queries(BENCH)
    qids = [q for q in sorted(gold) if q not in FLAWED_QIDS]
    light = build_light(stages, Modes("offline"), BENCH, args.pages)
    light.prepare(qids)
    rows = ranked_rows(light, qids, queries, gold)
    final, deploy, chosen, folds, metrics, per_source = train(rows, args.max_mean_k)

    signature = retriever_signature(light.settings())
    final.save_model(out / "xgboost_adaptive_k.json")
    (out / "deployment_policy.json").write_text(json.dumps({
        "cap": CAP, "features": list(FEATURE_NAMES), "feature_scaling": "none", "feature_scaler_artifact": None,
        "scale": deploy["scale"], "margin": deploy["margin"], "parameters": XGB_PARAMETERS, "seed": 4242,
        "training_oof_metrics": metrics,
        "note": f"Fit on the {len(rows)} benchmark queries; benchmark runs must read oof_k.json, not this model"},
        indent=1))
    (out / "dataset_manifest.json").write_text(json.dumps({
        "n_queries": len(rows), "settings": {**signature, "cap": CAP, "pages_source": args.pages,
                                             "embedder": light.settings()["embedder"]},
        "label": "smallest rank containing every qrel gold page; cap if any gold is outside top-cap",
        "calibration": f"max macro recall subject to mean K < {args.max_mean_k:g}", "features": list(FEATURE_NAMES),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1))
    (out / "oof_k.json").write_text(json.dumps({
        "n": len(rows), "metrics": metrics, "per_source": per_source, "folds": folds,
        "predictions": {r["qid"]: int(k) for r, k in zip(rows, chosen)}}, indent=1))
    write_jsonl(out / "feature_labels.jsonl", [{k: r[k] for k in ("qid", "source", "target_k", "reachable_at_cap",
                                                                  "gold_count", "gold_ranks", "features")}
                                               for r in rows])
    write_jsonl(out / "retrieval_scores.jsonl", [{"qid": r["qid"], "pages": r["pages"], "scores": r["scores"]}
                                                 for r in rows])
    print(json.dumps({"out": str(out), "n": len(rows), "oof": metrics,
                      "deploy_calibration": {k: deploy[k] for k in ("scale", "margin", "mean_k", "macro_recall")}},
                     indent=1))
    for source, m in per_source.items():
        print(source, round(m["mean_k"], 2), round(100 * m["macro_recall"], 2))


if __name__ == "__main__":
    main()
