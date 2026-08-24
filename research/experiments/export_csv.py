"""Export every ViDoRe experiment as CSV alongside its JSON artifact."""
import json
import random
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "data/benchmark/vidore_v3/results"
CSV = RES / "csv"
RESAMPLES = 10000

CSV.mkdir(exist_ok=True)
META = {
    "vidore_page":       ("ViDoRe markdown", "none", "control; the paper's simplest pipeline"),
    "vidore_fixed512":   ("ViDoRe markdown", "fixed 512/128", "chunking on flat text (near no-op: 11 pages split)"),
    "chandra_page":      ("chandra2", "none", "page-reassembled parse; the parser variable"),
    "chandra_fixed512":  ("chandra2", "fixed 512/128", "chunking on chandra2 (near no-op: 36 pages split)"),
    "chandra_blocks":    ("chandra2", "blocks 1200/200", "structured chunking; only possible on chandra2"),
    "chandra_no_figure": ("chandra2", "none", "drop VLM image descriptions (45.7% of chars)"),
    "chandra_no_boiler": ("chandra2", "none", "drop page headers/footers (37% of blocks)"),
    "chandra_prose":     ("chandra2", "none", "extracted prose only; minimal representation"),
}
PAPER_E2E = {"oracle": ("Oracle/Text, Gemini 3 Pro", 71.2),
             "retrieved_vidore": ("Jina-v4 text. + zerank-2, Gemini 3 Pro", 69.2),
             "retrieved_chandra2": ("ColEmbed-3B-v2 Text, Gemini 3 Pro", 64.9)}


def permutation(a, b):
    qids = sorted(set(a) & set(b))
    diffs = [b[q] - a[q] for q in qids]
    observed = sum(diffs) / len(diffs)
    rng = random.Random(0)
    extreme = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)) >= abs(observed)
                  for _ in range(RESAMPLES))
    return (observed, (extreme + 1) / (RESAMPLES + 1),
            sum(1 for d in diffs if d > 1e-9), sum(1 for d in diffs if d < -1e-9))


# -- 1. retrieval ladder -------------------------------------------------------
ladder = json.loads((RES / "physics_retrieval_ladder.json").read_text())
by_key = {(r["index"], r["arm"]): r for r in ladder}
rows = []
for r in ladder:
    corpus, chunking, purpose = META[r["index"]]
    rows.append({"index": r["index"], "corpus": corpus, "chunking": chunking, "retriever": r["arm"],
                 "ndcg@10": r["ndcg@10"], "recall@10": r["recall@10"], "units": r["units"],
                 "items_indexed": r["items"], "corpus_chars": r["chars"], "n_queries": 302,
                 "purpose": purpose})
pd.DataFrame(rows).to_csv(CSV / "1_retrieval_ladder.csv", index=False)

# -- 2. retrieval ablations ----------------------------------------------------
CONTRASTS = [("chunking, ViDoRe text", "vidore_page", "vidore_fixed512"),
             ("chunking flat, chandra2", "chandra_page", "chandra_fixed512"),
             ("chunking blocks, chandra2", "chandra_page", "chandra_blocks"),
             ("+ image descriptions", "chandra_no_figure", "chandra_page"),
             ("+ page headers/footers", "chandra_no_boiler", "chandra_page"),
             ("simplest -> richest chandra2", "chandra_prose", "chandra_page"),
             ("parser: ViDoRe -> chandra2", "vidore_page", "chandra_page")]
rows = []
for label, base, feature in CONTRASTS:
    for arm in ["bm25", "dense", "rrf", "alpha0.7"]:
        left, right = by_key[(base, arm)], by_key[(feature, arm)]
        delta, p, better, worse = permutation(left["per_question"], right["per_question"])
        rows.append({"contrast": label, "baseline": base, "treatment": feature, "retriever": arm,
                     "baseline_ndcg@10": left["ndcg@10"], "treatment_ndcg@10": right["ndcg@10"],
                     "delta_ndcg@10": round(100 * delta, 3), "p_value": round(p, 4),
                     "significant_at_0.05": p < 0.05, "queries_better": better,
                     "queries_worse": worse, "queries_tied": 302 - better - worse,
                     "n_queries": 302, "resamples": RESAMPLES})
pd.DataFrame(rows).to_csv(CSV / "2_retrieval_ablations.csv", index=False)

# -- 3. end-to-end summary -----------------------------------------------------
e2e = {name: json.loads((RES / f"physics_e2e_{name}.json").read_text()) for name in PAPER_E2E}
summary = {r["arm"]: r for r in json.loads((RES / "physics_e2e_summary.json").read_text())}
rows = []
for name, records in e2e.items():
    s, (comparator, published) = summary[name], PAPER_E2E[name]
    labels = pd.Series([r["judgment"] for r in records]).value_counts()
    rows.append({"arm": name, "context_source": comparator.split(",")[0] if name == "oracle" else name,
                 "n_queries": s["n"], "ndcg@10": round(s["ndcg@10"], 2) if s["ndcg@10"] else None,
                 "correct_only_pct": s["correct_only"], "correct_plus_partial_pct": s["correct_plus_partial"],
                 "n_correct": int(labels.get("Correct", 0)),
                 "n_partially_correct": int(labels.get("Partially Correct", 0)),
                 "n_incorrect": int(labels.get("Incorrect", 0)),
                 "mean_context_pages": s["ctx_pages"], "mean_context_chars": s["ctx_chars"],
                 "mean_gold_pages_in_context": s["gold_in_ctx"], "mean_gold_pages_total": 7.21,
                 "queries_with_zero_gold": sum(1 for r in records if r["gold_hit"] == 0),
                 "paper_comparator": comparator, "paper_score": published,
                 "generator": "openai/gpt-4o-mini", "judge": "openai/gpt-4o"})
pd.DataFrame(rows).to_csv(CSV / "3_e2e_summary.csv", index=False)

# -- 4. end-to-end contrasts ---------------------------------------------------
rows = []
for mode, label in [("correct_only", False), ("correct_plus_partial", True)]:
    binary = {n: {r["qid"]: float(r["judgment"] == "Correct" or
                                 (label and r["judgment"] == "Partially Correct"))
                  for r in recs} for n, recs in e2e.items()}
    for base, feature in [("oracle", "retrieved_vidore"), ("oracle", "retrieved_chandra2"),
                          ("retrieved_vidore", "retrieved_chandra2")]:
        delta, p, better, worse = permutation(binary[base], binary[feature])
        rows.append({"metric": mode, "baseline": base, "treatment": feature,
                     "baseline_pct": round(100 * sum(binary[base].values()) / 302, 2),
                     "treatment_pct": round(100 * sum(binary[feature].values()) / 302, 2),
                     "delta_pp": round(100 * delta, 2), "p_value": round(p, 4),
                     "significant_at_0.05": p < 0.05, "queries_better": better,
                     "queries_worse": worse, "queries_tied": 302 - better - worse})
pd.DataFrame(rows).to_csv(CSV / "4_e2e_contrasts.csv", index=False)

# -- 5. per-query end-to-end ---------------------------------------------------
rows = []
for name, records in e2e.items():
    for r in records:
        retrieved = r.get("retrieved") or []
        rows.append({"arm": name, "qid": r["qid"], "query": r["query"],
                     "gold_answer": r["gold_answer"], "model_answer": r["answer"],
                     "judgment": r["judgment"], "error": r["error"],
                     "n_context_pages": r["n_context_pages"], "context_chars": r["context_chars"],
                     "gold_pages_in_context": r["gold_hit"],
                     "retrieved_unit_ids": " | ".join(h["unit_id"] for h in retrieved),
                     "retrieved_is_gold": " | ".join(str(int(h["is_gold"])) for h in retrieved),
                     "retrieved_scores": " | ".join(f"{h['score']:.4f}" for h in retrieved)})
pd.DataFrame(rows).to_csv(CSV / "5_e2e_per_query.csv", index=False)

# -- 6. English Oracle generation + judge stability -----------------------------
second = json.loads((RES / "english_oracle_generation" / "stability.json").read_text())
rows = []
for path in sorted((RES / "english_oracle_generation").glob("*.json")):
    if path.stem == "stability":
        continue
    for r in json.loads(path.read_text()):
        rows.append({"subset": r["subset"], "qid": r["qid"], "query": r["query"],
                     "gold_answer": r["gold_answer"], "model_answer": r["answer"],
                     "judgment_pass1": r["judgment"], "judgment_pass2": second.get(r["qid"]),
                     "passes_agree": second.get(r["qid"]) == r["judgment"],
                     "gold_pages": r["gold_pages"], "context_chars": r["context_chars"],
                     "error": r["error"]})
english = pd.DataFrame(rows)
english.to_csv(CSV / "6_english_oracle_per_query.csv", index=False)

agg = english.groupby("subset").apply(lambda g: pd.Series({
    "n": len(g),
    "correct_only_pct": round(100 * (g.judgment_pass1 == "Correct").mean(), 1),
    "correct_plus_partial_pct": round(100 * g.judgment_pass1.isin(["Correct", "Partially Correct"]).mean(), 1),
    "pass2_correct_only_pct": round(100 * (g.judgment_pass2 == "Correct").mean(), 1),
    "judge_agreement_pct": round(100 * g.passes_agree.mean(), 1),
    "mean_context_chars": int(g.context_chars.mean()),
    "mean_gold_pages": round(g.gold_pages.mean(), 2),
}), include_groups=False).reset_index()
agg["paper_oracle_text_global"] = 70.6
agg.to_csv(CSV / "7_english_oracle_summary.csv", index=False)

for path in sorted(CSV.glob("*.csv")):
    frame = pd.read_csv(path)
    print(f"{path.name:36s} {len(frame):6d} rows x {len(frame.columns):2d} cols")
