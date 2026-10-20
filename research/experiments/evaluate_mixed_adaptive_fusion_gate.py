"""OOF query-adaptive BM25/dense fusion gate on cached mixed text."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "mixed_adaptive_fusion_gate"
ALPHAS = (0.00, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)


def metrics(qids, rankings, qrels, top_k=12):
    values, r10, r20 = [], [], []
    for qid in qids:
        gold = qrels[qid]
        pages = [pid for pid, _score in rankings[qid][:top_k]]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
    return {"recall@12": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20))}


def main() -> int:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [p for p in pages_all if p.text_source in {"pdf_inspector", "tesseract"} and p.text.strip()]
    queries = read_jsonl(DATASET / "queries.jsonl")
    qids = [str(q["query_id"]) for q in queries]
    qrels = defaultdict(set)
    for row in read_jsonl(CACHE / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    index = BM25Index(analyzer_name="auto").build({"chunk_id": p.page_id, "doc_id": p.doc_id, "text": p.text} for p in pages)
    all_ids = [p.page_id for p in pages_all]
    pos = {pid: i for i, pid in enumerate(all_ids)}
    emb = CACHE / "embedding_adaptive_openrouter_te3s_fulltext/embeddings"
    page_vecs = np.load(emb / "page_embeddings.npy", mmap_mode="r")[[pos[p.page_id] for p in pages]]
    query_vecs = np.load(emb / "query_embeddings.npy", mmap_mode="r")
    page_ids = [p.page_id for p in pages]
    page_norms = np.linalg.norm(page_vecs, axis=1)
    rankings = {alpha: {} for alpha in ALPHAS}
    features = []
    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        bm_hits = index.search(str(query["query"]), 100)
        bm = [(index.chunk_ids[p], float(s)) for p, s in bm_hits]
        bm_ids = [pid for pid, _s in bm]
        bm_max = max((s for _pid, s in bm), default=1.0)
        bm_norm = {pid: s / bm_max for pid, s in bm}
        qv = query_vecs[qi]
        ds = (page_vecs @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense = sorted(zip(page_ids, ds), key=lambda x: (-float(x[1]), x[0]))[:100]
        dense_ids = [pid for pid, _s in dense]
        dense_by_id = {pid: float(s) for pid, s in dense}
        dense_max = max(dense_by_id.values(), default=1.0)
        dense_norm = {pid: s / dense_max for pid, s in dense_by_id.items()}
        for alpha in ALPHAS:
            scores = {pid: alpha * bm_norm.get(pid, 0.0) + (1.0 - alpha) * dense_norm.get(pid, 0.0) for pid in set(bm_norm) | set(dense_norm)}
            rankings[alpha][qid] = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:100]
        bm_top_scores = np.asarray([s for _pid, s in bm[:20]], dtype=float)
        dense_top_scores = np.asarray([float(s) for _pid, s in dense[:20]], dtype=float)
        hybrid_scores = np.asarray([s for _pid, s in rankings[0.10][qid][:20]], dtype=float)
        probs = hybrid_scores / (hybrid_scores.sum() + 1e-9)
        features.append([len(str(query["query"]).split()), len(str(query["query"])), bm_top_scores[0] if len(bm_top_scores) else 0.0, (bm_top_scores[0] - bm_top_scores[1]) if len(bm_top_scores) > 1 else 0.0, dense_top_scores[0] if len(dense_top_scores) else 0.0, (dense_top_scores[0] - dense_top_scores[1]) if len(dense_top_scores) > 1 else 0.0, len(set(bm_ids[:10]) & set(dense_ids[:10])) / 10.0, float(-np.sum(probs * np.log(probs + 1e-9)))])
    features = np.asarray(features, dtype=float)
    targets = np.asarray([[len(set(pid for pid, _s in rankings[alpha][qid][:12]) & qrels[qid]) / max(len(qrels[qid]), 1) for alpha in ALPHAS] for qid in qids], dtype=float)
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    groups = np.asarray([str(q["source"]) for q in queries])
    oof_alpha = np.zeros(len(qids), dtype=float)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=2026).split(features, groups), start=1):
        scaler = StandardScaler().fit(features[train_idx])
        train_x, test_x = scaler.transform(features[train_idx]), scaler.transform(features[test_idx])
        models = []
        for col in range(len(ALPHAS)):
            model = ExtraTreesRegressor(n_estimators=250, min_samples_leaf=6, max_features=0.8, random_state=900 + col)
            model.fit(train_x, targets[train_idx, col])
            models.append(model)
        pred = np.column_stack([model.predict(test_x) for model in models])
        choice = np.argmax(pred, axis=1)
        for idx, selected in zip(test_idx, choice):
            oof_alpha[idx] = ALPHAS[int(selected)]
        test_qids = [qids[i] for i in test_idx]
        selected_rankings = {qid: rankings[float(oof_alpha[i])][qid] for i, qid in zip(test_idx, test_qids)}
        fold_rows.append({"fold": fold, "metrics": metrics(test_qids, selected_rankings, qrels), "alpha_histogram": {str(alpha): int(np.sum(oof_alpha[test_idx] == alpha)) for alpha in ALPHAS}})
    oof_rankings = {qid: rankings[float(alpha)][qid] for qid, alpha in zip(qids, oof_alpha)}
    fixed_metrics = {str(alpha): metrics(qids, rankings[alpha], qrels) for alpha in ALPHAS}
    report = {"indexed_pages": len(pages), "queries": len(qids), "features": int(features.shape[1]), "qrels_used_for_test_selection": False, "fixed": fixed_metrics, "oof_adaptive": metrics(qids, oof_rankings, qrels), "folds": fold_rows, "notes": ["Alpha is predicted from query/score features; test-fold qrels are not used for the gate."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mixed adaptive fusion gate", "", "| Method | Recall@12 | Recall@10 | Recall@20 |", "|---|---:|---:|---:|"]
    for name, value in [(f"fixed alpha={alpha:.2f}", fixed_metrics[str(alpha)]) for alpha in ALPHAS] + [("OOF adaptive alpha", report["oof_adaptive"])]:
        lines.append(f"| {name} | {value['recall@12']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
