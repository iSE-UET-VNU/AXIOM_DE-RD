"""OOF lightweight page reranker over BM25+dense candidates on mixed text."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from research.experiments.run_benchmark_adaptive_k import adaptive_k  # noqa: E402
from src.chunking_embedding.lexical import analyze  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "mixed_lightweight_reranker"


def metrics(qids, rankings, qrels, choices=None):
    values, r10, r20, ks = [], [], [], []
    for qid in qids:
        gold = qrels[qid]
        ranking = rankings[qid]
        k = int(choices[qid]) if choices is not None else 12
        pages = [pid for pid, _score in ranking[:k]]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(ks))}


def exact_budget(qids, rankings, choices, target=12):
    choices = dict(choices)
    total = sum(choices.values())
    while total > target * len(qids):
        available = [qid for qid in qids if choices[qid] > 1]
        qid = min(available, key=lambda q: (rankings[q][choices[q] - 1][1], q))
        choices[qid] -= 1
        total -= 1
    while total < target * len(qids):
        available = [qid for qid in qids if choices[qid] < len(rankings[qid])]
        qid = max(available, key=lambda q: (rankings[q][choices[q]][1], q))
        choices[qid] += 1
        total += 1
    return choices


def main() -> int:
    pages_all = [RawPage(**row) for row in read_jsonl(CACHE / "parsing/pages.jsonl")]
    pages = [p for p in pages_all if p.text_source in {"pdf_inspector", "tesseract"} and p.text.strip()]
    page_by_id = {p.page_id: p for p in pages}
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
    candidate_features: dict[str, np.ndarray] = {}
    candidate_ids: dict[str, list[str]] = {}
    base_rankings: dict[str, list[tuple[str, float]]] = {}
    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        query_tokens = set(analyze(str(query["query"])))
        bm_hits = index.search(str(query["query"]), 100)
        bm = [(index.chunk_ids[p], float(s)) for p, s in bm_hits]
        bm_rank = {pid: rank for rank, (pid, _s) in enumerate(bm, start=1)}
        bm_max = max((s for _pid, s in bm), default=1.0)
        bm_score = {pid: s / bm_max for pid, s in bm}
        qv = query_vecs[qi]
        ds = (page_vecs @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense = sorted(zip(page_ids, ds), key=lambda x: (-float(x[1]), x[0]))[:100]
        dense_rank = {pid: rank for rank, (pid, _s) in enumerate(dense, start=1)}
        dense_score_raw = {pid: float(s) for pid, s in dense}
        dense_max = max(dense_score_raw.values(), default=1.0)
        dense_score = {pid: s / dense_max for pid, s in dense_score_raw.items()}
        union = list(dict.fromkeys([pid for pid, _s in bm] + [pid for pid, _s in dense]))
        file_raw: dict[str, list[float]] = defaultdict(list)
        for pid in union:
            file_raw[pid.split("#page=", 1)[0]].append(0.10 * bm_score.get(pid, 0.0) + 0.90 * dense_score.get(pid, 0.0))
        file_score = {fid: sum(sorted(vals, reverse=True)[:2]) for fid, vals in file_raw.items()}
        file_max = max(file_score.values(), default=1.0)
        file_score = {fid: val / file_max for fid, val in file_score.items()}
        rows = []
        base_scores = {}
        for pid in union:
            tokens = set(analyze(page_by_id[pid].text[:12000]))
            lexical_coverage = len(tokens & query_tokens) / max(len(query_tokens), 1)
            bscore = bm_score.get(pid, 0.0)
            dscore = dense_score.get(pid, 0.0)
            hybrid = 0.10 * bscore + 0.90 * dscore
            base_scores[pid] = hybrid
            rows.append([bscore, dscore, float(bm_rank.get(pid, 0)), float(dense_rank.get(pid, 0)), float(bm_rank.get(pid, 0) > 0), float(dense_rank.get(pid, 0) > 0), float(lexical_coverage), float(file_score.get(pid.split("#page=", 1)[0], 0.0)), float(len(analyze(page_by_id[pid].text))), float(len(tokens))])
        candidate_ids[qid] = union
        candidate_features[qid] = np.asarray(rows, dtype=np.float32)
        base_rankings[qid] = sorted(base_scores.items(), key=lambda item: (-item[1], item[0]))

    source_by_qid = {qid: qid.split("::", 1)[0] for qid in qids}
    fold_by_qid = {}
    for source in sorted(set(source_by_qid.values())):
        source_qids = sorted(qid for qid in qids if source_by_qid[qid] == source)
        for i, qid in enumerate(source_qids):
            fold_by_qid[qid] = i % 5
    from sklearn.ensemble import ExtraTreesClassifier
    reranked = {}
    fold_rows = []
    for fold in range(5):
        train_qids = [qid for qid in qids if fold_by_qid[qid] != fold]
        test_qids = [qid for qid in qids if fold_by_qid[qid] == fold]
        train_x = np.concatenate([candidate_features[qid] for qid in train_qids], axis=0)
        train_y = np.concatenate([np.asarray([int(pid in qrels[qid]) for pid in candidate_ids[qid]], dtype=np.int64) for qid in train_qids])
        model = ExtraTreesClassifier(n_estimators=250, min_samples_leaf=3, max_features=0.8, class_weight="balanced", random_state=2026 + fold, n_jobs=-1)
        model.fit(train_x, train_y)
        for qid in test_qids:
            scores = model.predict_proba(candidate_features[qid])[:, 1]
            ranking = sorted(zip(candidate_ids[qid], scores), key=lambda item: (-float(item[1]), item[0]))
            reranked[qid] = [(pid, float(score)) for pid, score in ranking]
        fold_rows.append({"fold": fold + 1, "train_queries": len(train_qids), "test_queries": len(test_qids)})
    fixed_base = metrics(qids, base_rankings, qrels)
    fixed_rerank = metrics(qids, reranked, qrels)
    choices = {}
    for qid in qids:
        decision = adaptive_k([score for _pid, score in reranked[qid]], ignore_extreme=0, ignore_extreme_tail=0.1, retrieve_more=6)
        choices[qid] = int(decision["k"])
    choices = exact_budget(qids, reranked, choices)
    adaptive_metrics = metrics(qids, reranked, qrels, choices)
    report = {"indexed_pages": len(pages), "queries": len(qids), "candidate_union": "BM25 top100 + dense top100", "qrels_used_for_test_ranking": False, "fixed_base_hybrid": fixed_base, "fixed_oof_reranker": fixed_rerank, "adaptive_oof_reranker": adaptive_metrics, "folds": fold_rows, "notes": ["The page classifier is trained on qrels from other source-stratified folds only.", "Adaptive K is selected from reranker scores and calibrated to exactly mean K=12."]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mixed lightweight reranker", "", "| Method | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in (("Base hybrid fixed K=12", fixed_base), ("OOF learned reranker fixed K=12", fixed_rerank), ("OOF learned reranker + Adaptive-k", adaptive_metrics)):
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
