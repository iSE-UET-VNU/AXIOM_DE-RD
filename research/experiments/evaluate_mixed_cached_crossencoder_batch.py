"""Batched local cross-encoder reranking over mixed-text BM25+dense candidates."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import argparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.raw_hierarchical import RawPage, read_jsonl  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


CACHE = ROOT / "data/work/benchmark_hierarchical_local"
DATASET = ROOT / "data/raw/BENCHMARK"
OUTPUT = CACHE / "mixed_cached_crossencoder"
MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def metric(qids, rankings, qrels, choices=None):
    values, r10, r20, ks = [], [], [], []
    for qid in qids:
        gold = qrels[qid]
        k = int(choices[qid]) if choices is not None else 12
        pages = [pid for pid, _score in rankings[qid][:k]]
        denom = max(len(gold), 1)
        values.append(len(set(pages) & gold) / denom)
        r10.append(len(set(pages[:10]) & gold) / denom)
        r20.append(len(set(pages[:20]) & gold) / denom)
        ks.append(len(pages))
    return {"recall_at_returned_k": float(np.mean(values)), "recall@10": float(np.mean(r10)), "recall@20": float(np.mean(r20)), "mean_k": float(np.mean(ks))}


def calibrate(qids, rankings, choices, target=12):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-depth", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
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
    candidate_by_qid = {}
    pairs = []
    for qi, query in enumerate(queries):
        qid = str(query["query_id"])
        text = str(query["query"])
        bm = [index.chunk_ids[p] for p, _s in index.search(text, args.candidate_depth)]
        qv = query_vecs[qi]
        dense_scores = (page_vecs @ qv) / (page_norms * np.linalg.norm(qv) + 1e-9)
        dense = [page_ids[p] for p in np.argsort(-dense_scores, kind="stable")[:args.candidate_depth]]
        candidate_ids = list(dict.fromkeys(bm + dense))
        candidate_by_qid[qid] = candidate_ids
        pairs.extend((text, page_by_id[pid].text[:6000]) for pid in candidate_ids)
    print(f"scoring {len(pairs)} query-page pairs with {MODEL_NAME}", flush=True)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_NAME, device="cpu", max_length=384)
    scores = np.asarray(model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True), dtype=float)
    rankings = {}
    cursor = 0
    for qid in qids:
        candidate_ids = candidate_by_qid[qid]
        local_scores = scores[cursor:cursor + len(candidate_ids)]
        cursor += len(candidate_ids)
        order = np.argsort(-local_scores, kind="stable")
        rankings[qid] = [(candidate_ids[int(i)], float(local_scores[int(i)])) for i in order]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "rankings.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for qid in qids:
            handle.write(json.dumps({"query_id": qid, "ranking": rankings[qid]}, ensure_ascii=False, separators=(",", ":")) + "\n")
    fixed = metric(qids, rankings, qrels)
    raw_choices = {}
    for qid in qids:
        values = [score for _pid, score in rankings[qid]]
        # A moderate offset is used only to expose a variable score-gap budget;
        # exact total budget is enforced below without qrels.
        diffs = np.abs(np.diff(np.asarray(values, dtype=float)))
        cut = int(np.argmax(diffs)) + 1 if len(diffs) else 12
        raw_choices[qid] = max(1, min(len(values), cut + 6))
    choices = calibrate(qids, rankings, raw_choices)
    adaptive = metric(qids, rankings, qrels, choices)
    report = {"model": MODEL_NAME, "indexed_pages": len(pages), "candidate_depth_per_stream": args.candidate_depth, "batch_size": args.batch_size, "pairs": len(pairs), "qrels_used_for_ranking": False, "fixed_k12": fixed, "adaptive_exact_mean12": adaptive, "mean_raw_k": float(np.mean(list(raw_choices.values()))), "notes": ["All cross-encoder pairs are scored in one batched CPU inference call.", "Adaptive choices use score gaps only and are calibrated to total K=12*220."]}
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Mixed cached cross-encoder", "", "| Method | Recall at K | Recall@10 | Recall@20 | Mean K |", "|---|---:|---:|---:|---:|"]
    for name, value in (("Cross-encoder fixed K=12", fixed), ("Cross-encoder + Adaptive-k exact mean12", adaptive)):
        lines.append(f"| {name} | {value['recall_at_returned_k']:.2%} | {value['recall@10']:.2%} | {value['recall@20']:.2%} | {value['mean_k']:.2f} |")
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
