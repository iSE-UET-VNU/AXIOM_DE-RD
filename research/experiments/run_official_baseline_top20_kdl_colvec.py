"""Rerank the official light-BM25 top-20 pages with cached KDL chunks and ColVec.

Protocol is deliberately fixed: the official merged baseline's ``returned_pages``
are the only candidate pages.  KDL chunk BM25+dense (alpha=.7, depth=100) is
max-pooled to page, then optionally min-max fused with cached ColVec (.8).
All expensive assets are reused; no parser, encoder, model server, or API runs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.retrieval import alpha_fuse
from src.retrieval.sparse import BM25Index

BUNDLE = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2/ef7f11811803"
BASELINE = ROOT / "data/work/ohr_replacement_20260920/retrieval/merged_220_all_arms/runs/baseline_bm25.jsonl"
OLD_QRELS = ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20/bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
OHR_QRELS = ROOT / "data/work/ohr_replacement_20260920/inputs/qrels.jsonl"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source(qid: str) -> str:
    return qid.split("::", 1)[0]


def minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    return {key: 1.0 if high - low <= 1e-12 else (value - low) / (high - low) for key, value in values.items()}


def metrics(run: dict[str, dict[str, float]], gold: dict[str, dict[str, int]], qids: list[str], depth: int) -> dict[str, float]:
    results = []
    for qid in qids:
        ranking = sorted(run[qid], key=lambda page: (-run[qid][page], page))[:depth]
        positives = {page for page, grade in gold[qid].items() if grade > 0}
        dcg = sum((2 ** gold[qid].get(page, 0) - 1) / math.log2(rank + 1) for rank, page in enumerate(ranking, 1))
        ideal = sorted(gold[qid].values(), reverse=True)[:depth]
        idcg = sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
        results.append((len(set(ranking) & positives) / len(positives) if positives else 0.0, float(bool(set(ranking) & positives)), dcg / idcg if idcg else 0.0))
    return {f"recall@{depth}": 100 * float(np.mean([item[0] for item in results])), f"hit@{depth}": 100 * float(np.mean([item[1] for item in results])), f"ndcg@{depth}": 100 * float(np.mean([item[2] for item in results]))}


def report_by_source(run: dict[str, dict[str, float]], gold: dict[str, dict[str, int]], qids: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in ("all", "vidore_physics", "vidore_industrial", "ohrbench"):
        members = qids if group == "all" else [qid for qid in qids if source(qid) == group]
        result[group] = {"queries": len(members), **metrics(run, gold, members, 10), **metrics(run, gold, members, 20)}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/work/ohr_replacement_20260920/retrieval/baseline_top20_kdl_colvec")
    ap.add_argument("--candidate-run", type=Path, default=BASELINE, help="Cached light-retrieval JSONL with returned_pages/scores.")
    ap.add_argument("--candidate-label", default="official_baseline_bm25")
    ap.add_argument("--physics-dense-only", action="store_true", help="Use dense-only KDL scoring for French Physics pages; retain hybrid elsewhere.")
    ap.add_argument("--cohere-cache", type=Path, help="Use the first --candidate-k pages from cached Cohere reranking instead of --candidate-run.")
    ap.add_argument("--cohere-cache-order", choices=("reranked", "candidate"), default="reranked", help="When reading --cohere-cache, use Cohere's order or the original light-hybrid candidate order.")
    ap.add_argument("--candidate-k", type=int, default=20, help="Maximum pages/query when --cohere-cache is used.")
    ap.add_argument("--skip-colvec", action="store_true", help="Evaluate only the light candidate rank and KDL-text rank.")
    ap.add_argument("--dense-alpha", type=float, default=0.7, help="Dense weight for KDL alpha fusion; BM25 weight is 1-alpha.")
    ap.add_argument("--bm25-only", action="store_true", help="Use KDL BM25 only (still retrieve the top 100 chunks then max-pool to pages).")
    args = ap.parse_args()
    if not 0.0 <= args.dense_alpha <= 1.0:
        raise ValueError("--dense-alpha must be in [0, 1]")
    started = time.perf_counter()
    if args.cohere_cache:
        if args.candidate_k < 1:
            raise ValueError("--candidate-k must be positive")
        query_text = {str(row["query_id"]): str(row["query"]) for row in rows(BASELINE)}
        page_field = "reranked_pages" if args.cohere_cache_order == "reranked" else "candidate_pages"
        score_field = "rerank_scores" if args.cohere_cache_order == "reranked" else "candidate_scores"
        baseline_rows = {}
        for row in rows(args.cohere_cache):
            pages = row[page_field][:args.candidate_k]
            # The cache retains original hybrid page order but not its raw
            # scores.  Positional scores are sufficient here: KDL never uses
            # them, and they preserve that light ranking for reporting.
            scores = row.get(score_field)
            if scores is None:
                scores = [float(len(pages) - index) for index in range(len(pages))]
            baseline_rows[str(row["query_id"])] = {
                "query_id": row["query_id"], "query": query_text[str(row["query_id"])],
                "returned_pages": pages,
                "returned_scores": scores[:args.candidate_k],
            }
    else:
        baseline_rows = {str(row["query_id"]): row for row in rows(args.candidate_run)}
    gold: dict[str, dict[str, int]] = defaultdict(dict)
    # Match the official merged evaluator: binary Physics/Industrial page
    # qrels from the old bundle plus explicit OHR page qrels.
    for row in rows(OLD_QRELS):
        qid = str(row["query_id"])
        if qid.startswith(("vidore_physics::", "vidore_industrial::")):
            gold[qid][str(row["page_id"])] = 1
    for row in rows(OHR_QRELS):
        gold[str(row["query_id"])][f"{row['doc_id']}#page={int(row['page_index'])}"] = 1
    qids = sorted(set(baseline_rows) & set(gold))
    if len(qids) != 220:
        raise RuntimeError(f"Expected 220 shared qids, got {len(qids)}")
    candidate = {qid: list(baseline_rows[qid]["returned_pages"]) for qid in qids}
    if any(not value or len(value) > args.candidate_k for value in candidate.values()):
        raise RuntimeError("Candidate run violates its requested page budget")

    emb = BUNDLE / "embeddings/kdl"
    owners = np.asarray(json.loads((emb / "chunk_owners.json").read_text(encoding="utf-8")))
    texts = json.loads((emb / "chunk_texts.json").read_text(encoding="utf-8"))
    vectors = np.load(emb / "chunk_vectors.npy").astype(np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    query_ids = json.loads((emb / "query_ids.json").read_text(encoding="utf-8"))
    query_vectors = np.load(emb / "query_vectors.npy").astype(np.float32)
    query_vectors /= np.clip(np.linalg.norm(query_vectors, axis=1, keepdims=True), 1e-12, None)
    qvec = {qid: query_vectors[index] for index, qid in enumerate(query_ids)}
    if set(qids) != set(qvec):
        raise RuntimeError("KDL query-vector IDs do not match official baseline qids")

    union = set(page for pages in candidate.values() for page in pages)
    keep = np.asarray([index for index, page in enumerate(owners) if page in union], dtype=np.int32)
    local_owners, local_vectors = owners[keep], vectors[keep]
    bm25 = BM25Index(analyzer_name="plain").build([
        {"chunk_id": f"c{index}", "doc_id": str(local_owners[index]), "text": texts[int(original)]}
        for index, original in enumerate(keep)
    ])
    positions_by_page: dict[str, list[int]] = defaultdict(list)
    for index, page in enumerate(local_owners):
        positions_by_page[str(page)].append(index)

    colvec = np.load(BUNDLE / "colvec/colvec_scores.npy", mmap_mode="r")
    colvec_qids = json.loads((BUNDLE / "colvec/colvec_qids.json").read_text(encoding="utf-8"))
    colvec_pages = json.loads((BUNDLE / "colvec/colvec_keys.json").read_text(encoding="utf-8"))
    colrow, colcol = {qid: index for index, qid in enumerate(colvec_qids)}, {page: index for index, page in enumerate(colvec_pages)}

    baseline_scores = {qid: {page: float(score) for page, score in zip(candidate[qid], baseline_rows[qid]["returned_scores"])} for qid in qids}
    kdl_text: dict[str, dict[str, float]] = {}
    kdl_colvec: dict[str, dict[str, float]] = {}
    details = []
    for qid in qids:
        allowed_pages = [page for page in candidate[qid] if page in positions_by_page]
        allowed = {position for page in allowed_pages for position in positions_by_page[page]}
        lexical = [] if args.physics_dense_only and source(qid) == "vidore_physics" else bm25.search(str(baseline_rows[qid]["query"]), 100, allowed)
        dense_positions = np.asarray(sorted(allowed), dtype=np.int32)
        dense_scores = local_vectors[dense_positions] @ qvec[qid]
        chosen = dense_positions[np.argsort(-dense_scores)[:min(100, len(dense_positions))]]
        dense = [(int(position), float(local_vectors[position] @ qvec[qid])) for position in chosen]
        fused_chunks = lexical if args.bm25_only else alpha_fuse(lexical, dense, alpha=args.dense_alpha, top_k=100)
        page_scores: dict[str, float] = {}
        for position, score in fused_chunks:
            page = str(local_owners[position])
            page_scores[page] = max(page_scores.get(page, float("-inf")), float(score))
        # Preserve every baseline candidate even when KDL produced no text;
        # they receive a zero textual score and remain eligible for ColVec.
        page_scores = {page: page_scores.get(page, 0.0) for page in candidate[qid]}
        visual = {page: float(colvec[colrow[qid], colcol[page]]) for page in candidate[qid]}
        text_norm, visual_norm = minmax(page_scores), minmax(visual)
        kdl_text[qid] = page_scores
        kdl_colvec[qid] = {page: 0.2 * text_norm[page] + 0.8 * visual_norm[page] for page in candidate[qid]}
        details.append({"query_id": qid, "source": source(qid), "candidate_pages": candidate[qid], "baseline_scores": baseline_scores[qid], "kdl_text_scores": page_scores, "kdl_colvec_scores": kdl_colvec[qid]})

    elapsed = time.perf_counter() - started
    runs = {"official_baseline_bm25": baseline_scores, "kdl_chunk_hybrid": kdl_text, "kdl_chunk_hybrid_plus_colvec": kdl_colvec}
    text_protocol = (
        "KDL chunks 512/128, BM25-only, depth=100, max-pool page"
        if args.bm25_only else
        f"KDL chunks 512/128, BM25 + text-embedding-3-small dense_alpha={args.dense_alpha:.2f}, depth=100, max-pool page"
    )
    if args.physics_dense_only:
        text_protocol += "; Physics uses dense-only (cross-language ablation)"
    candidate_source = f"{args.cohere_cache} ({args.cohere_cache_order} order)" if args.cohere_cache else str(args.candidate_run)
    runs = {args.candidate_label: baseline_scores, "kdl_chunk_hybrid": kdl_text}
    if not args.skip_colvec:
        runs["kdl_chunk_hybrid_plus_colvec"] = kdl_colvec
    summary = {"protocol": {"candidate_source": candidate_source, "candidate_pages_per_query": f"up to {args.candidate_k}", "accurate_text": text_protocol, "accurate_visual": None if args.skip_colvec else "minmax KDL-text .2 + cached ColVec .8", "cache_only": True}, "cache": {"kdl_chunks": len(owners), "kdl_chunk_vectors_shape": list(vectors.shape), "colvec_scores_shape": list(colvec.shape), "candidate_union_pages": len(union), "candidate_union_chunks": len(keep), "candidate_pages_min": min(map(len, candidate.values())), "candidate_pages_mean": float(np.mean([len(value) for value in candidate.values()])), "candidate_pages_max": max(map(len, candidate.values()))}, "replay_timing_seconds": elapsed, "metrics": {name: report_by_source(run, gold, qids) for name, run in runs.items()}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_query.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    (args.output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Official baseline top-20 → cached KDL / ColVec", "", "All runs have the identical 20-page official BM25 candidate set; Recall@20 is therefore the fixed candidate ceiling.", "", "| Arm | All Recall@10 | All nDCG@10 | All Recall@20 |", "|---|---:|---:|---:|"]
    for name, result in summary["metrics"].items():
        lines.append(f"| {name} | {result['all']['recall@10']:.2f}% | {result['all']['ndcg@10']:.2f} | {result['all']['recall@20']:.2f}% |")
    lines.extend(["", f"Offline cache replay: {elapsed:.3f}s. This excludes preparation, KDL parsing, TE3 embedding, and ColVec inference."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
