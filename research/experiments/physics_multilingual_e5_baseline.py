"""Multilingual E5-small page-segment baseline for Physics.

This is a controlled semantic-text experiment after the cache-only skipping
screen.  PDF-inspector page text is split into fixed word windows, encoded by
``intfloat/multilingual-e5-small`` using the model's standard query/passage
prefixes, and max-pooled back to page scores.  The model is downloaded from
Hugging Face by the local environment on first use; no qrels are used during
encoding.  Fusion weights and RRF constants are fixed before evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_adaptive_evidence_fusion import _load_page_texts, _load_run  # noqa: E402
from physics_hierarchical_retrieval import _derived_metrics, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402

DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_multilingual_e5_baseline"
RESULT_ROOT = DEFAULT_ROOT / "results"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
BM25_PATH = RESULT_ROOT / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
C014_PATH = RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl"
CROSS_KF_PATH = RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl"


def _load_pages() -> tuple[list[str], list[str]]:
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    by_page = _load_page_texts(PARSED_RUN)
    texts = [by_page.get(page, "") for page in pages]
    if len(pages) != 1674 or any(not text.strip() for text in texts):
        raise RuntimeError("Expected non-empty PDF-inspector text for all pages")
    return pages, texts


def _segments(text: str, window: int = 240, overlap: int = 40) -> list[str]:
    words = str(text).split()
    if len(words) <= window:
        return [str(text)]
    step = max(1, window - overlap)
    result = []
    for start in range(0, len(words), step):
        result.append(" ".join(words[start:start + window]))
        if start + window >= len(words):
            break
    return result


def _load_run_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    return _load_run(path)


def _cached_scores(run: Mapping[str, list[dict[str, Any]]], pages: list[str], qids: list[str]) -> np.ndarray:
    position = {page: index for index, page in enumerate(pages)}
    scores = np.zeros((len(qids), len(pages)), dtype=np.float32)
    for query_index, qid in enumerate(qids):
        for item in run[qid][:100]:
            page = str(item["chunk_id"])
            if page in position:
                scores[query_index, position[page]] = max(0.0, float(item.get("score", 0.0)))
    return scores


def _normalise(row: np.ndarray) -> np.ndarray:
    positive = float(np.max(row)) if row.size else 0.0
    minimum = float(np.min(row)) if row.size else 0.0
    # Cosine scores may have a negative floor. Shift only for normalization;
    # this does not affect the ranking of the dense-only arm.
    shifted = row - minimum if minimum < 0.0 else row
    maximum = float(np.max(shifted)) if shifted.size else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(row)


def _run_from_scores(pages: list[str], scores: np.ndarray, qids: list[str]) -> dict[str, list[dict[str, Any]]]:
    output = {}
    for query_index, qid in enumerate(qids):
        order = np.argsort(-scores[query_index], kind="stable")[:100]
        output[qid] = [{"chunk_id": pages[int(position)], "doc_id": pages[int(position)], "score": float(scores[query_index, int(position)]), "rank": rank} for rank, position in enumerate(order, 1)]
    return output


def _rrf(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], constant: int = 20) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for stream in (left, right):
        for rank, item in enumerate(stream[:100], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
    order = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(order, 1)]


def _score_fusion(dense_scores: np.ndarray, parent_run: Mapping[str, list[dict[str, Any]]], pages: list[str], qids: list[str], weight_dense: float = 0.30) -> dict[str, list[dict[str, Any]]]:
    position = {page: index for index, page in enumerate(pages)}
    output = {}
    for query_index, qid in enumerate(qids):
        parent = np.zeros(len(pages), dtype=np.float32)
        for item in parent_run[qid][:100]:
            page = str(item["chunk_id"])
            if page in position:
                parent[position[page]] = max(0.0, float(item.get("score", 0.0)))
        combined = weight_dense * _normalise(dense_scores[query_index]) + (1.0 - weight_dense) * _normalise(parent)
        output[qid] = _run_from_scores(pages, combined[None, :], [qid])[qid]
    return output


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-words", type=int, default=240)
    parser.add_argument("--overlap", type=int, default=40)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    queries = [q.query for q in questions]
    qrels = benchmark.qrels()
    pages, texts = _load_pages()
    bm25_run = _load_run_file(BM25_PATH)
    c014_run = _load_run_file(C014_PATH)
    cross_kf_run = _load_run_file(CROSS_KF_PATH)
    segments: list[str] = []
    segment_pages: list[int] = []
    for page_index, text in enumerate(texts):
        page_segments = _segments(text, window=args.max_words, overlap=args.overlap)
        segments.extend("passage: " + segment for segment in page_segments)
        segment_pages.extend([page_index] * len(page_segments))
    model = SentenceTransformer(args.model, trust_remote_code=args.trust_remote_code)
    passage_vectors = model.encode(segments, batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    query_vectors = model.encode(["query: " + query for query in queries], batch_size=args.batch_size, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
    segment_scores = query_vectors @ passage_vectors.T
    dense_scores = np.full((len(qids), len(pages)), -np.inf, dtype=np.float32)
    segment_pages_np = np.asarray(segment_pages, dtype=np.int64)
    for query_index in range(len(qids)):
        np.maximum.at(dense_scores[query_index], segment_pages_np, segment_scores[query_index].astype(np.float32, copy=False))

    bm25_scores = _cached_scores(bm25_run, pages, qids)
    runs = {
        "bm25_control": bm25_run,
        "e5_dense": _run_from_scores(pages, dense_scores, qids),
        "bm25_plus_e5_a070": _run_from_scores(pages, np.asarray([0.70 * _normalise(row) for row in bm25_scores]) + np.asarray([0.30 * _normalise(row) for row in dense_scores]), qids),
        "rrf_e5_c014": {qid: _rrf(runs_q, c014_run[qid]) for qid, runs_q in zip(qids, _run_from_scores(pages, dense_scores, qids).values())},
        "rrf_e5_cross_kf": {qid: _rrf(runs_q, cross_kf_run[qid]) for qid, runs_q in zip(qids, _run_from_scores(pages, dense_scores, qids).values())},
        "score_e5_c014_a070": _score_fusion(dense_scores, c014_run, pages, qids, weight_dense=0.30),
        "score_e5_cross_kf_a070": _score_fusion(dense_scores, cross_kf_run, pages, qids, weight_dense=0.30),
    }
    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof = {}
    selections = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append({"fold": fold, "selected": winner, "train_page_recall@10": _metric_for_qids(metrics[winner], train), "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "page_embeddings.npy", passage_vectors)
    np.save(args.output_dir / "query_embeddings.npy", query_vectors)
    np.save(args.output_dir / "page_dense_scores.npy", dense_scores)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "model": args.model, "segmenting": {"window_words": args.max_words, "overlap_words": args.overlap, "segments": len(segments)}, "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["Multilingual E5 is queried in French and paired with French PDF-inspector passages.", "Page scores are max-pooled over fixed segments; no qrels are used in encoding or pooling.", "BM25 fusion weight and RRF constant are fixed before evaluation; OOF selection is among predeclared arms.", "This is a new semantic retrieval signal, not a data-skipping-only result."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics multilingual E5 baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "model": args.model, "segments": len(segments), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
