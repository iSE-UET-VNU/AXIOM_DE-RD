"""Rerank a small structural candidate pool with a multilingual cross-encoder."""

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
RESULT_ROOT = DEFAULT_ROOT / "results"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_cross_encoder_rerank"


def _normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min()) if len(values) else 0.0
    shifted = values - minimum if minimum < 0.0 else values
    maximum = float(shifted.max()) if len(shifted) else 0.0
    return shifted / maximum if maximum > 0.0 else np.zeros_like(values)


def _rrf_score(items: Sequence[dict[str, Any]], depth: int) -> dict[str, float]:
    return {str(item["chunk_id"]): 1.0 / (20 + rank) for rank, item in enumerate(items[:depth], 1)}


def _candidate_pages(parents: Sequence[Mapping[str, list[dict[str, Any]]]], qid: str, depth: int) -> list[str]:
    return sorted({str(item["chunk_id"]) for parent in parents for item in parent[qid][:depth]})


def _rerank_run(
    ce_scores: Mapping[str, np.ndarray],
    parents: Sequence[Mapping[str, list[dict[str, Any]]]],
    qids: list[str],
    ce_weight: float,
    prior_kind: str,
    candidate_depth: int,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        pages = _candidate_pages(parents, qid, candidate_depth)
        page_to_index = {page: index for index, page in enumerate(pages)}
        parent_maps = [_rrf_score(parent[qid], candidate_depth) for parent in parents]
        prior = np.asarray([
            max(rrf.get(page, 0.0) for rrf in parent_maps)
            if prior_kind == "max_rrf"
            else sum(rrf.get(page, 0.0) for rrf in parent_maps)
            for page in pages
        ], dtype=np.float32)
        ce = ce_scores[qid]
        combined = ce_weight * _normalise(ce) + (1.0 - ce_weight) * _normalise(prior)
        order = np.argsort(-combined, kind="stable")
        output[qid] = [{"chunk_id": pages[int(index)], "doc_id": pages[int(index)], "score": float(combined[int(index)]), "rank": rank} for rank, index in enumerate(order, 1)]
    return output


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-depth", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    queries = {q.qid: q.query for q in questions}
    qrels = benchmark.qrels()
    page_texts = _load_page_texts(PARSED_RUN)
    c014 = _load_run(RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl")
    cross_kf = _load_run(RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl")
    parents = [c014, cross_kf]

    pairs: list[tuple[str, str]] = []
    pair_keys: list[tuple[str, str]] = []
    for qid in qids:
        for page in _candidate_pages(parents, qid, args.candidate_depth):
            pairs.append((queries[qid], page_texts.get(page, "")))
            pair_keys.append((qid, page))
    model = CrossEncoder(args.model, max_length=args.max_length)
    values = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    ce_scores: dict[str, np.ndarray] = {qid: np.zeros(len(_candidate_pages(parents, qid, args.candidate_depth)), dtype=np.float32) for qid in qids}
    page_positions = {qid: {page: index for index, page in enumerate(_candidate_pages(parents, qid, args.candidate_depth))} for qid in qids}
    for (qid, page), score in zip(pair_keys, values):
        ce_scores[qid][page_positions[qid][page]] = float(np.asarray(score).reshape(-1)[0])

    runs = {
        "c014_control": c014,
        "cross_kf_control": cross_kf,
        "ce_only": _rerank_run(ce_scores, parents, qids, 1.0, "sum_rrf", args.candidate_depth),
        "ce_sum_rrf_w080": _rerank_run(ce_scores, parents, qids, 0.80, "sum_rrf", args.candidate_depth),
        "ce_max_rrf_w080": _rerank_run(ce_scores, parents, qids, 0.80, "max_rrf", args.candidate_depth),
    }
    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    candidates = list(runs)
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append({"fold": fold, "selected": winner, "train_page_recall@10": _metric_for_qids(metrics[winner], train), "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries=queries)
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "model": args.model, "candidate_depth_per_parent": args.candidate_depth, "pairs": len(pairs), "max_length": args.max_length, "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": [f"Cross-encoder sees only the union of c014 and cross-Kf top-{args.candidate_depth} pages per query.", "RRF-20 prior and CE weights are predeclared; no qrel features enter reranking.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics cross-encoder rerank", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "pairs": len(pairs), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
