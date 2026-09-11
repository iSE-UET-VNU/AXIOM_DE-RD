"""Offline multi-index fusion with the cached Tesseract page index.

Tesseract+BM25 is treated as an independent cheap text index.  The experiment
tests whether its candidate evidence complements the PDF-inspector hierarchy
through a small predeclared RRF/score-fusion family.  Five-fold selection is
used for the headline result; no new OCR, rendering or model inference runs.
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

from physics_hierarchical_retrieval import _derived_metrics, _paired_comparison, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
RESULT_ROOT = DEFAULT_ROOT / "results"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_tesseract_multindex_baseline"

BASELINE_PATHS = {
    "tesseract": RESULT_ROOT / "physics_tesseract_bm25/retrieval_french.jsonl",
    "c014": RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
    "cross_kf": RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
    "hierarchical": RESULT_ROOT / "physics_hierarchical_retrieval/oof_run.jsonl",
    "bm25": RESULT_ROOT / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
}


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["qid"])] = list(row["chunks"])
    return output


def _norm(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    values = {str(item["chunk_id"]): float(item.get("score", 0.0)) for item in items[:100]}
    maximum = max(values.values(), default=0.0)
    return {page: max(0.0, score) / maximum if maximum > 0.0 else 0.0 for page, score in values.items()}


def _rrf(*inputs: Sequence[dict[str, Any]], constant: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for items in inputs:
        for rank, item in enumerate(items[:100], 1):
            page = str(item["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


def _score_fusion(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]], weight: float) -> list[dict[str, Any]]:
    scores_left = _norm(left)
    scores_right = _norm(right)
    pages = set(scores_left) | set(scores_right)
    scores = {page: weight * scores_left.get(page, 0.0) + (1.0 - weight) * scores_right.get(page, 0.0) for page in pages}
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _tesseract_file_gate(
    tesseract: Sequence[dict[str, Any]],
    page_rank: Sequence[dict[str, Any]],
    *,
    k_files: int,
) -> list[dict[str, Any]]:
    """Use OCR only as a parent file gate, then rank retained pages normally."""
    file_scores: dict[str, float] = {}
    for item in tesseract[:100]:
        file_id = _file_id(str(item["chunk_id"]))
        file_scores[file_id] = max(file_scores.get(file_id, 0.0), float(item.get("score", 0.0)))
    selected_files = {
        file_id for file_id, _ in sorted(file_scores.items(), key=lambda item: (-item[1], item[0]))[:k_files]
    }
    retained = [item for item in page_rank[:100] if _file_id(str(item["chunk_id"])) in selected_files]
    return [{**item, "rank": rank} for rank, item in enumerate(retained, 1)]


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    loaded = {name: _load(path) for name, path in BASELINE_PATHS.items()}
    if any(set(run) != set(qids) for run in loaded.values()):
        raise RuntimeError("All cached runs must contain the same 302 qids")

    runs: dict[str, dict[str, list[dict[str, Any]]]] = {"tesseract": loaded["tesseract"]}
    for base in ("c014", "cross_kf", "hierarchical", "bm25"):
        runs[base] = loaded[base]
    for base in ("c014", "cross_kf", "hierarchical", "bm25"):
        for constant in (10, 20, 40):
            name = f"rrf_tesseract_{base}_k{constant}"
            runs[name] = {qid: _rrf(loaded["tesseract"][qid], loaded[base][qid], constant=constant) for qid in qids}
        for weight in (0.25, 0.50, 0.75):
            name = f"score_tesseract_{base}_w{weight:.2f}"
            runs[name] = {qid: _score_fusion(loaded["tesseract"][qid], loaded[base][qid], weight) for qid in qids}
        for k_files in (3, 5, 10):
            name = f"tesseract_file_gate_{base}_kf{k_files}"
            runs[name] = {
                qid: _tesseract_file_gate(loaded["tesseract"][qid], loaded[base][qid], k_files=k_files)
                for qid in qids
            }

    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections: list[dict[str, Any]] = []
    candidates = list(runs)
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        for qid in heldout:
            oof[qid] = runs[winner][qid]
        selections.append({"fold": fold, "selected": winner, "train_page_recall@10": _metric_for_qids(metrics[winner], train), "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "inputs": {name: str(path) for name, path in BASELINE_PATHS.items()},
        "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()},
        "best_full_set": best_full,
        "oof": {"metrics": oof_metric, "folds": selections},
        "notes": [
            "Tesseract BM25 is a cached independent text index; this experiment does not rerun OCR.",
            "RRF and score-fusion constants/weights are a small predeclared family, with OOF selection by training folds.",
            "All methods use full page IDs and the canonical derived metrics.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics Tesseract multi-index baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        item = metrics[name]
        lines.append(f"| {name} | {item['page_recall@10']:.2%} | {item['ndcg@10']:.2f} | {item['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set exploratory method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
