"""Use cached fine-grained hierarchy runs as secondary candidate indexes."""

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

from physics_hierarchical_retrieval import _derived_metrics, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402

RESULT_ROOT = ROOT / "data/benchmark/vidore_v3/results"
DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = RESULT_ROOT / "physics_fine_index_multistream"

PATHS = {
    "fine_block": RESULT_ROOT / "physics_hierarchical_retrieval/runs/fine-block.jsonl",
    "fine_paragraph": RESULT_ROOT / "physics_hierarchical_retrieval/runs/fine-paragraph.jsonl",
    "fine_sentence5": RESULT_ROOT / "physics_hierarchical_retrieval/runs/fine-sentence_group5.jsonl",
    "c014": RESULT_ROOT / "physics_cascade_diversity_search/runs/oof_selected.jsonl",
    "cross_kf": RESULT_ROOT / "physics_cross_kf_fusion/runs/oof_selected.jsonl",
    "bm25": RESULT_ROOT / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
}


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["qid"])] = list(row["chunks"])
    return result


def _rrf(streams: Sequence[Sequence[dict[str, Any]]], constant: int = 20) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    for stream in streams:
        for rank, row in enumerate(stream[:100], 1):
            page = str(row["chunk_id"])
            scores[page] = scores.get(page, 0.0) + 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda page: (-scores[page], page))[:100]
    return [{"chunk_id": page, "doc_id": page, "score": scores[page], "rank": rank} for rank, page in enumerate(ordered, 1)]


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
    loaded = {name: _load(path) for name, path in PATHS.items()}
    if any(set(run) != set(qids) for run in loaded.values()):
        raise RuntimeError("All fine and structural streams must contain 302 qids")

    specs = {
        "fine_block_only": ("fine_block",),
        "fine_paragraph_only": ("fine_paragraph",),
        "rrf_fine_block_c014": ("fine_block", "c014"),
        "rrf_fine_paragraph_c014": ("fine_paragraph", "c014"),
        "rrf_fine_block_paragraph_c014": ("fine_block", "fine_paragraph", "c014"),
        "rrf_fine_block_paragraph_cross_kf": ("fine_block", "fine_paragraph", "cross_kf"),
        "rrf_fine_all_structural": ("fine_block", "fine_paragraph", "fine_sentence5", "c014", "cross_kf"),
    }
    runs = {name: {qid: _rrf([loaded[stream][qid] for stream in streams]) for qid in qids} for name, streams in specs.items()}
    runs["c014_control"] = loaded["c014"]
    runs["cross_kf_control"] = loaded["cross_kf"]
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
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {"dataset": "vidore_v3/physics", "queries": len(qids), "inputs": {name: str(path) for name, path in PATHS.items()}, "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()}, "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections}, "notes": ["Fine-grained runs are cached hierarchy outputs; they are treated as secondary block/paragraph indexes.", "All methods use fixed RRF constant 20; OOF selection is among predeclared stream sets.", "All results use full page IDs and canonical derived metrics."]}
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics fine-index multistream", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
