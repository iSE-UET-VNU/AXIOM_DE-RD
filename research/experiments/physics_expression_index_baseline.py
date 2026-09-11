"""Fixed character-expression index baseline for Physics.

Character n-grams are a cheap expression index over PDF-inspector page text.
They are intended to tolerate French inflections, punctuation and small text
encoding/OCR variations that a word-only inverted index misses.  The test has
two fixed vectorizers and two fixed BM25 fusions; no qrel-driven n-gram or
weight tuning is performed.  The cached BM25 run remains the control.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics_adaptive_evidence_fusion import _load_page_texts, _load_run  # noqa: E402
from physics_hierarchical_retrieval import _derived_metrics, _paired_comparison, _stratified_folds, _write_run  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402

DEFAULT_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = DEFAULT_ROOT / "results/physics_expression_index_baseline"
PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
BM25_PATH = DEFAULT_ROOT / "results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"


def _normalise(scores: np.ndarray) -> np.ndarray:
    maximum = float(np.max(scores)) if scores.size else 0.0
    return scores / maximum if maximum > 0.0 else np.zeros_like(scores)


def _run_from_scores(page_units: list[str], scores: np.ndarray, qids: list[str], top_k: int = 100) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for query_index, qid in enumerate(qids):
        values = scores[query_index]
        order = np.argsort(-values, kind="stable")[:top_k]
        output[qid] = [
            {"chunk_id": page_units[int(position)], "doc_id": page_units[int(position)], "score": float(values[int(position)]), "rank": rank}
            for rank, position in enumerate(order, 1)
        ]
    return output


def _load_texts() -> tuple[list[str], list[str]]:
    metadata = json.loads((PAGE_VECTOR_DIR / "page_metadata.json").read_text(encoding="utf-8"))
    pages = [str(row["unit_id"]) for row in metadata]
    by_page = _load_page_texts(PARSED_RUN)
    texts = [by_page.get(page, "") for page in pages]
    if len(pages) != 1674 or any(not text.strip() for text in texts):
        raise RuntimeError("Expected non-empty text for all 1,674 pages")
    return pages, texts


def _cached_scores(run: Mapping[str, list[dict[str, Any]]], page_units: list[str], qids: list[str]) -> np.ndarray:
    positions = {page: index for index, page in enumerate(page_units)}
    scores = np.zeros((len(qids), len(page_units)), dtype=np.float32)
    for query_index, qid in enumerate(qids):
        for item in run[qid][:100]:
            page = str(item["chunk_id"])
            if page in positions:
                scores[query_index, positions[page]] = max(0.0, float(item.get("score", 0.0)))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    queries = [q.query for q in questions]
    qrels = benchmark.qrels()
    page_units, page_texts = _load_texts()
    bm25_run = _load_run(BM25_PATH)
    bm25_scores = _cached_scores(bm25_run, page_units, qids)

    vectorizers = {
        "char_wb_3_5": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, strip_accents="unicode"),
        "word_folded_1_2": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents="unicode"),
    }
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {"bm25_control": bm25_run}
    for name, vectorizer in vectorizers.items():
        page_matrix = vectorizer.fit_transform(page_texts)
        query_matrix = vectorizer.transform(queries)
        expression_scores = (query_matrix @ page_matrix.T).toarray().astype(np.float32, copy=False)
        runs[name] = _run_from_scores(page_units, expression_scores, qids)
        # One fixed calibrated fusion weight: BM25 remains the primary field.
        fused = 0.70 * np.asarray([_normalise(row) for row in bm25_scores]) + 0.30 * np.asarray([_normalise(row) for row in expression_scores])
        runs[f"bm25_plus_{name}_a070"] = _run_from_scores(page_units, fused, qids)

    metrics = {name: _derived_metrics(run, qids, qrels) for name, run in runs.items()}
    folds = _stratified_folds(qids, {q.qid: q for q in questions}, qrels)
    candidates = list(runs)
    oof: dict[str, list[dict[str, Any]]] = {}
    selections: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        train = set(qids) - set(heldout)
        winner = max(candidates, key=lambda name: (_metric_for_qids(metrics[name], train), name))
        oof.update({qid: runs[winner][qid] for qid in heldout})
        selections.append({"fold": fold, "selected": winner, "train_page_recall@10": _metric_for_qids(metrics[winner], train), "test_page_recall@10": _metric_for_qids(metrics[winner], set(heldout))})
    oof_metric = _derived_metrics(oof, qids, qrels)
    best_full = max(candidates, key=lambda name: (metrics[name]["page_recall@10"], metrics[name]["ndcg@10"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run(args.output_dir / "oof_run.jsonl", oof, qids, queries={q.qid: q.query for q in questions})
    report = {
        "dataset": "vidore_v3/physics", "queries": len(qids),
        "methods": {name: {"page_recall@10": value["page_recall@10"], "ndcg@10": value["ndcg@10"], "file_recall@3": value["file_metrics_by_k"]["3"]["file_recall"]} for name, value in metrics.items()},
        "best_full_set": best_full, "oof": {"metrics": oof_metric, "folds": selections},
        "notes": ["Character and folded word TF-IDF are fixed expression-index baselines over PDF-inspector text.", "BM25+expression fusion uses a fixed BM25 weight of 0.70; no qrel-driven weight search is used.", "All results use full page IDs and canonical derived metrics."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Physics expression index baseline", "", "| Method | Page recall@10 | nDCG@10 | File recall@3 |", "|---|---:|---:|---:|"]
    for name in sorted(candidates, key=lambda item: metrics[item]["page_recall@10"], reverse=True):
        value = metrics[name]
        lines.append(f"| {name} | {value['page_recall@10']:.2%} | {value['ndcg@10']:.2f} | {value['file_metrics_by_k']['3']['file_recall']:.2%} |")
    lines += ["", f"Best full-set method: `{best_full}`.", f"OOF page recall@10: **{oof_metric['page_recall@10']:.2%}**.", "", "| Fold | Selected | Test page recall@10 |", "|---:|---|---:|"]
    for row in selections:
        lines.append(f"| {row['fold']} | `{row['selected']}` | {row['test_page_recall@10']:.2%} |")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "best_full_set": best_full, "best_full_page_recall@10": round(metrics[best_full]["page_recall@10"] * 100, 2), "oof_page_recall@10": round(oof_metric["page_recall@10"] * 100, 2), "selections": selections}, ensure_ascii=False, indent=2))


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> float:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    return float(np.mean([float(row["page_recall@10"]) for row in rows])) if rows else 0.0


if __name__ == "__main__":
    main()
