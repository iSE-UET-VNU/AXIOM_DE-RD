"""Fair fixed-budget comparison for the dense adaptive-k experiment."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import statistics
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate(rows: list[dict[str, Any]], qrels: dict[str, set[str]], name: str) -> dict[str, Any]:
    coverage, recall10, hits10, file_recall, page_counts = [], [], [], [], []
    unique_files = []
    for row in rows:
        gold = qrels.get(row["query_id"], set())
        returned = row["returned_pages"]
        gold_files = {page.split("#page=", 1)[0] for page in gold}
        found = set(returned) & gold
        top10 = set(returned[:10]) & gold
        selected_files = set(row["selected_files"])
        coverage.append(len(found) / len(gold) if gold else 0.0)
        recall10.append(len(top10) / len(gold) if gold else 0.0)
        hits10.append(float(bool(top10)))
        file_recall.append(len(selected_files & gold_files) / len(gold_files) if gold_files else 0.0)
        page_counts.append(len(returned))
        unique_files.append(len({page.split("#page=", 1)[0] for page in returned}))
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "name": name,
        "queries": len(rows),
        "page_recall_at_returned_k": mean(coverage),
        "page_recall_at_10": mean(recall10),
        "page_hit_at_10": mean(hits10),
        "file_recall_at_kf3": mean(file_recall),
        "mean_pages": mean(page_counts),
        "median_pages": statistics.median(page_counts) if page_counts else 0.0,
        "mean_unique_files_in_output": mean(unique_files),
    }


def main() -> int:
    root = ROOT / "data/work/benchmark_hierarchical_local"
    embedding_root = root / "embedding_adaptive_openrouter_te3s_fulltext"
    pages = read_jsonl(root / "parsing/pages.jsonl")
    queries = read_jsonl(ROOT / "data/raw/BENCHMARK/queries.jsonl")
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(root / "qrels_page_level.jsonl"):
        qrels[str(row["query_id"])].add(str(row["page_id"]))
    page_ids = [row["page_id"] for row in pages]
    doc_ids = [row["doc_id"] for row in pages]
    vectors = np.load(embedding_root / "embeddings/page_embeddings.npy")
    query_vectors = np.load(embedding_root / "embeddings/query_embeddings.npy")
    by_file: dict[str, list[int]] = defaultdict(list)
    for index, doc_id in enumerate(doc_ids):
        by_file[doc_id].append(index)

    fixed_specs = {
        "global_fixed19": ("global", 19),
        "global_fixed24": ("global", 24),
        "hier_fixed27": ("hier", 27),
        "hier_fixed32": ("hier", 32),
    }
    fixed_rows = {name: [] for name in fixed_specs}
    for query, query_vector in zip(queries, query_vectors):
        scores = np.asarray(vectors @ query_vector, dtype=np.float32)
        scores[[i for i, page in enumerate(pages) if not str(page.get("text") or "").strip()]] = -1.0
        global_order = np.argsort(-scores, kind="stable")[:100]
        file_scores = {doc_id: max(float(scores[i]) for i in positions) for doc_id, positions in by_file.items()}
        ranked_files = sorted(file_scores, key=lambda doc_id: (-file_scores[doc_id], doc_id))
        selected = ranked_files[:3]
        hier_order = sorted(
            [i for doc_id in selected for i in by_file[doc_id]],
            key=lambda i: (-float(scores[i]), i),
        )[:100]
        for name, (view, budget) in fixed_specs.items():
            order = global_order if view == "global" else hier_order
            fixed_rows[name].append({
                "query_id": str(query["query_id"]),
                "selected_files": selected,
                "returned_pages": [page_ids[int(i)] for i in order[:budget]],
            })

    adaptive_files = {
        "global_largest_gap": embedding_root / "per_query_global_largest_gap.jsonl",
        "global_largest_gap_buffer5": embedding_root / "per_query_global_largest_gap_buffer5.jsonl",
        "hier_largest_gap": embedding_root / "per_query_hier_largest_gap.jsonl",
        "hier_largest_gap_buffer5": embedding_root / "per_query_hier_largest_gap_buffer5.jsonl",
    }
    all_metrics = {name: evaluate(rows, qrels, name) for name, rows in fixed_rows.items()}
    for name, path in adaptive_files.items():
        all_metrics[name] = evaluate(read_jsonl(path), qrels, name)
    report = {
        "contract_version": "adaptive-fixed-budget-comparison-v1",
        "note": "Fixed baselines use the same dense page scores and same top-3 file selection as their adaptive counterpart; only page budget is fixed.",
        "metrics": all_metrics,
    }
    output = embedding_root / "fair_budget_comparison"
    write_json(output / "report.json", report)
    lines = [
        "# Fair adaptive-vs-fixed budget comparison",
        "",
        "Fixed and adaptive rows use the same dense embeddings and top-3 file selection. The only difference is whether page K is fixed or selected per query.",
        "",
        "| Policy | Page recall at returned K | Page recall@10 | File recall@3 | Mean pages | Median pages |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ["global_fixed19", "global_largest_gap", "global_fixed24", "global_largest_gap_buffer5", "hier_fixed27", "hier_largest_gap", "hier_fixed32", "hier_largest_gap_buffer5"]
    for name in order:
        item = all_metrics[name]
        lines.append(f"| {name} | {item['page_recall_at_returned_k']:.2%} | {item['page_recall_at_10']:.2%} | {item['file_recall_at_kf3']:.2%} | {item['mean_pages']:.2f} | {item['median_pages']:.0f} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
