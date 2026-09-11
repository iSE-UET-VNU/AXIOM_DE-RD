"""Evaluate page and derived file retrieval for all Physics baseline runs.

Page metrics use the first 10 ranked pages. File metrics use the first 10
unique files encountered in the first 100 ranked pages. The latter is a
diagnostic derived from page retrieval, not an independently indexed file
retriever.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_baseline_comparison"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "file_level_report.json"

DEFAULT_RUNS = {
    "PDF-inspector + BM25": RUN_DIR / "bm25_french_bm25-french_vs-english.jsonl",
    "V-SPLADE (English query)": RUN_DIR / "vsplade_french_bm25-french_vs-english.jsonl",
    "Tesseract + BM25": ROOT / "data/benchmark/vidore_v3/results/physics_tesseract_bm25/retrieval_french.jsonl",
    "Jina-CLIP-v2": ROOT / "data/output/visual_retrieval/vidore_v3_physics_jina_clip_v2/retrieval_french.jsonl",
    "ColSmol-256M": ROOT / "data/output/visual_retrieval/vidore_v3_physics_colsmol/retrieval_french.jsonl",
}

DEFAULT_NDCG_SOURCES = {
    "PDF-inspector + BM25": (RUN_DIR / "metrics_french_bm25-french_vs-english.json", "bm25"),
    "V-SPLADE (English query)": (RUN_DIR / "metrics_french_bm25-french_vs-english.json", "vsplade"),
    "Tesseract + BM25": (
        ROOT / "data/benchmark/vidore_v3/results/physics_tesseract_bm25/metrics.json",
        None,
    ),
    "Jina-CLIP-v2": (
        ROOT / "data/output/visual_retrieval/vidore_v3_physics_jina_clip_v2/metrics.json",
        None,
    ),
    "ColSmol-256M": (
        ROOT / "data/output/visual_retrieval/vidore_v3_physics_colsmol/metrics.json",
        None,
    ),
}


def _load(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval run not found: {path}")
    out: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["qid"])] = row["chunks"]
    return out


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _unique_files(chunks: list[dict[str, Any]], depth: int) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:depth]:
        file_id = _file_id(str(chunk["chunk_id"]))
        if file_id not in seen:
            seen.add(file_id)
            files.append(file_id)
        if len(files) >= 10:
            break
    return files


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ndcg_at_k(ranked_pages: list[str], qrel: dict[str, int], k: int) -> float:
    """Return one query's graded nDCG using the benchmark's qrel gains."""

    def gain(relevance: int) -> float:
        return float((2**int(relevance)) - 1)

    dcg = sum(
        gain(qrel.get(page, 0)) / math.log2(rank + 1)
        for rank, page in enumerate(ranked_pages[:k], 1)
        if qrel.get(page, 0) > 0
    )
    ideal = sorted((int(value) for value in qrel.values()), reverse=True)[:k]
    idcg = sum(
        gain(value) / math.log2(rank + 1)
        for rank, value in enumerate(ideal, 1)
    )
    return dcg / idcg if idcg else 0.0


def _cached_ndcg(source: tuple[Path, str | None]) -> float | None:
    """Read the official nDCG already produced for a cached retrieval run."""

    path, method = source
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method is not None:
        return float(payload["methods"][method]["ndcg@10"])
    return float(payload["metrics"]["k=10"]["ndcg@10"])


def _load_physics_questions_and_qrels() -> tuple[list[str], dict[str, dict[str, int]]]:
    """Read the flat evaluation export with FastParquet.

    The local PyArrow version currently fails on the Physics corpus parquet's
    nested metadata, while FastParquet reads the same flat export correctly.
    This fallback keeps the report independent of that optional reader issue.
    """

    import pandas as pd

    dataset_dir = ROOT / "data/benchmark/vidore_v3/physics"
    corpus = pd.read_parquet(
        dataset_dir / "corpus.parquet",
        engine="fastparquet",
        columns=["corpus_id", "doc_id", "page_number_in_doc"],
    )
    pages = {
        int(row.corpus_id): (
            f"physics::{row.doc_id}#page={int(row.page_number_in_doc)}"
        )
        for row in corpus.itertuples(index=False)
    }
    queries = pd.read_parquet(
        dataset_dir / "queries.parquet",
        engine="fastparquet",
        columns=["query_id", "language"],
    )
    french_queries = queries[queries["language"].astype(str).str.lower() == "french"]
    qids = [
        f"physics::{int(query_id)}"
        for query_id in sorted(french_queries["query_id"].astype(int).tolist())
    ]
    qid_set = set(qids)

    qrels_frame = pd.read_parquet(
        dataset_dir / "qrels.parquet",
        engine="fastparquet",
        columns=["query_id", "corpus_id", "score"],
    )
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_frame.itertuples(index=False):
        qid = f"physics::{int(row.query_id)}"
        page = pages.get(int(row.corpus_id))
        if qid not in qid_set or page is None:
            continue
        qrels.setdefault(qid, {})[page] = int(row.score)
    return qids, qrels


def _evaluate(
    run: dict[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    page_hit: list[float] = []
    page_recall: list[float] = []
    page_precision: list[float] = []
    file_hit: list[float] = []
    file_recall: list[float] = []
    file_precision: list[float] = []
    file_candidate_counts: list[float] = []
    gold_file_counts: list[int] = []
    ndcg: list[float] = []
    per_query: list[dict[str, Any]] = []

    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page) for page in gold_pages}
        gold_file_counts.append(len(gold_files))
        pages = [str(chunk["chunk_id"]) for chunk in run[qid][:10]]
        found_pages = set(pages) & gold_pages
        page_hit.append(float(bool(found_pages)))
        page_recall.append(len(found_pages) / len(gold_pages) if gold_pages else 0.0)
        # Standard precision@10 over page slots.
        page_precision.append(len(found_pages) / 10.0)

        # Use enough page candidates to form ten unique files. Repeated pages
        # from one file must not consume multiple file-level ranks.
        files = _unique_files(run[qid], 100)
        found_files = set(files) & gold_files
        file_hit.append(float(bool(found_files)))
        file_recall.append(len(found_files) / len(gold_files) if gold_files else 0.0)
        file_candidate_counts.append(float(len(files)))
        file_precision.append(len(found_files) / len(files) if files else 0.0)
        ndcg.append(_ndcg_at_k(pages, qrels.get(qid, {}), 10))
        per_query.append(
            {
                "qid": qid,
                "gold_pages": sorted(gold_pages),
                "gold_files": sorted(gold_files),
                "top10_pages": pages,
                "top10_files_from_top100_pages": files,
                "page_hit@10": bool(found_pages),
                "page_recall@10": page_recall[-1],
                "page_precision@10": page_precision[-1],
                "file_hit@10": bool(found_files),
                "file_recall@10": file_recall[-1],
                "file_precision@10": file_precision[-1],
                "first_gold_file_rank": next(
                    (rank for rank, file_id in enumerate(files, 1) if file_id in gold_files),
                    None,
                ),
            }
        )

    file_metrics_by_k: dict[str, dict[str, float]] = {}
    for k in (1, 3, 5, 10):
        recalls: list[float] = []
        hits: list[float] = []
        precisions: list[float] = []
        for row in per_query:
            candidates = row["top10_files_from_top100_pages"][:k]
            found = set(candidates) & set(row["gold_files"])
            denominator = len(row["gold_files"])
            recalls.append(len(found) / denominator if denominator else 0.0)
            hits.append(float(bool(found)))
            precisions.append(len(found) / len(candidates) if candidates else 0.0)
        file_metrics_by_k[str(k)] = {
            "file_recall": _mean(recalls),
            "file_hit": _mean(hits),
            "file_precision": _mean(precisions),
        }

    return {
        "queries": len(qids),
        "mean_gold_files": _mean([float(value) for value in gold_file_counts]),
        "median_gold_files": statistics.median(gold_file_counts) if gold_file_counts else 0.0,
        "gold_file_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(Counter(gold_file_counts).items())
        },
        "ndcg@10": 100.0 * _mean(ndcg),
        "page_hit@10": _mean(page_hit),
        "page_recall@10": _mean(page_recall),
        "page_precision@10": _mean(page_precision),
        "file_hit@10_from_top100_pages": _mean(file_hit),
        "file_recall@10_from_top100_pages": _mean(file_recall),
        "file_precision@10_from_top100_pages": _mean(file_precision),
        "mean_unique_file_candidates": _mean(file_candidate_counts),
        "file_metrics_by_k": file_metrics_by_k,
        "per_query": per_query,
        "definitions": {
            "page_hit@10": "Fraction of queries with >=1 gold page in top 10 pages.",
            "page_recall@10": "Mean |gold pages ∩ top 10 pages| / |gold pages|.",
            "page_precision@10": "Mean |gold pages ∩ top 10 pages| / 10.",
            "file_hit@10_from_top100_pages": (
                "Fraction of queries with >=1 gold file among the first 10 unique "
                "files encountered in the top 100 pages."
            ),
            "file_recall@10_from_top100_pages": (
                "Mean |gold files ∩ derived top 10 files| / |gold files|."
            ),
            "file_precision@10_from_top100_pages": (
                "Mean |gold files ∩ derived top 10 files| / number of unique "
                "derived candidates."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override or add an arm, for example --run 'V-SPLADE=path.jsonl'.",
    )
    args = parser.parse_args()

    qids, qrels = _load_physics_questions_and_qrels()
    files = dict(DEFAULT_RUNS)
    files["PDF-inspector + BM25"] = args.run_dir / "bm25_french_bm25-french_vs-english.jsonl"
    files["V-SPLADE (English query)"] = args.run_dir / "vsplade_french_bm25-french_vs-english.jsonl"
    for override in args.run:
        if "=" not in override:
            raise ValueError(f"Expected NAME=PATH for --run, got {override!r}")
        name, path = override.split("=", 1)
        files[name.strip()] = Path(path.strip())

    loaded = {name: _load(path) for name, path in files.items()}
    expected_qids = set(qids)
    for name, run in loaded.items():
        missing = sorted(expected_qids - set(run))
        extra = sorted(set(run) - expected_qids)
        if missing or extra:
            raise RuntimeError(
                f"{name}: qid mismatch; missing={missing[:3]} extra={extra[:3]}"
            )
        short_runs = [qid for qid in qids if len(run[qid]) < 100]
        if short_runs:
            raise RuntimeError(
                f"{name}: expected 100 page candidates per query; "
                f"short rows include {short_runs[:3]}"
            )

    report = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "page_depth": 10,
        "file_candidate_depth": 100,
        "file_candidate_count": 10,
        "runs": {name: str(path) for name, path in files.items()},
        "methods": {name: _evaluate(loaded[name], qids, qrels) for name in files},
    }
    for name, source in DEFAULT_NDCG_SOURCES.items():
        if name in report["methods"]:
            cached = _cached_ndcg(source)
            if cached is not None:
                report["methods"][name]["ndcg@10"] = cached
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = [
        "# Physics baseline page and derived file retrieval",
        "",
        "All methods are evaluated on the same 302 French Physics queries. "
        "File metrics are derived from the first 100 page candidates by taking "
        "the first 10 unique files; this is not an independent file index.",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | Page precision@10 | "
        "File recall@10 | File precision@10 | File hit@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["methods"].items():
        markdown.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | "
            f"{metrics['page_hit@10']:.2%} | {metrics['page_recall@10']:.2%} | "
            f"{metrics['page_precision@10']:.2%} | "
            f"{metrics['file_recall@10_from_top100_pages']:.2%} | "
            f"{metrics['file_precision@10_from_top100_pages']:.2%} | "
            f"{metrics['file_hit@10_from_top100_pages']:.2%} |"
        )
    markdown += [
        "",
        "Definitions:",
        "- `page_hit@10`: at least one gold page in the top 10 pages.",
        "- `page_recall@10`: gold pages found divided by the number of gold pages.",
        "- `page_precision@10`: gold pages found divided by 10 page slots.",
        "- `file_recall@10`: fraction of gold files recovered in the derived top 10 files (primary file metric).",
        "- `file_precision@10`: gold files found divided by derived file candidates.",
        "- `file_hit@10`: at least one gold file in the derived top 10 files.",
        "",
        "Gold-file count: 261 queries have one gold file; 41 queries have two gold files.",
        "",
        "File-recall sensitivity by the number of unique file candidates:",
        "",
        "| Method | File recall@1 | File recall@3 | File recall@5 | File recall@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["methods"].items():
        by_k = metrics["file_metrics_by_k"]
        markdown.append(
            f"| {name} | {by_k['1']['file_recall']:.2%} | "
            f"{by_k['3']['file_recall']:.2%} | {by_k['5']['file_recall']:.2%} | "
            f"{by_k['10']['file_recall']:.2%} |"
        )
    markdown += [
        "",
        "Per-query details are stored in the `per_query` field of the JSON report.",
    ]
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(json.dumps({
        name: {
            key: value
            for key, value in metrics.items()
            if key not in {"definitions", "per_query"}
        }
        for name, metrics in report["methods"].items()
    }, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
