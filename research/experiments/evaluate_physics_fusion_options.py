"""Screen lightweight combinations of PDF-inspector/BM25 and V-SPLADE.

This experiment is deliberately offline.  It reuses the cached 302-query
Physics runs and cached V-SPLADE sparse vectors, so it does not render PDFs,
run a model, or call an API.

The screening arms are:

* BM25 and V-SPLADE baselines;
* score-normalised weighted fusion over several BM25 weights;
* rank-based reciprocal-rank fusion;
* BM25 over the learned sparse-token coordinates, and two ways to combine it
  with PDF-inspector text;
* file-level max/sum pooling over a weighted page ranking;
* a fixed formula/symbol router (a deployable heuristic, not a trained model);
* oracle page/file routing and candidate-union coverage as ceilings.

The learned-token BM25 arms are a controlled diagnostic: each active sparse
coordinate is represented as ``v<ID>``.  This preserves the exact fixed
vocabulary coordinates without depending on tokenizer decoding or a model
package being installed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402

from evaluate_physics_file_level import (  # noqa: E402
    _evaluate,
    _load_physics_questions_and_qrels,
)


DEFAULT_PAIR_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_OUTPUT_DIR = DEFAULT_PAIR_DIR / "fusion_options_screening"

FORMULA_SIGNAL = re.compile(
    r"\d|[\[\]{}()=±×÷*/^_\\]|\b(?:commutateur|équation|theoreme|théorème|"
    r"formule|calculer|calcul|relation|valeur numérique)\b",
    flags=re.IGNORECASE,
)


def _load_csr(path: Path) -> sparse.csr_matrix:
    payload = np.load(path, allow_pickle=True)
    shape = tuple(int(value) for value in payload["shape"])
    return sparse.csr_matrix(
        (
            payload["data"].astype(np.float32, copy=False),
            payload["indices"].astype(np.int32, copy=False),
            payload["indptr"].astype(np.int32, copy=False),
        ),
        shape=shape,
    )


def _load_run(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval run not found: {path}")
    output: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output[str(row["qid"])] = list(row["chunks"])
    return output


def _load_page_texts_from_run(path: Path) -> dict[str, str]:
    # The paired runs contain the PDF-inspector text on every page candidate.
    output: dict[str, str] = {}
    for chunks in _load_run(path).values():
        for chunk in chunks:
            page = str(chunk["chunk_id"])
            if page not in output:
                output[page] = str(chunk.get("text") or "")
    return output


def _normalise(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {position: 0.0 for position in scores}
    return {position: score / maximum for position, score in scores.items()}


def _fuse(
    left: list[tuple[int, float]],
    right: list[tuple[int, float]],
    *,
    method: str,
    alpha: float = 0.7,
    rank_constant: int = 60,
) -> list[tuple[int, float]]:
    left_map = dict(left)
    right_map = dict(right)
    positions = set(left_map) | set(right_map)
    if method == "weighted":
        left_norm = _normalise(left_map)
        right_norm = _normalise(right_map)
        scores = {
            position: alpha * left_norm.get(position, 0.0)
            + (1.0 - alpha) * right_norm.get(position, 0.0)
            for position in positions
        }
    elif method == "rrf":
        left_rank = {position: rank for rank, (position, _) in enumerate(left, 1)}
        right_rank = {position: rank for rank, (position, _) in enumerate(right, 1)}
        scores = {
            position: (
                1.0 / (rank_constant + left_rank[position])
                if position in left_rank
                else 0.0
            )
            + (
                1.0 / (rank_constant + right_rank[position])
                if position in right_rank
                else 0.0
            )
            for position in positions
        }
    else:
        raise ValueError(f"Unknown fusion method: {method}")
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _positions_from_run(
    run: dict[str, list[dict[str, Any]]], page_position: dict[str, int]
) -> dict[str, list[tuple[int, float]]]:
    output: dict[str, list[tuple[int, float]]] = {}
    for qid, chunks in run.items():
        hits: list[tuple[int, float]] = []
        for chunk in chunks:
            page = str(chunk["chunk_id"])
            if page in page_position:
                hits.append((page_position[page], float(chunk.get("score", 0.0))))
        output[qid] = hits
    return output


def _positions_to_run(
    ranked: dict[str, list[tuple[int, float]]], page_units: list[str], depth: int = 100
) -> dict[str, list[dict[str, Any]]]:
    return {
        qid: [
            {"chunk_id": page_units[position], "doc_id": page_units[position], "score": score}
            for position, score in hits[:depth]
        ]
        for qid, hits in ranked.items()
    }


def _token_text(row: sparse.csr_matrix, limit: int | None = None) -> str:
    indices = row.indices
    if limit is not None and len(indices) > limit:
        order = np.argsort(-row.data, kind="stable")[:limit]
        indices = indices[order]
    return " ".join(f"v{int(index)}" for index in indices)


def _build_token_bm25(
    page_vectors: sparse.csr_matrix,
    page_units: list[str],
    *,
    limit: int | None = None,
) -> BM25Index:
    return BM25Index(analyzer_name="plain").build(
        [
            {"chunk_id": unit, "doc_id": unit, "text": _token_text(page_vectors.getrow(i), limit)}
            for i, unit in enumerate(page_units)
        ]
    )


def _token_queries(query_vectors: sparse.csr_matrix, *, limit: int | None = None) -> list[str]:
    return [_token_text(query_vectors.getrow(i), limit) for i in range(query_vectors.shape[0])]


def _rank_index(index: BM25Index, queries: list[str], depth: int) -> dict[str, list[tuple[int, float]]]:
    output: dict[str, list[tuple[int, float]]] = {}
    for i, query in enumerate(queries):
        hits = index.search(query, depth)
        # Keep the same fixed candidate depth as the cached baseline runs.
        # Zero-score pages are only a deterministic tail when the learned
        # token field has fewer than `depth` lexical matches.
        seen = {position for position, _ in hits}
        for position in range(len(index.lengths)):
            if len(hits) >= depth:
                break
            if position not in seen:
                hits.append((position, 0.0))
        output[f"physics::{i}"] = hits
    return output


def _file_id(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _aggregate_files(
    base_ranked: list[tuple[int, float]],
    page_units: list[str],
    *,
    pool: str,
) -> list[tuple[int, float]]:
    by_file: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for position, score in base_ranked:
        by_file[_file_id(page_units[position])].append((position, score))
    file_scores: dict[str, float] = {}
    for file_id, pages in by_file.items():
        values = sorted((score for _, score in pages), reverse=True)
        if pool == "max":
            file_scores[file_id] = values[0]
        elif pool == "sum2":
            file_scores[file_id] = sum(values[:2])
        else:
            raise ValueError(f"Unknown file pool: {pool}")
    ordered_files = sorted(file_scores, key=lambda file_id: (-file_scores[file_id], file_id))
    output: list[tuple[int, float]] = []
    for file_id in ordered_files:
        pages = sorted(
            by_file[file_id], key=lambda item: (-item[1], page_units[item[0]])
        )
        output.extend(pages)
    return output


def _union_upper_bound(
    bm25: dict[str, list[dict[str, Any]]],
    visual: dict[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    page_hits: list[float] = []
    file_recalls: list[float] = []
    file_hits: list[float] = []
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        union_pages = {
            str(c["chunk_id"])
            for c in bm25[qid][:10] + visual[qid][:10]
        }
        page_hits.append(float(bool(union_pages & gold_pages)))
        gold_files = {_file_id(page) for page in gold_pages}
        files: list[str] = []
        for chunk in bm25[qid][:100] + visual[qid][:100]:
            file_id = _file_id(str(chunk["chunk_id"]))
            if file_id not in files:
                files.append(file_id)
        # This is a candidate-coverage ceiling, not a ranked top-3 result:
        # assume an oracle can pick the best three files from the union.
        available_gold = set(files) & gold_files
        file_hits.append(float(bool(available_gold)))
        file_recalls.append(
            min(3, len(available_gold)) / len(gold_files) if gold_files else 0.0
        )
    return {
        "page_hit@10_union_of_two_top10": sum(page_hits) / len(page_hits),
        "file_hit_union_candidates": sum(file_hits) / len(file_hits),
        "file_recall@3_union_candidate_coverage": sum(file_recalls) / len(file_recalls),
    }


def _oracle_router(
    bm25: dict[str, list[dict[str, Any]]],
    visual: dict[str, list[dict[str, Any]]],
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    *,
    objective: str,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page) for page in gold_pages}

        def utility(chunks: list[dict[str, Any]]) -> tuple[int, int]:
            top_pages = [str(c["chunk_id"]) for c in chunks[:10]]
            page_score = len(set(top_pages) & gold_pages)
            files: list[str] = []
            for chunk in chunks[:100]:
                file_id = _file_id(str(chunk["chunk_id"]))
                if file_id not in files:
                    files.append(file_id)
                if len(files) >= 3:
                    break
            file_score = len(set(files) & gold_files)
            return page_score, file_score

        left = utility(bm25[qid])
        right = utility(visual[qid])
        index = 0 if (left[0] > right[0] if objective == "page" else left[1] > right[1]) else 1
        output[qid] = [dict(chunk) for chunk in (bm25[qid] if index == 0 else visual[qid])]
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Physics fusion-option screening",
        "",
        "Offline comparison on 302 French Physics queries. BM25 uses PDF-inspector text; V-SPLADE uses cached English-query visual sparse vectors. File metrics are derived from the first 100 page candidates and use the first 3 unique files.",
        "",
        "| Method | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | File hit@3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["methods"].items():
        by_k = metrics["file_metrics_by_k"]["3"]
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_recall@10']:.2%} | {metrics['page_hit@10']:.2%} | {by_k['file_recall']:.2%} | {by_k['file_hit']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Weighted fusion alpha is the PDF-inspector/BM25 weight; alpha was swept for diagnosis and should be selected on a held-out set before claiming a final result.",
        "- `visual-token-BM25` treats active V-SPLADE coordinates as `v<ID>` terms. It is a diagnostic sparse lexical field, not a second model inference.",
        "- File aggregation ranks files by max or sum of the best two page scores, then pages within each file. It tests file discovery behaviour and can trade page ranking accuracy for file coverage.",
        "- Oracle routers use qrels and are ceilings only. The union numbers are candidate coverage ceilings, not ranked retrieval results.",
        "",
        "## Candidate-union ceiling",
        "",
        f"- Page hit@10 if either baseline retrieves a gold page: **{report['union_upper_bound']['page_hit@10_union_of_two_top10']:.2%}**.",
        f"- File recall@3 candidate coverage if an oracle picks three files from the combined stream: **{report['union_upper_bound']['file_recall@3_union_candidate_coverage']:.2%}**.",
        f"- Oracle page router: **{report['oracle']['page']['page_hit@10']:.2%}** page hit@10; oracle file router: **{report['oracle']['file']['file_metrics_by_k']['3']['file_recall']:.2%}** file recall@3.",
        "",
        "Per-query comparison is in `per_query.jsonl`; the full JSON report contains per-query metrics for every arm.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", type=Path, default=DEFAULT_PAIR_DIR)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--depth", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    bm25_path = args.pair_dir / "bm25_french_bm25-french_vs-english.jsonl"
    visual_path = args.pair_dir / "vsplade_french_bm25-french_vs-english.jsonl"
    bm25_run = _load_run(bm25_path)
    visual_run = _load_run(visual_path)
    page_metadata = json.loads(
        (args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8")
    )
    page_units = [str(row["unit_id"]) for row in page_metadata]
    page_position = {unit: i for i, unit in enumerate(page_units)}
    page_vectors = _load_csr(args.page_vector_dir / "page_vectors.npz")
    query_vectors = _load_csr(args.query_vector_dir / "query_vectors.npz")
    if page_vectors.shape[0] != len(page_units):
        raise RuntimeError("Page vectors and page metadata have different row counts")
    qids, qrels = _load_physics_questions_and_qrels()
    qids = sorted(qids, key=lambda qid: int(qid.rsplit("::", 1)[1]))
    if query_vectors.shape[0] != len(qids):
        raise RuntimeError("Query vectors and French Physics query count differ")
    if set(bm25_run) != set(qids) or set(visual_run) != set(qids):
        raise RuntimeError("Baseline runs do not contain exactly the 302 French qids")

    bm25 = _positions_from_run(bm25_run, page_position)
    visual = _positions_from_run(visual_run, page_position)
    benchmark = ViDoreV3(
        root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="french"
    )
    english_benchmark = ViDoreV3(
        root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="english"
    )
    french_questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    english_questions = sorted(
        list(english_benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    french_queries = [question.query for question in french_questions]
    english_queries = [question.query for question in english_questions]
    if len(french_queries) != len(qids) or len(english_queries) != len(qids):
        raise RuntimeError("French/English Physics query counts are not both 302")
    ranked: dict[str, dict[str, list[tuple[int, float]]]] = {
        "BM25 PDF-inspector": bm25,
        "V-SPLADE English": visual,
    }

    for alpha in (0.25, 0.50, 0.60, 0.70, 0.80, 0.90):
        name = f"weighted alpha={alpha:.2f}"
        ranked[name] = {
            qid: _fuse(bm25[qid], visual[qid], method="weighted", alpha=alpha)
            for qid in qids
        }
    for constant in (20, 60, 100):
        name = f"RRF k={constant}"
        ranked[name] = {
            qid: _fuse(bm25[qid], visual[qid], method="rrf", rank_constant=constant)
            for qid in qids
        }

    token_started = time.perf_counter()
    token_index = _build_token_bm25(page_vectors, page_units)
    token_queries = _token_queries(query_vectors)
    token_bm25 = _rank_index(token_index, token_queries, args.depth)
    ranked["V-token BM25"] = token_bm25
    for alpha in (0.25, 0.50, 0.75):
        name = f"two-field token/BM25 alpha={alpha:.2f}"
        ranked[name] = {
            qid: _fuse(bm25[qid], token_bm25[qid], method="weighted", alpha=alpha)
            for qid in qids
        }

    page_text_by_id = _load_page_texts_from_run(bm25_path)
    # Query expansion with the benchmark's English translation tests whether
    # bilingual lexical matching alone can recover English labels in pages.
    text_index = BM25Index(analyzer_name="plain").build(
        [
            {
                "chunk_id": unit,
                "doc_id": unit,
                "text": page_text_by_id.get(unit, ""),
            }
            for unit in page_units
        ]
    )
    ranked["BM25 bilingual query expansion"] = _rank_index(
        text_index,
        [f"{french_queries[i]} {english_queries[i]}" for i in range(len(qids))],
        args.depth,
    )

    # Straight concatenation is intentionally included as a simple baseline
    # for the proposed "PDF-inspector text + learned keywords" representation.
    visual_token_text = [
        _token_text(page_vectors.getrow(i)) for i in range(page_vectors.shape[0])
    ]
    joint_index = BM25Index(analyzer_name="plain").build(
        [
            {
                "chunk_id": unit,
                "doc_id": unit,
                "text": f"{page_text_by_id.get(unit, '')} {visual_token_text[i]}",
            }
            for i, unit in enumerate(page_units)
        ]
    )
    joint_queries = [
        f"{query} {token_queries[i]}" for i, query in enumerate(french_queries)
    ]
    ranked["joint PDF text + V tokens"] = _rank_index(joint_index, joint_queries, args.depth)
    token_seconds = time.perf_counter() - token_started

    weighted07 = ranked["weighted alpha=0.70"]
    for pool in ("max", "sum2"):
        name = f"file aggregation {pool} on weighted07"
        ranked[name] = {
            qid: _aggregate_files(weighted07[qid], page_units, pool=pool)
            for qid in qids
        }

    # A fixed, leakage-free heuristic: mathematical/symbol-heavy questions go
    # to lexical retrieval; ordinary natural-language questions go to visual
    # retrieval.  This is intentionally not tuned with qrels.
    router: dict[str, list[tuple[int, float]]] = {}
    for i, qid in enumerate(qids):
        router[qid] = bm25[qid] if FORMULA_SIGNAL.search(french_queries[i]) else visual[qid]
    ranked["heuristic router (formula->BM25)"] = router

    ranked_runs = {name: _positions_to_run(value, page_units, args.depth) for name, value in ranked.items()}
    methods = {name: _evaluate(run, qids, qrels) for name, run in ranked_runs.items()}

    oracle_page_run = _oracle_router(bm25_run, visual_run, qids, qrels, objective="page")
    oracle_file_run = _oracle_router(bm25_run, visual_run, qids, qrels, objective="file")
    # `benchmark` is constructed as a consistency check that the benchmark
    # object still exposes all 302 questions used by the flat qrel loader.
    if len(list(benchmark.questions())) != len(qids):
        raise RuntimeError("Benchmark object and flat qrel loader disagree on query count")
    oracle = {
        "page": _evaluate(oracle_page_run, qids, qrels),
        "file": _evaluate(oracle_file_run, qids, qrels),
    }

    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": len(page_units),
        "page_depth": 10,
        "candidate_depth": args.depth,
        "file_metric": "first 3 unique files from first 100 page candidates",
        "sources": {
            "pdf_text": str(bm25_path),
            "visual_run": str(visual_path),
            "page_vectors": str(args.page_vector_dir / "page_vectors.npz"),
            "query_vectors": str(args.query_vector_dir / "query_vectors.npz"),
        },
        "methods": methods,
        "union_upper_bound": _union_upper_bound(bm25_run, visual_run, qids, qrels),
        "oracle": oracle,
        "timing_seconds": {
            "total": time.perf_counter() - started,
            "token_field_build_and_search": token_seconds,
        },
        "notes": [
            "V-SPLADE query vectors are the cached English-query vectors evaluated against French qrels.",
            "Weighted alpha values are a same-set sweep and are exploratory, not a held-out final choice.",
            "Oracle results use qrels and are upper bounds only.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")

    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for i, qid in enumerate(qids):
            row: dict[str, Any] = {"qid": qid, "query_french": french_queries[i]}
            for name, metrics in methods.items():
                detail = next(item for item in metrics["per_query"] if item["qid"] == qid)
                gold_files = set(detail["gold_files"])
                top3_files = detail["top10_files_from_top100_pages"][:3]
                file_recall_at3 = (
                    len(set(top3_files) & gold_files) / len(gold_files)
                    if gold_files
                    else 0.0
                )
                row[name] = {
                    "page_hit@10": detail["page_hit@10"],
                    "page_recall@10": detail["page_recall@10"],
                    "file_recall@3": file_recall_at3,
                    "top10_pages": detail["top10_pages"],
                    "top3_files": top3_files,
                }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        name: {
            "ndcg@10": round(metrics["ndcg@10"], 2),
            "page_recall@10": round(100 * metrics["page_recall@10"], 2),
            "page_hit@10": round(100 * metrics["page_hit@10"], 2),
            "file_recall@3": round(100 * metrics["file_metrics_by_k"]["3"]["file_recall"], 2),
            "file_hit@3": round(100 * metrics["file_metrics_by_k"]["3"]["file_hit"], 2),
        }
        for name, metrics in methods.items()
    }
    print(json.dumps({"summary": summary, "union": report["union_upper_bound"], "timing": report["timing_seconds"], "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
