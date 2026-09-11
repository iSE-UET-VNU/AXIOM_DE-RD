"""Evaluate lightweight BM25 + V-SPLADE fusion on ViDoRe V3 Physics.

The BM25 leg uses page text already produced by the KDL + pdf-inspector run.
The V-SPLADE leg uses cached image-page sparse vectors and cached query vectors;
no model inference or PDF rendering is performed here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmarks.vidore_v3 import ViDoreV3, unit_id  # noqa: E402
from src.evaluation.pipeline_pages import canonical_doc, documents, page_blocks  # noqa: E402
from src.evaluation.run_retrieval import evaluate  # noqa: E402
from src.retrieval import runs  # noqa: E402
from src.retrieval.protocol import ScoredChunk  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_french_302q"
DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_OUTPUT = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion"


def _load_csr(path: Path) -> sparse.csr_matrix:
    payload = np.load(path)
    shape = tuple(int(x) for x in payload["shape"])
    return sparse.csr_matrix(
        (
            payload["data"].astype(np.float32, copy=False),
            payload["indices"].astype(np.int32, copy=False),
            payload["indptr"].astype(np.int32, copy=False),
        ),
        shape=shape,
    )


def _load_page_texts(parsed_run: Path) -> dict[str, str]:
    page_texts: dict[str, str] = {}
    for document in documents(parsed_run):
        doc = canonical_doc(document.get("document", {}).get("file_name"))
        for page, blocks in page_blocks(document).items():
            page_texts[unit_id("physics", doc, page)] = "\n".join(
                str(block.get("text") or "")
                for block in blocks
                if str(block.get("text") or "").strip()
            )
    return page_texts


def _top_scores(scores: np.ndarray, depth: int) -> list[tuple[int, float]]:
    depth = min(depth, len(scores))
    if depth <= 0:
        return []
    candidates = np.argpartition(-scores, depth - 1)[:depth]
    return sorted(
        ((int(position), float(scores[position])) for position in candidates),
        key=lambda item: (-item[1], item[0]),
    )


def _normalise(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    if maximum <= 0:
        return {position: 0.0 for position in scores}
    return {position: score / maximum for position, score in scores.items()}


def _fuse(
    bm25_hits: list[tuple[int, float]],
    splade_hits: list[tuple[int, float]],
    *,
    method: str,
    alpha: float,
) -> list[tuple[int, float]]:
    bm25 = dict(bm25_hits)
    splade = dict(splade_hits)
    positions = set(bm25) | set(splade)

    if method == "weighted":
        bm25_norm = _normalise(bm25)
        splade_norm = _normalise(splade)
        scored = {
            position: alpha * bm25_norm.get(position, 0.0)
            + (1.0 - alpha) * splade_norm.get(position, 0.0)
            for position in positions
        }
    elif method == "rrf":
        bm25_rank = {position: rank for rank, (position, _) in enumerate(bm25_hits, 1)}
        splade_rank = {position: rank for rank, (position, _) in enumerate(splade_hits, 1)}
        scored = {
            position: (1.0 / (60 + bm25_rank[position]) if position in bm25_rank else 0.0)
            + (1.0 / (60 + splade_rank[position]) if position in splade_rank else 0.0)
            for position in positions
        }
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    return sorted(scored.items(), key=lambda item: (-item[1], item[0]))


def _records(
    ranked: dict[str, list[tuple[int, float]]],
    questions: list[Any],
    units: list[str],
    page_texts: list[str],
    *,
    retriever_id: str,
    params: dict[str, Any],
    depth: int,
) -> list[runs.RunRecord]:
    records: list[runs.RunRecord] = []
    params_hash = runs.params_hash(params)
    for question in questions:
        hits = [
            ScoredChunk(
                chunk_id=units[position],
                doc_id=units[position],
                score=score,
                rank=rank,
                text=page_texts[position],
            )
            for rank, (position, score) in enumerate(ranked[question.qid][:depth], 1)
        ]
        records.append(
            runs.RunRecord.build(
                question.qid,
                question.query,
                retriever_id,
                "physics-page-bm25-vsplade",
                params_hash,
                hits,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["french", "english"], default="french")
    parser.add_argument(
        "--bm25-language",
        choices=["french", "english"],
        default=None,
        help="Language of the lexical query; defaults to the evaluation language.",
    )
    parser.add_argument(
        "--vsplade-language",
        choices=["french", "english"],
        default=None,
        help="Language of the V-SPLADE query vectors; defaults to the evaluation language.",
    )
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=None)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    bm25_language = args.bm25_language or args.language
    vsplade_language = args.vsplade_language or args.language
    query_vector_dir = args.query_vector_dir or (
        ROOT / f"data/output/vsplade/vidore_v3_physics_{vsplade_language}_302q"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    vectors_started = time.perf_counter()
    page_matrix = _load_csr(args.page_vector_dir / "page_vectors.npz")
    page_metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    query_matrix = _load_csr(query_vector_dir / "query_vectors.npz")
    vector_seconds = time.perf_counter() - vectors_started

    page_units = [str(row["unit_id"]) for row in page_metadata]
    if page_matrix.shape[0] != len(page_units):
        raise RuntimeError(f"Page vector/metadata mismatch: {page_matrix.shape} vs {len(page_units)}")

    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language=args.language)
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    bm25_benchmark = ViDoreV3(
        root=args.benchmark_root, subset="physics", language=bm25_language
    )
    bm25_questions = sorted(
        list(bm25_benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1])
    )
    if len(bm25_questions) != len(questions):
        raise RuntimeError(
            f"Evaluation/BM25 query count mismatch: {len(questions)} vs {len(bm25_questions)}"
        )
    if query_matrix.shape[0] != len(questions):
        raise RuntimeError(f"Query vector/question mismatch: {query_matrix.shape} vs {len(questions)}")

    parse_started = time.perf_counter()
    text_by_unit = _load_page_texts(args.parsed_run)
    page_texts = [text_by_unit.get(unit, "") for unit in page_units]
    parse_seconds = time.perf_counter() - parse_started
    missing_text = sum(not text.strip() for text in page_texts)
    if missing_text:
        print(f"Warning: {missing_text}/{len(page_texts)} pages have no pdf-inspector text", flush=True)

    index_started = time.perf_counter()
    bm25 = BM25Index(analyzer_name="plain").build(
        [
            {"chunk_id": unit, "doc_id": unit, "text": text}
            for unit, text in zip(page_units, page_texts)
        ]
    )
    index_seconds = time.perf_counter() - index_started

    score_started = time.perf_counter()
    splade_scores = (query_matrix @ page_matrix.T).toarray()
    score_seconds = time.perf_counter() - score_started

    methods = ("bm25", "vsplade", "weighted", "rrf")
    ranked_by_method: dict[str, dict[str, list[tuple[int, float]]]] = {
        method: {} for method in methods
    }
    retrieval_started = time.perf_counter()
    for query_index, question in enumerate(questions):
        # Translations have different qid ranges in ViDoRe V3. Pair them by
        # sorted query ordinal: each language contains the same 302 questions.
        bm25_hits = bm25.search(bm25_questions[query_index].query, args.depth)
        splade_hits = _top_scores(splade_scores[query_index], args.depth)
        ranked_by_method["bm25"][question.qid] = bm25_hits
        ranked_by_method["vsplade"][question.qid] = splade_hits
        ranked_by_method["weighted"][question.qid] = _fuse(
            bm25_hits, splade_hits, method="weighted", alpha=args.alpha
        )
        ranked_by_method["rrf"][question.qid] = _fuse(
            bm25_hits, splade_hits, method="rrf", alpha=args.alpha
        )
    retrieval_seconds = time.perf_counter() - retrieval_started

    metrics: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "language": args.language,
        "bm25_language": bm25_language,
        "vsplade_language": vsplade_language,
        "page_count": len(page_units),
        "queries": len(questions),
        "text_source": "KDL + pdf-inspector page text",
        "visual_source": "cached V-SPLADE rendered-page vectors",
        "depth": args.depth,
        "alpha": args.alpha,
        "methods": {},
        "timing_seconds": {
            "load_vectors": time.perf_counter() - started,
            "load_parsed_page_text": parse_seconds,
            "build_bm25_index": index_seconds,
            "score_vsplade": score_seconds,
            "rank_and_fuse": retrieval_seconds,
        },
        "coverage": {"pages_without_pdf_inspector_text": missing_text},
    }

    evaluate_started = time.perf_counter()
    for method in methods:
        retriever_id = f"physics-{method}-bm25-vsplade"
        params = {
            "method": method,
            "depth": args.depth,
            "alpha": args.alpha,
            "language": args.language,
            "bm25_language": bm25_language,
            "vsplade_language": vsplade_language,
        }
        records = _records(
            ranked_by_method[method],
            questions,
            page_units,
            page_texts,
            retriever_id=retriever_id,
            params=params,
            depth=args.depth,
        )
        suffix = (
            f"_{args.language}"
            if bm25_language == args.language and vsplade_language == args.language
            else f"_{args.language}_bm25-{bm25_language}_vs-{vsplade_language}"
        )
        run_path = args.output_dir / f"{method}{suffix}.jsonl"
        runs.write(run_path, records)
        method_metrics = evaluate(benchmark, records, k=10)
        metrics["methods"][method] = method_metrics

    metrics["timing_seconds"]["evaluate_and_write"] = time.perf_counter() - evaluate_started
    metrics["timing_seconds"]["load_vectors"] = vector_seconds
    metrics["timing_seconds"]["total"] = time.perf_counter() - started
    metrics_suffix = (
        args.language
        if bm25_language == args.language and vsplade_language == args.language
        else f"{args.language}_bm25-{bm25_language}_vs-{vsplade_language}"
    )
    metrics_path = args.output_dir / f"metrics_{metrics_suffix}.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "language": args.language,
        "bm25_language": bm25_language,
        "vsplade_language": vsplade_language,
        "page_count": len(page_units),
        "queries": len(questions),
        "metrics": {
            method: {
                key: value
                for key, value in method_metrics.items()
                if key in {"recall@10", "ndcg@10", "page_recall@10", "single_evidence_recall@10", "multi_evidence_recall@10"}
            }
            for method, method_metrics in metrics["methods"].items()
        },
        "timing_seconds": metrics["timing_seconds"],
        "output": str(args.output_dir),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
