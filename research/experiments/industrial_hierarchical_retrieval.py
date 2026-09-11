"""Run an offline Industrial file -> page retrieval cascade.

This experiment is deliberately separate from the Physics runner.  It reuses
the existing KDL + PDF-inspector parsed cache and ViDoRe Industrial qrels, but
does not render, encode, call an API, or modify production retrieval code.

Industrial currently has no cached V-SPLADE page/query vectors in this
repository, so this is the lexical hierarchical arm:

    page text -> file score/pool -> hard top-K files -> page reranking

Example::

    python research/experiments/industrial_hierarchical_retrieval.py
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    CascadeConfig,
    HierarchyCorpus,
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _paired_comparison,
    _safe_name,
    _stage_metrics,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = (
    ROOT / "data/output/vidore-v3-industrial-kdl-pdf-inspector/fcfa9a665c256e86"
)
DEFAULT_BENCHMARK_ROOT = ROOT / "data/raw/benchmarks/vidore_v3"
DEFAULT_BASELINE_RUN = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_discovery_bm25_english.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_hierarchical_retrieval"
)


def _build_indexes(
    parsed_run: Path,
    page_ids: Sequence[str],
) -> tuple[HierarchyCorpus, Any, dict[str, Any], dict[str, int]]:
    started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(
        parsed_run, subset="industrial", page_ids=page_ids
    )
    page_index = build_bm25(
        (page_id, corpus.pages[page_id].text) for page_id in corpus.page_order
    )
    file_texts = corpus.file_texts("all_text")
    file_index = build_bm25(file_texts.items())
    counts = {
        "files": len(corpus.files),
        "pages": len(corpus.pages),
        "pages_with_parser_text": sum(
            bool(node.metadata.get("has_parser_text")) for node in corpus.pages.values()
        ),
        "blocks": len(corpus.blocks),
        "paragraphs": len(corpus.paragraphs),
        "sentences": len(corpus.sentences),
        "evidence_atoms": len(corpus.evidence_atoms),
        "build_seconds": round(time.perf_counter() - started, 6),
        "page_text_bytes": sum(
            len(corpus.pages[page_id].text.encode("utf-8"))
            for page_id in corpus.page_order
        ),
        "file_text_bytes": sum(
            len(text.encode("utf-8")) for text in file_texts.values()
        ),
    }
    return corpus, page_index, {"all_text": file_index}, counts


def _page_run(
    corpus: HierarchyCorpus,
    page_index: Any,
    query: str,
    depth: int = 100,
) -> list[dict[str, Any]]:
    scores = _index_scores(page_index, query, top_k=len(corpus.page_order))
    ranked = sort_scores(scores)[:depth]
    return [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
        }
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]


def _validate_run(
    name: str,
    run: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    corpus_ids: set[str],
    *,
    exact_depth: int | None = None,
) -> None:
    if set(run) != set(qids):
        raise RuntimeError(f"{name} qid set does not match the 283 Industrial queries")
    for qid in qids:
        rows = run[qid]
        if exact_depth is not None and len(rows) != exact_depth:
            raise RuntimeError(
                f"{name} has {len(rows)} candidates for {qid}, expected {exact_depth}"
            )
        if len(rows) > 100:
            raise RuntimeError(f"{name} has more than 100 candidates for {qid}")
        ids = [str(row["chunk_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{name} contains duplicate page candidates for {qid}")
        unknown = set(ids) - corpus_ids
        if unknown:
            raise RuntimeError(f"{name} contains pages outside the corpus: {sorted(unknown)[:3]}")


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Industrial hierarchical retrieval",
        "",
        "Offline `file -> page` cascade using the cached KDL + PDF-inspector text.",
        "",
        "## Protocol",
        "",
        f"- Dataset: **ViDoRe V3 Industrial**, English, **{report['queries']} queries**.",
        f"- Corpus: **{report['index_counts']['files']} files**, **{report['index_counts']['pages']} pages**.",
        "- File stage: concatenate parsed page text per file, score with BM25 and page-score pooling, then keep hard top-K files.",
        "- Page stage: rerank pages inside the selected files with page BM25 plus the file parent score.",
        "- Metrics: top-10 page metrics and file recall derived from the first 100 page results.",
        "- No V-SPLADE Industrial artifact exists locally, so this run has no visual-semantic component.",
        "",
        "## Results",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | File candidate recall@3 | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in report["methods"].items():
        metrics = method["retrieval_metrics"]
        file_metrics = metrics["file_metrics_by_k"]["3"]["file_recall"]
        stage = method.get("stage_metrics", {})
        candidate = stage.get("file_candidate_recall", {}).get("@3", 0.0)
        runtime = method.get("timing_seconds", {}).get("retrieval", 0.0)
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {file_metrics:.2%} | {candidate:.2%} | {runtime:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The cached baseline is the existing page-BM25 run. The cascade rows use the same parsed cache but impose a hard file budget before page ranking.",
        "A file candidate recall loss indicates that the file stage discarded evidence before the page stage could recover it.",
        "",
        "## Caveats",
        "",
        f"- The parsed cache has **{report['index_counts']['pages_with_parser_text']} / {report['index_counts']['pages']}** pages with parser text; empty pages remain in the inventory but cannot receive lexical score.",
        "- This is a retrieval-only experiment; no rendering, OCR, VLM, V-SPLADE or API call is used.",
        "- File recall is a page-ranking-derived discovery diagnostic, not an independent file index evaluation.",
        "",
        "Stage traces are in `stage_traces.jsonl`; page runs are in `runs/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    load_started = time.perf_counter()
    benchmark = ViDoreV3(
        root=args.benchmark_root, subset="industrial", language="english"
    )
    questions_list = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    questions = {question.qid: question for question in questions_list}
    qids = [question.qid for question in questions_list]
    qrels = benchmark.qrels()
    page_ids = sorted(document.doc_id for document in benchmark.corpus())
    corpus_ids = set(page_ids)
    if len(qids) != 283:
        raise RuntimeError(f"Expected 283 Industrial English queries, got {len(qids)}")
    if len(page_ids) != 5244:
        raise RuntimeError(f"Expected 5,244 Industrial pages, got {len(page_ids)}")
    if benchmark.unreachable_n:
        raise RuntimeError(f"Industrial qrels contain {benchmark.unreachable_n} unreachable pages")
    if set(qrels) != set(qids):
        raise RuntimeError("Industrial qrels do not cover exactly the 283 queries")
    load_seconds = time.perf_counter() - load_started

    corpus, page_index, file_indexes, index_counts = _build_indexes(
        args.parsed_run, page_ids
    )
    if set(corpus.pages) != corpus_ids:
        raise RuntimeError("Hierarchy page inventory does not match Industrial corpus")
    if len(corpus.files) != 27:
        raise RuntimeError(f"Expected 27 Industrial files, got {len(corpus.files)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    baseline_run = _load_run(args.baseline_run)
    _validate_run("cached baseline", baseline_run, qids, corpus_ids, exact_depth=100)

    methods: dict[str, dict[str, Any]] = {
        "Cached page BM25 baseline": {
            "config": None,
            "retrieval_metrics": _derived_metrics(baseline_run, qids, qrels),
            "stage_metrics": {},
            "timing_seconds": {"retrieval": 0.0},
        }
    }
    traces_all: dict[str, dict[str, Any]] = {}

    # Rebuild a page-BM25 run from exactly the hierarchy input.  This makes the
    # cascade comparison auditable when parser pages are empty/missing text.
    rebuilt_started = time.perf_counter()
    rebuilt_run = {
        question.qid: _page_run(corpus, page_index, question.query)
        for question in questions_list
    }
    rebuilt_seconds = time.perf_counter() - rebuilt_started
    _validate_run(
        "rebuilt page BM25", rebuilt_run, qids, corpus_ids, exact_depth=100
    )
    rebuilt_metrics = _derived_metrics(rebuilt_run, qids, qrels)
    methods["Rebuilt page BM25 on KDL cache"] = {
        "config": {"source": "KDL + PDF-inspector page text"},
        "retrieval_metrics": rebuilt_metrics,
        "stage_metrics": {},
        "timing_seconds": {"retrieval": round(rebuilt_seconds, 6)},
    }
    _write_run(
        runs_dir / "rebuilt_page_bm25.jsonl",
        rebuilt_run,
        qids,
        queries={qid: questions[qid].query for qid in qids},
    )

    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes={},
        visual_scores={},
    )
    configs = [
        CascadeConfig(
            name=f"cascade-bm25-all_text-max-kf{k}",
            file_representation="all_text",
            file_pool="max",
            k_files=k,
            page_depth=100,
            final_depth=100,
            bm25_weight=1.0,
            parent_weight=0.15,
            file_direct_weight=0.50,
            file_pool_source="page_base",
        )
        for k in (1, 2, 3, 5, 10)
    ]

    retrieval_started = time.perf_counter()
    for config in configs:
        config_started = time.perf_counter()
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for index, question in enumerate(questions_list, 1):
            run, trace = retriever.retrieve(question.qid, question.query, config)
            if len(run) > 100:
                raise RuntimeError(
                    f"{config.name} returned {len(run)} pages for {question.qid}"
                )
            run_by_qid[question.qid] = run
            traces[question.qid] = trace
            if index % 100 == 0 or index == len(questions_list):
                print(f"[{config.name}] {index}/{len(questions_list)} queries")
        _validate_run(config.name, run_by_qid, qids, corpus_ids)
        metrics = _derived_metrics(run_by_qid, qids, qrels)
        stage = _stage_metrics(traces, qids, qrels)
        name = config.name
        methods[name] = {
            "config": asdict(config),
            "retrieval_metrics": metrics,
            "stage_metrics": stage,
            "timing_seconds": {
                "retrieval": round(time.perf_counter() - config_started, 6)
            },
        }
        traces_all[name] = traces
        _write_run(
            runs_dir / f"{_safe_name(name)}.jsonl",
            run_by_qid,
            qids,
            queries={qid: questions[qid].query for qid in qids},
        )
    retrieval_seconds = time.perf_counter() - retrieval_started

    baseline_metrics = methods["Cached page BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "Cached page BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(
                baseline_metrics, method["retrieval_metrics"]
            )

    trace_started = time.perf_counter()
    with (args.output_dir / "stage_traces.jsonl").open("w", encoding="utf-8") as handle:
        for name, traces in traces_all.items():
            for qid in qids:
                handle.write(
                    json.dumps(
                        {"method": name, **traces[qid]}, ensure_ascii=False
                    )
                    + "\n"
                )
    trace_seconds = time.perf_counter() - trace_started

    report: dict[str, Any] = {
        "dataset": "vidore_v3/industrial",
        "evaluation_language": "english",
        "queries": len(qids),
        "pages": len(page_ids),
        "index_counts": index_counts,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "benchmark_root": str(args.benchmark_root),
            "baseline_run": str(args.baseline_run),
            "visual_artifacts": [],
            "visual_status": "not_available_for_industrial",
        },
        "methods": methods,
        "timing_seconds": {
            "load_benchmark_and_qrels": round(load_seconds, 6),
            "index_build": index_counts["build_seconds"],
            "rebuilt_page_bm25": round(rebuilt_seconds, 6),
            "cascade_retrieval_all_configs": round(retrieval_seconds, 6),
            "write_stage_traces": round(trace_seconds, 6),
            "total": round(time.perf_counter() - started, 6),
        },
        "validation": {
            "query_count": len(qids),
            "baseline_candidates_per_query": 100,
            "corpus_pages": len(corpus_ids),
            "qrels_unreachable": benchmark.unreachable_n,
            "qrel_pages_in_corpus": all(
                set(qrels[qid]).issubset(corpus_ids) for qid in qids
            ),
            "hierarchy_page_inventory_matches": set(corpus.pages) == corpus_ids,
            "visual_vectors_used": False,
        },
        "notes": [
            "Industrial has no cached V-SPLADE page/query vectors locally; this experiment is BM25-only hierarchical retrieval.",
            "The cascade uses KDL + PDF-inspector page text and retains 205 parser-empty corpus pages in the inventory.",
            "File recall is derived from the first 100 final page candidates and is not an independent file index metric.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset": report["dataset"],
                "language": report["evaluation_language"],
                "configs": [asdict(config) for config in configs],
                "sources": report["sources"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        _markdown(report), encoding="utf-8"
    )

    summary = {
        name: {
            "page_recall@10": round(method["retrieval_metrics"]["page_recall@10"] * 100, 2),
            "file_recall@3": round(
                method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100,
                2,
            ),
            "page_recall_delta_pp": round(
                method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0),
                2,
            ),
        }
        for name, method in methods.items()
    }
    print(
        json.dumps(
            {
                "summary": summary,
                "timing_seconds": report["timing_seconds"],
                "output": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
