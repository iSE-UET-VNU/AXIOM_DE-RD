"""Offline fine-grained second-pass retrieval for ViDoRe Industrial.

The input remains KDL + PDF-inspector extracted text.  A first page-BM25 pass
is complemented by a second pass over structural blocks, paragraphs, or
sentence groups.  Fine scores are pooled back to pages and fused with the
page score.  This tests whether the current page unit is too coarse for the
technical Industrial queries, without adding a new embedding/model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    HierarchyCorpus,
    build_bm25,
    node_index_records,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _load_run,
    _paired_comparison,
    _safe_name,
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
    ROOT / "data/benchmark/vidore_v3/results/industrial_fine_grained_retrieval"
)


@dataclass(frozen=True)
class FineConfig:
    name: str
    unit: str
    fine_weight: float
    pool: str
    fine_depth: int = 1000
    page_weight: float = 1.0


def _aggregate(
    fine_hits: Sequence[tuple[str, float]],
    nodes: Mapping[str, Any],
    pool: str,
) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for node_id, score in fine_hits:
        page_id = nodes[node_id].page_id
        if not page_id:
            continue
        values.setdefault(page_id, []).append(max(0.0, float(score)))
    output: dict[str, float] = {}
    for page_id, scores in values.items():
        scores.sort(reverse=True)
        if pool == "max":
            output[page_id] = scores[0]
        elif pool == "sum_top2":
            output[page_id] = sum(scores[:2])
        elif pool == "sum_top3":
            output[page_id] = sum(scores[:3])
        elif pool == "coverage":
            output[page_id] = sum(1.0 / (5.0 + rank) for rank in range(1, min(10, len(scores)) + 1))
        else:
            raise ValueError(pool)
    return output


def _rows(
    corpus: HierarchyCorpus,
    scores: Mapping[str, float],
    components: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    ranked = sort_scores(scores)[:100]
    return [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
            "scores": {
                key: round(float(value), 8)
                for key, value in components.get(page_id, {}).items()
            },
        }
        for rank, (page_id, score) in enumerate(ranked, 1)
    ]


def _retrieve(
    corpus: HierarchyCorpus,
    page_index: Any,
    fine_index: Any,
    fine_nodes: Mapping[str, Any],
    query: str,
    config: FineConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page_hits = page_index.search(query, len(corpus.page_order))
    page_scores = {page_index.chunk_ids[pos]: float(score) for pos, score in page_hits}
    fine_hits_raw = fine_index.search(query, config.fine_depth)
    fine_hits = [(fine_index.chunk_ids[pos], float(score)) for pos, score in fine_hits_raw]
    fine_page_scores = _aggregate(fine_hits, fine_nodes, config.pool)
    page_norm = normalise_scores(page_scores)
    fine_norm = normalise_scores(fine_page_scores)
    final_scores = {
        page_id: config.page_weight * page_norm.get(page_id, 0.0)
        + config.fine_weight * fine_norm.get(page_id, 0.0)
        for page_id in corpus.page_order
    }
    components = {
        page_id: {
            "page_bm25": page_norm.get(page_id, 0.0),
            "fine_page_score": fine_norm.get(page_id, 0.0),
            "final_score": final_scores[page_id],
        }
        for page_id in corpus.page_order
    }
    run = _rows(corpus, final_scores, components)
    trace = {
        "first_page_hits": [
            {"page_id": page_index.chunk_ids[pos], "score": round(float(score), 8)}
            for pos, score in page_hits[:10]
        ],
        "fine_hits": [
            {
                "node_id": node_id,
                "page_id": fine_nodes[node_id].page_id,
                "score": round(score, 8),
                "text": fine_nodes[node_id].text[:300],
            }
            for node_id, score in fine_hits[:100]
        ],
        "final_pages": [row["chunk_id"] for row in run],
    }
    return run, trace


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="industrial", language="english")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    qids = [q.qid for q in questions]
    qrels = benchmark.qrels()
    page_ids = sorted(document.doc_id for document in benchmark.corpus())
    if len(questions) != 283 or len(page_ids) != 5244:
        raise RuntimeError("Unexpected Industrial inventory")

    build_started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="industrial", page_ids=page_ids)
    page_index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in corpus.page_order)
    configs = [
        FineConfig(name="fine-block-max-w025", unit="block", fine_weight=0.25, pool="max"),
        FineConfig(name="fine-block-sum2-w025", unit="block", fine_weight=0.25, pool="sum_top2"),
        FineConfig(name="fine-block-sum3-w025", unit="block", fine_weight=0.25, pool="sum_top3"),
        FineConfig(name="fine-block-max-w010", unit="block", fine_weight=0.10, pool="max"),
        FineConfig(name="fine-block-max-w040", unit="block", fine_weight=0.40, pool="max"),
        FineConfig(name="fine-paragraph-max-w025", unit="paragraph", fine_weight=0.25, pool="max"),
        FineConfig(name="fine-paragraph-sum2-w025", unit="paragraph", fine_weight=0.25, pool="sum_top2"),
        FineConfig(name="fine-sentence-group5-max-w025", unit="sentence_group5", fine_weight=0.25, pool="max"),
    ]
    fine_indexes: dict[str, tuple[Any, dict[str, Any]]] = {}
    for unit in sorted({config.unit for config in configs}):
        nodes = {node.node_id: node for node in corpus.nodes_for_unit(unit)}
        fine_indexes[unit] = (build_bm25(node_index_records(nodes.values(), corpus, include_context=True)), nodes)
    index_build = time.perf_counter() - build_started

    baseline = _load_run(args.baseline_run)
    methods: dict[str, dict[str, Any]] = {
        "Cached page BM25 baseline": {
            "config": None,
            "retrieval_metrics": _derived_metrics(baseline, qids, qrels),
            "timing_seconds": {"retrieval": 0.0},
        }
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    (args.output_dir / "traces").mkdir(exist_ok=True)
    for config in configs:
        fine_index, fine_nodes = fine_indexes[config.unit]
        config_started = time.perf_counter()
        runs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, Any] = {}
        for question in questions:
            run, trace = _retrieve(corpus, page_index, fine_index, fine_nodes, question.query, config)
            runs[question.qid] = run
            traces[question.qid] = trace
        metrics = _derived_metrics(runs, qids, qrels)
        methods[config.name] = {
            "config": asdict(config),
            "retrieval_metrics": metrics,
            "timing_seconds": {"retrieval": round(time.perf_counter() - config_started, 6)},
        }
        _write_run(args.output_dir / "runs" / f"{_safe_name(config.name)}.jsonl", runs, qids, queries={q.qid: q.query for q in questions})
        with (args.output_dir / "traces" / f"{_safe_name(config.name)}.jsonl").open("w", encoding="utf-8") as handle:
            for qid in qids:
                handle.write(json.dumps({"qid": qid, **traces[qid]}, ensure_ascii=False) + "\n")

    baseline_metrics = methods["Cached page BM25 baseline"]["retrieval_metrics"]
    for name, method in methods.items():
        if name != "Cached page BM25 baseline":
            method["comparison_to_baseline"] = _paired_comparison(baseline_metrics, method["retrieval_metrics"])

    lines = [
        "# Industrial fine-grained second-pass retrieval", "",
        "All page and fine-unit inputs are KDL + PDF-inspector extracted text.", "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Δ page pp | Retrieval s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, method in methods.items():
        m = method["retrieval_metrics"]
        f = m["file_metrics_by_k"]["3"]["file_recall"]
        d = method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0)
        lines.append(f"| {name} | {m['ndcg@10']:.2f} | {m['page_hit@10']:.2%} | {m['page_recall@10']:.2%} | {f:.2%} | {d:+.2f} | {method['timing_seconds']['retrieval']:.2f} |")
    lines += ["", "The fine pass is evaluated as a page-backed retrieval signal because the benchmark qrels are page-level.", "Block/paragraph/caption structure is inherited from the parser; no sentence qrels are fabricated.", ""]
    report = {
        "dataset": "vidore_v3/industrial", "evaluation_language": "english", "queries": len(qids), "pages": len(page_ids),
        "index_counts": {"files": len(corpus.files), "pages": len(corpus.pages), "blocks": len(corpus.blocks), "paragraphs": len(corpus.paragraphs), "sentences": len(corpus.sentences), "evidence_atoms": len(corpus.evidence_atoms), "index_build_seconds": round(index_build, 6)},
        "sources": {"parsed_run": str(args.parsed_run), "baseline_run": str(args.baseline_run)}, "methods": methods,
        "timing_seconds": {"index_build": round(index_build, 6), "total": round(time.perf_counter() - started, 6)},
        "notes": ["Fine units are lifted to pages with max/sum pooling.", "All retrieval uses extracted text; raw PDFs are not indexed."],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"page_recall@10": round(m["retrieval_metrics"]["page_recall@10"] * 100, 2), "file_recall@3": round(m["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "delta_pp": round(m.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2)} for name, m in methods.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
