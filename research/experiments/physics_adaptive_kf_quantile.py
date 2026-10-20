"""Evaluate a quartile-based adaptive file budget on Physics.

The policy proposed for this experiment uses the file score curve from the
existing Kf=3 hierarchical trace:

    Q4 (largest top1-top3 gap) -> Kf=3
    Q3                            -> Kf=4
    Q2                            -> Kf=5
    Q1 (smallest gap)             -> Kf=6

The quartile thresholds use scores only, not qrels.  Kf=4 and Kf=6 are run
from the cached KDL/PDF-inspector and V-SPLADE artifacts; no new encoding or
parsing is performed.  The fixed Kf=6 arm is the comparison baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import CascadeConfig  # noqa: E402
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    HierarchicalRetriever,
    _build_corpus_and_indexes,
    _derived_metrics,
    _file_id,
    _load_run,
    _load_visual_scores,
    _page_vector_units,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = (
    ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
)
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASE_TRACE = (
    ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/stage_traces.jsonl"
)
DEFAULT_BASE_RUN_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/runs"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_adaptive_kf_quantile"
)

EXPECTED_QUERIES = 302
EXPECTED_PAGES = 1674
KFS = (3, 4, 5, 6)
PAGE_DEPTH = 100
FINAL_DEPTH = 100
BASE_METHOD = "cascade-all_text-max-kf3-fusion"


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return tuple(float(value) for value in np.quantile(array, (0.25, 0.50, 0.75)))


def _load_file_gaps(path: Path, qids: Sequence[str]) -> dict[str, float]:
    gaps: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("method") != BASE_METHOD:
                continue
            qid = str(row["qid"])
            candidates = row.get("file_candidates") or []
            scores = [float(item.get("score") or 0.0) for item in candidates]
            if len(scores) < 3:
                raise RuntimeError(f"Not enough file candidates for {qid}")
            if qid in gaps:
                raise RuntimeError(f"Duplicate file-gap trace for {qid}")
            gaps[qid] = scores[0] - scores[2]
    if set(gaps) != set(qids):
        raise RuntimeError(
            f"Expected file gaps for {len(qids)} qids, got {len(gaps)}"
        )
    return gaps


def _choose_kf(gap: float, thresholds: Sequence[float]) -> tuple[str, int]:
    q25, q50, q75 = thresholds
    if gap < q25:
        return "Q1", 6
    if gap < q50:
        return "Q2", 5
    if gap < q75:
        return "Q3", 4
    return "Q4", 3


def _compact_run(run: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(row["chunk_id"]),
            "doc_id": str(row.get("doc_id") or row["chunk_id"]),
            "score": float(row.get("score") or 0.0),
            "rank": int(row.get("rank") or rank),
        }
        for rank, row in enumerate(run[:PAGE_DEPTH], 1)
    ]


def _selected_files(trace: Mapping[str, Any]) -> list[str]:
    """Return the actual file scope selected by the cascade trace.

    The trace field ``counts.selected_files`` is only a count.  The ordered
    file identities live in ``file_candidates``; use the configured Kf to
    recover the compact file scope for cost and ceiling metrics.
    """
    config = trace.get("config") or {}
    k_files = int(config.get("k_files") or 0)
    candidates = trace.get("file_candidates") or []
    return [str(row["node_id"]) for row in candidates[:k_files]]


def _write_run(path: Path, qids: Sequence[str], runs: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in qids:
            handle.write(
                json.dumps(
                    {"qid": qid, "chunks": _compact_run(runs[qid])},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def _stage_metrics(
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    traces: Mapping[str, Mapping[str, Any]],
    corpus: Any,
) -> dict[str, float]:
    scope_recall: list[float] = []
    page_ceiling: list[float] = []
    scope_pages: list[float] = []
    selected_kf: list[float] = []
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        selected = _selected_files(traces[qid])
        selected_set = set(selected)
        scope_recall.append(
            len(selected_set & gold_files) / len(gold_files) if gold_files else 0.0
        )
        selected_pages = {
            page_id
            for file_id in selected_set
            for page_id in corpus.file_to_pages.get(file_id, [])
        }
        page_ceiling.append(
            len(selected_pages & gold_pages) / len(gold_pages) if gold_pages else 0.0
        )
        scope_pages.append(float(len(selected_pages)))
        selected_kf.append(float(len(selected)))
    return {
        "file_scope_recall": _mean(scope_recall),
        "page_pool_recall_ceiling": _mean(page_ceiling),
        "avg_scope_pages": _mean(scope_pages),
        "avg_kf": _mean(selected_kf),
    }


def _merge_adaptive_runs(
    qids: Sequence[str],
    selected_kf: Mapping[str, int],
    runs_by_kf: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        qid: list(runs_by_kf[selected_kf[qid]][qid])
        for qid in qids
    }


def _compare(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    base = {str(row["qid"]): row for row in baseline["per_query"]}
    cand = {str(row["qid"]): row for row in candidate["per_query"]}
    diffs = [
        float(cand[qid]["page_recall@10"]) - float(base[qid]["page_recall@10"])
        for qid in base
    ]
    return {
        "page_recall_delta_pp": 100.0 * _mean(diffs),
        "better_queries": int(sum(value > 0 for value in diffs)),
        "same_queries": int(sum(value == 0 for value in diffs)),
        "worse_queries": int(sum(value < 0 for value in diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--base-trace", type=Path, default=DEFAULT_BASE_TRACE)
    parser.add_argument("--base-run-dir", type=Path, default=DEFAULT_BASE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    if len(qids) != EXPECTED_QUERIES:
        raise RuntimeError(f"Expected {EXPECTED_QUERIES} queries, got {len(qids)}")

    page_ids = _page_vector_units(args.page_vector_dir)
    if len(page_ids) != EXPECTED_PAGES or len(set(page_ids)) != EXPECTED_PAGES:
        raise RuntimeError("Physics page inventory is not exactly 1,674 unique pages")
    file_gaps = _load_file_gaps(args.base_trace, qids)
    thresholds = _quantiles([file_gaps[qid] for qid in qids])
    quartiles: dict[str, str] = {}
    selected_kf: dict[str, int] = {}
    for qid in qids:
        quartile, kf = _choose_kf(file_gaps[qid], thresholds)
        quartiles[qid] = quartile
        selected_kf[qid] = kf

    corpus, page_index, file_indexes, fine_indexes, index_counts = _build_corpus_and_indexes(
        args.parsed_run, page_ids, fine_units=()
    )
    visual_scores, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    retriever = HierarchicalRetriever(
        corpus,
        page_index=page_index,
        file_indexes=file_indexes,
        fine_indexes=fine_indexes,
        visual_scores=visual_scores,
    )

    runs_by_kf: dict[int, dict[str, list[dict[str, Any]]]] = {}
    traces_by_kf: dict[int, dict[str, dict[str, Any]]] = {}
    retrieval_seconds_by_kf: dict[str, float] = {}
    for kf in KFS:
        config = CascadeConfig(
            name=f"adaptive-kf{kf}",
            file_representation="all_text",
            file_pool="max",
            k_files=kf,
            page_depth=PAGE_DEPTH,
            final_depth=FINAL_DEPTH,
            bm25_weight=0.70,
            parent_weight=0.15,
        )
        arm_started = time.perf_counter()
        runs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for question in questions:
            run, trace = retriever.retrieve(question.qid, question.query, config)
            runs[question.qid] = run
            traces[question.qid] = trace
        runs_by_kf[kf] = runs
        traces_by_kf[kf] = traces
        retrieval_seconds_by_kf[str(kf)] = round(time.perf_counter() - arm_started, 6)

    adaptive_runs = _merge_adaptive_runs(qids, selected_kf, runs_by_kf)
    adaptive_traces = {
        qid: traces_by_kf[selected_kf[qid]][qid] for qid in qids
    }
    fixed_runs = {kf: runs_by_kf[kf] for kf in KFS}
    fixed_traces = {kf: traces_by_kf[kf] for kf in KFS}

    qrels = benchmark.qrels()
    metrics: dict[str, dict[str, Any]] = {}
    stage: dict[str, dict[str, float]] = {}
    for kf in KFS:
        metrics[f"fixed_kf{kf}"] = _derived_metrics(runs_by_kf[kf], qids, qrels)
        stage[f"fixed_kf{kf}"] = _stage_metrics(qids, qrels, traces_by_kf[kf], corpus)
    metrics["adaptive_quartile"] = _derived_metrics(adaptive_runs, qids, qrels)
    stage["adaptive_quartile"] = _stage_metrics(qids, qrels, adaptive_traces, corpus)

    per_query: list[dict[str, Any]] = []
    adaptive_metric_rows = {
        row["qid"]: row for row in metrics["adaptive_quartile"]["per_query"]
    }
    fixed_metric_rows = {
        kf: {row["qid"]: row for row in metrics[f"fixed_kf{kf}"]["per_query"]}
        for kf in KFS
    }
    for qid in qids:
        row = adaptive_metric_rows[qid]
        selected = _selected_files(adaptive_traces[qid])
        gold_files = {_file_id(page_id) for page_id in qrels[qid]}
        selected_pages = {
            page_id
            for file_id in selected
            for page_id in corpus.file_to_pages.get(str(file_id), [])
        }
        gold_pages = set(qrels[qid])
        per_query.append(
            {
                "qid": qid,
                "file_gap_top1_top3": file_gaps[qid],
                "quartile": quartiles[qid],
                "selected_kf": selected_kf[qid],
                "gold_page_count": len(gold_pages),
                "gold_file_count": len(gold_files),
                "selected_file_count": len(selected),
                "selected_scope_pages": len(selected_pages),
                "adaptive_file_scope_recall": len(set(selected) & gold_files) / len(gold_files),
                "adaptive_page_pool_recall_ceiling": len(selected_pages & gold_pages) / len(gold_pages),
                "adaptive_page_recall@10": row["page_recall@10"],
                "adaptive_page_hit@10": row["page_hit@10"],
                "adaptive_file_recall@3": len(
                    set(row["top10_files_from_top100_pages"][:3]) & gold_files
                ) / len(gold_files),
                "fixed_kf6_page_recall@10": fixed_metric_rows[6][qid]["page_recall@10"],
                "fixed_kf6_page_hit@10": fixed_metric_rows[6][qid]["page_hit@10"],
                "fixed_kf6_file_recall@3": len(
                    set(fixed_metric_rows[6][qid]["top10_files_from_top100_pages"][:3]) & gold_files
                ) / len(gold_files),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    for kf in KFS:
        _write_run(runs_dir / f"fixed_kf{kf}.jsonl", qids, runs_by_kf[kf])
    _write_run(runs_dir / "adaptive_quartile.jsonl", qids, adaptive_runs)
    (args.output_dir / "per_query.jsonl").write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in per_query
        )
        + "\n",
        encoding="utf-8",
    )

    baseline_metrics = metrics["fixed_kf6"]
    adaptive_metrics = metrics["adaptive_quartile"]
    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "method": "file-gap quartile adaptive Kf",
        "config": {
            "file_gap_source": BASE_METHOD,
            "file_gap": "score(file_rank_1) - score(file_rank_3)",
            "quartile_thresholds": list(thresholds),
            "policy": {"Q1": 6, "Q2": 5, "Q3": 4, "Q4": 3},
            "page_depth": PAGE_DEPTH,
            "final_depth": FINAL_DEPTH,
            "file_representation": "all_text",
            "file_pool": "max",
            "bm25_weight": 0.70,
            "parent_weight": 0.15,
            "qrels_used_for_budget_selection": False,
        },
        "selected_quartiles": dict(Counter(quartiles.values())),
        "selected_kf_counts": dict(Counter(str(value) for value in selected_kf.values())),
        "arms": {
            name: {
                "retrieval_metrics": {
                    key: value for key, value in value.items() if key != "per_query"
                },
                "stage_metrics": stage[name],
                "retrieval_seconds": retrieval_seconds_by_kf.get(name.removeprefix("fixed_kf"), None),
            }
            for name, value in metrics.items()
            if name.startswith("fixed_")
        },
        "adaptive": {
            "retrieval_metrics": {
                key: value for key, value in adaptive_metrics.items() if key != "per_query"
            },
            "stage_metrics": stage["adaptive_quartile"],
            "retrieval_seconds_all_fixed_arms": sum(retrieval_seconds_by_kf.values()),
        },
        "comparison_to_fixed_kf6": _compare(baseline_metrics, adaptive_metrics),
        "sources": {
            "base_trace": str(args.base_trace),
            "parsed_run": str(args.parsed_run),
            "visual_cache": visual_meta,
            "index_counts": index_counts,
        },
        "validation": {
            "qids": len(qids),
            "pages": len(page_ids),
            "unique_pages": len(set(page_ids)),
            "qrels": len(qrels),
            "all_fixed_arms_have_all_qids": all(set(runs_by_kf[kf]) == set(qids) for kf in KFS),
            "adaptive_has_all_qids": set(adaptive_runs) == set(qids),
        },
        "wall_seconds": round(time.perf_counter() - started, 6),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def metric_line(name: str, metric: Mapping[str, Any], stage_metric: Mapping[str, float]) -> str:
        file_recall = metric["file_metrics_by_k"]["3"]["file_recall"]
        return (
            f"| {name} | {metric['ndcg@10']:.2f} | {metric['page_hit@10']:.2%} | "
            f"{metric['page_recall@10']:.2%} | {file_recall:.2%} | "
            f"{stage_metric['file_scope_recall']:.2%} | "
            f"{stage_metric['page_pool_recall_ceiling']:.2%} | "
            f"{stage_metric['avg_kf']:.2f} | {stage_metric['avg_scope_pages']:.2f} |"
        )

    lines = [
        "# Physics adaptive Kf by file-score quartile",
        "",
        "## Policy",
        "",
        "- Signal: file score gap `score(file 1) - score(file 3)` from the existing Kf=3 hierarchical trace.",
        f"- Thresholds Q1/Q2/Q3: `{thresholds[0]:.6f}`, `{thresholds[1]:.6f}`, `{thresholds[2]:.6f}`.",
        "- Q1 -> Kf=6; Q2 -> Kf=5; Q3 -> Kf=4; Q4 -> Kf=3.",
        "- Thresholds are score-only; qrels are not used to choose the budget.",
        "",
        f"Selected queries: `{dict(Counter(quartiles.values()))}`; selected Kf: `{dict(Counter(str(value) for value in selected_kf.values()))}`.",
        "",
        "## Results",
        "",
        "`File recall@3` is derived from the first three unique files in the first 100 final page results. `File scope recall` is the recall of all files retained before page ranking. `Page ceiling` is the gold-page recall available inside that retained scope.",
        "",
        "| Arm | nDCG@10 | Page hit@10 | Page recall@10 | Derived file recall@3 | File scope recall | Page ceiling | Avg Kf | Avg scope pages |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kf in KFS:
        name = f"Fixed Kf={kf}"
        lines.append(metric_line(name, metrics[f"fixed_kf{kf}"], stage[f"fixed_kf{kf}"]))
    lines.append(metric_line("Adaptive quartile", adaptive_metrics, stage["adaptive_quartile"]))
    lines += [
        "",
        "## Adaptive versus fixed Kf=6",
        "",
        f"- Page recall delta: **{report['comparison_to_fixed_kf6']['page_recall_delta_pp']:+.2f} pp**.",
        f"- Query-level outcome: **{report['comparison_to_fixed_kf6']['better_queries']} better**, **{report['comparison_to_fixed_kf6']['same_queries']} tied**, **{report['comparison_to_fixed_kf6']['worse_queries']} worse**.",
        "",
        "## Interpretation",
        "",
        "- The adaptive policy is intentionally cost-aware: difficult/low-gap queries receive more files, while high-gap queries receive fewer.",
        "- The comparison must use page recall and cost together; a small accuracy loss may be acceptable only if the scope reduction is material.",
        "- This is a score-only exploratory policy. Kf=4 and Kf=6 are computed with the same cached hierarchical scorer as the existing Kf=3/Kf=5 arms.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "thresholds": thresholds,
                "selected_kf_counts": dict(Counter(str(value) for value in selected_kf.values())),
                "adaptive_page_recall@10": adaptive_metrics["page_recall@10"],
                "fixed_kf6_page_recall@10": baseline_metrics["page_recall@10"],
                "adaptive_avg_scope_pages": stage["adaptive_quartile"]["avg_scope_pages"],
                "fixed_kf6_avg_scope_pages": stage["fixed_kf6"]["avg_scope_pages"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
