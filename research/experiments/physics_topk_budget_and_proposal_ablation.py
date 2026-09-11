"""Fixed proposal-union budget experiment for Physics.

This experiment keeps the proposal depths fixed at the previously reported
policy (30 page candidates per page branch and 10 file-synopsis candidates)
and varies only the file/page budgets after proposal generation.

It also runs a small end-to-end ablation that removes one proposal signal at a
time.  The ablation is deliberately signal-level: when a branch is removed,
its proposal pages and its score are both removed.  No proposal depth is
increased and no qrels are used to build an index.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    HierarchyCorpus,
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_fielded_hierarchical_retrieval import (  # noqa: E402
    EXPECTED_PAGES,
    EXPECTED_QUERIES,
    FIELD_ORDER,
    FIELD_WEIGHTS,
    FILE_SYNOPSIS_DEPTH,
    PAGE_PROPOSAL_DEPTH,
    PARENT_WEIGHT,
    VISUAL_WEIGHT,
    FieldedBM25,
    QuerySignals,
    _build_field_records,
    _file_id,
    _index_scores,
    _load_visual_scores,
    _page_vector_units,
    _row,
)
from research.experiments.physics_hierarchical_retrieval import _ndcg_at_k  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_topk_budget_ablation"

FILE_BUDGETS = (3, 5, 10, 15, 20)
PAGE_BUDGETS = (10, 20, 50, 100)
BRANCHES = ("structural", "flat_bm25", "vsplade")


def _active_page_score(signals: QuerySignals, branches: set[str]) -> dict[str, float]:
    """Return the fixed page score using only active signal branches."""
    flat = normalise_scores(signals.flat_scores) if "flat_bm25" in branches else {}
    visual = normalise_scores(signals.visual_scores) if "vsplade" in branches else {}
    structural = normalise_scores(signals.field_section_scores) if "structural" in branches else {}

    if flat and visual:
        flat_weight, visual_weight = 1.0 - VISUAL_WEIGHT, VISUAL_WEIGHT
    elif flat:
        flat_weight, visual_weight = 1.0, 0.0
    elif visual:
        flat_weight, visual_weight = 0.0, 1.0
    elif structural:
        flat_weight, visual_weight = 0.0, 0.0
    else:
        return {}

    return {
        page_id: (
            flat_weight * flat.get(page_id, 0.0)
            + visual_weight * visual.get(page_id, 0.0)
            + (structural.get(page_id, 0.0) if not flat and not visual else 0.0)
        )
        for page_id in set(flat) | set(visual) | set(structural)
    }


def _retrieve_with_branches(
    corpus: HierarchyCorpus,
    signals: QuerySignals,
    *,
    branches: set[str],
    k_files: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run fixed-depth proposal, compact file selection and page ranking."""
    started = time.perf_counter()
    structural_ranked = (
        sort_scores(signals.field_section_scores)[:PAGE_PROPOSAL_DEPTH]
        if "structural" in branches
        else []
    )
    flat_ranked = (
        sort_scores(signals.flat_scores)[:PAGE_PROPOSAL_DEPTH]
        if "flat_bm25" in branches
        else []
    )
    visual_ranked = (
        sort_scores(signals.visual_scores)[:PAGE_PROPOSAL_DEPTH]
        if "vsplade" in branches
        else []
    )
    synopsis_ranked = (
        sort_scores(signals.file_scores)[:FILE_SYNOPSIS_DEPTH]
        if "structural" in branches
        else []
    )

    page_proposal_ids = {
        page_id for page_id, _ in structural_ranked
    } | {
        page_id for page_id, _ in flat_ranked
    } | {
        page_id for page_id, _ in visual_ranked
    }
    candidate_files = {_file_id(page_id) for page_id in page_proposal_ids}
    candidate_files.update(file_id for file_id, _ in synopsis_ranked)

    page_base = _active_page_score(signals, branches)
    evidence_values: dict[str, list[float]] = defaultdict(list)
    for page_id in page_proposal_ids:
        evidence_values[_file_id(page_id)].append(page_base.get(page_id, 0.0))
    compact_scores: dict[str, float] = {}
    for file_id, scores in evidence_values.items():
        compact_scores[file_id] = sum(sorted(scores, reverse=True)[:2])
    for file_id in candidate_files:
        compact_scores.setdefault(file_id, 0.0)

    compact_ranked = sorted(
        compact_scores.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    selected_files = {file_id for file_id, _ in compact_ranked[:k_files]}
    selected_page_ids = [
        page_id
        for page_id in corpus.page_order
        if _file_id(page_id) in selected_files
    ]

    selected_file_scores = {
        file_id: compact_scores.get(file_id, 0.0)
        for file_id in selected_files
    }
    parent_norm = normalise_scores(selected_file_scores)
    final_scores = {
        page_id: (1.0 - PARENT_WEIGHT) * page_base.get(page_id, 0.0)
        + PARENT_WEIGHT * parent_norm.get(_file_id(page_id), 0.0)
        for page_id in selected_page_ids
    }
    ranked_pages = sorted(
        final_scores.items(),
        key=lambda item: (-float(item[1]), item[0]),
    )
    run = [
        {
            "chunk_id": page_id,
            "doc_id": page_id,
            "text": corpus.pages[page_id].text,
            "score": round(float(score), 8),
            "rank": rank,
        }
        for rank, (page_id, score) in enumerate(ranked_pages[:100], 1)
    ]
    trace = {
        "branches": sorted(branches),
        "proposal_file_ids": sorted(candidate_files),
        "selected_files": [file_id for file_id, _ in compact_ranked[:k_files]],
        "page_ranked": [
            _row(page_id, score, rank)
            for rank, (page_id, score) in enumerate(ranked_pages[:100], 1)
        ],
        "structural_page_ids": [page_id for page_id, _ in structural_ranked],
        "flat_bm25_page_ids": [page_id for page_id, _ in flat_ranked],
        "vsplade_page_ids": [page_id for page_id, _ in visual_ranked],
        "synopsis_file_ids": [file_id for file_id, _ in synopsis_ranked],
        "counts": {
            "proposal_pages": len(page_proposal_ids),
            "proposal_files": len(candidate_files),
            "selected_files": len(selected_files),
            "pages_scanned": len(selected_page_ids),
            "page_ranked": len(ranked_pages),
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
    }
    return run, trace


def _metrics_at_k(
    run: Mapping[str, list[dict[str, Any]]],
    traces: Mapping[str, dict[str, Any]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    page_k: int,
    file_k: int,
    all_page_ids: Sequence[str],
) -> dict[str, float]:
    page_recall: list[float] = []
    page_hit: list[float] = []
    precision: list[float] = []
    ndcg: list[float] = []
    file_recall: list[float] = []
    file_hit: list[float] = []
    selected_page_coverage: list[float] = []
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        pages = [str(row["chunk_id"]) for row in run[qid][:page_k]]
        found_pages = set(pages) & gold_pages
        page_recall.append(len(found_pages) / len(gold_pages) if gold_pages else 0.0)
        page_hit.append(float(bool(found_pages)))
        precision.append(len(found_pages) / page_k if page_k else 0.0)
        ndcg.append(_ndcg_at_k(pages, qrels.get(qid, {}), page_k))

        selected = set(traces[qid].get("selected_files", [])[:file_k])
        found_files = selected & gold_files
        file_recall.append(len(found_files) / len(gold_files) if gold_files else 0.0)
        file_hit.append(float(bool(found_files)))

        selected_page_ids = {
            page_id
            for page_id in all_page_ids
            if _file_id(page_id) in selected
        }
        selected_page_coverage.append(
            len(selected_page_ids & gold_pages) / len(gold_pages)
            if gold_pages
            else 0.0
        )

    mean = lambda values: float(np.mean(values)) if values else 0.0
    return {
        "ndcg": 100.0 * mean(ndcg),
        "page_recall": mean(page_recall),
        "page_hit": mean(page_hit),
        "page_precision": mean(precision),
        "file_recall": mean(file_recall),
        "file_hit": mean(file_hit),
        "selected_page_coverage": mean(selected_page_coverage),
    }


def _proposal_metrics(
    traces: Mapping[str, dict[str, Any]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, float]:
    coverage: list[float] = []
    counts: list[float] = []
    pages: list[float] = []
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page_id) for page_id in gold_pages}
        proposal_files = set(traces[qid].get("proposal_file_ids", []))
        coverage.append(
            len(proposal_files & gold_files) / len(gold_files)
            if gold_files
            else 0.0
        )
        counts.append(float(len(proposal_files)))
        pages.append(float(traces[qid]["counts"]["proposal_pages"]))
    return {
        "proposal_file_coverage": float(np.mean(coverage)),
        "mean_proposal_files": float(np.mean(counts)),
        "median_proposal_files": float(np.median(counts)),
        "min_proposal_files": float(np.min(counts)),
        "max_proposal_files": float(np.max(counts)),
        "mean_proposal_pages": float(np.mean(pages)),
    }


def _build_query_states(
    benchmark: ViDoreV3,
    parsed_run: Path,
    page_vector_dir: Path,
    query_vector_dir: Path,
) -> tuple[HierarchyCorpus, list[Any], list[str], dict[str, QuerySignals], dict[str, Any]]:
    questions = sorted(
        list(benchmark.questions()),
        key=lambda question: int(question.qid.rsplit("::", 1)[1]),
    )
    qids = [question.qid for question in questions]
    page_ids = _page_vector_units(page_vector_dir)
    if len(qids) != EXPECTED_QUERIES or len(page_ids) != EXPECTED_PAGES:
        raise RuntimeError(f"Inventory mismatch: queries={len(qids)}, pages={len(page_ids)}")

    corpus = HierarchyCorpus.from_parsed_run(
        parsed_run,
        subset="physics",
        page_ids=page_ids,
    )
    field_records = _build_field_records(corpus, parsed_run)
    page_fields = (
        "title", "heading", "body", "caption", "table", "formula", "figure", "boilerplate"
    )
    page_context_fields = (*page_fields, "section_context")
    page_field_index = FieldedBM25.build(
        field_records.page_records,
        fields=page_fields,
        weights=FIELD_WEIGHTS,
    )
    page_context_index = FieldedBM25.build(
        field_records.page_records,
        fields=page_context_fields,
        weights=FIELD_WEIGHTS,
    )
    file_index = FieldedBM25.build(
        field_records.file_records,
        fields=page_context_fields,
        weights=FIELD_WEIGHTS,
    )
    flat_index = build_bm25(
        (page_id, corpus.pages[page_id].text)
        for page_id in corpus.page_order
    )
    visual_by_qid, visual_meta = _load_visual_scores(
        page_vector_dir,
        query_vector_dir,
        page_ids,
        qids,
    )
    states: dict[str, QuerySignals] = {}
    for question in questions:
        flat_scores = _index_scores(flat_index, question.query, top_k=len(page_ids))
        field_scores, field_components = page_field_index.score(question.query)
        context_scores, context_components = page_context_index.score(question.query)
        file_scores, file_components = file_index.score(question.query)
        states[question.qid] = QuerySignals(
            flat_scores=flat_scores,
            field_scores=field_scores,
            field_components=field_components,
            field_section_scores=context_scores,
            field_section_components=context_components,
            file_scores=file_scores,
            file_components=file_components,
            visual_scores=visual_by_qid[question.qid],
        )
    return corpus, questions, qids, states, visual_meta


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    started = time.perf_counter()
    benchmark = ViDoreV3(root=args.benchmark_root, subset="physics", language="french")
    corpus, questions, qids, states, visual_meta = _build_query_states(
        benchmark,
        args.parsed_run,
        args.page_vector_dir,
        args.query_vector_dir,
    )
    qrels = benchmark.qrels()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "proposal_depth_per_page_branch": PAGE_PROPOSAL_DEPTH,
        "file_synopsis_depth": FILE_SYNOPSIS_DEPTH,
        "file_budgets": list(FILE_BUDGETS),
        "page_budgets": list(PAGE_BUDGETS),
        "compact_strategy": "sum_top2",
        "visual_weight_when_both_active": VISUAL_WEIGHT,
        "parent_weight": PARENT_WEIGHT,
        "proposal_union_is_fixed": True,
        "qrels_used_for_indexing": False,
    }

    budget_rows: list[dict[str, Any]] = []
    per_query_rows: list[dict[str, Any]] = []
    full_branches = set(BRANCHES)
    for k_files in FILE_BUDGETS:
        runs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, dict[str, Any]] = {}
        for qid in qids:
            runs[qid], traces[qid] = _retrieve_with_branches(
                corpus,
                states[qid],
                branches=full_branches,
                k_files=k_files,
            )
        run_name = f"full_kf{k_files}"
        _write_jsonl(runs_dir / f"{run_name}.jsonl", [
            {"qid": qid, "results": runs[qid]} for qid in qids
        ])
        proposal = _proposal_metrics(traces, qids, qrels)
        for page_k in PAGE_BUDGETS:
            metric = _metrics_at_k(
                runs,
                traces,
                qids,
                qrels,
                page_k=page_k,
                file_k=k_files,
                all_page_ids=corpus.page_order,
            )
            budget_rows.append({
                "arm": run_name,
                "k_files": k_files,
                "k_pages": page_k,
                **metric,
                **proposal,
                "mean_selected_files": float(np.mean([
                    traces[qid]["counts"]["selected_files"] for qid in qids
                ])),
                "mean_pages_scanned": float(np.mean([
                    traces[qid]["counts"]["pages_scanned"] for qid in qids
                ])),
                "mean_query_retrieval_seconds": float(np.mean([
                    traces[qid]["timing_seconds"]["total"] for qid in qids
                ])),
            })
        for qid in qids:
            gold_pages = set(qrels.get(qid, {}))
            per_query_rows.append({
                "qid": qid,
                "arm": run_name,
                "proposal_files": traces[qid]["proposal_file_ids"],
                "selected_files": traces[qid]["selected_files"],
                "proposal_counts": traces[qid]["counts"],
                "page_ranks": [row["node_id"] for row in traces[qid]["page_ranked"]],
                "page_recall_at_k": {
                    str(page_k): (
                        len({str(row["chunk_id"]) for row in runs[qid][:page_k]} & gold_pages)
                        / len(gold_pages)
                        if gold_pages
                        else 0.0
                    )
                    for page_k in PAGE_BUDGETS
                },
            })

    ablation_rows: list[dict[str, Any]] = []
    ablations = {
        "full": set(BRANCHES),
        "without_structural": {"flat_bm25", "vsplade"},
        "without_flat_bm25": {"structural", "vsplade"},
        "without_vsplade": {"structural", "flat_bm25"},
    }
    for name, branches in ablations.items():
        runs = {}
        traces = {}
        for qid in qids:
            runs[qid], traces[qid] = _retrieve_with_branches(
                corpus,
                states[qid],
                branches=branches,
                k_files=3,
            )
        metric = _metrics_at_k(
            runs, traces, qids, qrels, page_k=10, file_k=3,
            all_page_ids=corpus.page_order,
        )
        metric100 = _metrics_at_k(
            runs, traces, qids, qrels, page_k=100, file_k=3,
            all_page_ids=corpus.page_order,
        )
        ablation_rows.append({
            "arm": name,
            "active_branches": sorted(branches),
            "page_recall@10": metric["page_recall"],
            "page_recall@100": metric100["page_recall"],
            "page_hit@10": metric["page_hit"],
            "file_recall@3": metric["file_recall"],
            "selected_page_coverage": metric["selected_page_coverage"],
            **_proposal_metrics(traces, qids, qrels),
            "mean_pages_scanned": float(np.mean([
                traces[qid]["counts"]["pages_scanned"] for qid in qids
            ])),
        })

    report = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(corpus.page_order),
        "files": len(corpus.file_order),
        "config": config,
        "visual_cache": visual_meta,
        "budget_rows": budget_rows,
        "proposal_ablation": ablation_rows,
        "timing_seconds": {
            "index_and_query_signal_build": round(time.perf_counter() - started, 6),
            "note": "Includes local field/BM25 index construction and cached V-SPLADE score loading; excludes parsing and model encoding.",
        },
        "notes": [
            "The proposal union stays fixed at top-30 structural pages + top-30 flat BM25 pages + top-30 V-SPLADE pages + top-10 structural file synopsis files.",
            "A plus sign in the proposal description means set union, not score addition.",
            "Budget rows are fixed full-set curves; no qrel-selected winner is reported.",
            "Page recall@K is computed over the top-K pages after compact file selection.",
            "File recall@Kf is computed over the selected top-Kf files, before ColVec.",
        ],
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(args.output_dir / "per_query.jsonl", per_query_rows)

    lines = [
        "# Physics top-K budget and proposal ablation",
        "",
        "Fixed proposal union: top-30 structural pages + top-30 flat BM25 pages + top-30 V-SPLADE pages + top-10 structural synopsis files.",
        "",
        "## File/page budget curves",
        "",
        "| Kf | Kp | Page recall@Kp | Page hit@Kp | File recall@Kf | Selected-page coverage | Proposal coverage | Mean proposal files | Mean pages scanned |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        lines.append(
            f"| {row['k_files']} | {row['k_pages']} | {row['page_recall']:.2%} | {row['page_hit']:.2%} | "
            f"{row['file_recall']:.2%} | {row['selected_page_coverage']:.2%} | "
            f"{row['proposal_file_coverage']:.2%} | {row['mean_proposal_files']:.2f} | "
            f"{row['mean_pages_scanned']:.2f} |"
        )
    lines += [
        "",
        "## Proposal branch ablation at Kf=3",
        "",
        "| Arm | Active branches | Page recall@10 | Page recall@100 | File recall@3 | Selected-page coverage | Proposal coverage | Mean proposal files |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['arm']} | {', '.join(row['active_branches'])} | {row['page_recall@10']:.2%} | "
            f"{row['page_recall@100']:.2%} | {row['file_recall@3']:.2%} | "
            f"{row['selected_page_coverage']:.2%} | {row['proposal_file_coverage']:.2%} | "
            f"{row['mean_proposal_files']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The budget curve varies only Kf and Kp; proposal depth is not tuned.",
        "Selected-page coverage is the useful pre-ColVec coverage metric when ColVec receives every page in the selected files; page recall@100 is only the current light ranker output.",
        "Proposal coverage is measured before compact file selection and is not file recall@Kf.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "budget_rows": len(budget_rows),
        "ablation_rows": len(ablation_rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
