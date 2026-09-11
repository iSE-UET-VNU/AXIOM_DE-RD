"""Data-skipping-inspired baselines for the Physics light retriever.

This is an offline, cache-only experiment.  It maps common lakehouse pruning
ideas onto the document hierarchy without pretending that a page retriever is
an analytical table scan:

* file metadata / manifest pruning -> file scores pooled from page regions;
* row groups / zone maps -> contiguous page regions with coarse scores;
* exact page verification -> the existing page BM25 + V-SPLADE score;
* soft pruning -> coarse scores are priors but all pages in selected files
  remain eligible;
* hard pruning -> only the best regions are eligible, and candidate recall is
  reported separately so a false negative is visible.

The experiment uses only the existing KDL + pdf-inspector and V-SPLADE
caches.  No qrels or modality labels are used during scoring.  The fixed arms
are intentionally small: 4-page and 8-page regions, with and without a hard
16-page-per-file budget.  The OOF selector chooses among these predeclared
arms only on training folds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import (  # noqa: E402
    HierarchyCorpus,
    HierarchyNode,
    aggregate_file_scores,
    build_bm25,
    normalise_scores,
    sort_scores,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _file_id,
    _index_scores,
    _load_run,
    _load_visual_scores,
    _paired_comparison,
    _page_number,
    _write_run,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
DEFAULT_BENCHMARK_ROOT = ROOT / "data/benchmark/vidore_v3"
DEFAULT_PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
DEFAULT_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_english_302q"
DEFAULT_BASELINE_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl"
DEFAULT_HIERARCHICAL_RUN = ROOT / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/oof_run.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_data_skipping_baselines"


@dataclass(frozen=True)
class Region:
    region_id: str
    file_id: str
    page_ids: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class SkipConfig:
    name: str
    region_pages: int
    file_k: int
    hard_region_budget_pages: int | None
    region_weight: float = 0.15
    parent_weight: float = 0.15
    page_bm25_weight: float = 0.60
    file_direct_weight: float = 0.50
    region_pool: str = "sum_top2"


def _regions(corpus: HierarchyCorpus, region_pages: int) -> list[Region]:
    """Build deterministic contiguous micro-partitions inside each file."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for page_id in corpus.page_order:
        by_file[_file_id(page_id)].append(page_id)

    output: list[Region] = []
    for file_id in corpus.file_order:
        pages = sorted(by_file.get(file_id, []), key=_page_number)
        for start in range(0, len(pages), region_pages):
            page_ids = tuple(pages[start : start + region_pages])
            if not page_ids:
                continue
            text = "\n".join(corpus.pages[page_id].text for page_id in page_ids).strip()
            output.append(
                Region(
                    region_id=f"{file_id}#region={start // region_pages}",
                    file_id=file_id,
                    page_ids=page_ids,
                    text=text,
                )
            )
    return output


def _load_qrels_and_questions(
    benchmark_root: Path,
) -> tuple[list[Any], list[str], dict[str, dict[str, int]]]:
    benchmark = ViDoreV3(root=benchmark_root, subset="physics", language="french")
    questions = sorted(
        list(benchmark.questions()), key=lambda question: int(question.qid.rsplit("::", 1)[1])
    )
    qids = [question.qid for question in questions]
    qrels = benchmark.qrels()
    if len(questions) != 302 or len(qrels) != 302:
        raise RuntimeError(f"Expected 302 Physics questions/qrels, got {len(questions)}/{len(qrels)}")
    return questions, qids, qrels


def _region_scores(
    regions: Sequence[Region],
    region_index: Any,
    query: str,
    page_scores: Mapping[str, float],
    visual_scores: Mapping[str, float],
    *,
    pool: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return lexical, visual and combined region scores."""
    region_bm25 = normalise_scores(
        _index_scores(region_index, query, top_k=len(regions))
    )
    by_region_visual: dict[str, float] = {}
    for region in regions:
        values = sorted(
            (max(0.0, float(visual_scores.get(page_id, 0.0))) for page_id in region.page_ids),
            reverse=True,
        )
        if pool == "max":
            by_region_visual[region.region_id] = values[0] if values else 0.0
        elif pool == "sum_top2":
            by_region_visual[region.region_id] = sum(values[:2])
        else:
            raise ValueError(f"Unsupported region pool: {pool}")
    region_visual = normalise_scores(by_region_visual)

    # This mirrors the fixed page alpha used by the strongest prior structural
    # screen.  It is a score at a coarser level, not a new learned parameter.
    combined = {
        region.region_id: 0.60 * region_bm25.get(region.region_id, 0.0)
        + 0.40 * region_visual.get(region.region_id, 0.0)
        for region in regions
    }
    return region_bm25, region_visual, combined


class DataSkippingRetriever:
    """File -> region -> exact page scorer."""

    def __init__(
        self,
        corpus: HierarchyCorpus,
        *,
        page_index: Any,
        file_index: Any,
        region_indexes: Mapping[int, Any],
        regions_by_size: Mapping[int, Sequence[Region]],
        visual_scores: Mapping[str, Mapping[str, float]],
    ) -> None:
        self.corpus = corpus
        self.page_index = page_index
        self.file_index = file_index
        self.region_indexes = dict(region_indexes)
        self.regions_by_size = {int(key): list(value) for key, value in regions_by_size.items()}
        self.visual_scores = visual_scores
        self.page_to_file = {page_id: _file_id(page_id) for page_id in corpus.page_order}

    def retrieve(
        self,
        qid: str,
        query: str,
        config: SkipConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        page_bm25 = _index_scores(self.page_index, query, top_k=len(self.corpus.page_order))
        page_bm25_norm = normalise_scores(page_bm25)
        visual = {
            page_id: float(score)
            for page_id, score in self.visual_scores.get(qid, {}).items()
            if page_id in self.corpus.pages
        }
        visual_norm = normalise_scores(visual)
        page_base = {
            page_id: config.page_bm25_weight * page_bm25_norm.get(page_id, 0.0)
            + (1.0 - config.page_bm25_weight) * visual_norm.get(page_id, 0.0)
            for page_id in self.corpus.page_order
        }

        regions = self.regions_by_size[config.region_pages]
        region_index = self.region_indexes[config.region_pages]
        region_bm25, region_visual, region_base = _region_scores(
            regions,
            region_index,
            query,
            page_base,
            visual_norm,
            pool=config.region_pool,
        )
        region_to_file = {region.region_id: region.file_id for region in regions}
        file_pool = aggregate_file_scores(
            region_base,
            region_to_file,
            self.corpus.file_order,
            config.region_pool,
        )
        file_pool_norm = normalise_scores(file_pool)
        file_direct_norm = normalise_scores(
            _index_scores(self.file_index, query, top_k=len(self.corpus.file_order))
        )
        file_scores = {
            file_id: config.file_direct_weight * file_direct_norm.get(file_id, 0.0)
            + (1.0 - config.file_direct_weight) * file_pool_norm.get(file_id, 0.0)
            for file_id in self.corpus.file_order
        }
        file_ranked = sort_scores(file_scores)
        selected_files = {file_id for file_id, _ in file_ranked[: config.file_k]}

        selected_regions: set[str] = set()
        region_counts: dict[str, int] = defaultdict(int)
        for file_id in selected_files:
            candidates = sorted(
                (region for region in regions if region.file_id == file_id),
                key=lambda region: (-region_base.get(region.region_id, 0.0), region.region_id),
            )
            if config.hard_region_budget_pages is None:
                selected_regions.update(region.region_id for region in candidates)
                continue
            used_pages = 0
            for region in candidates:
                if used_pages >= config.hard_region_budget_pages:
                    break
                selected_regions.add(region.region_id)
                region_counts[file_id] += 1
                used_pages += len(region.page_ids)

        allowed_pages = {
            page_id
            for region in regions
            if region.region_id in selected_regions
            for page_id in region.page_ids
        }
        region_by_page = {
            page_id: region
            for region in regions
            for page_id in region.page_ids
        }
        file_norm = normalise_scores(file_scores)
        final_scores: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for page_id in allowed_pages:
            region = region_by_page[page_id]
            region_score = region_base.get(region.region_id, 0.0)
            score = (
                (1.0 - config.region_weight - config.parent_weight) * page_base[page_id]
                + config.region_weight * region_score
                + config.parent_weight * file_norm.get(region.file_id, 0.0)
            )
            final_scores[page_id] = score
            components[page_id] = {
                "page_bm25": page_bm25_norm.get(page_id, 0.0),
                "vsplade_page": visual_norm.get(page_id, 0.0),
                "page_base": page_base[page_id],
                "region_bm25": region_bm25.get(region.region_id, 0.0),
                "region_visual": region_visual.get(region.region_id, 0.0),
                "region_base": region_score,
                "parent_file": file_norm.get(region.file_id, 0.0),
            }
        ranked = sort_scores(final_scores)[:100]
        run = [
            {
                "chunk_id": page_id,
                "doc_id": page_id,
                "text": self.corpus.pages[page_id].text,
                "score": round(float(score), 8),
                "rank": rank,
                "scores": {key: round(float(value), 8) for key, value in components[page_id].items()},
            }
            for rank, (page_id, score) in enumerate(ranked, 1)
        ]
        trace = {
            "qid": qid,
            "config": asdict(config),
            "selected_files": sorted(selected_files),
            "file_candidates": [file_id for file_id, _ in file_ranked],
            "selected_regions": sorted(selected_regions),
            "allowed_pages": sorted(allowed_pages),
            "region_ids_by_page": {
                page_id: region.region_id for page_id, region in region_by_page.items()
            },
            "region_counts": dict(region_counts),
            "counts": {
                "files_indexed": len(self.corpus.files),
                "pages_indexed": len(self.corpus.pages),
                "regions_indexed": len(regions),
                "selected_files": len(selected_files),
                "selected_regions": len(selected_regions),
                "allowed_pages": len(allowed_pages),
            },
        }
        return run, trace


def _metric_for_qids(
    metrics: Mapping[str, Any],
    qids: set[str],
) -> tuple[float, float]:
    rows = [row for row in metrics["per_query"] if row["qid"] in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    file_recall = []
    for row in rows:
        gold = set(row["gold_files"])
        found = set(row["top10_files_from_top100_pages"][:3])
        file_recall.append(len(found & gold) / len(gold) if gold else 0.0)
    return page, sum(file_recall) / len(file_recall)


def _oof_select(
    qids: Sequence[str],
    metrics_by_name: Mapping[str, Mapping[str, Any]],
    runs_by_name: Mapping[str, Mapping[str, list[dict[str, Any]]]],
    qrels: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    folds = [list(qids[index::5]) for index in range(5)]
    qid_set = set(qids)
    oof: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train_set = qid_set - heldout_set
        ranking = []
        for name, metrics in metrics_by_name.items():
            page, file = _metric_for_qids(metrics, train_set)
            ranking.append((page, file, name))
        _, _, winner = max(ranking, key=lambda item: (item[0], item[1], item[2]))
        for qid in heldout:
            oof[qid] = runs_by_name[winner][qid]
        selected.append(
            {
                "fold": fold,
                "heldout_qids": heldout,
                "selected_arm": winner,
                "train_page_recall@10": _metric_for_qids(metrics_by_name[winner], train_set)[0],
                "train_file_recall@3": _metric_for_qids(metrics_by_name[winner], train_set)[1],
            }
        )
    return oof, {
        "folds": selected,
        "selected_arm_counts": {
            name: sum(row["selected_arm"] == name for row in selected)
            for name in metrics_by_name
        },
    }


def _stage_metrics(
    traces: Mapping[str, dict[str, Any]],
    qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    file_recall: list[float] = []
    region_recall: list[float] = []
    page_candidate_recall: list[float] = []
    skip_fraction: list[float] = []
    for qid in qids:
        gold_pages = set(qrels.get(qid, {}))
        gold_files = {_file_id(page) for page in gold_pages}
        selected_files = set(traces[qid]["selected_files"])
        selected_regions = set(traces[qid]["selected_regions"])
        allowed_pages = set(traces[qid]["allowed_pages"])
        file_recall.append(len(selected_files & gold_files) / len(gold_files) if gold_files else 0.0)
        # A region is relevant when it contains a gold page.  This is the
        # document equivalent of row-group candidate recall.
        relevant_regions = {
            traces[qid]["region_ids_by_page"][page]
            for page in gold_pages
            if page in traces[qid]["region_ids_by_page"]
        }
        region_recall.append(
            len(selected_regions & relevant_regions) / len(relevant_regions)
            if relevant_regions else 0.0
        )
        page_candidate_recall.append(len(allowed_pages & gold_pages) / len(gold_pages) if gold_pages else 0.0)
        total = max(1, int(traces[qid]["counts"]["pages_indexed"]))
        skip_fraction.append(1.0 - len(allowed_pages) / total)
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "selected_file_recall": mean(file_recall),
        "selected_region_page_recall_proxy": mean(region_recall),
        "page_candidate_recall": mean(page_candidate_recall),
        "mean_page_skip_fraction": mean(skip_fraction),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    baseline = report["references"]["cached_page_bm25"]["retrieval_metrics"]
    hierarchical = report["references"]["current_hierarchical_oof"]["retrieval_metrics"]
    lines = [
        "# Physics data-skipping baselines",
        "",
        "Offline cache-only experiment mapping lakehouse pruning ideas to file -> region -> page retrieval.",
        "",
        "## Protocol",
        "",
        "- Dataset: **ViDoRe V3 Physics**, 302 French queries, 42 files, 1,674 pages.",
        "- V-SPLADE query cache: English translations scored against French qrels; this confound is retained.",
        "- Region: contiguous pages within one file; region sizes are fixed at 4 or 8 pages.",
        "- File stage: direct file BM25 plus pooled region score; hard file budget is fixed at 3.",
        "- Page stage: exact page BM25 + V-SPLADE score with region/file priors.",
        "- Hard arms use a fixed 16-page-per-selected-file region budget; candidate recall is reported separately.",
        "",
        "## Results",
        "",
        "| Arm | nDCG@10 | Page recall@10 | Page hit@10 | File recall@3 | Selected-file recall | Page candidate recall | Mean page skip | Δ vs hierarchical pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["methods"].items():
        metric = item["retrieval_metrics"]
        stage = item["stage_metrics"]
        delta = item.get("comparison_to_hierarchical", {}).get("page_recall_delta_pp", 0.0)
        lines.append(
            f"| {name} | {metric['ndcg@10']:.2f} | {metric['page_recall@10']:.2%} | "
            f"{metric['page_hit@10']:.2%} | {metric['file_metrics_by_k']['3']['file_recall']:.2%} | "
            f"{stage.get('selected_file_recall', 0.0):.2%} | {stage.get('page_candidate_recall', 0.0):.2%} | "
            f"{stage.get('mean_page_skip_fraction', 0.0):.2%} | {delta:+.2f} |"
        )
    lines += [
        "",
        "## References",
        "",
        f"- Cached page BM25: **{baseline['page_recall@10']:.2%}** page recall@10.",
        f"- Current hierarchical OOF: **{hierarchical['page_recall@10']:.2%}** page recall@10.",
        "- The requested 47.47% reference is not a Physics row in the local artifacts; it is present in the Industrial report as the Kf=10 hard BM25 cascade.",
        "",
        "## OOF selection",
        "",
        f"- OOF page recall@10: **{report['cv']['retrieval_metrics']['page_recall@10']:.2%}**.",
        f"- OOF nDCG@10: **{report['cv']['retrieval_metrics']['ndcg@10']:.2f}**.",
        f"- OOF file recall@3: **{report['cv']['retrieval_metrics']['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"- Selected arms: `{report['cv']['selected_arm_counts']}`.",
        "",
        "## Interpretation",
        "",
        "- Soft region priors test hierarchical metadata guidance without a false-negative page filter.",
        "- Hard arms are only useful if their page candidate recall remains high; skipped gold pages are irreversible at the next stage.",
        "- A positive page result with a materially lower file recall would be a ranking win but not necessarily a better discovery system.",
        "",
        "Per-query traces are in `traces.jsonl`; runs are in `runs/`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-run", type=Path, default=DEFAULT_PARSED_RUN)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--page-vector-dir", type=Path, default=DEFAULT_PAGE_VECTOR_DIR)
    parser.add_argument("--query-vector-dir", type=Path, default=DEFAULT_QUERY_VECTOR_DIR)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--hierarchical-run", type=Path, default=DEFAULT_HIERARCHICAL_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    started = time.perf_counter()

    questions, qids, qrels = _load_qrels_and_questions(args.benchmark_root)
    metadata = json.loads((args.page_vector_dir / "page_metadata.json").read_text(encoding="utf-8"))
    page_ids = [str(row["unit_id"]) for row in metadata]
    if len(page_ids) != 1674 or len(set(page_ids)) != 1674:
        raise RuntimeError(f"Expected 1,674 unique Physics pages, got {len(page_ids)}")
    corpus = HierarchyCorpus.from_parsed_run(args.parsed_run, subset="physics", page_ids=page_ids)
    page_index = build_bm25((page_id, corpus.pages[page_id].text) for page_id in corpus.page_order)
    file_index = build_bm25(corpus.file_texts("all_text").items())

    region_sizes = (4, 8)
    regions_by_size = {
        size: _regions(corpus, size) for size in region_sizes
    }
    region_indexes = {
        size: build_bm25((region.region_id, region.text) for region in regions)
        for size, regions in regions_by_size.items()
    }
    visual_scores, visual_meta = _load_visual_scores(
        args.page_vector_dir, args.query_vector_dir, page_ids, qids
    )
    retriever = DataSkippingRetriever(
        corpus,
        page_index=page_index,
        file_index=file_index,
        region_indexes=region_indexes,
        regions_by_size=regions_by_size,
        visual_scores=visual_scores,
    )

    configs = [
        SkipConfig("rowgroup4-soft", 4, 3, None),
        SkipConfig("rowgroup8-soft", 8, 3, None),
        SkipConfig("rowgroup4-hard16", 4, 3, 16),
        SkipConfig("rowgroup8-hard16", 8, 3, 16),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    methods: dict[str, dict[str, Any]] = {}
    runs_by_name: dict[str, dict[str, list[dict[str, Any]]]] = {}
    traces_by_name: dict[str, dict[str, dict[str, Any]]] = {}
    for config in configs:
        run_by_qid: dict[str, list[dict[str, Any]]] = {}
        trace_by_qid: dict[str, dict[str, Any]] = {}
        for question in questions:
            run, trace = retriever.retrieve(question.qid, question.query, config)
            run_by_qid[question.qid] = run
            trace_by_qid[question.qid] = trace
        metrics = _derived_metrics(run_by_qid, qids, qrels)
        stage = _stage_metrics(trace_by_qid, qids, qrels)
        runs_by_name[config.name] = run_by_qid
        traces_by_name[config.name] = trace_by_qid
        methods[config.name] = {
            "config": asdict(config),
            "retrieval_metrics": metrics,
            "stage_metrics": stage,
        }
        _write_run(
            runs_dir / f"{config.name}.jsonl",
            run_by_qid,
            qids,
            queries={question.qid: question.query for question in questions},
        )

    baseline_run = _load_run(args.baseline_run)
    hierarchical_run = _load_run(args.hierarchical_run)
    references = {
        "cached_page_bm25": {"retrieval_metrics": _derived_metrics(baseline_run, qids, qrels)},
        "current_hierarchical_oof": {"retrieval_metrics": _derived_metrics(hierarchical_run, qids, qrels)},
    }
    for item in methods.values():
        item["comparison_to_bm25"] = _paired_comparison(
            references["cached_page_bm25"]["retrieval_metrics"], item["retrieval_metrics"]
        )
        item["comparison_to_hierarchical"] = _paired_comparison(
            references["current_hierarchical_oof"]["retrieval_metrics"], item["retrieval_metrics"]
        )

    oof_run, cv_meta = _oof_select(qids, {name: item["retrieval_metrics"] for name, item in methods.items()}, runs_by_name, qrels)
    cv_metrics = _derived_metrics(oof_run, qids, qrels)
    cv_trace: dict[str, dict[str, Any]] = {}
    for fold in cv_meta["folds"]:
        for qid in fold["heldout_qids"]:
            cv_trace[qid] = traces_by_name[fold["selected_arm"]][qid]
    cv = {
        **cv_meta,
        "retrieval_metrics": cv_metrics,
        "stage_metrics": _stage_metrics(cv_trace, qids, qrels),
        "comparison_to_bm25": _paired_comparison(references["cached_page_bm25"]["retrieval_metrics"], cv_metrics),
        "comparison_to_hierarchical": _paired_comparison(references["current_hierarchical_oof"]["retrieval_metrics"], cv_metrics),
    }
    _write_run(args.output_dir / "oof_run.jsonl", oof_run, qids, queries={question.qid: question.query for question in questions})
    with (args.output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for name, traces in traces_by_name.items():
            for qid in qids:
                handle.write(json.dumps({"arm": name, **traces[qid]}, ensure_ascii=False) + "\n")

    report: dict[str, Any] = {
        "dataset": "vidore_v3/physics",
        "queries": len(qids),
        "pages": len(page_ids),
        "files": len(corpus.files),
        "references": references,
        "methods": methods,
        "cv": cv,
        "sources": {
            "parsed_run": str(args.parsed_run),
            "baseline_run": str(args.baseline_run),
            "hierarchical_run": str(args.hierarchical_run),
            "visual": visual_meta,
            "region_sizes": list(region_sizes),
            "region_counts": {str(size): len(regions) for size, regions in regions_by_size.items()},
        },
        "timing_seconds": {"total": round(time.perf_counter() - started, 6)},
        "notes": [
            "All scoring is qrel-free; qrels are used only after retrieval for evaluation.",
            "V-SPLADE query vectors are cached English translations while Physics qrels and BM25 queries are French.",
            "Hard region arms are candidate-pruning experiments and are not assumed safe without candidate-recall evidence.",
            "The requested 47.47% local artifact row belongs to Industrial, not Physics; Physics references are reported explicitly.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output_dir),
        "methods": {
            name: {
                "page_recall@10": round(item["retrieval_metrics"]["page_recall@10"] * 100, 2),
                "file_recall@3": round(item["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
                "page_candidate_recall": round(item["stage_metrics"]["page_candidate_recall"] * 100, 2),
            }
            for name, item in methods.items()
        },
        "oof": {
            "page_recall@10": round(cv_metrics["page_recall@10"] * 100, 2),
            "file_recall@3": round(cv_metrics["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
            "selected_arm_counts": cv["selected_arm_counts"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
