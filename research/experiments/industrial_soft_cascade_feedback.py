"""Evaluate a protected soft cascade for ViDoRe Industrial.

The hard file cascade can improve the local page ordering but can also hide a
page that the global first-pass BM25 ranked highly.  This runner treats the
legacy second pass as a feedback candidate generator rather than an exclusive
replacement: it interleaves a small number of its pages with pages from the
original global BM25 run.  The operation is deterministic, uses no qrels at
retrieval time, and reuses runs produced by
``industrial_legacy_second_retrieval.py``.

This is a cheap approximation to a protected iterative retrieval loop:

    global page BM25 -> file10 + legacy second pass -> protected union/rerank

It is intentionally a separate research adapter and does not change the
production retrieval service.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _load_run,
    _paired_comparison,
    _safe_name,
    _stratified_folds,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_BASELINE_RUN = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_discovery_bm25_english.jsonl"
)
DEFAULT_SECOND_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_legacy_second_retrieval"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/industrial_soft_cascade_feedback"
)


def _interleave(
    baseline: Mapping[str, list[dict[str, Any]]],
    feedback: Mapping[str, list[dict[str, Any]]],
    qids: Sequence[str],
    *,
    feedback_take: int,
    baseline_take: int,
) -> dict[str, list[dict[str, Any]]]:
    """Build a protected union while preserving source rank order."""

    if feedback_take <= 0 or baseline_take <= 0:
        raise ValueError("interleave group sizes must be positive")
    output: dict[str, list[dict[str, Any]]] = {}
    for qid in qids:
        feedback_rows = feedback[qid]
        baseline_rows = baseline[qid]
        feedback_pos = 0
        baseline_pos = 0
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        while len(rows) < 100 and (
            feedback_pos < len(feedback_rows) or baseline_pos < len(baseline_rows)
        ):
            for source, take, source_name in (
                (feedback_rows, feedback_take, "legacy_feedback"),
                (baseline_rows, baseline_take, "global_bm25_protected"),
            ):
                consumed = 0
                while consumed < take and len(rows) < 100:
                    if source_name == "legacy_feedback":
                        if feedback_pos >= len(source):
                            break
                        source_row = source[feedback_pos]
                        source_rank = feedback_pos + 1
                        feedback_pos += 1
                    else:
                        if baseline_pos >= len(source):
                            break
                        source_row = source[baseline_pos]
                        source_rank = baseline_pos + 1
                        baseline_pos += 1
                    page_id = str(source_row["chunk_id"])
                    if page_id in seen:
                        continue
                    seen.add(page_id)
                    row = deepcopy(source_row)
                    row["rank"] = len(rows) + 1
                    row["score"] = round(1.0 / (len(rows) + 1), 8)
                    scores = dict(row.get("scores") or {})
                    scores["interleave_source"] = source_name
                    scores["interleave_source_rank"] = source_rank
                    scores["interleave_feedback_take"] = feedback_take
                    scores["interleave_baseline_take"] = baseline_take
                    row["scores"] = scores
                    rows.append(row)
                    consumed += 1
            if feedback_pos >= len(feedback_rows) and baseline_pos >= len(baseline_rows):
                break
        if len(rows) != 100:
            raise RuntimeError(f"Interleave produced {len(rows)} rows for {qid}, expected 100")
        output[qid] = rows
    return output


def _metric_for_qids(
    metrics: Mapping[str, Any], qids: set[str]
) -> tuple[float, float]:
    rows = [row for row in metrics["per_query"] if str(row["qid"]) in qids]
    if not rows:
        return 0.0, 0.0
    page = sum(float(row["page_recall@10"]) for row in rows) / len(rows)
    file_recall: list[float] = []
    for row in rows:
        candidates = row["top10_files_from_top100_pages"][:3]
        gold = set(row["gold_files"])
        file_recall.append(len(set(candidates) & gold) / len(gold) if gold else 0.0)
    return page, sum(file_recall) / len(file_recall)


def _oof_selection(
    qids: Sequence[str],
    questions: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int]],
    methods: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Mapping[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Select one fixed arm on four folds and evaluate it on the fifth."""

    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    oof_run: dict[str, list[dict[str, Any]]] = {}
    selected: list[dict[str, Any]] = []
    for fold, heldout in enumerate(folds):
        heldout_set = set(heldout)
        train = all_qids - heldout_set
        ranking = [
            (*_metric_for_qids(method["retrieval_metrics"], train), name)
            for name, method in methods.items()
        ]
        _, _, winner = max(ranking, key=lambda row: (row[0], row[1], row[2]))
        for qid in heldout:
            oof_run[qid] = runs[winner][qid]
        selected.append(
            {
                "fold": fold,
                "heldout_qids": heldout,
                "selected_method": winner,
                "train_page_recall@10": _metric_for_qids(
                    methods[winner]["retrieval_metrics"], train
                )[0],
                "train_file_recall@3": _metric_for_qids(
                    methods[winner]["retrieval_metrics"], train
                )[1],
            }
        )
    return {
        "folds": selected,
        "selected_method_counts": {
            name: sum(row["selected_method"] == name for row in selected)
            for name in methods
        },
        "run": oof_run,
    }


def _load_run_from(path: Path) -> dict[str, list[dict[str, Any]]]:
    run = _load_run(path)
    return {str(qid): rows for qid, rows in run.items()}


def _write_compact_run(
    path: Path,
    run: Mapping[str, Sequence[Mapping[str, Any]]],
    qids: Sequence[str],
    queries: Mapping[str, str],
) -> None:
    """Persist ranking provenance without duplicating page text.

    The source runs already contain the extracted page text.  Copying it for
    every sensitivity arm can consume multiple gigabytes and provides no
    additional evidence for this rank-only experiment.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for qid in qids:
            compact_rows = []
            for row in run[qid]:
                compact_rows.append(
                    {
                        "chunk_id": row["chunk_id"],
                        "doc_id": row.get("doc_id", row["chunk_id"]),
                        "score": row.get("score", 0.0),
                        "rank": row.get("rank", len(compact_rows) + 1),
                        "scores": row.get("scores", {}),
                    }
                )
            handle.write(
                json.dumps(
                    {"qid": qid, "query": queries[qid], "chunks": compact_rows},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, default=DEFAULT_BASELINE_RUN)
    parser.add_argument("--second-dir", type=Path, default=DEFAULT_SECOND_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = ViDoreV3(
        root=ROOT / "data/raw/benchmarks/vidore_v3",
        subset="industrial",
        language="english",
    )
    questions_list = sorted(
        list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1])
    )
    qids = [q.qid for q in questions_list]
    questions = {q.qid: q for q in questions_list}
    qrels = benchmark.qrels()
    baseline = _load_run_from(args.baseline_run)

    # All source runs are generated by the prior cache-only legacy second-pass
    # experiment.  The best gamma values are included as sensitivity arms; the
    # interleaving ratios are the actual protected-cascade hypothesis.
    source_names = {
        "legacy-second-file10-max-gamma0.1": "legacy-second-file10-max-gamma0_1.jsonl",
        "legacy-second-file10-max-gamma0.12": "legacy-second-file10-max-gamma0_12.jsonl",
        "legacy-second-file10-max-gamma0.15": "legacy-second-file10-max-gamma0_15.jsonl",
    }
    source_runs = {
        name: _load_run_from(args.second_dir / "runs" / filename)
        for name, filename in source_names.items()
    }

    methods: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "Cached page BM25 baseline": baseline
    }

    def add_method(name: str, run: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> None:
        metrics = _derived_metrics(run, qids, qrels)
        methods[name] = {
            "config": config,
            "retrieval_metrics": metrics,
            "comparison_to_baseline": _paired_comparison(
                methods["Cached page BM25 baseline"]["retrieval_metrics"], metrics
            ),
        }
        runs[name] = run

    methods["Cached page BM25 baseline"] = {
        "config": None,
        "retrieval_metrics": _derived_metrics(baseline, qids, qrels),
        "timing_seconds": {"retrieval": 0.0},
    }
    for name, source_run in source_runs.items():
        add_method(
            name,
            source_run,
            {"source": "cached legacy second pass", "interleave": None},
        )

    ratios = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (1, 2), (2, 2)]
    for source_name, source_run in source_runs.items():
        for feedback_take, baseline_take in ratios:
            name = f"soft-{_safe_name(source_name)}-interleave-{feedback_take}to{baseline_take}"
            add_method(
                name,
                _interleave(
                    baseline,
                    source_run,
                    qids,
                    feedback_take=feedback_take,
                    baseline_take=baseline_take,
                ),
                {
                    "source": source_name,
                    "feedback_take": feedback_take,
                    "baseline_take": baseline_take,
                    "depth": 100,
                },
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    oof = _oof_selection(qids, questions, qrels, methods, runs)
    oof_metrics = _derived_metrics(oof["run"], qids, qrels)
    baseline_metrics = methods["Cached page BM25 baseline"]["retrieval_metrics"]
    restricted_name = "soft-legacy-second-file10-max-gamma0_1-interleave-3to1"
    restricted_methods = {
        name: methods[name]
        for name in ("Cached page BM25 baseline", restricted_name)
    }
    restricted_runs = {
        name: runs[name] for name in ("Cached page BM25 baseline", restricted_name)
    }
    restricted_oof = _oof_selection(
        qids, questions, qrels, restricted_methods, restricted_runs
    )
    restricted_metrics = _derived_metrics(restricted_oof["run"], qids, qrels)
    restricted_comparison = _paired_comparison(baseline_metrics, restricted_metrics)
    report = {
        "dataset": "vidore_v3/industrial",
        "evaluation_language": "english",
        "queries": len(qids),
        "methods": methods,
        "cv": {
            "folds": oof["folds"],
            "selected_method_counts": oof["selected_method_counts"],
            "retrieval_metrics": oof_metrics,
            "comparison_to_baseline": _paired_comparison(baseline_metrics, oof_metrics),
        },
        "cv_restricted_two_arm": {
            "candidate_methods": list(restricted_methods),
            "folds": restricted_oof["folds"],
            "selected_method_counts": restricted_oof["selected_method_counts"],
            "retrieval_metrics": restricted_metrics,
            "comparison_to_baseline": restricted_comparison,
        },
        "sources": {
            "baseline_run": str(args.baseline_run),
            "legacy_second_dir": str(args.second_dir),
        },
        "notes": [
            "The feedback source is restricted to the top-10-file legacy second-pass run.",
            "Interleaving protects global BM25 candidates from hard file-stage exclusion.",
            "No qrels are used in retrieval; qrels are used only for evaluation and OOF arm selection.",
            "Full-set sensitivity rows are exploratory; OOF is the primary generalisation check.",
        ],
    }
    query_texts = {q.qid: q.query for q in questions_list}
    _write_compact_run(args.output_dir / "oof_run.jsonl", oof["run"], qids, query_texts)
    best_name = max(
        (name for name in methods if name != "Cached page BM25 baseline"),
        key=lambda name: (
            methods[name]["retrieval_metrics"]["page_recall@10"],
            methods[name]["retrieval_metrics"]["ndcg@10"],
        ),
    )
    _write_compact_run(
        args.output_dir / "runs" / f"{_safe_name(best_name)}.jsonl",
        runs[best_name],
        qids,
        query_texts,
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Industrial protected soft cascade",
        "",
        "Legacy second-pass candidates are interleaved with global page BM25 candidates to protect against hard file-stage misses.",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Δ page pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, method in methods.items():
        metrics = method["retrieval_metrics"]
        file_recall = metrics["file_metrics_by_k"]["3"]["file_recall"]
        delta = method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0)
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {file_recall:.2%} | {delta:+.2f} |"
        )
    cv_metrics = report["cv"]["retrieval_metrics"]
    cv_compare = report["cv"]["comparison_to_baseline"]
    restricted_cv_metrics = report["cv_restricted_two_arm"]["retrieval_metrics"]
    restricted_cv_compare = report["cv_restricted_two_arm"]["comparison_to_baseline"]
    lines += [
        "",
        "## Out-of-fold selection",
        "",
        f"- Selected methods: `{report['cv']['selected_method_counts']}`.",
        f"- OOF nDCG@10: **{cv_metrics['ndcg@10']:.2f}**; page recall@10: **{cv_metrics['page_recall@10']:.2%}** ({cv_compare['page_recall_delta_pp']:+.2f}pp vs baseline).",
        f"- OOF file recall@3: **{cv_metrics['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"- OOF paired bootstrap page-recall CI: **{cv_compare['page_recall_ci95_pp']} pp**, p={cv_compare['page_recall_p_two_sided']:.4f}.",
        "",
        "## Restricted two-arm OOF check",
        "",
        f"- Candidates: `Cached page BM25 baseline` vs `{restricted_name}`.",
        f"- Selected methods: `{report['cv_restricted_two_arm']['selected_method_counts']}`.",
        f"- OOF nDCG@10: **{restricted_cv_metrics['ndcg@10']:.2f}**; page recall@10: **{restricted_cv_metrics['page_recall@10']:.2%}** ({restricted_cv_compare['page_recall_delta_pp']:+.2f}pp vs baseline).",
        f"- OOF file recall@3: **{restricted_cv_metrics['file_metrics_by_k']['3']['file_recall']:.2%}**; paired bootstrap CI: **{restricted_cv_compare['page_recall_ci95_pp']} pp**, p={restricted_cv_compare['page_recall_p_two_sided']:.4f}.",
        "",
        "Full-set rows are a sensitivity screen; the OOF row is the primary selection result.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "page_recall@10": round(method["retrieval_metrics"]["page_recall@10"] * 100, 2),
                    "file_recall@3": round(method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2),
                    "delta_pp": round(method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2),
                }
                for name, method in methods.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
