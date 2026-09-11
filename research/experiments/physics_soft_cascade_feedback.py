"""Evaluate protected soft-cascade feedback on ViDoRe Physics.

This is the Physics counterpart of the Industrial protected-cascade
experiment. It reuses the existing French BM25, weighted BM25+V-SPLADE,
Kf=3 V-SPLADE cascade and Kf=3 legacy-second runs. No rendering, encoding or
network call is performed. The experiment tests whether interleaving global
page candidates is still useful when the Physics cascade already has a strong
visual signal.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.industrial_soft_cascade_feedback import (  # noqa: E402
    _interleave,
    _write_compact_run,
)
from research.experiments.physics_hierarchical_retrieval import (  # noqa: E402
    _derived_metrics,
    _load_run,
    _paired_comparison,
    _safe_name,
    _stratified_folds,
)
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "data/benchmark/vidore_v3/results/physics_soft_cascade_feedback"
)


def _metric_for_qids(metrics: Mapping[str, Any], qids: set[str]) -> tuple[float, float]:
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
    folds = _stratified_folds(qids, questions, qrels)
    all_qids = set(qids)
    output: dict[str, list[dict[str, Any]]] = {}
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
            output[qid] = runs[winner][qid]
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
        "run": output,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    benchmark = ViDoreV3(
        root=ROOT / "data/raw/benchmarks/vidore_v3",
        subset="physics",
        language="french",
    )
    questions_list = sorted(
        list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1])
    )
    qids = [q.qid for q in questions_list]
    questions = {q.qid: q for q in questions_list}
    qrels = benchmark.qrels()
    result_root = ROOT / "data/benchmark/vidore_v3/results"
    paths = {
        "PDF-inspector + BM25 baseline": result_root
        / "physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
        "Global BM25 + V-SPLADE weighted fusion": result_root
        / "physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl",
        "Full cascade Kf=3": result_root
        / "physics_legacy_second_file_budget/runs/full-cascade-kf3.jsonl",
        "Full cascade Kf=3 + legacy second": result_root
        / "physics_legacy_second_file_budget/runs/full-cascade-kf3-legacy-second-selected-files-max-gamma0_25.jsonl",
    }
    runs = {name: _load_run(path) for name, path in paths.items()}
    for name, run in runs.items():
        # A hard Kf=3 cascade can contain fewer than 100 pages when the
        # selected files have a small total page count.  This is expected and
        # the protected interleave fills the remaining depth from its global
        # source.  Every source must still provide enough rows for page@10.
        if set(run) != set(qids) or any(len(run[qid]) < 10 for qid in qids):
            raise RuntimeError(f"{name} does not cover all 302 qids at depth >= 10")

    methods: dict[str, dict[str, Any]] = {}

    def add_method(
        name: str,
        run: dict[str, list[dict[str, Any]]],
        config: dict[str, Any] | None,
    ) -> None:
        metrics = _derived_metrics(run, qids, qrels)
        methods[name] = {
            "config": config,
            "retrieval_metrics": metrics,
        }
        if name != "PDF-inspector + BM25 baseline":
            methods[name]["comparison_to_baseline"] = _paired_comparison(
                methods["PDF-inspector + BM25 baseline"]["retrieval_metrics"], metrics
            )

    add_method("PDF-inspector + BM25 baseline", runs["PDF-inspector + BM25 baseline"], None)
    add_method(
        "Global BM25 + V-SPLADE weighted fusion",
        runs["Global BM25 + V-SPLADE weighted fusion"],
        {"source": "cached weighted fusion", "bm25_alpha": 0.70},
    )
    add_method(
        "Full cascade Kf=3",
        runs["Full cascade Kf=3"],
        {"source": "cached Kf=3 V-SPLADE cascade"},
    )
    add_method(
        "Full cascade Kf=3 + legacy second",
        runs["Full cascade Kf=3 + legacy second"],
        {"source": "cached Kf=3 cascade + legacy second", "gamma": 0.25},
    )

    soft_runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ratios = [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
    source = runs["Full cascade Kf=3 + legacy second"]
    for base_name in (
        "PDF-inspector + BM25 baseline",
        "Global BM25 + V-SPLADE weighted fusion",
    ):
        for feedback_take, baseline_take in ratios:
            name = (
                f"soft-{_safe_name(base_name)}-interleave-"
                f"{feedback_take}to{baseline_take}"
            )
            soft_runs[name] = _interleave(
                runs[base_name],
                source,
                qids,
                feedback_take=feedback_take,
                baseline_take=baseline_take,
            )
            add_method(
                name,
                soft_runs[name],
                {
                    "feedback_source": "Full cascade Kf=3 + legacy second",
                    "protected_source": base_name,
                    "feedback_take": feedback_take,
                    "baseline_take": baseline_take,
                },
            )

    all_runs = {**runs, **soft_runs}
    oof = _oof_selection(qids, questions, qrels, methods, all_runs)
    oof_metrics = _derived_metrics(oof["run"], qids, qrels)
    baseline_metrics = methods["PDF-inspector + BM25 baseline"]["retrieval_metrics"]
    report = {
        "dataset": "vidore_v3/physics",
        "evaluation_language": "french",
        "queries": len(qids),
        "pages": 1674,
        "files": 42,
        "methods": methods,
        "cv": {
            "folds": oof["folds"],
            "selected_method_counts": oof["selected_method_counts"],
            "retrieval_metrics": oof_metrics,
            "comparison_to_baseline": _paired_comparison(baseline_metrics, oof_metrics),
        },
        "sources": {name: str(path) for name, path in paths.items()},
        "notes": [
            "The protected source is the cached V-SPLADE Kf=3 cascade plus legacy second retrieval.",
            "Interleaving is a rank-level protected union; duplicate pages are removed and depth remains 100.",
            "No qrels or modality labels are used during retrieval. Full-set rows are sensitivity results; OOF is the primary check.",
            "V-SPLADE query vectors are the existing cache and may use English translations against French qrels.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runs").mkdir(exist_ok=True)
    query_texts = {q.qid: q.query for q in questions_list}
    best_name = max(
        (name for name in methods if name != "PDF-inspector + BM25 baseline"),
        key=lambda name: (
            methods[name]["retrieval_metrics"]["page_recall@10"],
            methods[name]["retrieval_metrics"]["ndcg@10"],
        ),
    )
    _write_compact_run(args.output_dir / "runs" / f"{_safe_name(best_name)}.jsonl", all_runs[best_name], qids, query_texts)
    _write_compact_run(args.output_dir / "oof_run.jsonl", oof["run"], qids, query_texts)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Physics protected soft cascade",
        "",
        "Interleaving global page candidates with the cached Kf=3 V-SPLADE + legacy second cascade.",
        "",
        "| Method | nDCG@10 | Page hit@10 | Page recall@10 | File recall@3 | Δ page pp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, method in methods.items():
        metrics = method["retrieval_metrics"]
        delta = method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0)
        lines.append(
            f"| {name} | {metrics['ndcg@10']:.2f} | {metrics['page_hit@10']:.2%} | "
            f"{metrics['page_recall@10']:.2%} | {metrics['file_metrics_by_k']['3']['file_recall']:.2%} | {delta:+.2f} |"
        )
    cv = report["cv"]
    cm = cv["retrieval_metrics"]
    cc = cv["comparison_to_baseline"]
    lines += [
        "",
        "## Out-of-fold selection",
        "",
        f"- Selected methods: `{cv['selected_method_counts']}`.",
        f"- OOF nDCG@10: **{cm['ndcg@10']:.2f}**; page recall@10: **{cm['page_recall@10']:.2%}** ({cc['page_recall_delta_pp']:+.2f}pp vs baseline).",
        f"- OOF file recall@3: **{cm['file_metrics_by_k']['3']['file_recall']:.2%}**.",
        f"- OOF paired bootstrap page-recall CI: **{cc['page_recall_ci95_pp']} pp**, p={cc['page_recall_p_two_sided']:.4f}.",
        "",
        "This Physics comparison is separate from Industrial and retains the English-query V-SPLADE cache caveat.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: {"page_recall@10": round(method["retrieval_metrics"]["page_recall@10"] * 100, 2), "file_recall@3": round(method["retrieval_metrics"]["file_metrics_by_k"]["3"]["file_recall"] * 100, 2), "delta_pp": round(method.get("comparison_to_baseline", {}).get("page_recall_delta_pp", 0.0), 2)} for name, method in methods.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
