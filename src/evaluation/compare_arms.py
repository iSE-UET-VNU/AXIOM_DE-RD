"""Compare arms on the subset every arm can reach.

    python -m src.evaluation.compare_arms data/benchmark/runs/*.report.json

Accuracy on an arm's *own* reachable set is not comparable across arms. A parser
that reaches more documents is scored on a different question set, and that set
is not necessarily easier or harder in a way we can separate -- so a delta
between two such numbers confounds coverage with quality.

Three columns, and only the third may carry a cross-arm claim:

1. **coverage** -- reachable / total. This *is* the parsing result.
2. **acc|own** -- recall on the arm's own reachable set. Descriptive only.
3. **acc|common** -- recall on the intersection of every arm's reachable set.

A parser answering 64/111 at 70% beats one answering 49/111 at 78%. Reporting
(1) beside (3) is what makes that visible; averaging them hides it.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import repo_root
from typing import Any
import argparse
import json

PROJECT_ROOT = repo_root(__file__)


def load_arms(paths: list[Path]) -> dict[str, dict[str, Any]]:
    arms: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index_id = payload.get("index_id") or Path(path).stem
        units = (payload.get("coverage") or {}).get("units_indexed")
        for arm, metrics in (payload.get("arms") or {}).items():
            arms[f"{index_id}::{arm}"] = {**metrics, "units_indexed": units}
    return arms


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def check_arm_sets(arms: dict[str, dict[str, Any]]) -> None:
    """Refuse an index whose arm set is a subset of another's.

    A crash partway through a run leaves the arms that already finished on disk.
    The ViDoRe dense run died on an embedder timeout and wrote bm25 for every
    subset and dense/rrf for none, which reads as a completed bm25-only sweep.
    Comparing that against a full sweep is a comparison across arms that did not
    all execute.
    """
    by_index: dict[str, set[str]] = {}
    for name in arms:
        index_id, _, arm = name.rpartition("::")
        by_index.setdefault(index_id or name, set()).add(arm)
    if len(by_index) < 2:
        return
    expected = set().union(*by_index.values())
    missing = {index: sorted(expected - present)
               for index, present in by_index.items() if present != expected}
    if missing:
        raise SystemExit(
            f"Arms are missing for some indexes: {missing}. Every index must carry "
            "the same arm set, or the comparison spans runs that did not all "
            "execute. Re-run the missing arms, or pass only the complete indexes."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    arms = load_arms([Path(p) for p in args.reports])
    if len(arms) < 2:
        raise SystemExit("Need at least two arms to compare.")

    stale = [n for n, m in arms.items() if "reachable_qids" not in m or "per_question" not in m]
    if stale:
        raise SystemExit(
            "These arms predate the coverage partition and carry no reachable "
            f"set or per-question scores: {stale}. Re-run them -- a comparison "
            "against an arm whose reachable set is unknown cannot be made honest."
        )

    check_arm_sets(arms)

    reach = {name: set(m["reachable_qids"]) for name, m in arms.items()}
    common = set.intersection(*reach.values())
    union = set.union(*reach.values())

    print(f"arms             : {len(arms)}")
    print(f"common reachable : {len(common)}      union: {len(union)}")
    if not common:
        raise SystemExit("Empty intersection: these arms share no reachable questions.")
    if len(common) < 0.5 * len(union):
        print(
            f"WARNING: the common subset is {len(common)}/{len(union)} of the union. "
            "Cross-arm claims rest on less than half the evidence."
        )

    print(
        f"\n{'arm':40} {'units':>8} {'coverage':>9} {'acc|own':>9} {'acc|common':>11}"
    )
    print("-" * 82)
    for name, metrics in sorted(arms.items()):
        scores = metrics["per_question"]
        own = mean([scores[q] for q in metrics["reachable_qids"] if q in scores])
        shared = mean([scores[q] for q in sorted(common) if q in scores])
        units = metrics.get("units_indexed")
        print(
            f"{name[-39:]:40} {units if units is not None else '?':>8} "
            f"{metrics.get('coverage', 0):9.4f} {_fmt(own):>9} {_fmt(shared):>11}"
        )

    unit_counts = {m.get("units_indexed") for m in arms.values()}
    if len(unit_counts) > 1:
        print(
            f"\nNOTE: arms indexed different corpus sizes {sorted(c for c in unit_counts if c)}. "
            "These are not the same pipeline with a different text source -- one "
            "arm is missing documents the other has."
        )
    print(
        "\nOnly acc|common may carry a cross-arm claim. coverage is the parsing "
        "result and belongs beside it, never averaged into it."
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
