"""Score one experiment arm at the answer level.

    python -m research.harness.run_answer --run runs/vlm_fixed_rrf.jsonl --arm vlm_fixed_rrf

Input is a retrieval run: one JSON object per line, ``{"qid", "chunks": [...]}``,
where each chunk carries ``chunk_id``, ``doc_id``, ``text`` and ``score`` in rank
order. Both retrieval paths emit that shape -- the in-memory harness and the
production stack through Methods-Hub -- so the outcome variable is measured by
identical code regardless of which system produced the ranking.

Three metric levels are reported together, as designed:

1. answer    -- accuracy, by answer type / level / modality
2. context   -- gold-document recall in the text ACTUALLY packed into the prompt
3. cost      -- chunks and characters spent per question

Level 2 is the mediator, and reporting it is what makes a null result readable.
An arm that does not move accuracy has either failed to retrieve more gold
(a retrievability effect) or retrieved the same gold and been misread
(a readability effect). Accuracy alone cannot tell those apart.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import json
import time

from .benchmarks import load as load_benchmark
from .benchmarks.base import check_single_taxonomy
from .generate import GENERATOR_MODEL, MAX_CONTEXT_CHARS, ContextChunk, generate
from .judge import JUDGE_MODEL, judge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "benchmark"


def load_run(path: Path) -> dict[str, list[ContextChunk]]:
    runs: dict[str, list[ContextChunk]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        runs[str(payload["qid"])] = [
            ContextChunk(
                chunk_id=str(chunk.get("chunk_id", "")),
                doc_id=str(chunk["doc_id"]),
                text=str(chunk.get("text", "")),
                score=float(chunk.get("score", 0.0)),
            )
            for chunk in payload.get("chunks", [])
        ]
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Retrieval run JSONL.")
    parser.add_argument("--arm", required=True, help="Arm name, used in the output file.")
    parser.add_argument("--benchmark", default="ise", help="Benchmark adapter name.")
    parser.add_argument("--questions", default=str(DATA / "questions.jsonl"))
    parser.add_argument("--out", default=str(DATA / "answers"))
    parser.add_argument("--generator", default=GENERATOR_MODEL)
    parser.add_argument("--judge", default=JUDGE_MODEL)
    parser.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    parser.add_argument(
        "--subset",
        default="resolvable",
        choices=("resolvable", "text_only", "all"),
        help="Which questions to score. Stage-1 file recall uses the full "
        "resolvable set; answer scoring is bounded by what the arm can parse.",
    )
    args = parser.parse_args()

    benchmark = _load(args)
    questions = list(benchmark.questions())
    runs = load_run(Path(args.run))
    scored = [q for q in questions if q.qid in runs]
    missing = [q.qid for q in questions if q.qid not in runs]

    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for question in scored:
        generation = generate(
            question.qid,
            question.query,
            runs[question.qid],
            model=args.generator,
            max_chars=args.max_context_chars,
        )
        verdict = judge(
            question.query,
            question.answer,
            generation,
            question.answer_type,
            model=args.judge,
            generator_model=args.generator,
        )
        gold = benchmark.gold_docs(question.qid)
        records.append(
            {
                **asdict(generation),
                "correct": verdict.correct,
                "grader": verdict.grader,
                "judge_error": verdict.error,
                "answer_type": question.answer_type,
                "level": question.level,
                "modalities": list(question.modalities),
                "taxonomy": getattr(question, "taxonomy", ""),
                "gold_doc_ids": gold.flat(),
                # Set semantics with any-of groups: a directory reference is one
                # piece of evidence, not one requirement per file underneath it.
                "context_recall": gold.recall(generation.context_doc_ids),
                "context_hit": bool(set(gold.flat()) & set(generation.context_doc_ids)),
                "multi_gold": gold.required > 1,
                **_finer_granularities(benchmark, question.qid, generation),
            }
        )

    report = summarize(args.arm, records, missing, args, time.perf_counter() - started)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.arm}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"{args.arm}.per_question.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"arm              : {args.arm}")
    print(f"scored           : {report['scored']}  (missing from run: {len(missing)})")
    print(f"accuracy         : {report['accuracy']}")
    print(f"  abstained      : {report['abstain_rate']}")
    print(f"  errors         : {report['error_rate']}")
    print(f"context recall   : {report['context_recall']}  (hit {report['context_hit']})")
    print(f"context spend    : {report['chunks_mean']} chunks / {report['chars_mean']} chars")
    print(f"by answer type   : {report['accuracy_by_answer_type']}")
    print(f"by level         : {report['accuracy_by_level']}")
    print(f"by modality      : {report['accuracy_by_modality']}")
    print(f"artifacts        : {out_dir}")


def _load(args: argparse.Namespace) -> Any:
    if args.benchmark == "ise":
        return load_benchmark("ise", questions_path=args.questions, subset=args.subset)
    return load_benchmark(args.benchmark)


def _finer_granularities(
    benchmark: Any, qid: str, generation: Any
) -> dict[str, Any]:
    """Page and region recall, only where the source actually has those labels.

    Absent rather than zero when unavailable. A 0.0 would average into the
    report and read as "the retriever missed every region" when the truth is
    "this dataset has no region annotation" -- the iSE lake has file labels only.
    """
    extra: dict[str, Any] = {}
    pages = benchmark.gold_pages(qid)
    if pages:
        seen = {c.split("#page=")[-1] for c in generation.context_doc_ids if "#page=" in c}
        extra["page_recall"] = len(set(pages) & seen) / len(pages)

    regions = benchmark.gold_regions(qid)
    if regions:
        found = {r.region_id for r in regions if r.doc_id in set(generation.context_doc_ids)}
        extra["region_recall"] = len(found) / len(regions)
        # The per-modality split is the reason these datasets are worth running;
        # an aggregate score hides exactly the table/figure failure under test.
        extra["evidence_modalities"] = sorted({r.modality for r in regions})
    return extra


def summarize(
    arm: str,
    records: list[dict[str, Any]],
    missing: list[str],
    args: argparse.Namespace,
    seconds: float,
) -> dict[str, Any]:
    n = len(records) or 1

    def mean(key: str, rows: list[dict[str, Any]] | None = None) -> float:
        rows = records if rows is None else rows
        return round(sum(float(row[key]) for row in rows) / (len(rows) or 1), 4)

    by_group: dict[str, dict[str, float]] = {}
    for label, key in (("answer_type", "answer_type"), ("level", "level")):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            grouped[str(row[key])].append(row)
        by_group[label] = {name: mean("correct", rows) for name, rows in sorted(grouped.items())}

    # Refuses to flatten labels from different vocabularies into one table.
    # Averaging MMDocIR's ``text-only`` against its ``Pure-text (Plain-text)``
    # renders perfectly and compares two different things.
    check_single_taxonomy(
        (row.get("taxonomy", "") for row in records), context="accuracy_by_modality"
    )
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        for modality in row["modalities"] or ["unknown"]:
            by_modality[modality].append(row)

    multi = [row for row in records if row["multi_gold"]]

    return {
        "arm": arm,
        "generator": args.generator,
        "judge": args.judge,
        "max_context_chars": args.max_context_chars,
        "subset": args.subset,
        "scored": len(records),
        "missing_from_run": missing,
        "seconds": round(seconds, 1),
        "accuracy": mean("correct"),
        "abstain_rate": mean("abstained"),
        "error_rate": round(sum(1 for row in records if row["error"]) / n, 4),
        "context_recall": mean("context_recall"),
        "context_hit": mean("context_hit"),
        "chunks_mean": mean("chunks_used"),
        "chars_mean": mean("chars_used"),
        "accuracy_by_answer_type": by_group["answer_type"],
        "accuracy_by_level": by_group["level"],
        "accuracy_by_modality": {
            name: mean("correct", rows) for name, rows in sorted(by_modality.items())
        },
        # Multi-gold questions are where evidence pooling beats max-pooling, so
        # they are reported apart rather than averaged away.
        "multi_gold": {
            "n": len(multi),
            "accuracy": mean("correct", multi) if multi else 0.0,
            "context_recall": mean("context_recall", multi) if multi else 0.0,
        },
        "graders": dict(Counter(row["grader"] for row in records).most_common()),
    }


if __name__ == "__main__":
    main()
