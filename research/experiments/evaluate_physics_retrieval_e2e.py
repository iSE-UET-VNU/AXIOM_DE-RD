"""Evaluate retrieval arms with a common ViDoRe Physics answer-generation loop.

This is a research adapter.  It intentionally reuses the existing cached
retrieval runs and the same generator/judge protocol as the valid custom
Physics E2E baseline.  Each arm is checkpointed so transient API failures do
not discard completed queries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log2
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.evaluation.benchmarks import load
from src.evaluation.benchmarks.vidore_v3_judge import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    ViDoreVerdict,
    parse_judgment,
    render_documents,
    score,
)
from src.evaluation.llm import complete
from src.evaluation.model_guard import assert_real


OUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_retrieval_e2e"
GENERATOR_MODEL = "deepseek/deepseek-v4-flash"
JUDGE_MODEL = "openai/gpt-4o"
DEFAULT_WORKERS = 12
DEFAULT_TOP_K = 10

ARMS: dict[str, Path] = {
    "bm25": ROOT
    / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/bm25_french_bm25-french_vs-english.jsonl",
    "weighted_bm25_vsplade": ROOT
    / "data/benchmark/vidore_v3/results/physics_vsplade_bm25_fusion/weighted_french_bm25-french_vs-english.jsonl",
    "full_cascade_kf3": ROOT
    / "data/benchmark/vidore_v3/results/physics_hierarchical_retrieval/runs/cascade-all_text-max-kf3-fusion.jsonl",
    "legacy_second_gamma025": ROOT
    / "data/benchmark/vidore_v3/results/physics_legacy_second_retrieval/runs/legacy-second-page100-max-gamma0_25.jsonl",
}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid", ""))
            if not qid:
                raise ValueError(f"Missing qid in {path}:{line_number}")
            if qid in runs:
                raise ValueError(f"Duplicate qid {qid!r} in {path}")
            chunks = row.get("chunks")
            if not isinstance(chunks, list):
                raise ValueError(f"Missing chunks list for {qid!r} in {path}")
            runs[qid] = chunks
    return runs


def _page_id(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or chunk.get("page_id") or "")


def _page_file(page_id: str) -> str:
    return page_id.split("#page=", 1)[0]


def _score_of(chunk: dict[str, Any], fallback: int) -> float:
    for key in ("score", "final_score", "rerank_score"):
        value = chunk.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(-fallback)


def _retrieval_metrics(
    run: dict[str, list[dict[str, Any]]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    """Compute the fixed Physics retrieval diagnostics from a JSONL run."""

    page_hits = []
    page_recalls = []
    file_hits = []
    file_recalls = []

    for qid, gold in qrels.items():
        chunks = run[qid]
        top10 = [_page_id(chunk) for chunk in chunks[:10]]
        top10_set = set(top10)
        gold_pages = set(gold)
        hit_count = len(top10_set & gold_pages)
        page_hits.append(float(hit_count > 0))
        page_recalls.append(hit_count / max(1, len(gold_pages)))

        selected_files: list[str] = []
        for chunk in chunks[:100]:
            page_id = _page_id(chunk)
            file_id = _page_file(page_id)
            if file_id and file_id not in selected_files:
                selected_files.append(file_id)
            if len(selected_files) == 3:
                break
        gold_files = {_page_file(page_id) for page_id in gold_pages}
        file_hit_count = len(set(selected_files) & gold_files)
        file_hits.append(float(file_hit_count > 0))
        file_recalls.append(file_hit_count / max(1, len(gold_files)))

    # Keep nDCG implementation local so this evaluator does not depend on the
    # evaluator CLI's run-file shape.  Scores are used only for ordering.
    ndcgs = []
    for qid, gold in qrels.items():
        ranked = sorted(
            enumerate(run[qid][:10]),
            key=lambda item: _score_of(item[1], item[0]),
            reverse=True,
        )
        dcg = 0.0
        for rank, (_, chunk) in enumerate(ranked, 1):
            relevance = gold.get(_page_id(chunk), 0)
            dcg += (2.0**relevance - 1.0) / log2(rank + 1.0)
        ideal = sorted(gold.values(), reverse=True)[:10]
        idcg = sum(
            (2.0**relevance - 1.0) / log2(rank + 1.0)
            for rank, relevance in enumerate(ideal, 1)
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)

    return {
        "n_queries": float(len(qrels)),
        "page_hit_at10": 100.0 * sum(page_hits) / len(page_hits),
        "page_recall_at10": 100.0 * sum(page_recalls) / len(page_recalls),
        "file_hit_at3": 100.0 * sum(file_hits) / len(file_hits),
        "file_recall_at3": 100.0 * sum(file_recalls) / len(file_recalls),
        "ndcg_at10": 100.0 * sum(ndcgs) / len(ndcgs),
    }


def _make_contexts(
    run: dict[str, list[dict[str, Any]]],
    queries: dict[str, Any],
    qrels: dict[str, dict[str, int]],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for qid, query in queries.items():
        chunks = run[qid][:top_k]
        documents = []
        page_ids = []
        total_chars = 0
        for rank, chunk in enumerate(chunks, 1):
            page_id = _page_id(chunk)
            text = str(chunk.get("text") or chunk.get("content") or "").strip()
            if not page_id or not text:
                continue
            page_ids.append(page_id)
            total_chars += len(text)
            documents.append(text)
        gold_pages = set(qrels[qid])
        contexts[qid] = {
            "qid": qid,
            "query": str(query.query),
            "gold_answer": str(query.answer),
            "context": render_documents(documents),
            "context_page_ids": page_ids,
            "context_pages": len(page_ids),
            "context_chars": total_chars,
            "gold_page_hit": bool(set(page_ids) & gold_pages),
            "gold_pages_in_context": len(set(page_ids) & gold_pages),
        }
    return contexts


def _identity(arm: str, run_path: Path, top_k: int, workers: int) -> dict[str, Any]:
    return {
        "arm": arm,
        "run_path": str(run_path),
        "top_k": top_k,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "prompt": ANSWER_PROMPT,
        "workers": workers,
    }


def _load_records(checkpoint: Path, identity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not checkpoint.exists():
        return {}
    state = _json_load(checkpoint)
    stored_identity = dict(state.get("identity") or {})
    # Worker count affects only scheduling; it must not invalidate a resumable
    # answer/judge checkpoint.
    stored_identity.pop("workers", None)
    expected_identity = dict(identity)
    expected_identity.pop("workers", None)
    if stored_identity != expected_identity:
        raise ValueError(
            f"Checkpoint identity mismatch: {checkpoint}. Remove it only if you intend to restart this arm."
        )
    return {str(row["qid"]): row for row in state.get("records", [])}


def _save_checkpoint(path: Path, identity: dict[str, Any], records: dict[str, dict[str, Any]]) -> None:
    _json_dump(
        path,
        {
            "identity": identity,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": list(records.values()),
        },
    )


def _generate_one(row: dict[str, Any]) -> tuple[str, str, str | None, float]:
    started = time.perf_counter()
    try:
        prompt = ANSWER_PROMPT.format(documents=row["context"], query=row["query"])
        answer = complete(
            GENERATOR_MODEL,
            prompt,
            temperature=0.0,
            max_output_tokens=512,
            timeout=90.0,
            max_retries=1,
        ).strip()
        return row["qid"], answer, None, time.perf_counter() - started
    except Exception as exc:  # API failures are recorded per query and retried on resume.
        return row["qid"], "", f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def _judge_one(row: dict[str, Any]) -> tuple[str, str, str, float]:
    started = time.perf_counter()
    try:
        reply = complete(
            JUDGE_MODEL,
            JUDGE_PROMPT.format(
                query=row["query"],
                true_answer=row["gold_answer"],
                test_answer=row["answer"],
            ),
            temperature=0.0,
            max_output_tokens=256,
            timeout=90.0,
            max_retries=1,
        )
        return row["qid"], parse_judgment(reply), "", time.perf_counter() - started
    except Exception as exc:  # Keep errors visible; retry on a later invocation.
        return row["qid"], "Incorrect", f"{type(exc).__name__}: {exc}", time.perf_counter() - started


def _evaluate_arm(
    arm: str,
    run_path: Path,
    queries: dict[str, Any],
    qrels: dict[str, dict[str, int]],
    top_k: int,
    workers: int,
    output_dir: Path,
) -> dict[str, Any]:
    run = _read_jsonl(run_path)
    expected_qids = set(queries)
    if set(run) != expected_qids:
        missing = sorted(expected_qids - set(run))[:5]
        extra = sorted(set(run) - expected_qids)[:5]
        raise ValueError(f"{arm}: qid mismatch; missing={missing}, extra={extra}")
    if any(len(run[qid]) < top_k for qid in expected_qids):
        raise ValueError(f"{arm}: every qid must contain at least {top_k} page candidates")

    contexts = _make_contexts(run, queries, qrels, top_k)
    identity = _identity(arm, run_path, top_k, workers)
    checkpoint = output_dir / f"{arm}.checkpoint.json"
    records = _load_records(checkpoint, identity)
    for qid in queries:
        base = contexts[qid]
        previous = records.get(qid, {})
        records[qid] = {**base, **previous}

    _save_checkpoint(checkpoint, identity, records)

    generation_started = time.perf_counter()
    generation_pending = [
        records[qid]
        for qid in queries
        if not records[qid].get("answer") or records[qid].get("generation_error")
    ]
    generation_sum = 0.0
    if generation_pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_generate_one, row) for row in generation_pending]
            for index, future in enumerate(as_completed(futures), 1):
                qid, answer, error, elapsed = future.result()
                records[qid]["answer"] = answer
                records[qid]["generation_error"] = error or ""
                records[qid]["generation_seconds"] = elapsed
                records[qid].pop("judgment", None)
                records[qid].pop("judge_error", None)
                generation_sum += elapsed
                if index % 10 == 0:
                    _save_checkpoint(checkpoint, identity, records)
    generation_wall = time.perf_counter() - generation_started
    _save_checkpoint(checkpoint, identity, records)

    judge_started = time.perf_counter()
    judge_pending = [
        records[qid]
        for qid in queries
        if records[qid].get("answer") and not records[qid].get("generation_error")
        and (not records[qid].get("judgment") or records[qid].get("judge_error"))
    ]
    judge_sum = 0.0
    if judge_pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_judge_one, row) for row in judge_pending]
            for index, future in enumerate(as_completed(futures), 1):
                qid, judgment, error, elapsed = future.result()
                records[qid]["judgment"] = judgment
                records[qid]["judge_error"] = error
                records[qid]["judge_seconds"] = elapsed
                judge_sum += elapsed
                if index % 10 == 0:
                    _save_checkpoint(checkpoint, identity, records)
    judge_wall = time.perf_counter() - judge_started
    _save_checkpoint(checkpoint, identity, records)

    ordered = [records[qid] for qid in queries]
    judgment_counts: dict[str, int] = {}
    for row in ordered:
        label = row.get("judgment") or "Incorrect"
        judgment_counts[label] = judgment_counts.get(label, 0) + 1
    errors = sum(bool(row.get("generation_error") or row.get("judge_error")) for row in ordered)
    scored = [ViDoreVerdict(row["qid"], row.get("judgment", "Incorrect")) for row in ordered]
    accuracy = score(scored)
    valid_rows = [
        row for row in ordered if not (row.get("generation_error") or row.get("judge_error"))
    ]
    valid_accuracy = score(
        [ViDoreVerdict(row["qid"], row.get("judgment", "Incorrect")) for row in valid_rows]
    )
    retrieval = _retrieval_metrics(run, qrels)

    result = {
        "arm": arm,
        "run_path": str(run_path),
        "protocol": {
            "dataset": "vidore_v3/physics",
            "language": "french",
            "n_queries": len(queries),
            "top_k_context_pages": top_k,
            "generator_model": GENERATOR_MODEL,
            "judge_model": JUDGE_MODEL,
            "answer_prompt": ANSWER_PROMPT,
        },
        "retrieval_metrics": retrieval,
        "e2e_metrics": {
            "scored_queries": len(scored),
            "valid_judgments": len(scored) - errors,
            "errors": errors,
            "correct_only_percent": 100.0 * accuracy["correct_only"],
            "correct_plus_partial_percent": 100.0 * accuracy["correct_plus_partial"],
            "valid_only_correct_percent": 100.0 * valid_accuracy["correct_only"],
            "valid_only_correct_plus_partial_percent": 100.0 * valid_accuracy["correct_plus_partial"],
            "judgment_counts": judgment_counts,
        },
        "context_metrics": {
            "mean_context_pages": sum(row["context_pages"] for row in ordered) / len(ordered),
            "mean_context_chars": sum(row["context_chars"] for row in ordered) / len(ordered),
            "gold_page_hit_percent": 100.0 * sum(row["gold_page_hit"] for row in ordered) / len(ordered),
            "mean_gold_pages_in_context": sum(row["gold_pages_in_context"] for row in ordered) / len(ordered),
        },
        "timing": {
            "generation_wall_seconds": generation_wall,
            "generation_sum_query_seconds": generation_sum,
            "judge_wall_seconds": judge_wall,
            "judge_sum_query_seconds": judge_sum,
            "total_wall_seconds": generation_wall + judge_wall,
        },
    }
    _json_dump(output_dir / f"{arm}.json", result)
    _json_dump(output_dir / f"{arm}.per_question.json", ordered)
    return result


def _write_report(output_dir: Path, results: list[dict[str, Any]], reference: dict[str, Any] | None) -> None:
    for result in results:
        e2e = result["e2e_metrics"]
        if "valid_only_correct_percent" not in e2e:
            rows = _json_load(output_dir / f"{result['arm']}.per_question.json")
            valid_rows = [
                row for row in rows
                if not (row.get("generation_error") or row.get("judge_error"))
            ]
            valid_score = score(
                [
                    ViDoreVerdict(row["qid"], row.get("judgment", "Incorrect"))
                    for row in valid_rows
                ]
            )
            e2e["valid_only_correct_percent"] = 100.0 * valid_score["correct_only"]
            e2e["valid_only_correct_plus_partial_percent"] = 100.0 * valid_score["correct_plus_partial"]

    report = {
        "protocol": {
            "dataset": "vidore_v3/physics",
            "language": "french",
            "note": "E2E answer accuracy uses the same generator, prompt, judge and top-10 page context for all arms.",
        },
        "arms": results,
        "existing_reference": reference,
    }
    _json_dump(output_dir / "report.json", report)

    lines = [
        "# Physics retrieval end-to-end evaluation",
        "",
        "All arms use the same French queries, top-10 retrieved page context, answer prompt, generator and judge.",
        "Results with non-zero API errors must be treated as incomplete.",
        "",
        "| Arm | Page recall@10 | File recall@3 | nDCG@10 | Gold page in context | Correct/302 | Correct+partial/302 | Valid correct | Valid correct+partial | Errors | E2E wall (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        retrieval = result["retrieval_metrics"]
        e2e = result["e2e_metrics"]
        context = result["context_metrics"]
        timing = result["timing"]
        lines.append(
            f"| {result['arm']} | {retrieval['page_recall_at10']:.2f}% | "
            f"{retrieval['file_recall_at3']:.2f}% | {retrieval['ndcg_at10']:.2f} | "
            f"{context['gold_page_hit_percent']:.2f}% | {e2e['correct_only_percent']:.2f}% | "
            f"{e2e['correct_plus_partial_percent']:.2f}% | "
            f"{e2e['valid_only_correct_percent']:.2f}% | "
            f"{e2e['valid_only_correct_plus_partial_percent']:.2f}% | {e2e['errors']} | "
            f"{timing['total_wall_seconds']:.1f} |"
        )
    if reference:
        lines.extend(
            [
                "",
                "## Existing reference",
                "",
                "This is the previously completed custom E2E hybrid reference; it is shown separately because its retrieval arm is not one of the four runs above.",
                "",
                f"- Correct only: {reference.get('correct_only_percent', 'n/a')}%",
                f"- Correct + partial: {reference.get('correct_plus_partial_percent', 'n/a')}%",
                f"- Source: `{reference.get('source', '')}`",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help=f"Comma-separated arms; available: {', '.join(ARMS)}",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild the report from existing arm artifacts without calling models",
    )
    args = parser.parse_args()
    selected = [name.strip() for name in args.arms.split(",") if name.strip()]
    unknown = [name for name in selected if name not in ARMS]
    if unknown:
        raise SystemExit(f"Unknown arms: {unknown}; available: {list(ARMS)}")

    if args.report_only:
        results = []
        for arm in selected:
            result_path = args.output_dir / f"{arm}.json"
            if not result_path.exists():
                raise FileNotFoundError(f"Existing result not found: {result_path}")
            results.append(_json_load(result_path))
        reference_path = ROOT / "data/benchmark/vidore_v3/results/physics_e2e/retrieved_kdl_pdf_inspector.summary.json"
        reference = None
        if reference_path.exists():
            old = _json_load(reference_path)
            correct_only = float(old.get("correct_only", 0.0))
            correct_plus_partial = float(old.get("correct_plus_partial", 0.0))
            if correct_only <= 1.0 and correct_plus_partial <= 1.0:
                correct_only *= 100.0
                correct_plus_partial *= 100.0
            reference = {
                "correct_only_percent": correct_only,
                "correct_plus_partial_percent": correct_plus_partial,
                "source": str(reference_path),
            }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_report(args.output_dir, results, reference)
        print(f"Report: {args.output_dir / 'report.md'}", flush=True)
        return

    load_dotenv(ROOT / ".env")
    try:
        assert_real([GENERATOR_MODEL, JUDGE_MODEL])
    except Exception as exc:
        # Direct OpenRouter IDs were successfully resolved before the first
        # checkpoint was created.  On resume, the registry endpoint can be
        # transiently unavailable even while chat completions work; do not
        # discard a resumable experiment for that metadata-only failure.
        checkpoint_exists = any(args.output_dir.glob("*.checkpoint.json"))
        if not checkpoint_exists:
            raise
        print(f"Model guard unavailable while resuming: {type(exc).__name__}: {exc}", flush=True)
    benchmark = load("vidore_v3", subset="physics", language="french")
    questions = list(benchmark.questions())
    queries = {str(query.qid): query for query in questions}
    qrels = {
        str(qid): {str(page_id): int(value) for page_id, value in rels.items()}
        for qid, rels in benchmark.qrels().items()
    }
    if len(queries) != 302 or len(qrels) != 302:
        raise ValueError(f"Expected 302 Physics queries/qrels, got {len(queries)}/{len(qrels)}")
    if set(queries) != set(qrels):
        raise ValueError("Query and qrel IDs do not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for arm in selected:
        print(f"Evaluating {arm} ...", flush=True)
        results.append(
            _evaluate_arm(
                arm=arm,
                run_path=ARMS[arm],
                queries=queries,
                qrels=qrels,
                top_k=args.top_k,
                workers=args.workers,
                output_dir=args.output_dir,
            )
        )
        print(f"Completed {arm}", flush=True)

    reference_path = ROOT / "data/benchmark/vidore_v3/results/physics_e2e/retrieved_kdl_pdf_inspector.summary.json"
    reference = None
    if reference_path.exists():
        old = _json_load(reference_path)
        correct_only = float(old.get("correct_only", 0.0))
        correct_plus_partial = float(old.get("correct_plus_partial", 0.0))
        # The historical custom summary stores percentages (49.3/91.1), while
        # this evaluator's score() returns fractions internally.
        if correct_only <= 1.0 and correct_plus_partial <= 1.0:
            correct_only *= 100.0
            correct_plus_partial *= 100.0
        reference = {
            "correct_only_percent": correct_only,
            "correct_plus_partial_percent": correct_plus_partial,
            "source": str(reference_path),
        }
    _write_report(args.output_dir, results, reference)
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
