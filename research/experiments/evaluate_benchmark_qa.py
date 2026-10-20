"""Evaluate QA from a saved benchmark second-retrieval run.

This runner is intentionally separate from the retrieval runner.  The old KDL
second-retrieval artifact stores only ranked ids; this command reconstructs the
deterministic fixed-overlap chunk text from ``enriched_pages.jsonl`` and then
calls the existing ViDoRe answer prompt and three-way judge.

It never uses qrels for context construction or ranking.  Qrels are not needed
for QA scoring because the merged benchmark stores the reference answer in
``queries.jsonl``.  The command writes a rich context run before making any
LLM calls and checkpoints every completed answer/judgment.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.experiments.run_benchmark_kdl_second import _read_jsonl  # noqa: E402
from src.chunking_embedding import ChunkEmbedConfig, FieldContext, route_document  # noqa: E402
from src.chunking_embedding.fields import Resources  # noqa: E402
from src.evaluation.generate import ContextChunk, generate  # noqa: E402
from src.evaluation.llm import complete  # noqa: E402
from src.evaluation.benchmarks.vidore_v3_judge import (  # noqa: E402
    ViDoreVerdict,
    ViDoreStyle,
    judge_answer,
    score,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _records_by_page(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        page_id = str(row.get("page_id") or "")
        record = row.get("record")
        if page_id and isinstance(record, dict):
            output[page_id] = record
    return output


def _chunk_text_index(
    enriched_path: Path,
    chunking_config_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Recreate text-only chunk records without calling the embedding API."""
    records = _records_by_page(enriched_path)
    loaded = load_config(chunking_config_path)
    config = ChunkEmbedConfig.from_mapping(loaded.get("chunking_embedding") or {})
    by_chunk: dict[str, dict[str, str]] = {}
    page_stats: dict[str, Any] = {"pages": len(records), "chunks": 0, "text_chunks": 0}
    for page_id, record in records.items():
        source_object_id = str(record.get("source_object_id") or "")
        rows = record.get("rows") or []
        extraction = rows[0].get("extraction") if rows and isinstance(rows[0], dict) else None
        if not source_object_id or not isinstance(extraction, dict):
            continue
        context = FieldContext(
            doc_id=source_object_id,
            chunker_name=config.chunker,
            chunker_params=config.chunker_params,
            max_rows_per_chunk=config.max_rows_per_chunk,
            language=extraction.get("language"),
        )
        chunks = route_document(extraction, context, Resources())
        page_stats["chunks"] += len(chunks)
        for chunk in chunks:
            if chunk.chunk_type != "text_chunk" or not str(chunk.text).strip():
                continue
            page_stats["text_chunks"] += 1
            by_chunk[chunk.chunk_id] = {
                "page_id": page_id,
                "text": str(chunk.text),
            }
    return by_chunk, {"chunking": config.resolved(), **page_stats}


def _materialize_context(
    second_root: Path,
    run_path: Path,
    enriched_path: Path,
    chunking_config: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_chunk, stats = _chunk_text_index(enriched_path, chunking_config)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for row in _read_jsonl(run_path):
        units: list[dict[str, Any]] = []
        for item in row.get("chunks") or []:
            chunk_id = str(item.get("chunk_id") or "")
            text_record = by_chunk.get(chunk_id)
            if text_record is None:
                missing.append(chunk_id)
                continue
            units.append({
                "chunk_id": chunk_id,
                "page_id": text_record["page_id"],
                "rank": int(item.get("rank") or len(units) + 1),
                "text": text_record["text"],
            })
        rows.append({"query_id": str(row["query_id"]), "chunks": units})
    if missing:
        raise RuntimeError(
            f"Could not reconstruct {len(missing)} ranked chunks; "
            f"first ids: {missing[:5]}"
        )
    output = second_root / "runs" / "legacy_chunks_top10_context.jsonl"
    _write_jsonl(output, rows)
    stats["queries"] = len(rows)
    stats["ranked_chunks"] = sum(len(row["chunks"]) for row in rows)
    stats["context_path"] = str(output)
    _write_json(second_root / "reports" / "context_materialization.json", stats)
    return rows, stats


def _load_queries(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in _read_jsonl(path):
        answers = row.get("answers") or []
        if not answers:
            raise ValueError(f"Query {row.get('query_id')} has no answer")
        result[str(row["query_id"])] = {
            "query_id": str(row["query_id"]),
            "query": str(row.get("query") or ""),
            "answer": "\n".join(str(answer) for answer in answers),
            "answers": [str(answer) for answer in answers],
            "answer_count": len(answers),
            "source": row.get("source"),
        }
    return result


def _judge_against_answers(
    query: dict[str, Any],
    answer: str,
    judge: str,
    generator: str,
) -> tuple[ViDoreVerdict, list[dict[str, Any]]]:
    """Judge against all accepted references and retain the best verdict.

    The merged benchmark has multiple annotations for some MPDocVQA answers
    (for example ``"Page 2"`` and ``"2"``).  Joining those strings into one
    reference is needlessly strict, so an answer is credited if it matches any
    accepted reference.  The alternatives are retained for auditability.
    """
    labels = {"Correct": 2, "Partially Correct": 1, "Incorrect": 0}
    candidates: list[dict[str, Any]] = []
    best: ViDoreVerdict | None = None
    for gold in query["answers"]:
        try:
            verdict = judge_answer(
                query["query_id"], query["query"], gold, answer,
                model=judge, generator_model=generator,
            )
        except Exception as error:  # keep one malformed judge reply auditable
            verdict = ViDoreVerdict(query["query_id"], "Incorrect", error=str(error))
        candidates.append({"gold_answer": gold, "judgment": verdict.judgment, "error": verdict.error})
        if best is None or labels.get(verdict.judgment, -1) > labels.get(best.judgment, -1):
            best = verdict
    assert best is not None
    return best, candidates


def _load_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(row["query_id"]): row for row in _read_jsonl(path)}


def _evaluate_one(
    query: dict[str, Any],
    context_row: dict[str, Any],
    generator: str,
    judge: str,
    max_context_chars: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    chunks = [
        ContextChunk(
            chunk_id=str(item["chunk_id"]),
            doc_id=str(item["page_id"]),
            text=str(item["text"]),
            score=0.0,
        )
        for item in context_row.get("chunks") or []
    ]
    started = time.perf_counter()
    generation = generate(
        query["query_id"],
        query["query"],
        chunks,
        model=generator,
        max_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        render_prompt=ViDoreStyle().render_prompt,
    )
    generation_seconds = time.perf_counter() - started
    judge_started = time.perf_counter()
    alternatives: list[dict[str, Any]] = []
    if generation.error:
        verdict = ViDoreVerdict(query["query_id"], "Incorrect", error=generation.error)
    else:
        try:
            verdict, alternatives = _judge_against_answers(
                query, generation.answer, judge, generator,
            )
        except Exception as error:  # keep the checkpoint resumable
            verdict = ViDoreVerdict(query["query_id"], "Incorrect", error=str(error))
    return {
        "query_id": query["query_id"],
        "source": query.get("source"),
        "query": query["query"],
        "answer_count": query["answer_count"],
        "gold_answer": query["answer"],
        "gold_answers": query["answers"],
        "answer": generation.answer,
        "generation_error": generation.error,
        "judgment": verdict.judgment,
        "explanation": verdict.explanation,
        "judge_error": verdict.error,
        "judge_alternatives": alternatives if not generation.error else [],
        "correct": verdict.judgment == "Correct" and not verdict.error,
        "credited": verdict.judgment in ("Correct", "Partially Correct") and not verdict.error,
        "chunks_used": generation.chunks_used,
        "chars_used": generation.chars_used,
        "context_pages": generation.context_doc_ids,
        "generation_seconds": round(generation_seconds, 3),
        "judge_seconds": round(time.perf_counter() - judge_started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--chunking-config", type=Path, default=ROOT / "configs/pipeline.yaml")
    parser.add_argument("--generator", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--judge", default="openai/gpt-4o-mini")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--force-context", action="store_true")
    parser.add_argument(
        "--rejudge-alternatives",
        action="store_true",
        help="Rejudge completed multi-answer queries against each accepted reference.",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Reconstruct and validate ranked chunk text, without calling QA models.",
    )
    args = parser.parse_args()
    load_dotenv_file(ROOT)

    second_root = args.second_root
    run_path = second_root / "runs" / "legacy_chunks_top10.jsonl"
    enriched_path = second_root / "checkpoints" / "enriched_pages.jsonl"
    context_path = second_root / "runs" / "legacy_chunks_top10_context.jsonl"
    if args.force_context or not context_path.is_file():
        print("Materializing exact ranked chunk text...", flush=True)
        _, context_stats = _materialize_context(second_root, run_path, enriched_path, args.chunking_config)
        print(json.dumps(context_stats, ensure_ascii=False), flush=True)
    if args.materialize_only:
        print(f"Context artifact: {context_path}", flush=True)
        return 0
    context_rows = _load_rows(context_path)
    queries = _load_queries(args.dataset_root / "queries.jsonl")
    selected = [queries[qid] for qid in context_rows if qid in queries]
    if len(selected) != len(context_rows):
        raise RuntimeError("Context run contains unknown query ids")

    out_dir = second_root / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "legacy_chunks_top10_qa.jsonl"
    existing = _load_rows(rows_path)
    # A row with a provider error is a checkpoint of an attempt, not a valid
    # evaluation.  Keep successful rows and retry only incomplete/error rows.
    pending = [
        q for q in selected
        if q["query_id"] not in existing
        or existing[q["query_id"]].get("generation_error")
        or existing[q["query_id"]].get("judge_error")
        or (
            args.rejudge_alternatives
            and existing[q["query_id"]].get("answer_count", 1) > 1
        )
    ]
    print(f"QA: {len(selected)} queries, pending={len(pending)}, workers={args.workers}", flush=True)
    order = [q["query_id"] for q in selected]
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    _evaluate_one, query, context_rows[query["query_id"]], args.generator,
                    args.judge, args.max_context_chars, args.max_output_tokens,
                ): query["query_id"]
                for query in pending
            }
            for future in as_completed(futures):
                qid = futures[future]
                existing[qid] = future.result()
                _write_jsonl(rows_path, [existing[item] for item in order if item in existing])
                print(f"QA progress: {len(existing)}/{len(selected)} ({qid})", flush=True)

    scored = [existing[qid] for qid in order if qid in existing]
    verdicts = [ViDoreVerdict(row["query_id"], row.get("judgment", "Incorrect"), error=row.get("judge_error")) for row in scored]
    valid = [v for v in verdicts if not v.error and v.judgment in {"Correct", "Partially Correct", "Incorrect"}]
    score_report = score(valid) if valid else {"n": 0, "correct_only": 0.0, "correct_plus_partial": 0.0}
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({str(row.get("source") or "unknown") for row in scored}):
        source_rows = [row for row in scored if str(row.get("source") or "unknown") == source]
        source_valid = [
            row for row in source_rows
            if not row.get("generation_error") and not row.get("judge_error")
        ]
        by_source[source] = {
            "queries": len(source_rows),
            "valid": len(source_valid),
            "correct_only": round(
                100 * sum(bool(row.get("correct")) for row in source_valid) / max(1, len(source_valid)), 2
            ),
            "correct_plus_partial": round(
                100 * sum(bool(row.get("credited")) for row in source_valid) / max(1, len(source_valid)), 2
            ),
            "judgments": {
                label: sum(row.get("judgment") == label for row in source_valid)
                for label in ("Correct", "Partially Correct", "Incorrect")
            },
        }
    report = {
        "contract_version": "benchmark-qa-v1",
        "dataset_root": str(args.dataset_root),
        "second_root": str(second_root),
        "arm": "legacy_chunks_top10",
        "generator": args.generator,
        "judge": args.judge,
        "queries": len(selected),
        "completed": len(scored),
        "judge_valid": len(valid),
        "errors": len(scored) - len(valid),
        "correct_only": round(100 * score_report["correct_only"], 2),
        "correct_plus_partial": round(100 * score_report["correct_plus_partial"], 2),
        "judgments": {label: sum(1 for v in valid if v.judgment == label) for label in ("Correct", "Partially Correct", "Incorrect")},
        "by_source": by_source,
        "context_mean_chunks": round(sum(r["chunks_used"] for r in scored) / max(1, len(scored)), 2),
        "context_mean_chars": round(sum(r["chars_used"] for r in scored) / max(1, len(scored))),
        "generation_seconds_sum": round(sum(float(r.get("generation_seconds", 0)) for r in scored), 2),
        "judge_seconds_sum": round(sum(float(r.get("judge_seconds", 0)) for r in scored), 2),
        "context_path": str(context_path),
        "qa_rows_path": str(rows_path),
    }
    _write_json(out_dir / "legacy_chunks_top10_qa.report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
