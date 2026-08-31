"""Run DocBench on-demand retrieval and QA as one unit per question.

This entry point is intentionally separate from ``run_docbench_e2e.py``.
It keeps the same pipeline contract:

    pdf-inspector page preparation -> BM25 page retrieval ->
    KDL + pdf-inspector for missing pages -> fixed 512/128 chunks ->
    text-embedding-3-small -> hybrid BM25+dense baseline_legacy ->
    generator -> DocBench judge

The scheduling difference is that each worker owns a complete question:

    retrieve(question) -> generate(question) -> judge(question) -> checkpoint

Several complete questions may run concurrently, but a worker never starts its
next question before the current question has been checkpointed.  The shared
``OnDemandPerQueryRunner`` still deduplicates parsed pages and may micro-batch
KDL requests submitted by different question workers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Lock
from typing import Any
import argparse
import logging
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.on_demand_per_query import OnDemandPerQueryRunner  # noqa: E402
from research.data_discovery.run_docbench_e2e import (  # noqa: E402
    DOCBENCH_JUDGE_PROMPT,
    _apply_kdl_overrides,
    _build_report,
    _check_vllm_endpoint,
    _encode_retrieval_result,
    _hash_payload,
    _load_docbench,
    _load_or_build_page_index,
    _parse_score,
    _positive,
    _prepare_chunking_config,
    _read_latest_jsonl,
    _require_env,
    _resolve_path,
    _should_retry_unanswerable,
    _generate_with_unanswerable_retry,
    _update_summary,
    _validate_baseline_chunking,
    _write_json,
    _write_ordered_jsonl,
)
from src.evaluation.generate import ContextChunk  # noqa: E402
from src.evaluation.llm import complete  # noqa: E402
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402
from src.ingestion.parsing.kdl_health import KDLHostUnavailableError  # noqa: E402
from src.utils.observability import (  # noqa: E402
    JsonEventLogger,
    configure_run_logging,
    utc_now_iso,
)


LOGGER = logging.getLogger("docbench_per_question")


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv_file(ROOT)
    config_path = _resolve_path(args.config)
    config = load_config(config_path)
    docbench_config = dict(config.get("docbench") or {})
    scope = str(
        args.retrieval_scope or docbench_config.get("retrieval_scope") or "file"
    ).lower()
    if scope not in {"file", "lake"}:
        raise ValueError("retrieval scope must be 'file' or 'lake'")

    docbench_root = _resolve_path(args.docbench_root or docbench_config.get("root"))
    if docbench_root is None:
        raise ValueError("--docbench-root is required")
    output_dir = _resolve_path(
        args.output_dir
        or docbench_config.get("output_dir")
        or ROOT / "data" / "benchmark" / "docbench_on_demand_basic_per_question"
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logging(
        output_dir / "logs" / f"{scope}_on_demand_per_query.log",
        level=getattr(logging, str(args.log_level).upper()),
    )
    event_logger = JsonEventLogger(
        output_dir / "logs" / f"{scope}_events.jsonl",
        run_name="on-demand-per-query",
    )
    run_started_at = utc_now_iso()
    event_logger.emit("run_started", scope=scope, config=str(config_path))

    top_k_pages = _positive(
        args.top_k_pages or docbench_config.get("top_k_pages") or 10,
        "top-k-pages",
    )
    top_k_chunks = _positive(
        args.top_k_chunks or docbench_config.get("top_k_chunks") or 10,
        "top-k-chunks",
    )
    depth = _positive(
        args.depth or docbench_config.get("chunk_retrieval_depth") or 100,
        "depth",
    )
    alpha = float(
        args.alpha
        if args.alpha is not None
        else docbench_config.get("chunk_retrieval_alpha", 0.7)
    )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    configured_workers = docbench_config.get(
        "question_workers",
        docbench_config.get("query_workers", 2),
    )
    question_workers = _positive(
        args.question_workers or configured_workers,
        "question-workers",
    )
    microbatch_window = float(
        args.kdl_microbatch_window_seconds
        if args.kdl_microbatch_window_seconds is not None
        else docbench_config.get("kdl_microbatch_window_seconds", 0.30)
    )
    microbatch_max_pages = _positive(
        args.kdl_microbatch_max_pages
        or docbench_config.get("kdl_microbatch_max_pages")
        or 32,
        "kdl-microbatch-max-pages",
    )
    max_context_chars = _positive(
        args.max_context_chars or docbench_config.get("max_context_chars") or 12000,
        "max-context-chars",
    )
    max_unit_chars = _positive(
        args.max_unit_chars or docbench_config.get("max_unit_chars") or 8000,
        "max-unit-chars",
    )
    max_output_tokens = _positive(
        args.max_output_tokens or docbench_config.get("max_output_tokens") or 512,
        "max-output-tokens",
    )
    generator_model = str(
        args.generator
        or docbench_config.get("generator")
        or "deepseek/deepseek-v4-flash"
    )
    judge_model = str(
        args.judge or docbench_config.get("judge") or "openai/gpt-4o"
    )
    if generator_model == judge_model:
        raise ValueError("generator and judge should be different models")

    _require_env("VLLM_API_BASE")
    _require_env("OPENROUTER_API_KEY")
    if not args.skip_endpoint_check:
        _check_vllm_endpoint()

    documents, all_questions = _load_docbench(docbench_root)
    if args.max_documents is not None:
        if args.max_documents <= 0:
            raise ValueError("max-documents must be positive")
        selected_documents = documents[: args.max_documents]
    else:
        selected_documents = documents
    selected_doc_ids = {item["doc_id"] for item in selected_documents}
    questions = [
        item for item in all_questions if item["doc_id"] in selected_doc_ids
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[: args.limit]
    if not selected_documents:
        raise RuntimeError(f"No DocBench documents found under {docbench_root}")
    if not questions:
        raise RuntimeError("No DocBench questions selected")
    question_ids = {str(item["qid"]) for item in questions}
    if len(question_ids) != len(questions):
        raise RuntimeError("DocBench question ids must be unique")

    index_documents = documents if scope == "lake" else selected_documents
    index_started = time.perf_counter()
    page_index, corpus_fingerprint = _load_or_build_page_index(
        index_documents,
        output_dir=output_dir,
        scope=scope,
        force_rebuild=args.force_rebuild_index,
    )
    index_seconds = time.perf_counter() - index_started
    if not page_index.pages:
        raise RuntimeError("The pdf-inspector page index is empty")

    raw_chunking_config = dict(config.get("chunking_embedding") or {})
    chunking_config = _prepare_chunking_config(raw_chunking_config, output_dir)
    _validate_baseline_chunking(chunking_config)

    parser_config = resolve_parser_config(
        ROOT,
        config.get("parsing") or {},
        output_dir / "parser-assets" / scope,
    )
    parser_config = _apply_kdl_overrides(parser_config, args)
    parser_config = _with_kdl_event_log(
        parser_config, output_dir / "logs" / f"{scope}_kdl_events.jsonl"
    )
    if parser_config.get("provider") != "kdl_pdf_inspector":
        raise ValueError(
            "DocBench on-demand basic requires parsing.provider=kdl_pdf_inspector"
        )

    retrieval_config = {
        "pipeline": "on-demand-basic-per-question",
        "execution": "retrieve -> QA/judge per question",
        "light_preparation": "pdf_inspector",
        "light_retrieval": "bm25",
        "downstream": (
            "KDL + pdf-inspector -> fixed_overlap 512/128 -> "
            "text-embedding-3-small -> hybrid baseline_legacy"
        ),
        "retrieval_scope": scope,
        "indexed_documents": len(index_documents),
        "evaluated_documents": len(selected_documents),
        "top_k_pages": top_k_pages,
        "top_k_chunks": top_k_chunks,
        "depth": depth,
        "alpha": alpha,
        "question_workers": question_workers,
        "microbatch_window_seconds": microbatch_window,
        "microbatch_max_pages": microbatch_max_pages,
        "corpus_fingerprint": corpus_fingerprint,
        "parser_config_hash": _hash_payload(parser_config),
        "chunking_config_hash": _hash_payload(chunking_config),
    }
    retrieval_config_hash = _hash_payload(retrieval_config)
    retrieval_path = output_dir / "retrieval" / f"{scope}_baseline_legacy.jsonl"
    timings_path = output_dir / "retrieval" / f"{scope}_timings.jsonl"
    qa_path = output_dir / "qa" / f"{scope}_baseline_legacy.jsonl"

    # Only reuse rows produced under the current retrieval contract.  This
    # prevents a run with changed parser/concurrency settings from silently
    # mixing old retrieval results into a new experiment.
    retrieval_rows = {
        qid: row
        for qid, row in _read_latest_jsonl(retrieval_path).items()
        if row.get("retrieval_config_hash") == retrieval_config_hash
    }
    timing_rows = {
        qid: row
        for qid, row in _read_latest_jsonl(timings_path).items()
        if qid in retrieval_rows
    }
    qa_config_hash = _hash_payload(
        {
            "arm": "baseline_legacy",
            "generator": generator_model,
            "judge": judge_model,
            "max_context_chars": max_context_chars,
            "max_unit_chars": max_unit_chars,
            "max_output_tokens": max_output_tokens,
            "skip_judge": args.skip_judge,
            "retrieval_config_hash": retrieval_config_hash,
        }
    )
    qa_rows = {
        qid: row
        for qid, row in _read_latest_jsonl(qa_path).items()
        if row.get("qa_config_hash") == qa_config_hash
    }

    runner = OnDemandPerQueryRunner(
        page_index,
        parser_config=parser_config,
        chunking_config=chunking_config,
        project_root=ROOT,
        work_dir=output_dir / "work" / scope,
        cache_dir=output_dir / "cache" / scope,
        top_k_pages=top_k_pages,
        top_k_chunks=top_k_chunks,
        depth=depth,
        alpha=alpha,
        query_workers=question_workers,
        microbatch_window_seconds=microbatch_window,
        microbatch_max_pages=microbatch_max_pages,
        force_reparse=args.force_reparse,
        event_logger=event_logger,
    )
    checkpoint_lock = Lock()
    abort_event = Event()

    def checkpoint_retrieval(
        question: dict[str, Any],
        row: dict[str, Any],
        timing: dict[str, Any],
    ) -> None:
        qid = str(question["qid"])
        with checkpoint_lock:
            retrieval_rows[qid] = row
            timing_rows[qid] = timing
            _write_ordered_jsonl(retrieval_path, retrieval_rows, questions)
            _write_ordered_jsonl(timings_path, timing_rows, questions)

    def checkpoint_qa(question: dict[str, Any], row: dict[str, Any]) -> None:
        qid = str(question["qid"])
        with checkpoint_lock:
            qa_rows[qid] = row
            _write_ordered_jsonl(qa_path, qa_rows, questions)

    def process_question(number: int, question: dict[str, Any]) -> str:
        qid = str(question["qid"])
        started = time.perf_counter()
        retrieval_row = retrieval_rows.get(qid)
        retrieval_reusable = bool(
            retrieval_row
            and retrieval_row.get("status") == "ok"
            and retrieval_row.get("retrieval_config_hash") == retrieval_config_hash
        )

        if not retrieval_reusable:
            if abort_event.is_set():
                LOGGER.warning(
                    "Skipping qid=%s because the KDL host circuit is open",
                    qid,
                )
                return qid
            try:
                retrieval = runner.run_query(
                    question["question"], query_id=qid
                )
                retrieval_row = _encode_retrieval_result(
                    question,
                    retrieval,
                    retrieval_config,
                    retrieval_config_hash,
                )
                timing = {"qid": qid, **retrieval.timing}
            except KDLHostUnavailableError as error:
                message = f"{type(error).__name__}: {error}"
                abort_event.set()
                retrieval_row = {
                    **question,
                    "pipeline": "pdf_inspector -> bm25 -> baseline_legacy",
                    "retrieval_config": retrieval_config,
                    "retrieval_config_hash": retrieval_config_hash,
                    "status": "aborted",
                    "error": message,
                    "finished_at_utc": utc_now_iso(),
                    "hits": [],
                    "chunks": [],
                    "timing": {},
                }
                timing = {
                    "qid": qid,
                    "status": "aborted",
                    "error": message,
                    "finished_at_utc": utc_now_iso(),
                }
                checkpoint_retrieval(question, retrieval_row, timing)
                event_logger.emit(
                    "run_aborted",
                    qid=qid,
                    error_type=type(error).__name__,
                    error=message,
                )
                raise
            except Exception as error:  # noqa: BLE001 - checkpoint per-question failure
                message = f"{type(error).__name__}: {error}"
                retrieval_row = {
                    **question,
                    "pipeline": "pdf_inspector -> bm25 -> baseline_legacy",
                    "retrieval_config": retrieval_config,
                    "retrieval_config_hash": retrieval_config_hash,
                    "status": "error",
                    "error": message,
                    "finished_at_utc": utc_now_iso(),
                    "hits": [],
                    "chunks": [],
                    "timing": {},
                }
                timing = {
                    "qid": qid,
                    "status": "error",
                    "error": message,
                    "finished_at_utc": utc_now_iso(),
                }
            checkpoint_retrieval(question, retrieval_row, timing)

        if args.skip_qa:
            LOGGER.info(
                "Question %d/%d: qid=%s retrieval=%s elapsed=%.2fs",
                number,
                len(questions),
                qid,
                retrieval_row.get("status"),
                time.perf_counter() - started,
            )
            return qid

        qa_row = qa_rows.get(qid)
        qa_reusable = bool(
            qa_row
            and qa_row.get("status") == "ok"
            and qa_row.get("qa_config_hash") == qa_config_hash
            and not _should_retry_unanswerable(qa_row)
        )
        if not qa_reusable:
            qa_row = _answer_and_judge(
                question,
                retrieval_row,
                generator_model=generator_model,
                judge_model=judge_model,
                max_context_chars=max_context_chars,
                max_unit_chars=max_unit_chars,
                max_output_tokens=max_output_tokens,
                skip_judge=args.skip_judge,
                qa_config_hash=qa_config_hash,
                previous_row=qa_row,
            )
            checkpoint_qa(question, qa_row)

        LOGGER.info(
            "Question %d/%d: qid=%s retrieval=%s qa=%s elapsed=%.2fs",
            number,
            len(questions),
            qid,
            retrieval_row.get("status"),
            qa_row.get("status"),
            time.perf_counter() - started,
        )
        return qid

    try:
        pending: list[dict[str, Any]] = []
        for question in questions:
            qid = str(question["qid"])
            retrieval_done = (
                qid in retrieval_rows
                and retrieval_rows[qid].get("status") == "ok"
            )
            qa_done = (
                qid in qa_rows
                and qa_rows[qid].get("status") == "ok"
                and qa_rows[qid].get("qa_config_hash") == qa_config_hash
                and not _should_retry_unanswerable(qa_rows[qid])
            )
            if not (retrieval_done if args.skip_qa else retrieval_done and qa_done):
                pending.append(question)
        LOGGER.info(
            "Running per-question pipeline: pending=%d/%d workers=%d scope=%s",
            len(pending),
            len(questions),
            question_workers,
            scope,
        )
        # Each future handles retrieve + QA/judge before it completes.  The
        # pool therefore parallelizes whole questions, not just retrieval.
        with ThreadPoolExecutor(max_workers=question_workers) as pool:
            futures = {
                pool.submit(process_question, number, question): question
                for number, question in enumerate(pending, start=1)
            }
            for future in as_completed(futures):
                # process_question checkpoints ordinary failures itself.  A
                # programming or filesystem error is still surfaced here.
                future.result()

        timing_summary = runner.timing_summary(
            light_preparation_seconds=index_seconds
        )
        timing_summary["run_started_at_utc"] = run_started_at
        timing_summary["run_finished_at_utc"] = utc_now_iso()
        if args.skip_qa:
            report = {
                "arm": "baseline_legacy",
                "pipeline": "pdf_inspector -> bm25 -> KDL+pdf_inspector -> fixed_chunk_512 -> text-embedding-3-small -> hybrid(alpha=0.7)",
                "retrieval_scope": scope,
                "questions_expected": len(questions),
                "questions_retrieved": sum(
                    row.get("status") == "ok"
                    for row in retrieval_rows.values()
                    if str(row.get("qid")) in question_ids
                ),
                "accuracy": None,
            }
        else:
            report = _build_report(
                questions,
                retrieval_rows,
                qa_rows,
                scope=scope,
                index_documents=index_documents,
                selected_documents=selected_documents,
                page_index=page_index,
                runner=runner,
                top_k_pages=top_k_pages,
                top_k_chunks=top_k_chunks,
                depth=depth,
                alpha=alpha,
                generator_model=generator_model,
                judge_model=judge_model,
            )

        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{scope}_baseline_legacy.json"
        _write_json(report_path, report)
        _write_json(reports_dir / f"{scope}_timing_summary.json", timing_summary)
        manifest = {
            "contract_version": "docbench-on-demand-basic-per-question-v1",
            "pipeline": "pdf_inspector -> bm25 -> baseline_legacy -> generator -> judge",
            "execution": "retrieve -> QA/judge per question",
            "retrieval_scope": scope,
            "docbench_root": str(docbench_root.resolve()),
            "config": str(config_path),
            "documents_available": len(documents),
            "documents_indexed": len(index_documents),
            "documents_selected": len(selected_documents),
            "questions_selected": len(questions),
            "indexed_pages": len(page_index.pages),
            "corpus_fingerprint": corpus_fingerprint,
            "parser_config": parser_config,
            "chunking_config": chunking_config,
            "retrieval_config": retrieval_config,
            "qa_config_hash": qa_config_hash,
            "retrieval_output": str(retrieval_path),
            "timings_output": str(timings_path),
            "qa_output": str(qa_path),
            "report_output": str(report_path),
            "timing_summary": timing_summary,
            "report": report,
        }
        _write_json(output_dir / f"manifest_{scope}.json", manifest)
        _update_summary(output_dir / "summary.json", scope, report, timing_summary)
        event_logger.emit("run_completed", scope=scope, questions=len(questions))
        print(
            __import__("json").dumps(
                {"report": report, "timing": timing_summary},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        runner.close()


def _with_kdl_event_log(config: dict[str, Any], path: Path) -> dict[str, Any]:
    result = dict(config)
    kdl = dict(result.get("kdl") or {})
    kdl["event_log_path"] = str(path)
    result["kdl"] = kdl
    return result


def _answer_and_judge(
    question: dict[str, Any],
    retrieval: dict[str, Any],
    *,
    generator_model: str,
    judge_model: str,
    max_context_chars: int,
    max_unit_chars: int,
    max_output_tokens: int,
    skip_judge: bool,
    qa_config_hash: str,
    previous_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    qid = str(question["qid"])
    started_at = utc_now_iso()
    if retrieval.get("status") != "ok":
        return {
            **question,
            "arm": "baseline_legacy",
            "status": "error",
            "score": None,
            "error": f"retrieval failed: {retrieval.get('error')}",
            "qa_config_hash": qa_config_hash,
            "started_at_utc": started_at,
            "finished_at_utc": utc_now_iso(),
        }

    context = [
        ContextChunk(
            chunk_id=str(item.get("chunk_id") or ""),
            doc_id=str(item.get("doc_id") or ""),
            text=str(item.get("text") or "").strip()[:max_unit_chars],
            score=float(item.get("score") or 0.0),
        )
        for item in retrieval.get("chunks") or []
        if str(item.get("text") or "").strip()
    ]
    generation, retry_count, initial_answer = _generate_with_unanswerable_retry(
        qid,
        question["question"],
        context,
        model=generator_model,
        max_chars=max_context_chars,
        max_output_tokens=max_output_tokens,
        previous_row=previous_row,
    )
    result: dict[str, Any] = {
        **question,
        "arm": "baseline_legacy",
        "generator": generator_model,
        "judge": judge_model,
        "sys_ans": generation.answer,
        "chunks_used": generation.chunks_used,
        "chars_used": generation.chars_used,
        "context_doc_ids": generation.context_doc_ids,
        "context_page_ids": [item.doc_id for item in context],
        "context_unit_ids": [item.chunk_id for item in context],
        "retrieved_page_count": len(retrieval.get("hits") or []),
        "retrieval_timing": retrieval.get("timing") or {},
        "qa_config_hash": qa_config_hash,
        "unanswerable_retry_count": retry_count,
        "started_at_utc": started_at,
    }
    if initial_answer is not None:
        result["initial_sys_ans"] = initial_answer
        LOGGER.info("QA qid=%s abstained; regenerated answer once", qid)
    if generation.error:
        result.update(
            {
                "status": "error",
                "score": None,
                "judge_raw": "",
                "error": generation.error,
            }
        )
        return result
    if not generation.answer.strip() or (
        generation.abstained and initial_answer is None
    ):
        result.update(
            {"status": "ok", "score": 0, "judge_raw": "abstained", "error": None}
        )
        return result
    if skip_judge:
        result.update(
            {"status": "ok", "score": None, "judge_raw": "skipped", "error": None}
        )
        return result

    prompt = (
        DOCBENCH_JUDGE_PROMPT.replace("{{question}}", question["question"])
        .replace("{{sys_ans}}", generation.answer)
        .replace("{{ref_ans}}", question["answer"])
        .replace("{{ref_text}}", question["evidence"])
    )
    try:
        judge_raw = complete(
            judge_model,
            prompt,
            temperature=0.0,
            max_output_tokens=16,
        )
        result.update(
            {
                "status": "ok",
                "score": _parse_score(judge_raw),
                "judge_raw": judge_raw,
                "error": None,
            }
        )
    except Exception as error:  # noqa: BLE001 - persist per-question judge failure
        result.update(
            {
                "status": "error",
                "score": None,
                "judge_raw": "",
                "error": repr(error),
            }
        )
    result["finished_at_utc"] = utc_now_iso()
    return result


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docbench-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "pipeline.docbench-on-demand-basic.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retrieval-scope", choices=("file", "lake"))
    parser.add_argument("--top-k-pages", type=int)
    parser.add_argument("--top-k-chunks", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--question-workers",
        "--query-workers",
        dest="question_workers",
        type=int,
        help="Concurrent complete question pipelines.",
    )
    parser.add_argument("--kdl-microbatch-window-seconds", type=float)
    parser.add_argument("--kdl-microbatch-max-pages", type=int)
    parser.add_argument("--kdl-max-workers", type=int)
    parser.add_argument("--kdl-render-processes", type=int)
    parser.add_argument("--kdl-bbox-max-workers", type=int)
    parser.add_argument("--kdl-request-workers", type=int)
    parser.add_argument("--kdl-request-batch-size", type=int)
    parser.add_argument("--kdl-max-model-sequences", type=int)
    parser.add_argument("--kdl-host-failure-threshold", type=int)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--max-unit-chars", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--generator")
    parser.add_argument("--judge")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
