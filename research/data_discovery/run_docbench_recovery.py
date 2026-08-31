"""Recover an interrupted DocBench on-demand run without redoing good QIDs.

The normal ``run_docbench_e2e.py`` entry point starts every question again
when it is invoked a second time.  This entry point instead treats the
existing retrieval JSONL and parser cache as a checkpoint:

* a QID is affected when one of its selected pages is absent from the parsed
  page cache, or when its previous retrieval row was not successful;
* successful, unaffected rows are preserved;
* only affected and (by default) previously missing QIDs are submitted;
* the shared parser cache is reused, so KDL is called only for missing pages;
* each completed QID is atomically checkpointed back into the existing JSONL.

QA is optional.  With ``--run-qa`` only QIDs processed by this recovery are
answered, preserving existing QA rows.  ``--qa-all-missing`` can be used to
complete every missing QA row as a separate explicit choice.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import json
import logging
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.on_demand_per_query import (  # noqa: E402
    OnDemandPerQueryRunner,
    _record_page_text,
)
from research.data_discovery.run_docbench_e2e import (  # noqa: E402
    _apply_kdl_overrides,
    _check_vllm_endpoint,
    _encode_retrieval_result,
    _hash_payload,
    _load_docbench,
    _load_or_build_page_index,
    _positive,
    _prepare_chunking_config,
    _read_latest_jsonl,
    _require_env,
    _resolve_path,
    _should_retry_unanswerable,
    _validate_baseline_chunking,
    _write_json,
    _write_ordered_jsonl,
)
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


LOGGER = logging.getLogger("docbench_recovery")


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
        or ROOT / "data" / "benchmark" / "docbench_on_demand_basic"
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

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
    query_workers = _positive(
        args.query_workers or docbench_config.get("query_workers") or 4,
        "query-workers",
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

    chunking_config = _prepare_chunking_config(
        dict(config.get("chunking_embedding") or {}), output_dir
    )
    _validate_baseline_chunking(chunking_config)
    parser_config = resolve_parser_config(
        ROOT,
        config.get("parsing") or {},
        output_dir / "parser-assets" / scope,
    )
    parser_config = _apply_kdl_overrides(parser_config, args)
    if parser_config.get("provider") != "kdl_pdf_inspector":
        raise ValueError(
            "DocBench recovery requires parsing.provider=kdl_pdf_inspector"
        )
    if not args.dry_run:
        _require_env("VLLM_API_BASE")
        _require_env("OPENROUTER_API_KEY")
        if not args.skip_endpoint_check:
            _check_vllm_endpoint()

    retrieval_config = {
        "pipeline": "on-demand-basic",
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
        "corpus_fingerprint": corpus_fingerprint,
        "parser_config_hash": _hash_payload(parser_config),
        "chunking_config_hash": _hash_payload(chunking_config),
    }
    retrieval_config_hash = _hash_payload(retrieval_config)
    retrieval_path = output_dir / "retrieval" / f"{scope}_baseline_legacy.jsonl"
    timings_path = output_dir / "retrieval" / f"{scope}_timings.jsonl"
    qa_path = output_dir / "qa" / f"{scope}_baseline_legacy.jsonl"

    existing_retrieval = _read_latest_jsonl(retrieval_path)
    retrieval_rows = {
        qid: row
        for qid, row in existing_retrieval.items()
        if qid in {str(item["qid"]) for item in questions}
        and row.get("retrieval_config_hash") == retrieval_config_hash
    }
    existing_timings = _read_latest_jsonl(timings_path)
    timing_rows = {
        qid: row
        for qid, row in existing_timings.items()
        if qid in retrieval_rows
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
        query_workers=query_workers,
        microbatch_window_seconds=microbatch_window,
        microbatch_max_pages=microbatch_max_pages,
    )

    try:
        empty_cache_records = _empty_cached_page_count(runner)
        removed_empty_cache = (
            0 if args.dry_run else _remove_empty_cached_pages(runner)
        )
        page_lookup = _page_lookup(page_index)
        cached_page_ids = _cached_page_ids(runner)
        affected: dict[str, list[str]] = {}
        for qid, row in retrieval_rows.items():
            missing = _missing_pages(row, page_lookup, cached_page_ids)
            if row.get("status") != "ok" or missing:
                affected[qid] = missing

        question_by_qid = {str(item["qid"]): item for item in questions}
        missing_qids = [
            qid for qid in question_by_qid if qid not in retrieval_rows
        ]
        pending_qids = set(affected)
        if not args.only_affected:
            pending_qids.update(missing_qids)
        pending = [
            question
            for question in questions
            if str(question["qid"]) in pending_qids
        ]

        dry_summary = {
            "output_dir": str(output_dir),
            "scope": scope,
            "questions_selected": len(questions),
            "existing_retrieval_rows": len(retrieval_rows),
            "affected_existing_qids": len(affected),
            "missing_retrieval_qids": len(missing_qids),
            "pending_qids": len(pending),
            "cached_page_records": len(cached_page_ids),
            "empty_cache_records_found": empty_cache_records,
            "empty_cache_records_removed": removed_empty_cache,
            "affected_qids": sorted(affected),
            "missing_qids": sorted(missing_qids),
            "pending_qids_list": [str(item["qid"]) for item in pending],
        }
        if args.dry_run:
            print(json.dumps(dry_summary, ensure_ascii=False, indent=2))
            return 0

        if not args.no_backup:
            backup_dir = _backup_outputs(
                output_dir,
                retrieval_path,
                timings_path,
                qa_path,
            )
            dry_summary["backup_dir"] = str(backup_dir)

        if not pending:
            LOGGER.info("No retrieval rows require recovery")
        else:
            _run_retrieval_recovery(
                runner,
                pending,
                questions,
                retrieval_rows,
                timing_rows,
                retrieval_config,
                retrieval_config_hash,
                retrieval_path,
                timings_path,
                page_lookup,
                query_workers,
            )

        qa_summary: dict[str, Any] = {"enabled": False}
        if args.run_qa:
            qa_summary = _run_qa_recovery(
                args,
                config,
                questions,
                retrieval_rows,
                pending,
                qa_path,
                retrieval_config_hash,
            )

        cached_after = _cached_page_ids(runner)
        unresolved = {
            str(question["qid"]): _missing_pages(
                retrieval_rows.get(str(question["qid"]), {}),
                page_lookup,
                cached_after,
            )
            for question in questions
            if str(question["qid"]) in pending_qids
        }
        unresolved = {qid: pages for qid, pages in unresolved.items() if pages}
        summary = {
            **dry_summary,
            "retrieval_rows_after": len(retrieval_rows),
            "retrieval_rows_ok_after": sum(
                row.get("status") == "ok" for row in retrieval_rows.values()
            ),
            "unresolved_parse_pages_after": unresolved,
            "qa": qa_summary,
            "index_seconds": round(index_seconds, 3),
        }
        recovery_report = output_dir / "reports" / f"{scope}_recovery_summary.json"
        _write_json(recovery_report, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        runner.close()


def _run_retrieval_recovery(
    runner: OnDemandPerQueryRunner,
    pending: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    timing_rows: dict[str, dict[str, Any]],
    retrieval_config: dict[str, Any],
    retrieval_config_hash: str,
    retrieval_path: Path,
    timings_path: Path,
    page_lookup: dict[tuple[str, int], str],
    query_workers: int,
) -> None:
    def execute(question: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        qid = str(question["qid"])
        try:
            result = runner.run_query(question["question"], query_id=qid)
            row = _encode_retrieval_result(
                question,
                result,
                retrieval_config,
                retrieval_config_hash,
            )
            cached = _cached_page_ids(runner)
            missing = _missing_pages(row, page_lookup, cached)
            row["parse_missing_pages"] = missing
            row["parse_complete"] = not missing
            timing = {
                "qid": qid,
                **result.timing,
                "parse_missing_pages": missing,
                "parse_complete": not missing,
            }
        except Exception as error:  # noqa: BLE001 - checkpoint a failed QID
            message = f"{type(error).__name__}: {error}"
            row = {
                **question,
                "pipeline": "pdf_inspector -> bm25 -> baseline_legacy",
                "retrieval_config": retrieval_config,
                "retrieval_config_hash": retrieval_config_hash,
                "status": "error",
                "error": message,
                "hits": [],
                "chunks": [],
                "parse_complete": False,
            }
            timing = {"qid": qid, "status": "error", "error": message}
        return row, timing

    with ThreadPoolExecutor(max_workers=query_workers) as pool:
        futures = {pool.submit(execute, question): question for question in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            question = futures[future]
            row, timing = future.result()
            qid = str(question["qid"])
            retrieval_rows[qid] = row
            timing_rows[qid] = timing
            _write_ordered_jsonl(retrieval_path, retrieval_rows, questions)
            _write_ordered_jsonl(timings_path, timing_rows, questions)
            LOGGER.info(
                "Recovery retrieval %d/%d: qid=%s status=%s parse_complete=%s",
                number,
                len(pending),
                qid,
                row.get("status"),
                row.get("parse_complete"),
            )


def _run_qa_recovery(
    args: argparse.Namespace,
    config: dict[str, Any],
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    pending: list[dict[str, Any]],
    qa_path: Path,
    retrieval_config_hash: str,
) -> dict[str, Any]:
    from research.data_discovery.run_docbench_per_question import _answer_and_judge

    docbench_config = dict(config.get("docbench") or {})
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
    qa_workers = _positive(
        args.qa_workers or docbench_config.get("qa_workers") or 4,
        "qa-workers",
    )
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
    qa_rows = _read_latest_jsonl(qa_path)
    pending_qids = {str(item["qid"]) for item in pending}
    pending_qids.update(
        str(question["qid"])
        for question in questions
        if _should_retry_unanswerable(qa_rows.get(str(question["qid"])))
    )
    if args.qa_all_missing:
        pending_qids.update(
            str(question["qid"])
            for question in questions
            if not (
                str(question["qid"]) in qa_rows
                and qa_rows[str(question["qid"])].get("status") == "ok"
                and qa_rows[str(question["qid"])].get("qa_config_hash")
                == qa_config_hash
            )
        )
    qa_pending = [
        question
        for question in questions
        if str(question["qid"]) in pending_qids
    ]

    def execute(question: dict[str, Any]) -> dict[str, Any]:
        qid = str(question["qid"])
        return _answer_and_judge(
            question,
            retrieval_rows.get(qid) or {},
            generator_model=generator_model,
            judge_model=judge_model,
            max_context_chars=max_context_chars,
            max_unit_chars=max_unit_chars,
            max_output_tokens=max_output_tokens,
            skip_judge=args.skip_judge,
            qa_config_hash=qa_config_hash,
            previous_row=qa_rows.get(qid),
        )

    with ThreadPoolExecutor(max_workers=qa_workers) as pool:
        futures = {pool.submit(execute, question): question for question in qa_pending}
        for number, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            qid = str(row["qid"])
            qa_rows[qid] = row
            _write_ordered_jsonl(qa_path, qa_rows, questions)
            LOGGER.info(
                "Recovery QA %d/%d: qid=%s status=%s",
                number,
                len(qa_pending),
                qid,
                row.get("status"),
            )
    _write_ordered_jsonl(qa_path, qa_rows, questions)
    return {
        "enabled": True,
        "qa_pending": len(qa_pending),
        "qa_rows_after": len(qa_rows),
        "qa_ok_after": sum(row.get("status") == "ok" for row in qa_rows.values()),
        "qa_config_hash": qa_config_hash,
    }


def _page_lookup(page_index: Any) -> dict[tuple[str, int], str]:
    return {
        (str(Path(page.file_path).resolve()), int(page.page_index)): page.page_id
        for page in page_index.pages
    }


def _cached_page_ids(runner: OnDemandPerQueryRunner) -> set[str]:
    with runner._state_lock:  # noqa: SLF001 - recovery validates the persisted cache
        return {
            page_id
            for page_id, record in runner._records_by_page.items()  # noqa: SLF001
            if _record_page_text(record)
        }


def _remove_empty_cached_pages(runner: OnDemandPerQueryRunner) -> int:
    with runner._state_lock:  # noqa: SLF001 - repair before worker submission
        empty = [
            page_id
            for page_id, record in runner._records_by_page.items()  # noqa: SLF001
            if not _record_page_text(record)
        ]
        for page_id in empty:
            runner._records_by_page.pop(page_id, None)  # noqa: SLF001
            runner._page_texts.pop(page_id, None)  # noqa: SLF001
        if empty:
            runner._save_parser_cache()  # noqa: SLF001

    if empty:
        with runner._chunk_lock:  # noqa: SLF001
            stale = [
                chunk_id
                for chunk_id, chunk in runner._chunks_by_id.items()  # noqa: SLF001
                if chunk.page_id in empty
            ]
            for chunk_id in stale:
                runner._chunks_by_id.pop(chunk_id, None)  # noqa: SLF001
            runner._chunk_index = None  # noqa: SLF001
            runner._chunk_embedder = None  # noqa: SLF001
            if stale:
                runner._save_chunk_cache()  # noqa: SLF001
            if runner._chunks_by_id:  # noqa: SLF001
                runner._rebuild_chunk_index()  # noqa: SLF001
    return len(empty)


def _empty_cached_page_count(runner: OnDemandPerQueryRunner) -> int:
    with runner._state_lock:  # noqa: SLF001 - read-only cache validation
        return sum(
            not _record_page_text(record)
            for record in runner._records_by_page.values()  # noqa: SLF001
        )


def _missing_pages(
    row: dict[str, Any],
    page_lookup: dict[tuple[str, int], str],
    cached_page_ids: set[str],
) -> list[str]:
    missing: set[str] = set()
    for path, indices in (row.get("selected_pages") or {}).items():
        for index in indices:
            page_id = page_lookup.get((str(Path(path).resolve()), int(index)))
            if page_id is None:
                missing.add(f"{path}#page_index={index}")
            elif page_id not in cached_page_ids:
                missing.add(page_id)
    return sorted(missing)


def _backup_outputs(
    output_dir: Path,
    retrieval_path: Path,
    timings_path: Path,
    qa_path: Path,
) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_{}".format(
        time.time_ns() % 1_000_000_000
    )
    backup_dir = output_dir / "recovery-backups" / stamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for path in (retrieval_path, timings_path, qa_path):
        if path.is_file():
            shutil.copy2(path, backup_dir / path.name)
    return backup_dir


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
    parser.add_argument("--query-workers", type=int)
    parser.add_argument("--qa-workers", type=int)
    parser.add_argument("--kdl-microbatch-window-seconds", type=float)
    parser.add_argument("--kdl-microbatch-max-pages", type=int)
    parser.add_argument("--kdl-max-workers", type=int)
    parser.add_argument("--kdl-render-processes", type=int)
    parser.add_argument("--kdl-bbox-max-workers", type=int)
    parser.add_argument("--kdl-request-workers", type=int)
    parser.add_argument("--kdl-request-batch-size", type=int)
    parser.add_argument("--kdl-max-model-sequences", type=int)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--max-unit-chars", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--generator")
    parser.add_argument("--judge")
    parser.add_argument(
        "--only-affected",
        action="store_true",
        help="Do not include QIDs absent from the existing retrieval JSONL.",
    )
    parser.add_argument(
        "--run-qa",
        action="store_true",
        help="Run QA only for QIDs processed by this recovery.",
    )
    parser.add_argument(
        "--qa-all-missing",
        action="store_true",
        help="With --run-qa, also process every missing/non-ok QA row.",
    )
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
