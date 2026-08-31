"""Run the DocBench page-context pipeline.

This runner is the page-context arm of the DocBench discovery experiment:

    pdf-inspector page index -> BM25 page discovery ->
    KDL + pdf-inspector accurate parse -> page text -> generator -> judge

It deliberately does not import or execute the chunking, embedding, or
chunk-level hybrid-retrieval stages.  The BM25 step is the light page
discovery step; there is no second retrieval after accurate parsing.

Results are written under a ``pages``-specific set of files so this runner can
share an output directory with the baseline without overwriting its manifest or
QA rows.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import logging
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import (  # noqa: E402
    PageIndex,
    run_from_parse_artifacts,
    run_selected_pages,
)
from research.data_discovery.run_docbench_e2e import (  # noqa: E402
    DOMAIN_ORDER,
    DOCBENCH_JUDGE_PROMPT,
    TYPE_ORDER,
    _apply_kdl_overrides,
    _check_vllm_endpoint,
    _load_docbench,
    _load_or_build_page_index,
    _parse_score,
    _resolve_path,
    _generate_with_unanswerable_retry,
    _should_retry_unanswerable,
    _type_group,
)
from src.evaluation.generate import ContextChunk  # noqa: E402
from src.evaluation.llm import complete  # noqa: E402
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402
from src.ingestion.parsing.kdl_health import (  # noqa: E402
    KDLHostHealth,
    KDLHostUnavailableError,
)
from src.utils.observability import (  # noqa: E402
    JsonEventLogger,
    configure_run_logging,
    utc_now_iso,
)


LOGGER = logging.getLogger("docbench_pages")
PIPELINE_NAME = (
    "pdf_inspector -> bm25_pages -> KDL+pdf_inspector -> "
    "page_text -> generator -> DocBench judge"
)


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_started = time.perf_counter()

    load_dotenv_file(ROOT)
    config_path = _resolve_path(args.config)
    assert config_path is not None
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
        or ROOT / "data" / "benchmark" / "docbench_pages"
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logging(
        output_dir / "logs" / f"{scope}_on_demand_just_parse.log",
        level=getattr(logging, str(args.log_level).upper()),
    )
    event_logger = JsonEventLogger(
        output_dir / "logs" / f"{scope}_events.jsonl",
        run_name="on-demand-just-parse",
    )
    run_started_at = utc_now_iso()
    event_logger.emit("run_started", scope=scope, config=str(config_path))

    top_k_pages = _positive(
        args.top_k_pages or docbench_config.get("top_k_pages") or 10,
        "top-k-pages",
    )
    workers = _positive(
        args.workers
        or docbench_config.get("qa_workers")
        or docbench_config.get("query_workers")
        or 4,
        "workers",
    )
    max_context_chars = _positive(
        args.max_context_chars
        or docbench_config.get("max_context_chars")
        or 12000,
        "max-context-chars",
    )
    max_page_chars = _positive(
        args.max_page_chars
        or docbench_config.get("max_unit_chars")
        or max_context_chars,
        "max-page-chars",
    )
    max_output_tokens = _positive(
        args.max_output_tokens
        or docbench_config.get("max_output_tokens")
        or 512,
        "max-output-tokens",
    )
    generator_model = str(
        args.generator
        or docbench_config.get("generator")
        or "deepseek/deepseek-v4-flash"
    )
    judge_model = str(args.judge or docbench_config.get("judge") or "openai/gpt-4o")
    if not args.skip_judge and generator_model == judge_model:
        raise ValueError("generator and judge should be different models")

    _require_env("OPENROUTER_API_KEY")
    # Parser artifacts make retrying generation independent of the KDL host.
    # In that mode no parser request is made, so neither the endpoint variable
    # nor the /models health check is required.
    if not args.reuse_parser_artifacts:
        _require_env("VLLM_API_BASE")
        if not args.skip_endpoint_check:
            _check_vllm_endpoint()

    documents, all_questions = _load_docbench(docbench_root)
    selected_documents = _select_documents(documents, args.max_documents)
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
    question_ids = [str(item["qid"]) for item in questions]
    if len(set(question_ids)) != len(question_ids):
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

    parser_config = resolve_parser_config(
        ROOT,
        config.get("parsing") or {},
        output_dir / "parser-assets" / f"pages_{scope}",
    )
    parser_config = _apply_kdl_overrides(parser_config, args)
    parser_config = _with_kdl_event_log(
        parser_config, output_dir / "logs" / f"{scope}_kdl_events.jsonl"
    )
    if parser_config.get("provider") != "kdl_pdf_inspector":
        raise ValueError(
            "DocBench pages pipeline requires parsing.provider=kdl_pdf_inspector"
        )

    discovery_started = time.perf_counter()
    hits_by_qid = {
        str(question["qid"]): page_index.search(
            question["question"], top_k=top_k_pages
        )
        for question in questions
    }
    discovery_seconds = time.perf_counter() - discovery_started
    selected = _union_pages(hits_by_qid)
    discovery_rows = {
        str(question["qid"]): {
            **question,
            "contract_version": "docbench-pages-v1",
            "pipeline": PIPELINE_NAME,
            "retrieval_scope": scope,
            "status": "ok",
            "hits": [hit.as_dict() for hit in hits_by_qid[str(question["qid"])]],
            "selected_pages": _selected_page_records(
                hits_by_qid[str(question["qid"])]
            ),
        }
        for question in questions
    }
    discovery_path = output_dir / "discovery" / f"{scope}_pages.jsonl"
    _write_ordered_jsonl(discovery_path, discovery_rows, questions)
    LOGGER.info(
        "Discovery: pages=%d questions=%d selected_unique_pages=%d scope=%s",
        len(page_index.pages),
        len(questions),
        sum(len(indices) for indices in selected.values()),
        scope,
    )

    parse_started = time.perf_counter()
    quarantined_pages = 0
    kdl_health_summary: dict[str, Any] = {}
    if args.reuse_parser_artifacts:
        parsed_pipeline = run_from_parse_artifacts(
            selected,
            parser_artifacts_dir=args.reuse_parser_artifacts,
            project_root=ROOT,
            chunking_config=None,
        )
        page_texts = _page_texts(parsed_pipeline.enriched.enriched_data, page_index)
        quarantined_pages = len(parsed_pipeline.ingestion.quarantined_documents)
    else:
        (
            page_texts,
            quarantined_pages,
            kdl_health_summary,
        ) = _parse_selected_pages_checkpointed(
            selected,
            parser_config=parser_config,
            project_root=ROOT,
            work_dir=output_dir / "work" / f"pages_{scope}",
            cache_path=output_dir / "cache" / f"pages_{scope}" / "parsed_pages.json",
            corpus_fingerprint=corpus_fingerprint,
            parser_config_hash=_hash_payload(parser_config),
            page_index=page_index,
            event_logger=event_logger,
        )
    parse_seconds = time.perf_counter() - parse_started
    if not page_texts:
        raise RuntimeError("Accurate ingestion returned no page text")
    LOGGER.info(
        "Parse: parsed_pages=%d quarantined=%d parse_seconds=%.2f",
        len(page_texts),
        quarantined_pages,
        parse_seconds,
    )

    qa_path = output_dir / "qa" / f"{scope}_pages.jsonl"
    qa_config = {
        "arm": "pages",
        "pipeline": PIPELINE_NAME,
        "generator": generator_model,
        "judge": None if args.skip_judge else judge_model,
        "max_context_chars": max_context_chars,
        "max_page_chars": max_page_chars,
        "max_output_tokens": max_output_tokens,
        "skip_judge": bool(args.skip_judge),
        "corpus_fingerprint": corpus_fingerprint,
        "parser_config_hash": _hash_payload(parser_config),
    }
    qa_config_hash = _hash_payload(qa_config)
    qa_rows = {
        qid: row
        for qid, row in _read_latest_jsonl(qa_path).items()
        if row.get("qa_config_hash") == qa_config_hash
    }
    generation_seconds, judge_seconds = _run_qa(
        questions,
        discovery_rows,
        page_texts,
        qa_rows,
        qa_path,
        generator_model=generator_model,
        judge_model=judge_model,
        max_context_chars=max_context_chars,
        max_page_chars=max_page_chars,
        max_output_tokens=max_output_tokens,
        workers=workers,
        skip_judge=args.skip_judge,
        qa_config_hash=qa_config_hash,
    )

    report = _build_report(
        questions,
        discovery_rows,
        qa_rows,
        scope=scope,
        index_documents=index_documents,
        selected_documents=selected_documents,
        page_index=page_index,
        parsed_page_count=len(page_texts),
        quarantined_count=quarantined_pages,
        generator_model=generator_model,
        judge_model=None if args.skip_judge else judge_model,
        top_k_pages=top_k_pages,
    )
    timing = {
        "pipeline": PIPELINE_NAME,
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": utc_now_iso(),
        "discovery_seconds": round(discovery_seconds, 3),
        "index_seconds": round(index_seconds, 3),
        "parse_clean_enrich_seconds": round(parse_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "judge_seconds": round(judge_seconds, 3),
        "total_seconds": round(time.perf_counter() - run_started, 3),
        "questions": len(questions),
        "workers": workers,
        "kdl_host_health": kdl_health_summary,
    }
    report_path = output_dir / "reports" / f"{scope}_pages.json"
    timing_path = output_dir / "reports" / f"{scope}_pages_timing_summary.json"
    _write_json(report_path, report)
    _write_json(timing_path, timing)
    manifest = {
        "contract_version": "docbench-pages-v1",
        "pipeline": PIPELINE_NAME,
        "post_parse_chunking": False,
        "post_parse_embedding": False,
        "post_parse_retrieval": False,
        "retrieval_scope": scope,
        "docbench_root": str(docbench_root.resolve()),
        "config": str(config_path),
        "documents_available": len(documents),
        "documents_indexed": len(index_documents),
        "documents_selected": len(selected_documents),
        "questions_selected": len(questions),
        "indexed_pages": len(page_index.pages),
        "parsed_pages": len(page_texts),
        "quarantined_pages": quarantined_pages,
        "corpus_fingerprint": corpus_fingerprint,
        "top_k_pages": top_k_pages,
        "parser_config": parser_config,
        "parser_artifacts_reused": bool(args.reuse_parser_artifacts),
        "parser_artifacts_dir": (
            str(args.reuse_parser_artifacts)
            if args.reuse_parser_artifacts
            else None
        ),
        "kdl_host_health": kdl_health_summary,
        "qa_config": qa_config,
        "qa_config_hash": qa_config_hash,
        "discovery_output": str(discovery_path),
        "qa_output": str(qa_path),
        "report_output": str(report_path),
        "timing_summary": timing,
        "report": report,
    }
    _write_json(output_dir / f"manifest_{scope}_pages.json", manifest)
    _update_summary(output_dir / "summary.json", f"{scope}_pages", report, timing)
    event_logger.emit("run_completed", scope=scope, questions=len(questions))
    print(json.dumps({"report": report, "timing": timing}, ensure_ascii=False, indent=2))
    return 0


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
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--workers",
        "--qa-workers",
        dest="workers",
        type=int,
        help="Concurrent generator/judge question workers.",
    )
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument(
        "--max-page-chars",
        "--max-unit-chars",
        dest="max_page_chars",
        type=int,
        help="Maximum characters retained from one parsed page before generation.",
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--generator")
    parser.add_argument("--judge")
    parser.add_argument("--kdl-max-workers", type=int)
    parser.add_argument("--kdl-render-processes", type=int)
    parser.add_argument("--kdl-bbox-max-workers", type=int)
    parser.add_argument("--kdl-request-workers", type=int)
    parser.add_argument("--kdl-request-batch-size", type=int)
    parser.add_argument("--kdl-max-model-sequences", type=int)
    parser.add_argument("--kdl-host-failure-threshold", type=int)
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument(
        "--reuse-parser-artifacts",
        type=Path,
        help="Reuse persisted KDL result.json files instead of calling KDL.",
    )
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _select_documents(
    documents: list[dict[str, Any]], max_documents: int | None
) -> list[dict[str, Any]]:
    if max_documents is None:
        return documents
    if max_documents <= 0:
        raise ValueError("max-documents must be positive")
    return documents[:max_documents]


def _union_pages(hits_by_qid: dict[str, list[Any]]) -> dict[str, list[int]]:
    selected: dict[str, set[int]] = defaultdict(set)
    for hits in hits_by_qid.values():
        for hit in hits:
            selected[hit.evidence.file_path].add(int(hit.evidence.page_index))
    return {
        path: sorted(indices)
        for path, indices in sorted(selected.items())
        if indices
    }


def _with_kdl_event_log(config: dict[str, Any], path: Path) -> dict[str, Any]:
    result = dict(config)
    kdl = dict(result.get("kdl") or {})
    kdl["event_log_path"] = str(path)
    result["kdl"] = kdl
    return result


def _parse_selected_pages_checkpointed(
    selected: dict[str, list[int]],
    *,
    parser_config: dict[str, Any],
    project_root: Path,
    work_dir: Path,
    cache_path: Path,
    corpus_fingerprint: str,
    parser_config_hash: str,
    page_index: PageIndex,
    event_logger: JsonEventLogger,
) -> tuple[dict[str, str], int, dict[str, Any]]:
    """Parse one page at a time and checkpoint after every successful page.

    A KDL host failure can happen in the middle of a large page union.  Page
    granularity ensures that completed pages are durable before the next KDL
    request starts, so a rerun only submits missing pages.
    """

    page_lookup = {
        (str(Path(page.file_path).resolve()), int(page.page_index)): page.page_id
        for page in page_index.pages
    }
    ordered_pages = [
        (
            str(Path(path).resolve()),
            int(page_index),
            page_lookup[
                (str(Path(path).resolve()), int(page_index))
            ],
        )
        for path, indices in sorted(selected.items())
        for page_index in sorted(indices)
        if (str(Path(path).resolve()), int(page_index)) in page_lookup
    ]
    cache: dict[str, Any] = {
        "contract_version": "docbench-pages-parse-checkpoint-v1",
        "corpus_fingerprint": corpus_fingerprint,
        "parser_config_hash": parser_config_hash,
        "records": {},
        "failed": {},
    }
    if cache_path.is_file():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(loaded, dict)
                and loaded.get("corpus_fingerprint") == corpus_fingerprint
                and loaded.get("parser_config_hash") == parser_config_hash
            ):
                cache.update(loaded)
        except (OSError, ValueError):
            LOGGER.warning("Ignoring unreadable page parse checkpoint: %s", cache_path)

    records = dict(cache.get("records") or {})
    failed = dict(cache.get("failed") or {})
    pending = [item for item in ordered_pages if item[2] not in records]
    LOGGER.info(
        "Parse checkpoint: pending=%d/%d cached=%d",
        len(pending),
        len(ordered_pages),
        len(records),
    )
    runtime_parser_config = dict(parser_config)
    runtime_kdl = dict(runtime_parser_config.get("kdl") or {})
    runtime_kdl["_host_health"] = KDLHostHealth(
        str(
            runtime_kdl.get("endpoint_url")
            or os.getenv("VLLM_API_BASE")
            or os.getenv("KDL_NANO_ENDPOINT_URL")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/"),
        failure_threshold=int(runtime_kdl.get("host_failure_threshold", 3)),
        abort_on_open=(
            str(runtime_kdl.get("host_abort_on_open", True)).lower()
            not in {"0", "false", "no", "off"}
        ),
        event_logger=(
            JsonEventLogger(runtime_kdl.get("event_log_path"), run_name="kdl")
            if runtime_kdl.get("event_log_path")
            else None
        ),
    )
    runtime_parser_config["kdl"] = runtime_kdl
    for number, (path, page_index_value, page_id) in enumerate(pending, start=1):
        started = time.perf_counter()
        event_logger.emit(
            "page_parse_started",
            page_id=page_id,
            page_index=page_index_value,
            pending_number=number,
            pending_total=len(pending),
        )
        try:
            result = run_selected_pages(
                {path: [page_index_value]},
                parser_config=runtime_parser_config,
                project_root=project_root,
                work_dir=(
                    work_dir
                    / f"page_{hashlib.sha1(page_id.encode()).hexdigest()[:16]}"
                ),
                one_page_inputs=True,
                chunking_config=None,
            )
            if result.enriched.enriched_data:
                record = result.enriched.enriched_data[0].__dict__
                records[page_id] = record
                failed.pop(page_id, None)
                cache["records"] = records
                cache["failed"] = failed
                _write_json(cache_path, cache)
                elapsed = time.perf_counter() - started
                event_logger.emit(
                    "page_parse_completed",
                    page_id=page_id,
                    elapsed_seconds=round(elapsed, 6),
                )
                LOGGER.info(
                    "Parse page %d/%d: page_id=%s status=ok elapsed=%.2fs",
                    number,
                    len(pending),
                    page_id,
                    elapsed,
                )
            else:
                failed[page_id] = {
                    "error": "KDL returned no enriched record",
                    "timestamp_utc": utc_now_iso(),
                }
                cache["failed"] = failed
                _write_json(cache_path, cache)
                LOGGER.warning("Parse page %s returned no enriched record", page_id)
        except KDLHostUnavailableError as error:
            cache["records"] = records
            cache["failed"] = failed
            cache["last_error"] = {
                "timestamp_utc": utc_now_iso(),
                "page_id": page_id,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(cache_path, cache)
            event_logger.emit(
                "run_aborted",
                page_id=page_id,
                error_type=type(error).__name__,
                error=str(error),
            )
            LOGGER.error(
                "Stopping page pipeline at page_id=%s; checkpoint preserved for resume",
                page_id,
            )
            raise
        except Exception as error:  # noqa: BLE001 - retain page-level progress
            failed[page_id] = {
                "error": f"{type(error).__name__}: {error}",
                "timestamp_utc": utc_now_iso(),
            }
            cache["failed"] = failed
            _write_json(cache_path, cache)
            event_logger.emit(
                "page_parse_failed",
                page_id=page_id,
                error_type=type(error).__name__,
                error=str(error),
            )
            LOGGER.exception("Parse page failed page_id=%s", page_id)

    page_texts = _page_texts_from_dicts(records.values(), page_index)
    return page_texts, len(failed), runtime_kdl["_host_health"].snapshot()


def _selected_page_records(hits: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "page_id": hit.page_id,
            "source_uri": hit.evidence.source_uri,
            "page_index": hit.evidence.page_index,
            "page_number": hit.evidence.page_number,
            "rank": hit.rank,
            "score": hit.score,
        }
        for hit in hits
    ]


def _page_texts(records: list[Any], index: PageIndex) -> dict[str, str]:
    page_ids = {
        (str(Path(page.file_path).resolve()), int(page.page_index)): page.page_id
        for page in index.pages
    }
    output: dict[str, str] = {}
    for record in records:
        metadata = record.metadata or {}
        source_metadata = metadata.get("source_metadata") or {}
        path = source_metadata.get("discovery_original_path")
        indices = source_metadata.get("discovery_page_indices")
        if not path or not isinstance(indices, list) or len(indices) != 1:
            continue
        page_id = page_ids.get((str(Path(path).resolve()), int(indices[0])))
        if not page_id:
            continue
        text = "\n\n".join(
            str(row.get("text") or "").strip()
            for row in (record.rows or [])
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ).strip()
        if text:
            output[page_id] = text
    return output


def _page_texts_from_dicts(records: Any, index: PageIndex) -> dict[str, str]:
    page_ids = {
        (str(Path(page.file_path).resolve()), int(page.page_index)): page.page_id
        for page in index.pages
    }
    output: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        metadata = dict(record.get("metadata") or {})
        source_metadata = metadata.get("source_metadata") or {}
        path = source_metadata.get("discovery_original_path")
        indices = source_metadata.get("discovery_page_indices")
        if not path or not isinstance(indices, list) or len(indices) != 1:
            continue
        page_id = page_ids.get((str(Path(path).resolve()), int(indices[0])))
        if not page_id:
            continue
        text = "\n\n".join(
            str(row.get("text") or "").strip()
            for row in (record.get("rows") or [])
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ).strip()
        if text:
            output[page_id] = text
    return output


def _run_qa(
    questions: list[dict[str, Any]],
    discovery_rows: dict[str, dict[str, Any]],
    page_texts: dict[str, str],
    qa_rows: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    generator_model: str,
    judge_model: str,
    max_context_chars: int,
    max_page_chars: int,
    max_output_tokens: int,
    workers: int,
    skip_judge: bool,
    qa_config_hash: str,
) -> tuple[float, float]:
    pending = [
        question
        for question in questions
        if not (
            str(question["qid"]) in qa_rows
            and qa_rows[str(question["qid"])].get("status") == "ok"
            and not _should_retry_unanswerable(qa_rows[str(question["qid"])])
        )
    ]
    LOGGER.info("Pages QA: pending=%d/%d workers=%d", len(pending), len(questions), workers)

    def execute(question: dict[str, Any]) -> dict[str, Any]:
        qid = str(question["qid"])
        previous_row = qa_rows.get(qid)
        started_at = utc_now_iso()
        discovery = discovery_rows[qid]
        context = _context_for_question(
            discovery,
            page_texts,
            max_page_chars=max_page_chars,
        )
        generation_started = time.perf_counter()
        generation, retry_count, initial_answer = _generate_with_unanswerable_retry(
            qid,
            question["question"],
            context,
            model=generator_model,
            max_chars=max_context_chars,
            max_output_tokens=max_output_tokens,
            previous_row=previous_row,
        )
        generation_seconds = time.perf_counter() - generation_started
        row: dict[str, Any] = {
            **question,
            "contract_version": "docbench-pages-v1",
            "pipeline": PIPELINE_NAME,
            "arm": "pages",
            "status": "ok" if not generation.error else "error",
            "generator": generator_model,
            "judge": None if skip_judge else judge_model,
            "sys_ans": generation.answer,
            "chunks_used": generation.chunks_used,
            "chars_used": generation.chars_used,
            "context_doc_ids": generation.context_doc_ids,
            "context_page_ids": [item.chunk_id for item in context],
            "context_unit_ids": [item.chunk_id for item in context],
            "discovered_page_count": len(discovery.get("hits") or []),
            "parsed_context_page_count": len(context),
            "qa_config_hash": qa_config_hash,
            "unanswerable_retry_count": retry_count,
            "error": generation.error,
            "judge_raw": "",
            "score": None,
            "generation_seconds": round(generation_seconds, 6),
            "judge_seconds": 0.0,
            "started_at_utc": started_at,
        }
        if initial_answer is not None:
            row["initial_sys_ans"] = initial_answer
            LOGGER.info("QA qid=%s abstained; regenerated answer once", qid)
        if generation.error:
            row["finished_at_utc"] = utc_now_iso()
            return row
        if not generation.answer.strip() or (
            generation.abstained and initial_answer is None
        ):
            row.update({"score": 0, "judge_raw": "abstained"})
            row["finished_at_utc"] = utc_now_iso()
            return row
        if skip_judge:
            row.update({"judge_raw": "skipped"})
            row["finished_at_utc"] = utc_now_iso()
            return row

        prompt = (
            DOCBENCH_JUDGE_PROMPT.replace("{{question}}", question["question"])
            .replace("{{sys_ans}}", generation.answer)
            .replace("{{ref_ans}}", question["answer"])
            .replace("{{ref_text}}", question["evidence"])
        )
        judge_started = time.perf_counter()
        try:
            judge_raw = complete(
                judge_model,
                prompt,
                temperature=0.0,
                max_output_tokens=16,
            )
            row.update({"judge_raw": judge_raw, "score": _parse_score(judge_raw)})
        except Exception as error:  # noqa: BLE001 - persist per-question failures
            row.update({"status": "error", "error": repr(error)})
        row["judge_seconds"] = round(time.perf_counter() - judge_started, 6)
        row["finished_at_utc"] = utc_now_iso()
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(execute, question): question
            for question in pending
        }
        for number, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            qa_rows[str(row["qid"])] = row
            _write_ordered_jsonl(output_path, qa_rows, questions)
            LOGGER.info(
                "Pages QA %d/%d: qid=%s status=%s score=%s",
                number,
                len(pending),
                row["qid"],
                row["status"],
                row.get("score"),
            )
    _write_ordered_jsonl(output_path, qa_rows, questions)
    return (
        sum(float(row.get("generation_seconds") or 0.0) for row in qa_rows.values()),
        sum(float(row.get("judge_seconds") or 0.0) for row in qa_rows.values()),
    )


def _context_for_question(
    discovery: dict[str, Any],
    page_texts: dict[str, str],
    *,
    max_page_chars: int,
) -> list[ContextChunk]:
    context: list[ContextChunk] = []
    for hit in discovery.get("hits") or []:
        page_id = str(hit.get("page_id") or "")
        text = page_texts.get(page_id, "").strip()
        if not text:
            continue
        context.append(
            ContextChunk(
                chunk_id=page_id,
                doc_id=str(hit.get("source_uri") or ""),
                text=text[:max_page_chars],
                score=float(hit.get("score") or 0.0),
            )
        )
    return context


def _build_report(
    questions: list[dict[str, Any]],
    discovery_rows: dict[str, dict[str, Any]],
    qa_rows: dict[str, dict[str, Any]],
    *,
    scope: str,
    index_documents: list[dict[str, Any]],
    selected_documents: list[dict[str, Any]],
    page_index: PageIndex,
    parsed_page_count: int,
    quarantined_count: int,
    generator_model: str,
    judge_model: str | None,
    top_k_pages: int,
) -> dict[str, Any]:
    ordered = [
        qa_rows[str(question["qid"])]
        for question in questions
        if str(question["qid"]) in qa_rows
    ]
    scored = [
        row
        for row in ordered
        if row.get("status") == "ok" and row.get("score") is not None
    ]
    scores = [int(row["score"]) for row in scored]
    by_domain: dict[str, list[int]] = defaultdict(list)
    by_type: dict[str, list[int]] = defaultdict(list)
    for row in scored:
        by_domain[str(row.get("domain") or "Unknown")].append(int(row["score"]))
        by_type[str(row.get("type_group") or _type_group(row.get("type")))].append(
            int(row["score"])
        )
    gold_hits = []
    context_pages = []
    for question in questions:
        discovery = discovery_rows.get(str(question["qid"]), {})
        gold_hits.append(
            int(
                any(
                    hit.get("source_uri") == question["doc_id"]
                    for hit in discovery.get("hits") or []
                )
            )
        )
        context_pages.append(
            len(qa_rows.get(str(question["qid"]), {}).get("context_page_ids") or [])
        )
    return {
        "arm": "pages",
        "pipeline": PIPELINE_NAME,
        "retrieval_scope": scope,
        "generator": generator_model,
        "judge": judge_model,
        "documents": len(selected_documents),
        "indexed_documents": len(index_documents),
        "indexed_pages": len(page_index.pages),
        "parsed_pages": parsed_page_count,
        "quarantined_pages": quarantined_count,
        "questions_expected": len(questions),
        "questions_completed": len(scored),
        "errors": sum(row.get("status") != "ok" for row in ordered),
        "missing": max(len(questions) - len(scored), 0),
        "accuracy": _mean(scores),
        "score_1": scores.count(1),
        "score_0": scores.count(0),
        "bm25_gold_document_hit_rate": _mean(gold_hits),
        "mean_context_pages": _mean(context_pages),
        "top_k_pages": top_k_pages,
        "post_parse_chunking": False,
        "post_parse_embedding": False,
        "post_parse_retrieval": False,
        "accuracy_by_domain": {
            key: _mean(by_domain[key]) for key in DOMAIN_ORDER if key in by_domain
        },
        "accuracy_by_type_group": {
            key: _mean(by_type[key]) for key in TYPE_ORDER if key in by_type
        },
    }


def _mean(values: list[int | float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _hash_payload(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _read_latest_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["qid"])] = row
    return rows


def _write_ordered_jsonl(
    path: Path,
    rows: dict[str, dict[str, Any]],
    questions: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = "".join(
        json.dumps(rows[str(question["qid"])], ensure_ascii=False) + "\n"
        for question in questions
        if str(question["qid"]) in rows
    )
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _update_summary(
    path: Path, run_name: str, report: dict[str, Any], timing: dict[str, Any]
) -> None:
    value: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                value = loaded
        except (OSError, ValueError):
            pass
    runs = value.setdefault("runs", {})
    runs[run_name] = {"report": report, "timing_summary": timing}
    _write_json(path, value)


def _positive(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"{name} is required; set it in .env or the shell")


if __name__ == "__main__":
    raise SystemExit(main())
