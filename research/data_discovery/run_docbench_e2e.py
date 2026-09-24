"""Run the DocBench on-demand basic pipeline from the repository.

The pipeline is intentionally page selective:

    pdf-inspector page preparation -> BM25 page retrieval ->
    KDL + pdf-inspector for missing pages -> fixed 512/128 chunks ->
    text-embedding-3-small -> hybrid BM25+dense baseline_legacy ->
    generator -> DocBench judge

``--retrieval-scope file`` restricts each question to the DocBench document it
belongs to. ``--retrieval-scope lake`` searches the complete DocBench PDF lake
without a per-question document filter.
When ``--ppocr-jsonl`` is supplied, PP-OCRv5 text is merged into the light
page index before BM25; this path does not invoke Tesseract.
The KDL endpoint can be remote (for example a vLLM server exposed from
Colab); PDF rendering, page indexing, chunking, embeddings, generation and
judging run on the local machine.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Any, Callable
import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.on_demand_per_query import (  # noqa: E402
    OnDemandPerQueryRunner,
    OnDemandQueryResult,
)
from research.data_discovery.pipeline import (  # noqa: E402
    PageIndex,
    PdfInspectorPageParser,
    build_page_index,
)
from src.evaluation.generate import (  # noqa: E402
    ABSTAIN,
    ContextChunk,
    Generation,
    generate,
)
from src.evaluation.benchmarks.vidore_v3_judge import ANSWER_PROMPT as VIDORe_ANSWER_PROMPT  # noqa: E402
from src.evaluation.llm import complete  # noqa: E402
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402
from src.ingestion.parsing.kdl_health import KDLHostUnavailableError  # noqa: E402
from src.utils.observability import (  # noqa: E402
    JsonEventLogger,
    configure_run_logging,
    utc_now_iso,
)


LOGGER = logging.getLogger("docbench_on_demand")

DOMAIN_ORDER = ("Aca.", "Fin.", "Gov.", "Laws", "News")
DOMAIN_FILE_RANGES = {
    "Aca.": range(0, 49),
    "Fin.": range(49, 89),
    "Gov.": range(89, 133),
    "Laws": range(133, 179),
    "News": range(179, 229),
}
TYPE_ORDER = ("Text.", "Multi.", "Meta.", "Una.")

DOCBENCH_JUDGE_PROMPT = """Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with 0, 0.5, or 1: 0 denotes wrong, 0.5 denotes partially correct,
   and 1 denotes fully correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {{question}}
User Answer: {{sys_ans}}
Reference Answer: {{ref_ans}}
Reference Text: {{ref_text}}

Evaluation Form (score ONLY):
- Correctness (0, 0.5, or 1):"""


def _make_query_page_scope(
    page_index: PageIndex,
    questions: list[dict[str, Any]],
    scope: str,
) -> Callable[[str, str], set[str] | None]:
    """Build the page filter for a DocBench retrieval run.

    In ``file`` mode, each question can retrieve only pages from its own
    ``doc_id``.  In ``lake`` mode, returning ``None`` preserves global-corpus
    retrieval.  Page ``source_uri`` values are set to DocBench ``doc_id`` by
    ``_load_or_build_page_index``.
    """
    normalized_scope = str(scope).lower()
    if normalized_scope not in {"file", "lake"}:
        raise ValueError("retrieval scope must be 'file' or 'lake'")

    if normalized_scope == "lake":
        return lambda _qid, _query: None

    question_doc_ids = {
        str(question["qid"]): str(question["doc_id"])
        for question in questions
    }
    page_ids_by_doc: dict[str, set[str]] = defaultdict(set)
    for page in page_index.pages:
        page_ids_by_doc[str(page.source_uri)].add(str(page.page_id))

    missing_docs = sorted(
        {
            doc_id
            for doc_id in question_doc_ids.values()
            if doc_id not in page_ids_by_doc
        }
    )
    if missing_docs:
        raise RuntimeError(
            "DocBench question documents are missing from the page index: "
            + ", ".join(missing_docs[:10])
        )

    def page_scope_for_query(qid: str, _query: str) -> set[str] | None:
        try:
            doc_id = question_doc_ids[str(qid)]
        except KeyError as error:
            raise KeyError(f"No DocBench document mapping for qid={qid!r}") from error
        return page_ids_by_doc[doc_id]

    return page_scope_for_query


def _is_unanswerable_answer(answer: Any) -> bool:
    """Return whether an answer contains DocBench's abstention sentinel."""

    return ABSTAIN in str(answer or "").upper()


def _unanswerable_retry_count(row: dict[str, Any] | None) -> int:
    """Read the durable, at-most-once retry marker from a QA row."""

    if not row:
        return 0
    try:
        return max(int(row.get("unanswerable_retry_count", 0) or 0), 0)
    except (TypeError, ValueError):
        return 0


def _should_retry_unanswerable(row: dict[str, Any] | None) -> bool:
    """Whether a completed row needs the one allowed unanswerable retry."""

    return bool(row) and _is_unanswerable_answer(row.get("sys_ans")) and (
        _unanswerable_retry_count(row) < 1
    )


def _generate_with_unanswerable_retry(
    qid: str,
    question: str,
    context: list[ContextChunk],
    *,
    model: str,
    max_chars: int,
    max_chunks: int | None = None,
    max_output_tokens: int,
    previous_row: dict[str, Any] | None = None,
) -> tuple[Generation, int, str | None]:
    """Generate an answer and retry once if it abstains.

    The retry count is stored in the QA checkpoint so a second process does not
    keep regenerating a question whose retry also abstained.  The returned
    third value is the first answer, when a retry was performed, and is kept
    only for auditability.
    """

    retry_count = _unanswerable_retry_count(previous_row)
    generation = generate(
        qid,
        question,
        context,
        model=model,
        max_chars=max_chars,
        max_chunks=max_chunks,
        max_output_tokens=max_output_tokens,
        render_prompt=lambda query, texts: VIDORe_ANSWER_PROMPT.format(
            documents="\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts)),
            query=query,
        ),
    )
    if _is_unanswerable_answer(generation.answer) and retry_count < 1:
        retry_generation = generate(
            qid,
            question,
            context,
            model=model,
            max_chars=max_chars,
            max_chunks=max_chunks,
            max_output_tokens=max_output_tokens,
            render_prompt=lambda query, texts: VIDORe_ANSWER_PROMPT.format(
                documents="\n\n".join(f"[{i + 1}] {text}" for i, text in enumerate(texts)),
                query=query,
            ),
        )
        return retry_generation, retry_count + 1, generation.answer
    return generation, retry_count, None


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
    configure_run_logging(
        output_dir / "logs" / f"{scope}_on_demand_basic.log",
        level=getattr(logging, str(args.log_level).upper()),
    )
    event_logger = JsonEventLogger(
        output_dir / "logs" / f"{scope}_events.jsonl",
        run_name="on-demand-basic",
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
    query_workers = _positive(
        args.query_workers or docbench_config.get("query_workers") or 4,
        "query-workers",
    )
    qa_workers = _positive(
        args.qa_workers or docbench_config.get("qa_workers") or 4,
        "qa-workers",
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
    parse_retry_attempts = _non_negative(
        args.parse_retry_attempts
        if args.parse_retry_attempts is not None
        else docbench_config.get("parse_retry_attempts", 2),
        "parse-retry-attempts",
    )
    parse_retry_backoff_seconds = _non_negative_float(
        args.parse_retry_backoff_seconds
        if args.parse_retry_backoff_seconds is not None
        else docbench_config.get("parse_retry_backoff_seconds", 0.0),
        "parse-retry-backoff-seconds",
    )
    max_context_chars = _positive(
        args.max_context_chars or docbench_config.get("max_context_chars") or 12000,
        "max-context-chars",
    )
    max_context_chunks = _positive(
        args.max_context_chunks or docbench_config.get("max_context_chunks") or top_k_chunks,
        "max-context-chunks",
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
    judge_model = str(args.judge or docbench_config.get("judge") or "openai/gpt-4o")
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
    questions = [item for item in all_questions if item["doc_id"] in selected_doc_ids]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[: args.limit]
    if not selected_documents:
        raise RuntimeError(f"No DocBench documents found under {docbench_root}")
    if not questions:
        raise RuntimeError("No DocBench questions selected")
    question_by_qid = {str(item["qid"]): item for item in questions}
    if len(question_by_qid) != len(questions):
        raise RuntimeError("DocBench question ids must be unique")

    index_documents = documents if scope == "lake" else selected_documents
    ppocr_jsonl = _resolve_path(
        args.ppocr_jsonl or docbench_config.get("ppocr_jsonl")
    )
    if ppocr_jsonl is not None and not ppocr_jsonl.is_file():
        raise FileNotFoundError(f"PP-OCRv5 JSONL does not exist: {ppocr_jsonl}")
    index_started = time.perf_counter()
    page_index, corpus_fingerprint = _load_or_build_page_index(
        index_documents,
        output_dir=output_dir,
        scope=scope,
        force_rebuild=args.force_rebuild_index,
        ppocr_jsonl=ppocr_jsonl,
    )
    preparation_metadata = dict(page_index.metadata)
    index_seconds = time.perf_counter() - index_started
    if not page_index.pages:
        raise RuntimeError("The pdf-inspector page index is empty")

    raw_config = dict(config.get("chunking_embedding") or {})
    chunking_config = _prepare_chunking_config(raw_config, output_dir)
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
        "pipeline": (
            "on-demand-basic-ppocrv5" if ppocr_jsonl else "on-demand-basic"
        ),
        "light_preparation": (
            "pdf_inspector -> ppocrv5 (no tesseract)"
            if ppocr_jsonl
            else "pdf_inspector"
        ),
        "light_retrieval": "bm25",
        "downstream": (
            "KDL + pdf-inspector -> fixed_overlap "
            f"512/{int((chunking_config.get('chunker_params') or {}).get('overlap', 0))} -> "
            "text-embedding-3-small -> hybrid baseline_legacy"
        ),
        "retrieval_scope": scope,
        "query_page_scope": (
            "per_question_document" if scope == "file" else "global_lake"
        ),
        "indexed_documents": len(index_documents),
        "evaluated_documents": len(selected_documents),
        "top_k_pages": top_k_pages,
        "top_k_chunks": top_k_chunks,
        "depth": depth,
        "alpha": alpha,
        "parse_retry_attempts": parse_retry_attempts,
        "parse_retry_backoff_seconds": parse_retry_backoff_seconds,
        "corpus_fingerprint": corpus_fingerprint,
        "preparation": preparation_metadata,
        "parser_config_hash": _hash_payload(parser_config),
        "chunking_config_hash": _hash_payload(chunking_config),
    }
    retrieval_config_hash = _hash_payload(retrieval_config)
    retrieval_path = output_dir / "retrieval" / f"{scope}_baseline_legacy.jsonl"
    timings_path = output_dir / "retrieval" / f"{scope}_timings.jsonl"
    retrieval_rows = {
        qid: row
        for qid, row in _read_latest_jsonl(retrieval_path).items()
        if row.get("retrieval_config_hash") == retrieval_config_hash
    }
    timing_rows = {
        qid: row
        for qid, row in _read_latest_jsonl(timings_path).items()
        if retrieval_rows.get(qid, {}).get("status") == "ok"
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
        parse_retry_attempts=parse_retry_attempts,
        parse_retry_backoff_seconds=parse_retry_backoff_seconds,
        force_reparse=args.force_reparse,
        page_scope_for_query=_make_query_page_scope(page_index, questions, scope),
        event_logger=event_logger,
    )
    try:
        _run_retrieval(
            runner,
            questions,
            retrieval_rows,
            timing_rows,
            retrieval_config,
            retrieval_config_hash,
            retrieval_path,
            timings_path,
            query_workers,
        )
        unique_bm25_pages_to_parse = _unique_bm25_pages_to_parse(retrieval_rows)
        timing_summary = runner.timing_summary(light_preparation_seconds=index_seconds)
        timing_summary["unique_bm25_pages_to_parse"] = unique_bm25_pages_to_parse
        timing_summary["light_preparation"] = retrieval_config["light_preparation"]
        timing_summary["preparation_metadata"] = preparation_metadata
        timing_summary["run_started_at_utc"] = run_started_at
        timing_summary["run_finished_at_utc"] = utc_now_iso()
        if not args.skip_qa:
            qa_rows = _run_qa(
                questions,
                retrieval_rows,
                output_dir / "qa" / f"{scope}_baseline_legacy.jsonl",
                generator_model=generator_model,
                judge_model=judge_model,
                max_context_chars=max_context_chars,
                max_context_chunks=max_context_chunks,
                max_unit_chars=max_unit_chars,
                max_output_tokens=max_output_tokens,
                qa_workers=qa_workers,
                skip_judge=args.skip_judge,
                retrieval_config_hash=retrieval_config_hash,
            )
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
                max_context_chunks=max_context_chunks,
                chunk_overlap=int((chunking_config.get("chunker_params") or {}).get("overlap", 0)),
                depth=depth,
                alpha=alpha,
                generator_model=generator_model,
                judge_model=judge_model,
                unique_bm25_pages_to_parse=unique_bm25_pages_to_parse,
            )
        else:
            report = {
                "arm": "baseline_legacy",
                "retrieval_scope": scope,
                "query_page_scope": retrieval_config["query_page_scope"],
                "questions_expected": len(questions),
                "questions_retrieved": sum(
                    row.get("status") == "ok" for row in retrieval_rows.values()
                ),
                "unique_bm25_pages_to_parse": unique_bm25_pages_to_parse,
                "accuracy": None,
            }

        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_version = _next_report_version(reports_dir, scope)
        report["report_version"] = report_version
        timing_summary["report_version"] = report_version
        report_path = reports_dir / f"{scope}_baseline_legacy_ver{report_version}.json"
        timing_path = reports_dir / f"{scope}_timing_summary_ver{report_version}.json"
        _write_json(report_path, report)
        _write_json(timing_path, timing_summary)
        manifest = {
            "contract_version": "docbench-on-demand-basic-v1",
            "report_version": report_version,
            "pipeline": retrieval_config["pipeline"],
            "light_preparation": retrieval_config["light_preparation"],
            "light_retrieval": retrieval_config["light_retrieval"],
            "retrieval_scope": scope,
            "query_page_scope": retrieval_config["query_page_scope"],
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
            "retrieval_output": str(retrieval_path),
            "timings_output": str(timings_path),
            "report_output": str(report_path),
            "timing_summary_output": str(timing_path),
            "timing_summary": timing_summary,
            "report": report,
        }
        _write_json(output_dir / f"manifest_{scope}_ver{report_version}.json", manifest)
        _update_summary(
            output_dir / f"summary_ver{report_version}.json",
            scope,
            report,
            timing_summary,
        )
        event_logger.emit("run_completed", scope=scope, questions=len(questions))
        print(
            json.dumps(
                {"report": report, "timing": timing_summary},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    finally:
        runner.close()


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docbench-root", type=Path, required=False)
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
    parser.add_argument(
        "--parse-retry-attempts",
        type=int,
        help=(
            "Retries after the initial KDL batch for pages that are missing "
            "or quarantined (default: 2)."
        ),
    )
    parser.add_argument(
        "--parse-retry-backoff-seconds",
        type=float,
        help="Delay before each unresolved-page retry batch (default: 0).",
    )
    parser.add_argument("--kdl-max-workers", type=int)
    parser.add_argument("--kdl-render-processes", type=int)
    parser.add_argument("--kdl-bbox-max-workers", type=int)
    parser.add_argument("--kdl-request-workers", type=int)
    parser.add_argument("--kdl-request-batch-size", type=int)
    parser.add_argument("--kdl-max-model-sequences", type=int)
    parser.add_argument(
        "--kdl-host-failure-threshold",
        type=int,
        help="Open the KDL circuit after this many consecutive failures (default: 3).",
    )
    parser.add_argument(
        "--kdl-host-recovery-seconds",
        type=float,
        help="Cooldown before a KDL recovery probe (default: 30).",
    )
    parser.add_argument(
        "--kdl-host-recovery-attempts",
        type=int,
        help="Maximum KDL recovery probes before aborting (default: 5).",
    )
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--max-context-chunks", type=int)
    parser.add_argument("--max-unit-chars", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--generator")
    parser.add_argument("--judge")
    parser.add_argument(
        "--ppocr-jsonl",
        type=Path,
        help=(
            "PP-OCRv5 notebook JSONL or result ZIP. Its page text is appended "
            "to the native pdf-inspector text before BM25; this path never "
            "invokes Tesseract."
        ),
    )
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _positive(value: Any, name: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _non_negative(value: Any, name: str) -> int:
    number = int(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"{name} is required; set it in .env or the shell")


def _check_vllm_endpoint() -> None:
    import requests

    base = str(os.environ["VLLM_API_BASE"]).rstrip("/")
    headers = {"ngrok-skip-browser-warning": "true"}
    if os.getenv("VLLM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['VLLM_API_KEY']}"
    try:
        response = requests.get(f"{base}/models", headers=headers, timeout=30)
    except requests.RequestException as error:
        raise RuntimeError(f"Cannot reach VLLM_API_BASE={base}: {error}") from error
    if response.status_code != 200:
        raise RuntimeError(
            f"VLLM endpoint check failed: HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )
    payload = response.json()
    models = [str(item.get("id")) for item in payload.get("data", [])]
    expected = os.getenv("VLLM_MODEL_NAME")
    if expected and models and expected not in models:
        raise RuntimeError(
            f"VLLM_MODEL_NAME={expected!r} not found in /models: {models}"
        )
    LOGGER.info("VLLM endpoint ready: %s; models=%s", base, models)


def _load_docbench(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"DocBench root is not a directory: {root}")
    # The exported ``0. BENCHMARK`` bundle is a flat manifest rather than the
    # historical numeric-folder DocBench layout.  Keep the same downstream
    # pipeline and normalize it to the runner's document/question contract.
    if (root / "documents.jsonl").is_file() and (root / "queries.jsonl").is_file():
        return _load_benchmark_bundle(root)
    data_root = root / "data" if (root / "data").is_dir() else root
    documents: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()
    seen_qids: set[str] = set()
    folders = sorted(
        (
            path
            for path in data_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=_folder_sort_key,
    )
    for folder in folders:
        pdfs = sorted(path for path in folder.glob("*.pdf") if path.is_file())
        qa_candidates = sorted(folder.glob("*_qa.jsonl")) + sorted(
            folder.glob("*_qa.json")
        )
        if not pdfs or not qa_candidates:
            continue
        if len(pdfs) > 1:
            LOGGER.warning("Multiple PDFs in %s; using %s", folder, pdfs[0].name)
        pdf_path = pdfs[0].resolve()
        doc_id = pdf_path.stem
        if doc_id in seen_doc_ids:
            raise ValueError(f"Duplicate DocBench document id from PDF stems: {doc_id}")
        seen_doc_ids.add(doc_id)
        file_index = int(folder.name) if folder.name.isdigit() else None
        document = {
            "folder": folder.name,
            "file_index": file_index,
            "domain": _domain_for_file_index(file_index),
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "qa_path": str(qa_candidates[0].resolve()),
        }
        documents.append(document)
        for index, item in enumerate(_read_qa_file(qa_candidates[0])):
            if (
                not isinstance(item, dict)
                or not str(item.get("question") or "").strip()
            ):
                continue
            qid = str(item.get("qid") or f"{folder.name}:{index}")
            if qid in seen_qids:
                raise ValueError(f"Duplicate DocBench question id: {qid}")
            seen_qids.add(qid)
            questions.append(
                {
                    "qid": qid,
                    "question": str(item.get("question") or ""),
                    "answer": str(item.get("answer") or ""),
                    "evidence": str(item.get("evidence") or ""),
                    "type": str(
                        item.get("type") or item.get("question_type") or "unknown"
                    ),
                    "type_group": _type_group(
                        item.get("type") or item.get("question_type")
                    ),
                    "folder": folder.name,
                    "file_index": file_index,
                    "domain": _domain_for_file_index(file_index),
                    "doc_id": doc_id,
                }
            )
    return documents, questions


def _load_benchmark_bundle(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the flat ``0. BENCHMARK`` export.

    Paths in ``documents.jsonl`` are rooted at the bundle directory.  Qrels are
    retained on each question so the report can calculate file recall and
    graded NDCG without changing the legacy DocBench format.
    """
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    documents: list[dict[str, Any]] = []
    by_doc_id: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(root / "documents.jsonl"):
        doc_id = str(item.get("doc_id") or "").strip()
        raw_path = str(item.get("path") or "").strip()
        if not doc_id or not raw_path:
            continue
        pdf_path = (root / raw_path).resolve()
        if not pdf_path.is_file():
            raise FileNotFoundError(f"Benchmark document path does not exist: {pdf_path}")
        if doc_id in by_doc_id:
            raise ValueError(f"Duplicate benchmark document id: {doc_id}")
        document = {
            "folder": str(item.get("source") or "benchmark"),
            "file_index": None,
            "domain": str(item.get("source") or "Unknown"),
            "doc_id": doc_id,
            "source": str(item.get("source") or ""),
            "pdf_path": str(pdf_path),
            "qa_path": str((root / "queries.jsonl").resolve()),
        }
        documents.append(document)
        by_doc_id[doc_id] = document

    qrels_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    qrels_path = root / "qrels.jsonl"
    if qrels_path.is_file():
        for item in read_jsonl(qrels_path):
            qrels_by_qid[str(item.get("query_id") or "")].append(item)

    questions: list[dict[str, Any]] = []
    seen_qids: set[str] = set()
    for item in read_jsonl(root / "queries.jsonl"):
        qid = str(item.get("query_id") or "").strip()
        query = str(item.get("query") or "").strip()
        if not qid or not query:
            continue
        if qid in seen_qids:
            raise ValueError(f"Duplicate benchmark question id: {qid}")
        seen_qids.add(qid)
        gold = qrels_by_qid.get(qid, [])
        gold_doc_ids = [str(row.get("doc_id") or "") for row in gold if row.get("doc_id")]
        missing = sorted(set(gold_doc_ids) - set(by_doc_id))
        if missing:
            raise ValueError(f"Benchmark qrels reference unknown documents: {missing[:5]}")
        answers = item.get("answers") or []
        answer = answers[0] if isinstance(answers, list) and answers else item.get("answer", "")
        gold_page_qrels = _benchmark_page_qrels(
            qid,
            gold,
            by_doc_id=by_doc_id,
            page_maps={},
        )
        # Preserve evidence paths for auditability; the actual reference text is
        # the answer in this export, just as in the legacy QA files.
        evidence = "\n".join(
            str(ev.get("source_page_path") or ev.get("source_corpus_id") or "")
            for row in gold for ev in (row.get("evidence") or []) if isinstance(ev, dict)
        )
        questions.append({
            "qid": qid,
            "question": query,
            "answer": str(answer),
            "evidence": evidence,
            "type": "unknown",
            "type_group": "Una.",
            "folder": str(item.get("source") or "benchmark"),
            "file_index": None,
            "domain": str(item.get("source") or "Unknown"),
            "doc_id": gold_doc_ids[0] if gold_doc_ids else "",
            "source": str(item.get("source") or ""),
            "gold_qrels": gold,
            "gold_page_qrels": gold_page_qrels,
        })
    if not documents:
        raise RuntimeError(f"No documents found in benchmark bundle: {root}")
    return sorted(documents, key=lambda item: str(item["doc_id"])), questions


def _benchmark_page_qrels(
    qid: str,
    qrels: list[dict[str, Any]],
    *,
    by_doc_id: dict[str, dict[str, Any]],
    page_maps: dict[str, dict[int, tuple[str, int]]],
) -> list[dict[str, Any]]:
    """Normalize bundle evidence to the page ids emitted by the runner.

    ViDoRe evidence identifies pages by the source corpus id, while MPDocVQA
    carries the PDF page number directly.  Both become ``doc_id#page=N`` here,
    so page retrieval metrics use the same identifiers as the page index.
    """
    page_grades: dict[str, float] = {}
    source = str(qid).split("::", 1)[0]
    source_map = page_maps.get(source)
    for qrel in qrels:
        doc_id = str(qrel.get("doc_id") or "")
        document = by_doc_id.get(doc_id)
        if not document:
            continue
        source_doc_id = str((document.get("metadata") or {}).get("source_doc_id") or "")
        for evidence in qrel.get("evidence") or []:
            page_number: int | None = None
            if evidence.get("pdf_page_number") is not None:
                page_number = int(evidence["pdf_page_number"])
            elif evidence.get("source_corpus_id") is not None:
                if source_map is None:
                    source_map = _load_vidore_page_map(source)
                    page_maps[source] = source_map
                mapped = source_map.get(int(evidence["source_corpus_id"]))
                if mapped is None:
                    raise ValueError(
                        f"{qid}: source_corpus_id={evidence['source_corpus_id']} "
                        f"is not present in the {source} corpus map"
                    )
                mapped_doc_id, zero_based_page = mapped
                if source_doc_id and mapped_doc_id != source_doc_id:
                    raise ValueError(
                        f"{qid}: qrel document {source_doc_id!r} does not match "
                        f"corpus page document {mapped_doc_id!r}"
                    )
                page_number = zero_based_page + 1
            if page_number is None or page_number <= 0:
                continue
            page_id = f"{doc_id}#page={page_number}"
            relevance = float(evidence.get("score") or qrel.get("relevance") or 0.0)
            page_grades[page_id] = max(page_grades.get(page_id, 0.0), relevance)
    return [
        {"page_id": page_id, "relevance": relevance}
        for page_id, relevance in sorted(page_grades.items())
    ]


@lru_cache(maxsize=None)
def _load_vidore_page_map(source: str) -> dict[int, tuple[str, int]]:
    """Load ``corpus_id -> (source document, zero-based page)`` for ViDoRe."""
    subset = source.removeprefix("vidore_")
    corpus_dir = ROOT / "data" / "raw" / f"vidore_v3_{subset}" / "corpus"
    mapping_path = corpus_dir / "corpus_mapping.parquet"
    paths = (
        [mapping_path]
        if mapping_path.is_file()
        else sorted(corpus_dir.glob("*.parquet"))
    )
    if not paths:
        raise FileNotFoundError(
            f"Missing ViDoRe {source} corpus mapping under {corpus_dir}. "
            "Download the corpus metadata so page-level qrels can be resolved."
        )
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to resolve ViDoRe source_corpus_id page qrels"
        ) from error

    mapping: dict[int, tuple[str, int]] = {}
    for path in paths:
        table = parquet.read_table(
            path, columns=["corpus_id", "doc_id", "page_number_in_doc"]
        )
        for row in table.to_pylist():
            corpus_id = int(row["corpus_id"])
            if corpus_id in mapping:
                raise ValueError(f"Duplicate ViDoRe corpus_id={corpus_id} in {corpus_dir}")
            mapping[corpus_id] = (
                str(row["doc_id"]),
                int(row["page_number_in_doc"]),
            )
    return mapping


def _read_qa_file(path: Path) -> list[Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else [value]
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _folder_sort_key(path: Path) -> tuple[int, Any]:
    try:
        return (0, int(path.name))
    except ValueError:
        return (1, path.name)


def _domain_for_file_index(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return "Unknown"
    for domain, values in DOMAIN_FILE_RANGES.items():
        if number in values:
            return domain
    return "Unknown"


def _type_group(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-").rstrip(".")
    if normalized in {"text-only", "text"}:
        return "Text."
    if normalized in {"multimodal-f", "multimodal-t", "multimodal", "multi"}:
        return "Multi."
    if normalized in {"meta-data", "metadata", "meta"}:
        return "Meta."
    return "Una."


def _load_or_build_page_index(
    documents: list[dict[str, Any]],
    *,
    output_dir: Path,
    scope: str,
    force_rebuild: bool,
    ppocr_jsonl: Path | None = None,
) -> tuple[PageIndex, str]:
    paths = [Path(item["pdf_path"]).resolve() for item in documents]
    fingerprint = _corpus_fingerprint(paths)
    ppocr_fingerprint = _artifact_fingerprint(ppocr_jsonl) if ppocr_jsonl else ""
    index_suffix = f"_{ppocr_fingerprint}" if ppocr_fingerprint else ""
    index_dir = output_dir / "indexes" / f"page_bm25_{scope}_{fingerprint}{index_suffix}"
    if (
        not force_rebuild
        and (index_dir / "pages.jsonl").is_file()
        and (index_dir / "bm25.json").is_file()
    ):
        LOGGER.info("Loading cached pdf-inspector/BM25 index: %s", index_dir)
        index = PageIndex.load(index_dir)
        return index, fingerprint

    source_uri_by_path = {
        str(Path(item["pdf_path"]).resolve()): str(item["doc_id"]) for item in documents
    }

    def source_uri(path: Path) -> str:
        return source_uri_by_path[str(path.resolve())]

    LOGGER.info(
        "Building pdf-inspector/BM25 index: documents=%d scope=%s",
        len(documents),
        scope,
    )
    index = build_page_index(
        paths,
        parser=PdfInspectorPageParser(),
        source_uri=source_uri,
    )
    if ppocr_jsonl is not None:
        LOGGER.info("Merging PP-OCRv5 page text into the light index: %s", ppocr_jsonl)
        index.apply_ppocrv5_jsonl(ppocr_jsonl, strict=True)
    else:
        index.metadata["preparation"] = "pdf_inspector"
    index.save(index_dir)
    return index, fingerprint


def _corpus_fingerprint(paths: list[Path]) -> str:
    parts = []
    for path in sorted(paths, key=lambda item: str(item)):
        stat = path.stat()
        parts.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _artifact_fingerprint(path: Path | None) -> str:
    """Hash the PP-OCR artifact so a changed notebook export cannot reuse BM25."""
    if path is None:
        return "native"
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"ppocrv5-{digest.hexdigest()[:16]}"


def _prepare_chunking_config(
    config: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    result = dict(config)
    params = dict(result.get("embedder_params") or {})
    params["cache_dir"] = str(output_dir / "embedding-cache")
    result["embedder_params"] = params
    return result


def _validate_baseline_chunking(config: dict[str, Any]) -> None:
    if str(config.get("chunker") or "") != "fixed_overlap":
        raise ValueError("baseline_legacy requires chunker=fixed_overlap")
    params = config.get("chunker_params") or {}
    if int(params.get("n_words", 0)) != 512 or int(params.get("overlap", 0)) not in {0, 128}:
        raise ValueError(
            "baseline_legacy supports fixed_overlap 512 words with overlap 128 "
            "(standard) or 0 (no-overlap experiment)"
        )
    model = str((config.get("embedder_params") or {}).get("model") or "")
    if model != "openai/text-embedding-3-small":
        raise ValueError(
            "baseline_legacy requires embedder model openai/text-embedding-3-small"
        )


def _apply_kdl_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    result = dict(config)
    kdl = dict(result.get("kdl") or {})
    names = {
        "max_workers": args.kdl_max_workers,
        "render_processes": args.kdl_render_processes,
        "bbox_max_workers": args.kdl_bbox_max_workers,
        "request_workers": args.kdl_request_workers,
        "request_batch_size": args.kdl_request_batch_size,
        "max_model_sequences": args.kdl_max_model_sequences,
        "host_failure_threshold": getattr(args, "kdl_host_failure_threshold", None),
        "host_recovery_seconds": getattr(args, "kdl_host_recovery_seconds", None),
        "host_recovery_attempts": getattr(args, "kdl_host_recovery_attempts", None),
    }
    for key, value in names.items():
        if value is not None:
            kdl[key] = value
    result["kdl"] = kdl
    return result


def _with_kdl_event_log(config: dict[str, Any], path: Path) -> dict[str, Any]:
    result = dict(config)
    kdl = dict(result.get("kdl") or {})
    kdl["event_log_path"] = str(path)
    result["kdl"] = kdl
    return result


def _run_retrieval(
    runner: OnDemandPerQueryRunner,
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    timing_rows: dict[str, dict[str, Any]],
    retrieval_config: dict[str, Any],
    retrieval_config_hash: str,
    retrieval_path: Path,
    timings_path: Path,
    query_workers: int,
) -> None:
    pending = [
        question
        for question in questions
        if not (
            str(question["qid"]) in retrieval_rows
            and retrieval_rows[str(question["qid"])].get("status") == "ok"
            and retrieval_rows[str(question["qid"])].get("parse_complete", True)
            and retrieval_rows[str(question["qid"])].get("retrieval_config_hash")
            == retrieval_config_hash
        )
    ]

    def execute(
        question: dict[str, Any],
    ) -> tuple[str, OnDemandQueryResult | None, str | None, Exception | None]:
        if abort_event.is_set():
            return (
                str(question["qid"]),
                None,
                "run aborted because the KDL host circuit is open",
                None,
            )
        try:
            result = runner.run_query(
                question["question"], query_id=str(question["qid"])
            )
            return str(question["qid"]), result, None, None
        except KDLHostUnavailableError as error:
            abort_event.set()
            return str(question["qid"]), None, f"{type(error).__name__}: {error}", error
        except Exception as error:  # noqa: BLE001 - persist per-query failures
            return str(question["qid"]), None, f"{type(error).__name__}: {error}", None

    LOGGER.info(
        "Running on-demand retrieval: pending=%d/%d workers=%d",
        len(pending),
        len(questions),
        query_workers,
    )
    if not pending:
        return
    abort_event = Event()
    host_failure: Exception | None = None
    with ThreadPoolExecutor(max_workers=query_workers) as pool:
        futures = {pool.submit(execute, question): question for question in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            question = futures[future]
            qid, result, error, detected_host_failure = future.result()
            if detected_host_failure is not None and host_failure is None:
                host_failure = detected_host_failure
                abort_event.set()
            finished_at = utc_now_iso()
            if result is None:
                row = {
                    **question,
                    "pipeline": retrieval_config["pipeline"],
                    "retrieval_config": retrieval_config,
                    "retrieval_config_hash": retrieval_config_hash,
                    "status": "aborted" if detected_host_failure else "error",
                    "error": error,
                    "finished_at_utc": finished_at,
                    "hits": [],
                    "chunks": [],
                    "timing": {},
                }
                timing = {
                    "qid": qid,
                    "status": row["status"],
                    "error": error,
                    "finished_at_utc": finished_at,
                }
            else:
                row = _encode_retrieval_result(
                    question, result, retrieval_config, retrieval_config_hash
                )
                timing = {"qid": qid, **result.timing}
            retrieval_rows[qid] = row
            timing_rows[qid] = timing
            _write_ordered_jsonl(retrieval_path, retrieval_rows, questions)
            _write_ordered_jsonl(timings_path, timing_rows, questions)
            LOGGER.info(
                "Retrieval %d/%d: qid=%s status=%s",
                number,
                len(pending),
                qid,
                row["status"],
            )
    if host_failure is not None:
        LOGGER.error(
            "Stopping pipeline because KDL host is unavailable; "
            "successful checkpoints are preserved for resume"
        )
        raise host_failure


def _encode_retrieval_result(
    question: dict[str, Any],
    result: OnDemandQueryResult,
    retrieval_config: dict[str, Any],
    retrieval_config_hash: str,
) -> dict[str, Any]:
    return {
        **question,
        "pipeline": retrieval_config["pipeline"],
        "retrieval_config": retrieval_config,
        "retrieval_config_hash": retrieval_config_hash,
        "status": "ok",
        "error": None,
        "parse_complete": not result.failed_page_ids,
        "parse_failed_page_ids": list(result.failed_page_ids),
        "parsed_page_ids": list(result.parsed_page_ids),
        "cached_page_ids": list(result.cached_page_ids),
        "hits": [hit.as_dict() for hit in result.hits],
        "selected_pages": result.selected_pages,
        "chunks": [
            {
                "chunk_id": chunk.record_id,
                "doc_id": chunk.page_id,
                "text": chunk.text,
                "rank": rank,
                "score": 1.0 / rank,
            }
            for rank, chunk in enumerate(result.ranked_chunks, start=1)
        ],
        "timing": result.timing,
    }


def _run_qa(
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    generator_model: str,
    judge_model: str,
    max_context_chars: int,
    max_context_chunks: int,
    max_unit_chars: int,
    max_output_tokens: int,
    qa_workers: int,
    skip_judge: bool,
    retrieval_config_hash: str,
) -> dict[str, dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_config_hash = _hash_payload(
        {
            "arm": "baseline_legacy",
            "generator": generator_model,
            "judge": judge_model,
            "max_context_chars": max_context_chars,
            "max_context_chunks": max_context_chunks,
            "max_unit_chars": max_unit_chars,
            "max_output_tokens": max_output_tokens,
            "skip_judge": skip_judge,
            "retrieval_config_hash": retrieval_config_hash,
        }
    )
    rows = _read_latest_jsonl(output_path)
    pending = [
        question
        for question in questions
        if not (
            str(question["qid"]) in rows
            and rows[str(question["qid"])].get("qa_config_hash") == qa_config_hash
            and rows[str(question["qid"])].get("status") == "ok"
            and not _should_retry_unanswerable(rows[str(question["qid"])])
        )
    ]
    LOGGER.info("QA baseline_legacy: pending=%d/%d", len(pending), len(questions))

    def execute(question: dict[str, Any]) -> dict[str, Any]:
        qid = str(question["qid"])
        previous_row = rows.get(qid)
        retrieval = retrieval_rows.get(qid) or {}
        if retrieval.get("status") != "ok":
            return {
                **question,
                "arm": "baseline_legacy",
                "status": "error",
                "score": None,
                "error": f"retrieval failed: {retrieval.get('error')}",
                "qa_config_hash": qa_config_hash,
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
        inference_started = time.perf_counter()
        generation, retry_count, initial_answer = _generate_with_unanswerable_retry(
            qid,
            question["question"],
            context,
            model=generator_model,
            max_chars=max_context_chars,
            max_chunks=max_context_chunks,
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
            "infer_time_seconds": round(time.perf_counter() - inference_started, 6),
            "qa_config_hash": qa_config_hash,
            "unanswerable_retry_count": retry_count,
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
        except Exception as error:  # noqa: BLE001 - persist the failed question
            result.update(
                {
                    "status": "error",
                    "score": None,
                    "judge_raw": "",
                    "error": repr(error),
                }
            )
        return result

    with ThreadPoolExecutor(max_workers=qa_workers) as pool:
        futures = {pool.submit(execute, question): question for question in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows[str(row["qid"])] = row
            _write_ordered_jsonl(output_path, rows, questions)
            LOGGER.info(
                "QA %d/%d: qid=%s status=%s",
                number,
                len(pending),
                row["qid"],
                row["status"],
            )
    _write_ordered_jsonl(output_path, rows, questions)
    return rows


def _parse_score(text: str) -> float:
    prefix = str(text or "")[:200]
    match = re.search(
        r"correctness\s*:\s*(0(?:\.5)?|1(?:\.0)?)\b",
        prefix,
        flags=re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    match = re.search(r"\b(0(?:\.5)?|1(?:\.0)?)\b", prefix)
    if match:
        return float(match.group(1))
    raise ValueError(f"judge response has no 0/0.5/1 score: {prefix!r}")


def _benchmark_retrieval_metrics(
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    *,
    page_index: PageIndex,
) -> dict[str, Any]:
    """Score file recall separately and all other retrieval metrics by page.

    The raw Light ranking contains pages and the Accurate ranking contains
    chunks whose ``doc_id`` is their source page.  Only ``file_recall`` maps
    either ranking to source files; recall/NDCG use page qrels directly.
    """
    page_to_file = {str(page.page_id): str(page.source_uri) for page in page_index.pages}
    metric_values: dict[str, dict[str, list[float]]] = {
        "light_page": defaultdict(list),
        "accurate_page": defaultdict(list),
        "light_file": defaultdict(list),
    }

    def score(stage: str, ranked: list[str], grades: dict[str, float]) -> None:
        for k in (10, 20):
            # Truncate before deduplication. Repeated chunks from one page and
            # repeated pages from one file therefore still consume rank.
            unique: list[str] = []
            for item in ranked[:k]:
                if item and item not in unique:
                    unique.append(item)
            found = sum(1 for doc_id in grades if doc_id in unique)
            gains = [grades.get(doc_id, 0.0) for doc_id in unique]
            ideal_gains = sorted(grades.values(), reverse=True)[:k]
            dcg = sum(
                (2.0**gain - 1.0) / (1.0 if rank == 1 else math.log2(rank))
                for rank, gain in enumerate(gains, start=1)
            )
            ideal = sum(
                (2.0**gain - 1.0) / (1.0 if rank == 1 else math.log2(rank))
                for rank, gain in enumerate(ideal_gains, start=1)
            )
            metric_values[stage][f"recall_at_{k}"].append(found / len(grades))
            metric_values[stage][f"ndcg_at_{k}"].append(dcg / ideal if ideal else 0.0)

    for question in questions:
        qrels = question.get("gold_qrels") or []
        if not qrels:
            continue
        grades: dict[str, float] = {}
        for qrel in qrels:
            doc_id = str(qrel.get("doc_id") or "")
            if doc_id:
                grades[doc_id] = max(grades.get(doc_id, 0.0), float(qrel.get("relevance") or 0.0))
        if not grades:
            continue
        row = retrieval_rows.get(str(question["qid"]), {})
        light_pages = [str(hit.get("page_id") or "") for hit in row.get("hits") or []]
        light_files = [str(hit.get("source_uri") or "") for hit in row.get("hits") or []]
        accurate_pages = [str(chunk.get("doc_id") or "") for chunk in row.get("chunks") or []]
        accurate_files = [page_to_file.get(page_id, "") for page_id in accurate_pages]

        page_grades = {
            str(item.get("page_id") or ""): float(item.get("relevance") or 0.0)
            for item in question.get("gold_page_qrels") or []
            if item.get("page_id")
        }
        if page_grades:
            score("light_page", light_pages, page_grades)
            if accurate_pages:
                score("accurate_page", accurate_pages, page_grades)
        score("light_file", light_files, grades)

    light_page_values = metric_values["light_page"]
    accurate_page_values = metric_values["accurate_page"]
    light_file_values = metric_values["light_file"]
    questions_n = len(light_page_values["recall_at_10"])
    accurate_n = len(accurate_page_values["recall_at_10"])
    file_questions_n = len(light_file_values["recall_at_20"])
    return {
        # Explicit names document the unit of each metric.
        "light_file_recall_at_20": _mean(light_file_values["recall_at_20"]),
        "light_page_ndcg_at_10": _mean(light_page_values["ndcg_at_10"]),
        "light_page_ndcg_at_20": _mean(light_page_values["ndcg_at_20"]),
        "light_page_recall_at_10": _mean(light_page_values["recall_at_10"]),
        "light_page_recall_at_20": _mean(light_page_values["recall_at_20"]),
        "accurate_page_ndcg_at_10": _mean(accurate_page_values["ndcg_at_10"]),
        "accurate_page_ndcg_at_20": _mean(accurate_page_values["ndcg_at_20"]),
        "accurate_page_recall_at_10": _mean(accurate_page_values["recall_at_10"]),
        "accurate_page_recall_at_20": _mean(accurate_page_values["recall_at_20"]),
        "page_qrel_questions": float(questions_n) if questions_n else None,
        "accurate_page_qrel_questions": float(accurate_n) if accurate_n else None,
        "file_qrel_questions": float(file_questions_n) if file_questions_n else None,
        # Backward-compatible names: all unqualified retrieval metrics are page-level.
        "file_recall": _mean(light_file_values["recall_at_20"]),
        "ndcg_at_10": _mean(light_page_values["ndcg_at_10"]),
        "ndcg_at_20": _mean(light_page_values["ndcg_at_20"]),
        "recall_at_10": _mean(light_page_values["recall_at_10"]),
        "recall_at_20": _mean(light_page_values["recall_at_20"]),
        "light_ndcg_at_10": _mean(light_page_values["ndcg_at_10"]),
        "light_ndcg_at_20": _mean(light_page_values["ndcg_at_20"]),
        "light_recall_at_10": _mean(light_page_values["recall_at_10"]),
        "light_recall_at_20": _mean(light_page_values["recall_at_20"]),
        "accurate_ndcg_at_10": _mean(accurate_page_values["ndcg_at_10"]),
        "accurate_ndcg_at_20": _mean(accurate_page_values["ndcg_at_20"]),
        "accurate_recall_at_10": _mean(accurate_page_values["recall_at_10"]),
        "accurate_recall_at_20": _mean(accurate_page_values["recall_at_20"]),
        "metric_units": {
            "file_recall": "file",
            "light_recall": "page",
            "light_ndcg": "page",
            "accurate_recall": "page",
            "accurate_ndcg": "page",
        },
    }


def _light_pipeline_label(page_index: PageIndex) -> str:
    preparation = str(page_index.metadata.get("preparation") or "pdf_inspector")
    if preparation == "pdf_inspector -> ppocrv5":
        return "pdf_inspector -> ppocrv5 -> bm25 -> baseline_legacy"
    return "pdf_inspector -> bm25 -> baseline_legacy"


def _unique_bm25_pages_to_parse(retrieval_rows: dict[str, dict[str, Any]]) -> int:
    """Count distinct BM25 pages selected before parsing across all queries."""
    return len(
        {
            str(hit.get("page_id"))
            for row in retrieval_rows.values()
            for hit in row.get("hits") or []
            if hit.get("page_id")
        }
    )


def _next_report_version(reports_dir: Path, scope: str) -> int:
    """Return an unused version shared by a report and its timing summary."""
    pattern = re.compile(
        rf"^{re.escape(scope)}_(?:baseline_legacy|timing_summary)_ver(\d+)\.json$"
    )
    versions = [
        int(match.group(1))
        for path in reports_dir.glob(f"{scope}_*_ver*.json")
        if (match := pattern.match(path.name))
    ]
    return max(versions, default=0) + 1


def _build_report(
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    qa_rows: dict[str, dict[str, Any]],
    *,
    scope: str,
    index_documents: list[dict[str, Any]],
    selected_documents: list[dict[str, Any]],
    page_index: PageIndex,
    runner: OnDemandPerQueryRunner,
    top_k_pages: int,
    top_k_chunks: int,
    max_context_chunks: int,
    chunk_overlap: int,
    depth: int,
    alpha: float,
    generator_model: str,
    judge_model: str,
    unique_bm25_pages_to_parse: int,
) -> dict[str, Any]:
    ordered = [
        qa_rows[str(question["qid"])]
        for question in questions
        if str(question["qid"]) in qa_rows
    ]
    completed = [
        row
        for row in ordered
        if row.get("status") == "ok" and row.get("score") is not None
    ]
    scores = [float(row["score"]) for row in completed]
    errors = [row for row in ordered if row.get("status") != "ok"]
    page_hits = []
    context_pages = []
    for question in questions:
        row = retrieval_rows.get(str(question["qid"]), {})
        hits = row.get("hits") or []
        page_hits.append(
            any(hit.get("source_uri") == question["doc_id"] for hit in hits)
        )
        qa_row = qa_rows.get(str(question["qid"]), {})
        context_pages.append(len(qa_row.get("context_page_ids") or []))

    retrieval_metrics = _benchmark_retrieval_metrics(
        questions, retrieval_rows, page_index=page_index
    )
    infer_times = [
        float(row.get("infer_time_seconds") or 0.0)
        for row in completed
        if row.get("infer_time_seconds") is not None
    ]

    by_domain: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in completed:
        by_domain[str(row.get("domain") or "Unknown")].append(float(row["score"]))
        by_type[str(row.get("type_group") or "Una.")].append(float(row["score"]))

    return {
        "arm": "baseline_legacy",
        "pipeline": (
            f"{_light_pipeline_label(page_index)} -> KDL+pdf_inspector -> "
            f"fixed_chunk_512_overlap_{chunk_overlap} -> "
            "text-embedding-3-small -> hybrid(alpha=0.7) -> generator -> DocBench judge"
        ),
        "retrieval_scope": scope,
        "query_page_scope": (
            "per_question_document" if scope == "file" else "global_lake"
        ),
        "generator": generator_model,
        "judge": judge_model,
        "documents": len(selected_documents),
        "indexed_documents": len(index_documents),
        "indexed_pages": len(page_index.pages),
        "questions_expected": len(questions),
        "questions_completed": len(completed),
        "errors": len(errors),
        "missing": max(len(questions) - len(completed), 0),
        "accuracy": _mean(scores),
        "correct_only": _mean([int(score >= 1.0) for score in scores]),
        "correct_plus_partial": _mean([int(score > 0.0) for score in scores]),
        "score_1": scores.count(1),
        "score_partial": scores.count(0.5),
        "score_0": scores.count(0),
        "bm25_gold_document_hit_rate": _mean([int(value) for value in page_hits]),
        "unique_bm25_pages_to_parse": unique_bm25_pages_to_parse,
        **retrieval_metrics,
        "retrieval_qrel_questions": retrieval_metrics["page_qrel_questions"],
        "infer_time_seconds_per_query": _mean(infer_times),
        "mean_context_pages": _mean(context_pages),
        "top_k_pages": top_k_pages,
        "top_k_chunks": top_k_chunks,
        "max_context_chunks": max_context_chunks,
        "chunk_retrieval_depth": depth,
        "chunk_retrieval_alpha": alpha,
        "accuracy_by_domain": {
            key: _mean(by_domain[key]) for key in DOMAIN_ORDER if key in by_domain
        },
        "accuracy_by_type_group": {
            key: _mean(by_type[key]) for key in TYPE_ORDER if key in by_type
        },
        "parsed_page_cache": runner.page_texts.__len__(),
        "prepared_chunk_cache": runner.prepared_chunk_count,
    }


def _mean(values: list[int | float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


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
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _update_summary(
    path: Path, scope: str, report: dict[str, Any], timing: dict[str, Any]
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
    runs[scope] = {"report": report, "timing_summary": timing}
    _write_json(path, value)


if __name__ == "__main__":
    raise SystemExit(main())
