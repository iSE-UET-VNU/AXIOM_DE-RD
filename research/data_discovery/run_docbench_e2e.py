"""Run the DocBench on-demand basic pipeline from the repository.

The pipeline is intentionally page selective:

    pdf-inspector page preparation -> BM25 page retrieval ->
    KDL + pdf-inspector for missing pages -> fixed 512/128 chunks ->
    text-embedding-3-small -> hybrid BM25+dense baseline_legacy ->
    generator -> DocBench judge

``--retrieval-scope file`` searches only the selected/evaluated DocBench
documents. ``--retrieval-scope lake`` searches the complete DocBench PDF lake.
The KDL endpoint can be remote (for example a vLLM server exposed from
Colab); PDF rendering, page indexing, chunking, embeddings, generation and
judging run on the local machine.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event
from typing import Any
import argparse
import hashlib
import json
import logging
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
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {{question}}
User Answer: {{sys_ans}}
Reference Answer: {{ref_ans}}
Reference Text: {{ref_text}}

Evaluation Form (score ONLY):
- Correctness:"""


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
        max_output_tokens=max_output_tokens,
    )
    if _is_unanswerable_answer(generation.answer) and retry_count < 1:
        retry_generation = generate(
            qid,
            question,
            context,
            model=model,
            max_chars=max_chars,
            max_output_tokens=max_output_tokens,
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
        force_reparse=args.force_reparse,
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
        timing_summary = runner.timing_summary(light_preparation_seconds=index_seconds)
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
                depth=depth,
                alpha=alpha,
                generator_model=generator_model,
                judge_model=judge_model,
            )
        else:
            report = {
                "arm": "baseline_legacy",
                "retrieval_scope": scope,
                "questions_expected": len(questions),
                "questions_retrieved": sum(
                    row.get("status") == "ok" for row in retrieval_rows.values()
                ),
                "accuracy": None,
            }

        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{scope}_baseline_legacy.json"
        _write_json(report_path, report)
        _write_json(reports_dir / f"{scope}_timing_summary.json", timing_summary)
        manifest = {
            "contract_version": "docbench-on-demand-basic-v1",
            "pipeline": "pdf_inspector -> bm25 -> baseline_legacy",
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
            "retrieval_output": str(retrieval_path),
            "timings_output": str(timings_path),
            "report_output": str(report_path),
            "timing_summary": timing_summary,
            "report": report,
        }
        _write_json(output_dir / f"manifest_{scope}.json", manifest)
        _update_summary(output_dir / "summary.json", scope, report, timing_summary)
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
    parser.add_argument("--kdl-max-workers", type=int)
    parser.add_argument("--kdl-render-processes", type=int)
    parser.add_argument("--kdl-bbox-max-workers", type=int)
    parser.add_argument("--kdl-request-workers", type=int)
    parser.add_argument("--kdl-request-batch-size", type=int)
    parser.add_argument("--kdl-max-model-sequences", type=int)
    parser.add_argument(
        "--kdl-host-failure-threshold",
        type=int,
        help="Stop after this many consecutive KDL host failures (default: 3).",
    )
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
) -> tuple[PageIndex, str]:
    paths = [Path(item["pdf_path"]).resolve() for item in documents]
    fingerprint = _corpus_fingerprint(paths)
    index_dir = output_dir / "indexes" / f"page_bm25_{scope}_{fingerprint}"
    if (
        not force_rebuild
        and (index_dir / "pages.jsonl").is_file()
        and (index_dir / "bm25.json").is_file()
    ):
        LOGGER.info("Loading cached pdf-inspector/BM25 index: %s", index_dir)
        return PageIndex.load(index_dir), fingerprint

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
    index.save(index_dir)
    return index, fingerprint


def _corpus_fingerprint(paths: list[Path]) -> str:
    parts = []
    for path in sorted(paths, key=lambda item: str(item)):
        stat = path.stat()
        parts.append(f"{path}|{stat.st_size}|{stat.st_mtime_ns}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


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
    if int(params.get("n_words", 0)) != 512 or int(params.get("overlap", 0)) != 128:
        raise ValueError(
            "baseline_legacy requires fixed_overlap 512 words / 128 overlap"
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
                    "pipeline": "pdf_inspector -> bm25 -> baseline_legacy",
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
        "pipeline": "pdf_inspector -> bm25 -> baseline_legacy",
        "retrieval_config": retrieval_config,
        "retrieval_config_hash": retrieval_config_hash,
        "status": "ok",
        "error": None,
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


def _parse_score(text: str) -> int:
    prefix = str(text or "")[:200]
    match = re.search(r"correctness\s*:\s*([01])\b", prefix, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b([01])\b", prefix)
    if match:
        return int(match.group(1))
    raise ValueError(f"judge response has no 0/1 score: {prefix!r}")


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
    depth: int,
    alpha: float,
    generator_model: str,
    judge_model: str,
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
    scores = [int(row["score"]) for row in completed]
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

    by_domain: dict[str, list[int]] = defaultdict(list)
    by_type: dict[str, list[int]] = defaultdict(list)
    for row in completed:
        by_domain[str(row.get("domain") or "Unknown")].append(int(row["score"]))
        by_type[str(row.get("type_group") or "Una.")].append(int(row["score"]))

    return {
        "arm": "baseline_legacy",
        "pipeline": "pdf_inspector -> bm25 -> KDL+pdf_inspector -> fixed_chunk_512 -> text-embedding-3-small -> hybrid(alpha=0.7) -> generator -> DocBench judge",
        "retrieval_scope": scope,
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
        "score_1": scores.count(1),
        "score_0": scores.count(0),
        "bm25_gold_document_hit_rate": _mean([int(value) for value in page_hits]),
        "mean_context_pages": _mean(context_pages),
        "top_k_pages": top_k_pages,
        "top_k_chunks": top_k_chunks,
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
