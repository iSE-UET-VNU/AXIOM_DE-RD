"""Run the on-demand basic baseline on a manifest-backed PDF benchmark.

The input bundle contains ``documents.jsonl``, ``queries.jsonl`` and
``qrels.jsonl`` alongside a ``data/`` PDF directory.  Unlike DocBench, a
query is not pre-assigned to one document, so this runner intentionally uses
lake retrieval over the complete document collection:

    pdf-inspector -> BM25 pages -> KDL + pdf-inspector for selected pages ->
    fixed 512/128 chunks -> text-embedding-3-small + hybrid baseline_legacy

The qrels are used only after retrieval to calculate derived file metrics;
they never constrain candidate generation or page parsing.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import csv
import json
import logging
import math
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery import run_docbench_e2e as legacy  # noqa: E402
from research.data_discovery.on_demand_per_query import OnDemandPerQueryRunner  # noqa: E402
from research.data_discovery.pipeline import (  # noqa: E402
    PageIndex,
    PdfInspectorPageParser,
    build_page_index,
)
from src.evaluation.benchmarks.vidore_v3_judge import (  # noqa: E402
    ANSWER_PROMPT as VIDORE_ANSWER_PROMPT,
    JUDGE_PROMPT as VIDORE_JUDGE_PROMPT,
    parse_judgment,
    render_documents,
)
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


LOGGER = logging.getLogger("benchmark_bundle_on_demand")
PIPELINE = (
    "pdf_inspector -> bm25 -> KDL+pdf_inspector -> fixed_chunk_512 -> "
    "text-embedding-3-small -> hybrid(alpha=0.7) -> generator -> ternary judge"
)

BUNDLE_JUDGE_PROMPT = """Evaluate the user answer against the question and reference answer.

Score only one of these values:
- 1: fully correct; it contains all material information and no material error.
- 0.5: partially correct; it contains material correct information but misses a
  material part, or includes a non-critical error.
- 0: incorrect, unsupported, empty, or abstains.

Question: {{question}}
User Answer: {{sys_ans}}
Reference Answer: {{ref_ans}}
Reference Text: {{ref_text}}

Evaluation Form (score ONLY):
- Correctness:"""

# Keep the ViDoRe wording intact; only adapt its placeholders to the generic
# QA runner's interpolation contract.
VIDORE_BUNDLE_JUDGE_PROMPT = (
    VIDORE_JUDGE_PROMPT.replace("{query}", "{{question}}")
    .replace("{true_answer}", "{{ref_ans}}")
    .replace("{test_answer}", "{{sys_ans}}")
)


def _render_vidore_answer_prompt(question: str, texts: Sequence[str]) -> str:
    return VIDORE_ANSWER_PROMPT.format(
        documents=render_documents(texts), query=question
    )


def _parse_vidore_score(reply: str) -> float:
    labels = {
        "Correct": 1.0,
        "Partially Correct": 0.5,
        "Incorrect": 0.0,
    }
    return labels[parse_judgment(reply)]


def _qa_prompt_settings(profile: str) -> dict[str, Any]:
    if profile == "vidore_v3":
        return {
            "generation_render_prompt": _render_vidore_answer_prompt,
            "generation_prompt_schema": "vidore_v3_figure_25",
            "judge_prompt": VIDORE_BUNDLE_JUDGE_PROMPT,
            "score_parser": _parse_vidore_score,
            "judge_score_schema": "vidore_v3_correct_partial_incorrect",
            "judge_max_output_tokens": 256,
        }
    if profile == "grounded_bundle":
        return {
            "generation_render_prompt": None,
            "generation_prompt_schema": "grounded_abstain_v1",
            "judge_prompt": BUNDLE_JUDGE_PROMPT,
            "score_parser": _parse_ternary_score,
            "judge_score_schema": "ternary_0_partial_0.5_correct_1",
            "judge_max_output_tokens": 16,
        }
    raise ValueError(
        "qa_prompt_profile must be 'vidore_v3' or 'grounded_bundle'"
    )


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    load_dotenv_file(ROOT)
    config_path = _resolve_path(args.config)
    config = load_config(config_path)
    bundle_config = dict(config.get("benchmark_bundle") or {})
    bundle_root = _resolve_path(args.bundle_root or bundle_config.get("root"))
    if bundle_root is None:
        raise ValueError("--bundle-root is required")
    bundle_root = _bundle_root(bundle_root)

    scope = str(args.retrieval_scope or bundle_config.get("retrieval_scope") or "lake")
    if scope != "lake":
        raise ValueError(
            "This benchmark has no query-owned document. Use --retrieval-scope lake."
        )
    output_dir = _resolve_path(
        args.output_dir
        or bundle_config.get("output_dir")
        or ROOT / "data/benchmark/benchmark_bundle_on_demand_basic_lake"
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    top_k_pages = _positive(
        args.top_k_pages or bundle_config.get("top_k_pages") or 10,
        "top-k-pages",
    )
    legacy_eval_top_k_chunks = _positive(
        args.legacy_eval_top_k_chunks
        or bundle_config.get("legacy_eval_top_k_chunks")
        or args.top_k_chunks
        or bundle_config.get("top_k_chunks")
        or 10,
        "legacy-eval-top-k-chunks",
    )
    generation_top_k_chunks = _positive(
        args.generation_top_k_chunks
        or bundle_config.get("generation_top_k_chunks")
        or args.top_k_chunks
        or bundle_config.get("top_k_chunks")
        or legacy_eval_top_k_chunks,
        "generation-top-k-chunks",
    )
    depth = _positive(
        args.depth or bundle_config.get("chunk_retrieval_depth") or 100,
        "depth",
    )
    alpha = float(
        args.alpha
        if args.alpha is not None
        else bundle_config.get("chunk_retrieval_alpha", 0.7)
    )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    query_workers = _positive(
        args.query_workers or bundle_config.get("query_workers") or 4,
        "query-workers",
    )
    qa_workers = _positive(
        args.qa_workers or bundle_config.get("qa_workers") or 4,
        "qa-workers",
    )
    microbatch_window = float(
        args.kdl_microbatch_window_seconds
        if args.kdl_microbatch_window_seconds is not None
        else bundle_config.get("kdl_microbatch_window_seconds", 0.30)
    )
    microbatch_max_pages = _positive(
        args.kdl_microbatch_max_pages
        or bundle_config.get("kdl_microbatch_max_pages")
        or 32,
        "kdl-microbatch-max-pages",
    )
    max_context_chars = _positive(
        args.max_context_chars or bundle_config.get("max_context_chars") or 12000,
        "max-context-chars",
    )
    max_unit_chars = _positive(
        args.max_unit_chars or bundle_config.get("max_unit_chars") or 8000,
        "max-unit-chars",
    )
    max_output_tokens = _positive(
        args.max_output_tokens or bundle_config.get("max_output_tokens") or 512,
        "max-output-tokens",
    )
    generator_model = str(
        args.generator
        or bundle_config.get("generator")
        or "deepseek/deepseek-v4-flash"
    )
    judge_model = str(
        args.judge or bundle_config.get("judge") or "openai/gpt-4o-mini"
    )
    if generator_model == judge_model:
        raise ValueError("generator and judge should be different models")
    qa_prompt_profile = str(
        args.qa_prompt_profile
        or bundle_config.get("qa_prompt_profile")
        or "vidore_v3"
    )
    qa_prompt_settings = _qa_prompt_settings(qa_prompt_profile)

    # Embedding and KDL parsing are still needed when answer generation is skipped.
    legacy._require_env("VLLM_API_BASE")
    legacy._require_env("OPENROUTER_API_KEY")
    if not args.skip_endpoint_check:
        legacy._check_vllm_endpoint()

    documents, questions = _load_bundle(bundle_root)
    if args.source:
        requested_sources = set(args.source)
        unknown_sources = requested_sources - {str(item["source"]) for item in questions}
        if unknown_sources:
            raise ValueError(f"Unknown --source value(s): {sorted(unknown_sources)}")
        questions = [item for item in questions if item["source"] in requested_sources]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        questions = questions[: args.limit]
    if not documents:
        raise RuntimeError(f"No PDF documents found in bundle: {bundle_root}")
    if not questions:
        raise RuntimeError("No benchmark questions selected")

    index_started = time.perf_counter()
    page_index, corpus_fingerprint = _load_or_build_page_index(
        documents,
        output_dir=output_dir,
        force_rebuild=args.force_rebuild_index,
    )
    index_seconds = time.perf_counter() - index_started
    if not page_index.pages:
        raise RuntimeError("The pdf-inspector page index is empty")

    raw_chunking_config = dict(config.get("chunking_embedding") or {})
    chunking_config = legacy._prepare_chunking_config(raw_chunking_config, output_dir)
    legacy._validate_baseline_chunking(chunking_config)
    parser_config = resolve_parser_config(
        ROOT,
        config.get("parsing") or {},
        output_dir / "parser-assets" / scope,
    )
    parser_config = legacy._apply_kdl_overrides(parser_config, args)
    if parser_config.get("provider") != "kdl_pdf_inspector":
        raise ValueError("baseline_legacy requires parsing.provider=kdl_pdf_inspector")

    retrieval_config = {
        "pipeline": "on-demand-basic",
        "light_preparation": "pdf_inspector",
        "light_retrieval": "bm25",
        "downstream": (
            "KDL + pdf-inspector -> fixed_overlap 512/128 -> "
            "text-embedding-3-small -> hybrid baseline_legacy"
        ),
        "retrieval_scope": scope,
        "indexed_documents": len(documents),
        "evaluated_queries": len(questions),
        "top_k_pages": top_k_pages,
        # This remains the legacy retrieval ranking depth for compatibility.
        "top_k_chunks": legacy_eval_top_k_chunks,
        "depth": depth,
        "alpha": alpha,
        "corpus_fingerprint": corpus_fingerprint,
        "parser_config_hash": legacy._hash_payload(parser_config),
        "chunking_config_hash": legacy._hash_payload(chunking_config),
    }
    retrieval_config_hash = legacy._hash_payload(retrieval_config)
    retrieval_path = output_dir / "retrieval" / f"{scope}_baseline_legacy.jsonl"
    timings_path = output_dir / "retrieval" / f"{scope}_timings.jsonl"
    retrieval_rows: dict[str, dict[str, Any]] = {}
    timing_rows: dict[str, dict[str, Any]] = {}

    runner = OnDemandPerQueryRunner(
        page_index,
        parser_config=parser_config,
        chunking_config=chunking_config,
        project_root=ROOT,
        work_dir=output_dir / "work" / scope,
        cache_dir=output_dir / "cache" / scope,
        top_k_pages=top_k_pages,
        top_k_chunks=legacy_eval_top_k_chunks,
        depth=depth,
        alpha=alpha,
        query_workers=query_workers,
        microbatch_window_seconds=microbatch_window,
        microbatch_max_pages=microbatch_max_pages,
        force_reparse=args.force_reparse,
    )
    try:
        legacy._run_retrieval(
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
        _add_file_metrics(retrieval_rows, questions, page_index)
        legacy._write_ordered_jsonl(retrieval_path, retrieval_rows, questions)
        timing_summary = runner.timing_summary(light_preparation_seconds=index_seconds)
        qa_rows: dict[str, dict[str, Any]] = {}
        if not args.skip_qa:
            qa_rows = legacy._run_qa(
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
                judge_prompt=qa_prompt_settings["judge_prompt"],
                score_parser=qa_prompt_settings["score_parser"],
                judge_score_schema=qa_prompt_settings["judge_score_schema"],
                context_chunk_limit=(
                    generation_top_k_chunks
                    if generation_top_k_chunks != legacy_eval_top_k_chunks
                    else None
                ),
                generation_render_prompt=qa_prompt_settings[
                    "generation_render_prompt"
                ],
                generation_prompt_schema=qa_prompt_settings[
                    "generation_prompt_schema"
                ],
                judge_max_output_tokens=qa_prompt_settings[
                    "judge_max_output_tokens"
                ],
            )
        report = _build_report(
            questions,
            retrieval_rows,
            qa_rows,
            page_index=page_index,
            runner=runner,
            top_k_pages=top_k_pages,
            legacy_eval_top_k_chunks=legacy_eval_top_k_chunks,
            generation_top_k_chunks=generation_top_k_chunks,
            depth=depth,
            alpha=alpha,
            generator_model=generator_model,
            judge_model=judge_model,
            qa_prompt_profile=qa_prompt_profile,
            light_preparation_seconds=index_seconds,
        )
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{scope}_baseline_legacy.json"
        _write_json(report_path, report)
        _write_json(reports_dir / f"{scope}_timing_summary.json", timing_summary)
        experiment_summary = _experiment_summary_row(report)
        summary_stem = reports_dir / f"{scope}_experiment_summary"
        _write_json(summary_stem.with_suffix(".json"), experiment_summary)
        _write_summary_csv(summary_stem.with_suffix(".csv"), experiment_summary)
        _write_summary_markdown(summary_stem.with_suffix(".md"), experiment_summary)
        manifest = {
            "contract_version": "benchmark-bundle-on-demand-basic-v1",
            "pipeline": PIPELINE,
            "retrieval_scope": scope,
            "bundle_root": str(bundle_root),
            "config": str(config_path),
            "documents_indexed": len(documents),
            "questions_selected": len(questions),
            "indexed_pages": len(page_index.pages),
            "corpus_fingerprint": corpus_fingerprint,
            "legacy_eval_top_k_chunks": legacy_eval_top_k_chunks,
            "generation_top_k_chunks": generation_top_k_chunks,
            "qa_prompt_profile": qa_prompt_profile,
            "parser_config": parser_config,
            "chunking_config": chunking_config,
            "retrieval_config": retrieval_config,
            "retrieval_output": str(retrieval_path),
            "timings_output": str(timings_path),
            "report_output": str(report_path),
            "experiment_summary_outputs": {
                "json": str(summary_stem.with_suffix(".json")),
                "csv": str(summary_stem.with_suffix(".csv")),
                "markdown": str(summary_stem.with_suffix(".md")),
            },
            "timing_summary": timing_summary,
            "report": report,
        }
        _write_json(output_dir / f"manifest_{scope}.json", manifest)
        legacy._update_summary(output_dir / "summary.json", scope, report, timing_summary)
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
    parser.add_argument("--bundle-root", type=Path, required=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/pipeline.benchmark-bundle-on-demand-basic.yaml",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retrieval-scope", choices=("lake",), default="lake")
    parser.add_argument("--source", action="append", help="Limit to a source name.")
    parser.add_argument("--top-k-pages", type=int)
    parser.add_argument(
        "--top-k-chunks",
        type=int,
        help="Deprecated shared fallback for both chunk limits.",
    )
    parser.add_argument(
        "--legacy-eval-top-k-chunks",
        type=int,
        help="Rank this many Legacy chunks before deriving file retrieval metrics.",
    )
    parser.add_argument(
        "--generation-top-k-chunks",
        type=int,
        help="Limit the ranked chunks included in the generator context.",
    )
    parser.add_argument("--depth", type=int)
    parser.add_argument("--alpha", type=float)
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
        "--qa-prompt-profile",
        choices=("vidore_v3", "grounded_bundle"),
        help="Generator and judge prompt profile for QA.",
    )
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument("--skip-qa", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _load_bundle(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    documents_path = root / "documents.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_path = root / "qrels.jsonl"
    documents_data = _read_jsonl(documents_path)
    queries_data = _read_jsonl(queries_path)
    qrels_data = _read_jsonl(qrels_path)

    resolved_root = root.resolve()
    documents: list[dict[str, Any]] = []
    documents_by_id: dict[str, dict[str, Any]] = {}
    for item in documents_data:
        doc_id = _required_string(item, "doc_id", documents_path)
        relative_path = _required_string(item, "path", documents_path)
        pdf_path = (resolved_root / relative_path).resolve()
        try:
            pdf_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"Document path escapes bundle root: {relative_path}") from error
        if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
            raise FileNotFoundError(f"Bundle document is not a readable PDF: {pdf_path}")
        if doc_id in documents_by_id:
            raise ValueError(f"Duplicate document id: {doc_id}")
        document = {
            "doc_id": doc_id,
            "source": str(item.get("source") or "unknown"),
            "pdf_path": str(pdf_path),
            "metadata": dict(item.get("metadata") or {}),
        }
        documents.append(document)
        documents_by_id[doc_id] = document

    queries: list[dict[str, Any]] = []
    queries_by_id: dict[str, dict[str, Any]] = {}
    for item in queries_data:
        qid = _required_string(item, "query_id", queries_path)
        question = _required_string(item, "query", queries_path)
        if qid in queries_by_id:
            raise ValueError(f"Duplicate query id: {qid}")
        answers = [str(value).strip() for value in item.get("answers") or [] if str(value).strip()]
        query = {
            "qid": qid,
            "question": question,
            "answer": "\n\n".join(answers),
            "evidence": "",
            "source": str(item.get("source") or "unknown"),
            "source_query_id": str(item.get("source_query_id") or ""),
            "gold_doc_ids": [],
        }
        queries.append(query)
        queries_by_id[qid] = query

    gold_relevance: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for item in qrels_data:
        qid = _required_string(item, "query_id", qrels_path)
        doc_id = _required_string(item, "doc_id", qrels_path)
        if qid not in queries_by_id:
            raise ValueError(f"Qrel refers to unknown query id: {qid}")
        if doc_id not in documents_by_id:
            raise ValueError(f"Qrel refers to unknown document id: {doc_id}")
        try:
            relevance = float(item.get("relevance", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid qrel relevance for {qid}/{doc_id}") from error
        if relevance > 0.0:
            previous = gold_relevance[qid].get(doc_id, 0.0)
            gold_relevance[qid][doc_id] = max(previous, relevance)
    missing_qrels = [
        item["qid"] for item in queries if not gold_relevance[item["qid"]]
    ]
    if missing_qrels:
        raise ValueError(f"Queries without a positive qrel: {missing_qrels[:5]}")
    for item in queries:
        relevance = gold_relevance[item["qid"]]
        item["gold_doc_ids"] = sorted(relevance)
        item["qrel_relevance"] = {
            doc_id: relevance[doc_id] for doc_id in sorted(relevance)
        }
    return documents, queries


def _bundle_root(root: Path) -> Path:
    candidate = root.resolve()
    if _is_bundle_root(candidate):
        return candidate
    nested = candidate / "0. BENCHMARK"
    if _is_bundle_root(nested):
        return nested
    raise FileNotFoundError(
        "Bundle root must contain documents.jsonl, queries.jsonl, qrels.jsonl, and data/. "
        f"Received: {root}"
    )


def _is_bundle_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "data").is_dir()
        and all(
            (path / name).is_file()
            for name in ("documents.jsonl", "queries.jsonl", "qrels.jsonl")
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required bundle manifest is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(value)
    return rows


def _required_string(item: dict[str, Any], key: str, path: Path) -> str:
    value = str(item.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing {key!r} in {path}")
    return value


def _load_or_build_page_index(
    documents: list[dict[str, Any]],
    *,
    output_dir: Path,
    force_rebuild: bool,
) -> tuple[PageIndex, str]:
    paths = [Path(item["pdf_path"]).resolve() for item in documents]
    fingerprint = legacy._corpus_fingerprint(paths)
    index_dir = output_dir / "indexes" / f"page_bm25_lake_{fingerprint}"
    if (
        not force_rebuild
        and (index_dir / "pages.jsonl").is_file()
        and (index_dir / "bm25.json").is_file()
    ):
        LOGGER.info("Loading cached pdf-inspector/BM25 index: %s", index_dir)
        return PageIndex.load(index_dir), fingerprint
    source_uri_by_path = {
        str(Path(item["pdf_path"]).resolve()): str(item["doc_id"])
        for item in documents
    }

    def source_uri(path: Path) -> str:
        return source_uri_by_path[str(path.resolve())]

    LOGGER.info("Building pdf-inspector/BM25 index: documents=%d scope=lake", len(documents))
    index = build_page_index(paths, parser=PdfInspectorPageParser(), source_uri=source_uri)
    index.save(index_dir)
    return index, fingerprint


def _add_file_metrics(
    retrieval_rows: dict[str, dict[str, Any]],
    questions: Iterable[dict[str, Any]],
    page_index: PageIndex,
) -> None:
    source_by_page_id = {page.page_id: page.source_uri for page in page_index.pages}
    for question in questions:
        qid = str(question["qid"])
        row = retrieval_rows.get(qid)
        if not row or row.get("status") != "ok":
            continue
        qrel_relevance = {
            str(doc_id): float(score)
            for doc_id, score in dict(question["qrel_relevance"]).items()
        }
        light_files = _unique(
            str(item.get("source_uri") or "") for item in row.get("hits") or []
        )
        hybrid_files = _unique(
            source_by_page_id.get(str(item.get("doc_id") or ""), "")
            for item in row.get("chunks") or []
        )
        row["light_bm25_top_files"] = light_files
        row["baseline_legacy_top_files"] = hybrid_files
        row["file_metrics"] = {
            "light_bm25": _file_metrics(light_files, qrel_relevance),
            "baseline_legacy": _file_metrics(hybrid_files, qrel_relevance),
        }


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _file_metrics(
    ranked_files: list[str],
    relevance: dict[str, float] | set[str],
) -> dict[str, Any]:
    relevance_by_file = (
        {str(doc_id): float(score) for doc_id, score in relevance.items()}
        if isinstance(relevance, dict)
        else {str(doc_id): 1.0 for doc_id in relevance}
    )
    gold_doc_ids = set(relevance_by_file)
    top_three = ranked_files[:3]
    top_ten = ranked_files[:10]
    found = set(top_three) & gold_doc_ids
    found_at_ten = set(top_ten) & gold_doc_ids
    first_rank = next(
        (
            index
            for index, value in enumerate(ranked_files, start=1)
            if value in gold_doc_ids
        ),
        None,
    )
    return {
        "file_hit@3": bool(found),
        "file_recall@3": len(found) / len(gold_doc_ids) if gold_doc_ids else 0.0,
        "file_ndcg@10": _ndcg_at_k(top_ten, relevance_by_file, k=10),
        "file_recall@10": (
            len(found_at_ten) / len(gold_doc_ids) if gold_doc_ids else 0.0
        ),
        "first_gold_file_rank": first_rank,
    }


def _ndcg_at_k(
    ranked_files: list[str], relevance_by_file: dict[str, float], *, k: int
) -> float:
    def gain(relevance: float) -> float:
        return (2.0**relevance) - 1.0

    dcg = sum(
        gain(relevance_by_file.get(doc_id, 0.0)) / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_files[:k], start=1)
    )
    ideal = sorted(relevance_by_file.values(), reverse=True)[:k]
    idcg = sum(
        gain(value) / math.log2(rank + 1)
        for rank, value in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg else 0.0


def _build_report(
    questions: list[dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
    qa_rows: dict[str, dict[str, Any]],
    *,
    page_index: PageIndex,
    runner: OnDemandPerQueryRunner,
    top_k_pages: int,
    legacy_eval_top_k_chunks: int,
    generation_top_k_chunks: int,
    depth: int,
    alpha: float,
    generator_model: str,
    judge_model: str,
    qa_prompt_profile: str,
    light_preparation_seconds: float,
) -> dict[str, Any]:
    successful_retrievals = [
        retrieval_rows[str(question["qid"])]
        for question in questions
        if retrieval_rows.get(str(question["qid"]), {}).get("status") == "ok"
    ]

    def file_mean(arm: str, metric: str) -> float | None:
        values = [
            float((row.get("file_metrics") or {}).get(arm, {}).get(metric, 0.0))
            for row in successful_retrievals
        ]
        return _mean(values)

    def retrieval_timing_mean(name: str) -> float | None:
        return _mean(
            [
                float((row.get("timing") or {}).get(name, 0.0))
                for row in successful_retrievals
            ]
        )

    qa_ordered = [
        qa_rows[str(question["qid"])]
        for question in questions
        if str(question["qid"]) in qa_rows
    ]
    completed = [
        row
        for row in qa_ordered
        if row.get("status") == "ok" and row.get("score") is not None
    ]
    scores = [float(row["score"]) for row in completed]
    errors = [row for row in qa_ordered if row.get("status") != "ok"]
    sources: defaultdict[str, list[float]] = defaultdict(list)
    for row in completed:
        sources[str(row.get("source") or "unknown")].append(float(row["score"]))
    generation_seconds = [
        float(row["generation_seconds"])
        for row in qa_ordered
        if row.get("generation_seconds") is not None
    ]
    judge_seconds = [
        float(row["judge_seconds"])
        for row in qa_ordered
        if row.get("judge_seconds") is not None
    ]
    qa_by_qid = {str(row["qid"]): row for row in qa_ordered}
    end_to_end_inference_seconds = []
    for question in questions:
        qid = str(question["qid"])
        retrieval = retrieval_rows.get(qid) or {}
        qa = qa_by_qid.get(qid) or {}
        if retrieval.get("status") != "ok" or qa.get("generation_seconds") is None:
            continue
        kdl_service = float(
            (retrieval.get("timing") or {}).get("kdl_batch_service_seconds", 0.0)
        )
        end_to_end_inference_seconds.append(
            kdl_service
            + float(qa["generation_seconds"])
            + float(qa.get("judge_seconds") or 0.0)
        )
    return {
        "arm": "baseline_legacy",
        "pipeline": PIPELINE,
        "retrieval_scope": "lake",
        "retrieval_metric_scope": "file",
        "documents": len({page.source_uri for page in page_index.pages}),
        "indexed_pages": len(page_index.pages),
        "questions_expected": len(questions),
        "questions_retrieved": len(successful_retrievals),
        "retrieval_errors": len(questions) - len(successful_retrievals),
        "light_bm25_file_hit@3": file_mean("light_bm25", "file_hit@3"),
        "light_bm25_file_recall@3": file_mean("light_bm25", "file_recall@3"),
        "light_bm25_file_ndcg@10": file_mean("light_bm25", "file_ndcg@10"),
        "light_bm25_file_recall@10": file_mean("light_bm25", "file_recall@10"),
        "baseline_legacy_file_hit@3": file_mean("baseline_legacy", "file_hit@3"),
        "baseline_legacy_file_recall@3": file_mean("baseline_legacy", "file_recall@3"),
        "baseline_legacy_file_ndcg@10": file_mean(
            "baseline_legacy", "file_ndcg@10"
        ),
        "baseline_legacy_file_recall@10": file_mean(
            "baseline_legacy", "file_recall@10"
        ),
        "page_qrel_metrics": None,
        "page_qrel_note": (
            "qrels source_corpus_id values are not local PDF page numbers; "
            "page metrics require a provenance map."
        ),
        "generator": generator_model,
        "judge": judge_model,
        "qa_prompt_profile": qa_prompt_profile,
        "questions_completed": len(completed),
        "qa_errors": len(errors),
        "accuracy": _mean([float(score == 1.0) for score in scores]),
        "correct_only_accuracy": _mean([float(score == 1.0) for score in scores]),
        "correct_plus_partial_accuracy": _mean(
            [float(score >= 0.5) for score in scores]
        ),
        "graded_accuracy": _mean(scores),
        "score_1": sum(score == 1.0 for score in scores),
        "score_partial": sum(score == 0.5 for score in scores),
        "score_0": sum(score == 0.0 for score in scores),
        "accuracy_by_source": {
            key: _mean([float(score == 1.0) for score in values])
            for key, values in sorted(sources.items())
        },
        "top_k_pages": top_k_pages,
        # Retained for existing report consumers; it is the Legacy ranking depth.
        "top_k_chunks": legacy_eval_top_k_chunks,
        "legacy_eval_top_k_chunks": legacy_eval_top_k_chunks,
        "generation_top_k_chunks": generation_top_k_chunks,
        "chunk_retrieval_depth": depth,
        "chunk_retrieval_alpha": alpha,
        "parsed_page_cache": len(runner.page_texts),
        "prepared_chunk_cache": runner.prepared_chunk_count,
        "time_light_preparation_seconds": round(light_preparation_seconds, 3),
        "latency_seconds_per_query_mean": {
            "light_retrieval": retrieval_timing_mean("light_retrieval_seconds"),
            "parsing": retrieval_timing_mean("parsing_seconds"),
            "chunk_embed_index": retrieval_timing_mean("chunk_embed_index_seconds"),
            "retrieval": retrieval_timing_mean("retrieval_seconds"),
        },
        "inference_seconds_per_query_mean": {
            "kdl_service": retrieval_timing_mean("kdl_batch_service_seconds"),
            "generation": _mean(generation_seconds),
            "judge": _mean(judge_seconds),
            "end_to_end_model_total": _mean(end_to_end_inference_seconds),
        },
        "inference_time_note": (
            "KDL service is the observed service time for each query's shared "
            "micro-batch; it is not an additive GPU-cost allocation."
        ),
    }


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


def _mean(values: list[int | float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _parse_ternary_score(text: str) -> float:
    prefix = str(text or "")[:200]
    match = re.search(
        r"correctness\s*:\s*(0(?:\.5)?|1(?:\.0)?)\b",
        prefix,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(r"\b(0(?:\.5)?|1(?:\.0)?)\b", prefix)
    if not match:
        raise ValueError(f"judge response has no 0, 0.5, or 1 score: {prefix!r}")
    score = float(match.group(1))
    if score not in {0.0, 0.5, 1.0}:
        raise ValueError(f"judge returned an unsupported score: {prefix!r}")
    return score


def _experiment_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    latency = dict(report.get("latency_seconds_per_query_mean") or {})
    inference = dict(report.get("inference_seconds_per_query_mean") or {})
    return {
        "metric_scope": report.get("retrieval_metric_scope"),
        "e2e_correct_only_accuracy": report.get("correct_only_accuracy"),
        "e2e_correct_plus_partial_accuracy": report.get(
            "correct_plus_partial_accuracy"
        ),
        "light_file_recall@3": report.get("light_bm25_file_recall@3"),
        "light_file_ndcg@10": report.get("light_bm25_file_ndcg@10"),
        "light_file_recall@10": report.get("light_bm25_file_recall@10"),
        "legacy_file_ndcg@10": report.get("baseline_legacy_file_ndcg@10"),
        "legacy_file_recall@10": report.get("baseline_legacy_file_recall@10"),
        "time_light_preparation_seconds": report.get(
            "time_light_preparation_seconds"
        ),
        "latency_light_retrieval_seconds_per_query": latency.get(
            "light_retrieval"
        ),
        "latency_parsing_seconds_per_query": latency.get("parsing"),
        "latency_chunk_embed_index_seconds_per_query": latency.get(
            "chunk_embed_index"
        ),
        "latency_retrieval_seconds_per_query": latency.get("retrieval"),
        "inference_kdl_service_seconds_per_query": inference.get("kdl_service"),
        "inference_generation_seconds_per_query": inference.get("generation"),
        "inference_judge_seconds_per_query": inference.get("judge"),
        "inference_e2e_model_total_seconds_per_query": inference.get(
            "end_to_end_model_total"
        ),
        "questions_expected": report.get("questions_expected"),
        "questions_retrieved": report.get("questions_retrieved"),
        "questions_completed": report.get("questions_completed"),
    }


def _write_summary_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_summary_markdown(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# On-demand Basic Experiment Summary",
        "",
        "All retrieval metrics in this table are file-level, derived from qrels.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {_format_summary_value(value)} |" for key, value in row.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_summary_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
