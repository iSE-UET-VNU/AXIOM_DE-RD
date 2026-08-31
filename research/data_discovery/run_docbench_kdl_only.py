"""Benchmark the DocBench page-selection path with KDL only.

The runner builds or reuses the cheap pdf-inspector page index, selects the
top-K pages for the requested questions, and sends the union to one KDL
ingestion batch.  It intentionally stops before cleaning, enrichment,
chunking, embedding, retrieval, generation, and judging so KDL throughput can
be compared independently.

Example:

    python -m research.data_discovery.run_docbench_kdl_only \
      --config configs/pipeline.docbench-on-demand-basic.yaml \
      --docbench-root /path/to/DocBench \
      --max-pages 512
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import logging
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import (  # noqa: E402
    _ingest_selected_pages,
)
from research.data_discovery.run_docbench_e2e import (  # noqa: E402
    _apply_kdl_overrides,
    _check_vllm_endpoint,
    _load_docbench,
    _load_or_build_page_index,
    _require_env,
    _resolve_path,
)
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402
from src.utils.observability import (  # noqa: E402
    configure_run_logging,
    utc_now_iso,
)


LOGGER = logging.getLogger("docbench_kdl_only")


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    started = time.perf_counter()
    run_started_at = utc_now_iso()
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
        or ROOT / "data" / "benchmark" / "docbench_kdl_only"
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_run_logging(
        output_dir / "logs" / f"{scope}_kdl_only.log",
        level=getattr(logging, str(args.log_level).upper()),
    )

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
            raise ValueError("--limit must be positive")
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

    top_k_pages = int(
        args.top_k_pages or docbench_config.get("top_k_pages") or 10
    )
    if top_k_pages <= 0:
        raise ValueError("--top-k-pages must be positive")
    selected = _union_pages(page_index, questions, top_k_pages)
    selected = _limit_pages(selected, args.max_pages)
    selected_page_count = sum(len(indices) for indices in selected.values())
    if selected_page_count <= 0:
        raise RuntimeError("Page discovery selected no pages")

    parser_config = resolve_parser_config(
        ROOT,
        config.get("parsing") or {},
        output_dir / "parser-assets" / scope,
    )
    parser_config = _apply_kdl_overrides(parser_config, args)
    kdl = dict(parser_config.get("kdl") or {})
    if args.kdl_scheduler is not None:
        kdl["scheduler"] = args.kdl_scheduler
    kdl["event_log_path"] = str(
        output_dir / "logs" / f"{scope}_kdl_events.jsonl"
    )
    parser_config["kdl"] = kdl
    if parser_config.get("provider") != "kdl_pdf_inspector":
        raise ValueError(
            "KDL-only DocBench benchmark requires "
            "parsing.provider=kdl_pdf_inspector"
        )

    LOGGER.info(
        "KDL-only: scope=%s questions=%d indexed_pages=%d selected_pages=%d "
        "scheduler=%s max_workers=%s render_processes=%s bbox_max_workers=%s",
        scope,
        len(questions),
        len(page_index.pages),
        selected_page_count,
        kdl.get("scheduler"),
        kdl.get("max_workers"),
        kdl.get("render_processes"),
        kdl.get("bbox_max_workers"),
    )

    parse_started = time.perf_counter()
    ingestion_output = _ingest_selected_pages(
        selected,
        parser_config=parser_config,
        project_root=ROOT,
        work_dir=output_dir / "work" / scope,
        one_page_inputs=True,
    )
    parse_seconds = time.perf_counter() - parse_started

    reported_latencies = [
        float((record.metadata or {}).get("latency_seconds"))
        for record in ingestion_output.parsed_data
        if (record.metadata or {}).get("latency_seconds") is not None
    ]
    parsed_pages = len(ingestion_output.parsed_data)
    result = {
        "contract_version": "docbench-kdl-only-v1",
        "pipeline": "pdf_inspector -> bm25 page selection -> KDL",
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": utc_now_iso(),
        "config": str(config_path),
        "docbench_root": str(docbench_root.resolve()),
        "retrieval_scope": scope,
        "questions": len(questions),
        "documents_selected": len(selected_documents),
        "documents_indexed": len(index_documents),
        "indexed_pages": len(page_index.pages),
        "discovered_unique_pages": selected_page_count,
        "requested_pages": selected_page_count,
        "parsed_pages": parsed_pages,
        "quarantined_pages": len(ingestion_output.quarantined_documents),
        "top_k_pages": top_k_pages,
        "max_pages_sample": args.max_pages,
        "corpus_fingerprint": corpus_fingerprint,
        "parser_config": parser_config,
        "timing": {
            "page_index_seconds": round(index_seconds, 3),
            "kdl_ingestion_wall_seconds": round(parse_seconds, 3),
            "reported_kdl_latency_sum_seconds": round(
                sum(reported_latencies), 3
            ),
            "reported_kdl_latency_mean_seconds": round(
                sum(reported_latencies) / len(reported_latencies), 3
            )
            if reported_latencies
            else None,
            "parsed_pages_per_wall_second": round(
                parsed_pages / parse_seconds, 4
            )
            if parse_seconds > 0
            else None,
            "end_to_end_seconds": round(time.perf_counter() - started, 3),
        },
        "notes": [
            "No chunking, embedding, chunk retrieval, generation, or judging was run.",
            "kdl_ingestion_wall_seconds includes ingestion result validation and page artifact handling.",
            "reported_kdl_latency_sum_seconds is aggregate per-page KDL metadata and is not wall-clock time.",
        ],
    }
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{scope}_kdl_only.json"
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "KDL-only complete: parsed=%d/%d wall=%.3fs pages_per_second=%.4f report=%s",
        parsed_pages,
        selected_page_count,
        parse_seconds,
        result["timing"]["parsed_pages_per_wall_second"] or 0.0,
        report_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _union_pages(
    page_index: Any,
    questions: list[dict[str, Any]],
    top_k_pages: int,
) -> dict[str, list[int]]:
    selected: defaultdict[str, set[int]] = defaultdict(set)
    for question in questions:
        for hit in page_index.search(question["question"], top_k=top_k_pages):
            selected[hit.evidence.file_path].add(int(hit.evidence.page_index))
    return {
        path: sorted(indices)
        for path, indices in sorted(selected.items())
        if indices
    }


def _limit_pages(
    selected: dict[str, list[int]], max_pages: int | None
) -> dict[str, list[int]]:
    if max_pages is None:
        return selected
    if max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    remaining = max_pages
    limited: dict[str, list[int]] = {}
    for path in sorted(selected):
        if remaining <= 0:
            break
        indices = sorted(selected[path])[:remaining]
        if indices:
            limited[path] = indices
            remaining -= len(indices)
    return limited


def _select_documents(
    documents: list[dict[str, Any]], max_documents: int | None
) -> list[dict[str, Any]]:
    if max_documents is None:
        return documents
    if max_documents <= 0:
        raise ValueError("--max-documents must be positive")
    return documents[:max_documents]


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
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force-rebuild-index", action="store_true")
    parser.add_argument("--skip-endpoint-check", action="store_true")
    parser.add_argument(
        "--kdl-scheduler",
        choices=("parsebench_document", "global_two_phase"),
    )
    parser.add_argument("--kdl-max-workers", type=int)
    parser.add_argument("--kdl-render-processes", type=int)
    parser.add_argument("--kdl-bbox-max-workers", type=int)
    parser.add_argument("--kdl-request-workers", type=int)
    parser.add_argument("--kdl-request-batch-size", type=int)
    parser.add_argument("--kdl-max-model-sequences", type=int)
    parser.add_argument("--kdl-host-failure-threshold", type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
