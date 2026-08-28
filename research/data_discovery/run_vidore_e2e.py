"""Run page-discovery E2E QA settings on a ViDoRe V3 subset.

The default mode chooses pages with BM25 and ingests the union of pages
selected by the requested questions once through the production parser (KDL
over the vLLM endpoint from ``.env``).  ``--on-demand-per-query`` switches to
the online mode: each query retrieves its own BM25 pages, parses only missing
pages, incrementally chunks/embeds/indexes them, and performs scoped hybrid
retrieval.  Its parser and chunk caches persist below ``--work-dir``.

Arms:

``chunks``
    Retrieve chunks from the accurately ingested selected pages and send the
    ranked chunks to the generator.

``pages``
    Send all accurately ingested text from the selected pages directly to the
    generator, without a second chunk/embed retrieval step.

The generator and judge are OpenRouter model ids.  Results are checkpointed
after every completed request so a remote endpoint failure can be resumed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.pipeline import (  # noqa: E402
    PageIndex,
    run_from_parse_artifacts,
    run_selected_pages,
)
from research.data_discovery.on_demand_per_query import (  # noqa: E402
    OnDemandPerQueryRunner,
)
from src.evaluation.benchmarks import load  # noqa: E402
from src.evaluation.benchmarks.vidore_v3_judge import (  # noqa: E402
    ANSWER_PROMPT,
    ViDoreVerdict,
    judge_answer,
    render_documents,
    score,
)
from src.evaluation.llm import complete  # noqa: E402
from src.retrieval.fusion import alpha_fuse  # noqa: E402
from src.retrieval.index import LocalIndex  # noqa: E402
from src.retrieval.protocol import ChunkRecord  # noqa: E402
from src.retrieval.retrievers import build as build_retriever  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402
from src.utils.config import load_config, resolve_parser_config  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402


@dataclass(frozen=True)
class PreparedChunk:
    record_id: str
    page_id: str
    text: str
    vector: np.ndarray


def main(argv: list[str] | None = None) -> int:
    args = _arguments().parse_args(argv)
    run_started = time.perf_counter()
    load_dotenv_file(ROOT)
    _require_env("OPENROUTER_API_KEY")
    _require_env("VLLM_API_BASE")

    parser_config = _resolved_parser_config(args.parser_config, args.work_dir)
    chunking_config = _chunking_config(args)
    index_started = time.perf_counter()
    index = PageIndex.load(args.index_dir)
    index_load_seconds = time.perf_counter() - index_started
    if not index.pages:
        raise RuntimeError(f"Discovery index is empty: {args.index_dir}")

    benchmark = load(
        "vidore_v3",
        root=args.benchmark_root,
        subset=args.subset,
        language=args.language,
    )
    questions = list(benchmark.questions())
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError("No questions selected")

    if args.resume_arm:
        if args.on_demand_per_query:
            raise ValueError(
                "--resume-arm is not supported with --on-demand-per-query; "
                "reuse the per-query cache and rerun the mode instead."
            )
        _resume_arm_from_checkpoint(args.resume_arm, questions, args)
        return 0

    if args.on_demand_per_query:
        return _run_on_demand_per_query(
            args,
            index=index,
            questions=questions,
            parser_config=parser_config,
            chunking_config=chunking_config,
            index_load_seconds=index_load_seconds,
            run_started=run_started,
        )

    discovery_started = time.perf_counter()
    hits_by_qid = {
        question.qid: index.search(question.query, top_k=args.top_k_pages)
        for question in questions
    }
    discovery_seconds = time.perf_counter() - discovery_started
    selected = _union_pages(hits_by_qid)
    print(
        f"discovery pages={len(index.pages)} questions={len(questions)} "
        f"selected_unique_pages={sum(len(v) for v in selected.values())} "
        f"vllm={os.environ.get('VLLM_API_BASE')} model="
        f"{os.environ.get('VLLM_MODEL_NAME', 'kdl-frontier-parser-nano')}",
        flush=True,
    )

    started = time.perf_counter()
    if args.reuse_parser_artifacts:
        pipeline = run_from_parse_artifacts(
            selected,
            parser_artifacts_dir=args.reuse_parser_artifacts,
            project_root=ROOT,
            chunking_config=None,
        )
    else:
        pipeline = run_selected_pages(
            selected,
            parser_config=parser_config,
            # Both arms share this accurate ingestion result.  Chunking and
            # embeddings are intentionally run only below for the ``chunks`` arm;
            # the ``pages`` arm never enters that stage.
            chunking_config=None,
            project_root=ROOT,
            work_dir=args.work_dir,
            one_page_inputs=True,
        )
    ingestion_seconds = time.perf_counter() - started
    print(
        f"ingested={len(pipeline.ingestion.parsed_data)} "
        f"quarantined={len(pipeline.ingestion.quarantined_documents)} "
        f"chunks={len(pipeline.chunking_embedding.retrieval_records) if pipeline.chunking_embedding else 0} "
        f"pipeline_seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )

    page_text_started = time.perf_counter()
    page_texts = _page_texts(pipeline.enriched.enriched_data, index)
    page_text_seconds = time.perf_counter() - page_text_started
    if not page_texts:
        raise RuntimeError("Accurate ingestion returned no page text")
    manifest = {
            "contract_version": "vidore-v3-discovery-e2e-v1",
            "subset": args.subset,
            "language": args.language,
            "questions": len(questions),
            "discovery_pages": len(index.pages),
            "selected_unique_pages": len(page_texts),
            "ingested_pages": len(page_texts),
            "quarantined_documents": len(pipeline.ingestion.quarantined_documents),
            "vllm_api_base": os.environ.get("VLLM_API_BASE"),
            "vllm_model": os.environ.get("VLLM_MODEL_NAME"),
            "parser_config": str(args.parser_config),
            "parser_artifacts_reused": bool(args.reuse_parser_artifacts),
            "parser_artifacts_dir": (
                str(args.reuse_parser_artifacts)
                if args.reuse_parser_artifacts
                else None
            ),
            "kdl_scheduler": ((parser_config.get("kdl") or {}).get("scheduler")),
            "generator": args.generator,
            "judge": args.judge if not args.skip_judge else None,
            "timing_seconds": {
                "discovery_retrieval": round(discovery_seconds, 3),
                "ingestion_cleaning_enrichment": round(ingestion_seconds, 3),
                "page_text_extraction": round(page_text_seconds, 3),
            },
        }
    _write_json(args.output_dir / "manifest.json", manifest)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = {arm.strip() for arm in args.arms.split(",") if arm.strip()}
    unknown = arms - {"pages", "chunks"}
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")

    arm_timings: dict[str, Any] = {}
    if "pages" in arms:
        arm_timings["pages"] = _run_arm(
            "pages", questions, hits_by_qid, page_texts, None, args
        )

    if "chunks" in arms:
        from src.chunking_embedding.stage import run as run_chunking_embedding

        chunking_started = time.perf_counter()
        pipeline.chunking_embedding = run_chunking_embedding(
            [record.__dict__ for record in pipeline.enriched.enriched_data],
            chunking_config,
        )
        print(
            f"chunks={len(pipeline.chunking_embedding.retrieval_records)} "
            f"chunk_embed_seconds={time.perf_counter() - chunking_started:.1f}",
            flush=True,
        )
        prepared_chunks = _prepared_chunks(pipeline, page_texts, index)
        chunk_retrieval_started = time.perf_counter()
        chunk_ranked = _retrieve_chunks(
            prepared_chunks,
            hits_by_qid,
            questions,
            chunking_config,
            top_k=args.top_k_chunks,
            depth=args.depth,
            alpha=args.alpha,
            batch_size=args.retrieval_batch_size,
        )
        chunk_retrieval_seconds = time.perf_counter() - chunk_retrieval_started
        arm_timings["chunks"] = _run_arm(
            "chunks", questions, hits_by_qid, page_texts, chunk_ranked, args
        )
        arm_timings["chunks"].update({
            "chunk_embed_seconds": round(time.perf_counter() - chunking_started, 3),
            "chunk_retrieval_seconds": round(chunk_retrieval_seconds, 3),
            "embedding_batch_size": int((chunking_config.get("embedder_params") or {}).get("batch_size", 64)),
            "retrieval_batch_size": args.retrieval_batch_size,
        })
    manifest["timing_seconds"].update({
        "arms": arm_timings,
        "total": round(time.perf_counter() - run_started, 3),
    })
    _write_json(args.output_dir / "manifest.json", manifest)
    return 0


def _run_on_demand_per_query(
    args: argparse.Namespace,
    *,
    index: PageIndex,
    questions: list[Any],
    parser_config: dict[str, Any],
    chunking_config: dict[str, Any],
    index_load_seconds: float,
    run_started: float,
) -> int:
    """Evaluate the stateful online pipeline, one query at a time."""
    if args.reuse_parser_artifacts:
        raise ValueError(
            "--reuse-parser-artifacts is a batch-mode option. "
            "Use --on-demand-cache-dir for per-query parser/chunk cache."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parser_config = _on_demand_parser_config(parser_config, args)
    cache_dir = args.on_demand_cache_dir or (args.work_dir / "on-demand-cache")
    runner = OnDemandPerQueryRunner(
        index,
        parser_config=parser_config,
        chunking_config=chunking_config,
        project_root=ROOT,
        work_dir=args.work_dir,
        cache_dir=cache_dir,
        top_k_pages=args.top_k_pages,
        top_k_chunks=args.top_k_chunks,
        depth=args.depth,
        alpha=args.alpha,
        query_workers=args.query_workers,
        microbatch_window_seconds=args.kdl_microbatch_window_seconds,
        microbatch_max_pages=args.kdl_microbatch_max_pages,
        force_reparse=args.force_reparse_on_demand,
    )
    try:
        query_results = runner.run_queries(
            ((question.qid, question.query) for question in questions),
            max_workers=args.query_workers,
        )
        timings_by_qid = {
            qid: result.timing for qid, result in query_results.items()
        }
        hits_by_qid = {
            qid: result.hits for qid, result in query_results.items()
        }
        chunk_ranked = {
            qid: result.ranked_chunks for qid, result in query_results.items()
        }
        page_texts = runner.page_texts
        timing_summary = runner.timing_summary(
            light_preparation_seconds=index_load_seconds
        )

        retrieval_path = args.output_dir / (
            f"{args.subset}_{args.language}_on_demand_per_query_retrieval.jsonl"
        )
        with retrieval_path.open("w", encoding="utf-8") as handle:
            for question in questions:
                result = query_results[question.qid]
                handle.write(
                    json.dumps(
                        {
                            "contract_version": "on-demand-per-query-retrieval-v1",
                            "pipeline": "pdf_inspector -> bm25 -> KDL -> fixed512 -> te3s -> hybrid",
                            "qid": question.qid,
                            "query": question.query,
                            "retriever_id": "on_demand_per_query_alpha0.7",
                            "hits": [hit.as_dict() for hit in result.hits],
                            "selected_pages": result.selected_pages,
                            "chunks": [
                                {
                                    "chunk_id": chunk.record_id,
                                    "doc_id": chunk.page_id,
                                    "text": chunk.text,
                                }
                                for chunk in result.ranked_chunks
                            ],
                            "timing": result.timing,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        timings_path = args.output_dir / (
            f"{args.subset}_{args.language}_on_demand_per_query_timings.jsonl"
        )
        with timings_path.open("w", encoding="utf-8") as handle:
            for question in questions:
                handle.write(
                    json.dumps(timings_by_qid[question.qid], ensure_ascii=False)
                    + "\n"
                )

        arms = {arm.strip() for arm in args.arms.split(",") if arm.strip()}
        unknown = arms - {"pages", "chunks"}
        if unknown:
            raise ValueError(f"Unknown arms: {sorted(unknown)}")
        arm_timings: dict[str, Any] = {}
        if "pages" in arms:
            arm_timings["pages"] = _run_arm(
                "pages",
                questions,
                hits_by_qid,
                page_texts,
                None,
                args,
                output_label="on_demand_per_query",
                timings_by_qid=timings_by_qid,
            )
        if "chunks" in arms:
            arm_timings["chunks"] = _run_arm(
                "chunks",
                questions,
                hits_by_qid,
                page_texts,
                chunk_ranked,
                args,
                output_label="on_demand_per_query",
                timings_by_qid=timings_by_qid,
            )

        selected_page_ids = {
            hit.page_id
            for result in query_results.values()
            for hit in result.hits
        }
        manifest = {
            "contract_version": "vidore-v3-on-demand-per-query-v1",
            "pipeline": (
                "light preparation=pdf-inspector; light retrieval=BM25; "
                "per-query KDL parse/cache -> fixed 512 chunks -> "
                "text-embedding-3-small -> hybrid alpha=0.7"
            ),
            "subset": args.subset,
            "language": args.language,
            "questions": len(questions),
            "discovery_pages": len(index.pages),
            "selected_unique_pages": len(selected_page_ids),
            "parsed_pages": len(page_texts),
            "prepared_chunks": runner.prepared_chunk_count,
            "query_workers": args.query_workers,
            "parser_config": str(args.parser_config),
            "parser_cache_dir": str(cache_dir),
            "kdl_config": dict(parser_config.get("kdl") or {}),
            "retrieval_output": str(retrieval_path),
            "timing_output": str(timings_path),
            "timing_summary": timing_summary,
            "arms": arm_timings,
            "total_runtime_seconds": round(time.perf_counter() - run_started, 3),
        }
        _write_json(args.output_dir / "manifest.json", manifest)
        print(
            f"on-demand-per-query questions={len(questions)} "
            f"selected_unique_pages={len(selected_page_ids)} "
            f"parsed_pages={len(page_texts)} "
            f"chunks={runner.prepared_chunk_count} "
            f"wall_seconds={timing_summary['online_stage_wall_seconds']:.3f}",
            flush=True,
        )
        return 0
    finally:
        runner.close()


def _on_demand_parser_config(
    parser_config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    """Apply the notebook's KDL concurrency settings without editing YAML."""
    config = dict(parser_config)
    kdl = dict(config.get("kdl") or {})
    kdl.update({
        "max_workers": args.kdl_max_workers,
        "render_processes": args.kdl_render_processes,
        "bbox_max_workers": args.kdl_bbox_max_workers,
        "request_workers": args.kdl_request_workers,
        "request_batch_size": args.kdl_request_batch_size,
        "max_model_sequences": args.kdl_max_model_sequences,
    })
    config["kdl"] = kdl
    return config


def _resolved_parser_config(path: Path, work_dir: Path) -> dict[str, Any]:
    config = load_config(path)
    parser = resolve_parser_config(ROOT, config.get("parsing") or {}, work_dir / "parser-assets")
    # Null values in the checked-in config intentionally fall through to .env.
    kdl = parser.get("kdl")
    if isinstance(kdl, dict):
        kdl = dict(kdl)
        kdl.pop("endpoint_url", None) if not kdl.get("endpoint_url") else None
        kdl.pop("model", None) if not kdl.get("model") else None
        parser["kdl"] = kdl
    return parser


def _chunking_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.chunking_config)
    section = dict(config.get("chunking_embedding") or {})
    if args.embedder:
        section["embedder"] = args.embedder
    if args.embedder_model:
        params = dict(section.get("embedder_params") or {})
        params["model"] = args.embedder_model
        section["embedder_params"] = params
    return section


def _union_pages(hits_by_qid: dict[str, list[Any]]) -> dict[str, list[int]]:
    selected: dict[str, set[int]] = {}
    for hits in hits_by_qid.values():
        for hit in hits:
            selected.setdefault(hit.evidence.file_path, set()).add(hit.evidence.page_index)
    return {path: sorted(indices) for path, indices in sorted(selected.items())}


def _page_texts(records: Iterable[Any], index: PageIndex) -> dict[str, str]:
    ids = {
        (str(Path(page.file_path).resolve()), page.page_index): page.page_id
        for page in index.pages
    }
    out: dict[str, str] = {}
    for record in records:
        metadata = record.metadata or {}
        source_metadata = metadata.get("source_metadata") or {}
        path = source_metadata.get("discovery_original_path")
        indices = source_metadata.get("discovery_page_indices")
        if not path or not isinstance(indices, list) or len(indices) != 1:
            continue
        page_id = ids.get((str(Path(path).resolve()), int(indices[0])))
        if not page_id:
            continue
        text = "\n\n".join(
            str(row.get("text") or "").strip()
            for row in (record.rows or [])
            if isinstance(row, dict) and str(row.get("text") or "").strip()
        ).strip()
        if text:
            out[page_id] = text
    return out


def _prepared_chunks(
    pipeline: Any,
    page_texts: dict[str, str],
    index: PageIndex,
) -> list[PreparedChunk]:
    output = pipeline.chunking_embedding
    if output is None:
        return []
    page_ids = {
        (str(Path(page.file_path).resolve()), page.page_index): page.page_id
        for page in index.pages
    }
    object_to_page: dict[str, str] = {}
    for record in pipeline.enriched.enriched_data:
        source_metadata = (record.metadata or {}).get("source_metadata") or {}
        indices = source_metadata.get("discovery_page_indices")
        path = source_metadata.get("discovery_original_path")
        if path and isinstance(indices, list) and len(indices) == 1:
            page_id = page_ids.get((str(Path(path).resolve()), int(indices[0])))
            if page_id and page_id in page_texts:
                object_to_page[str(record.source_object_id)] = page_id

    vectors = {
        str(item["record_id"]): np.asarray(item["embedding"], dtype=np.float32)
        for item in output.vector_records
    }
    chunks: list[PreparedChunk] = []
    for record in output.retrieval_records:
        if record.retrieval_type != "text_chunk":
            continue
        page_id = object_to_page.get(str(record.source_object_id), "")
        text = str((record.payload or {}).get("text") or "").strip()
        vector = vectors.get(record.record_id)
        if page_id and text and vector is not None:
            chunks.append(PreparedChunk(record.record_id, page_id, text, vector))
    return chunks


def _retrieve_chunks(
    chunks: list[PreparedChunk],
    hits_by_qid: dict[str, list[Any]],
    questions: list[Any],
    chunking_config: dict[str, Any],
    *,
    top_k: int,
    depth: int,
    alpha: float,
    batch_size: int,
) -> dict[str, list[PreparedChunk]]:
    if not chunks:
        return {question.qid: [] for question in questions}
    payload = [
        {"chunk_id": chunk.record_id, "doc_id": chunk.page_id, "text": chunk.text}
        for chunk in chunks
    ]
    bm25 = BM25Index(analyzer_name="auto").build(payload)
    embedder = _make_embedder(chunking_config)
    local_index = LocalIndex(
        index_id="vidore.physics.discovery.accurate.chunks",
        records=[ChunkRecord(chunk.record_id, chunk.page_id, chunk.text) for chunk in chunks],
        bm25=bm25,
        vectors=np.asarray([chunk.vector for chunk in chunks], dtype=np.float32),
        embedder=embedder,
        embeddings_model=str((chunking_config.get("embedder_params") or {}).get("model") or chunking_config.get("embedder") or ""),
        metric="cosine",
    )
    retriever = build_retriever("alpha", local_index, alpha=alpha, depth=depth)
    by_id = {chunk.record_id: chunk for chunk in chunks}
    result: dict[str, list[PreparedChunk]] = {}
    for start in range(0, len(questions), max(1, batch_size)):
        question_batch = questions[start : start + max(1, batch_size)]
        query_vectors = embedder.embed([question.query for question in question_batch])
        if len(query_vectors) != len(question_batch):
            raise RuntimeError(
                f"query embedder returned {len(query_vectors)} vectors for "
                f"{len(question_batch)} queries"
            )
        for question, query_vector in zip(question_batch, query_vectors):
            allowed_pages = {hit.page_id for hit in hits_by_qid[question.qid]}
            scope = sorted(allowed_pages)
            allowed_positions = local_index.scope_positions(scope)
            sparse_hits = retriever.sparse.raw(question.query, depth, scope)
            vector = np.asarray(query_vector, dtype=np.float32)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
            scores = local_index.vectors @ vector
            if allowed_positions is not None:
                mask = np.zeros(len(scores), dtype=bool)
                mask[list(allowed_positions)] = True
                scores = np.where(mask, scores, -np.inf)
            count = min(depth, int(np.isfinite(scores).sum()))
            dense_hits: list[tuple[str, float]] = []
            if count:
                positions = np.argpartition(-scores, count - 1)[:count]
                positions = positions[np.argsort(-scores[positions])]
                dense_hits = [
                    (local_index.record_at(int(position)).chunk_id, float(scores[position]))
                    for position in positions
                ]
            fused = alpha_fuse(dense_hits, sparse_hits, alpha, top_k)
            result[question.qid] = [by_id[chunk_id] for chunk_id, _score in fused]
    return result


def _make_embedder(config: dict[str, Any]) -> Any:
    from src.chunking_embedding.embedders import create_embedder

    return create_embedder(config["embedder"], dict(config.get("embedder_params") or {}))


def _run_arm(
    arm: str,
    questions: list[Any],
    hits_by_qid: dict[str, list[Any]],
    page_texts: dict[str, str],
    chunk_ranked: dict[str, list[PreparedChunk]] | None,
    args: argparse.Namespace,
    *,
    output_label: str = "discovery",
    timings_by_qid: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path = args.output_dir / f"{args.subset}_{args.language}_{output_label}_{arm}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    rows = {str(row["qid"]): row for row in existing}
    order = [question.qid for question in questions]
    for question in questions:
        if arm == "pages":
            context_ids = [hit.page_id for hit in hits_by_qid[question.qid]]
            context_texts = [page_texts[page_id] for page_id in context_ids if page_id in page_texts]
        else:
            ranked = chunk_ranked.get(question.qid, []) if chunk_ranked else []
            context_ids = [chunk.page_id for chunk in ranked]
            context_texts = [chunk.text for chunk in ranked]
        row = rows.setdefault(question.qid, {})
        row.update({
            "arm": arm,
            "qid": question.qid,
            "query": question.query,
            "gold_answer": question.answer,
            "n_context_pages": len(set(context_ids)),
            "n_context_units": len(context_texts),
            "context_chars": sum(len(text) for text in context_texts),
            "_context": render_documents(context_texts),
        })
        if timings_by_qid and question.qid in timings_by_qid:
            timing = dict(timings_by_qid[question.qid])
            row["online_timing"] = timing
            row["online_latency_seconds"] = timing.get(
                "wall_clock_seconds", timing.get("overall_seconds")
            )

    generation_started = time.perf_counter()
    pending = [row for row in rows.values() if not row.get("answer")]
    if pending:
        print(f"{arm}: generating {len(pending)} answers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_generate, row["query"], row["_context"], args.generator): row["qid"]
                for row in pending
            }
            for future in as_completed(futures):
                qid = futures[future]
                answer, error = future.result()
                rows[qid]["answer"] = answer
                rows[qid]["generation_error"] = error
                _save_rows(path, rows, order)
    generation_seconds = time.perf_counter() - generation_started

    judge_started = time.perf_counter()
    if not args.skip_judge:
        pending = [row for row in rows.values() if row.get("answer") and not row.get("judgment")]
        if pending:
            print(f"{arm}: judging {len(pending)} answers", flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_judge, row, args): row["qid"] for row in pending}
                for future in as_completed(futures):
                    qid = futures[future]
                    verdict = future.result()
                    rows[qid]["judgment"] = verdict.judgment
                    rows[qid]["judge_error"] = verdict.error
                    _save_rows(path, rows, order)
    judge_seconds = time.perf_counter() - judge_started

    final = []
    for qid in order:
        row = dict(rows[qid])
        row.pop("_context", None)
        final.append(row)
    _write_json(path, final)
    if not args.skip_judge:
        verdicts = [ViDoreVerdict(row["qid"], row.get("judgment", "Incorrect")) for row in final]
        scored = score(verdicts)
        summary = {
            "arm": arm,
            "n": scored["n"],
            "correct_only": round(100 * scored["correct_only"], 2),
            "correct_plus_partial": round(100 * scored["correct_plus_partial"], 2),
            "context_pages": round(sum(row["n_context_pages"] for row in final) / len(final), 2),
            "context_units": round(sum(row["n_context_units"] for row in final) / len(final), 2),
            "context_chars": round(sum(row["context_chars"] for row in final) / len(final)),
            "generation_seconds": round(generation_seconds, 3),
            "judge_seconds": round(judge_seconds, 3),
        }
        _write_json(
            path.with_suffix(".summary.json"),
            summary,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary
    return {
        "arm": arm,
        "n": len(final),
        "generation_seconds": round(generation_seconds, 3),
        "judge_seconds": round(judge_seconds, 3),
    }


def _resume_arm_from_checkpoint(
    arm: str,
    questions: list[Any],
    args: argparse.Namespace,
) -> None:
    """Resume generation and judging from an arm checkpoint.

    This deliberately skips discovery and ingestion.  The arm checkpoint
    already contains the rendered context for every question, so changing
    ``--workers`` does not require re-running the expensive parser phase.
    """
    path = args.output_dir / f"{args.subset}_{args.language}_discovery_{arm}.json"
    if not path.exists():
        raise FileNotFoundError(f"Arm checkpoint not found: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    rows = {str(row["qid"]): row for row in existing}
    order = [question.qid for question in questions]
    missing = [qid for qid in order if qid not in rows]
    if missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} question(s); cannot resume safely"
        )

    generation_started = time.perf_counter()
    pending = [row for row in rows.values() if not row.get("answer")]
    if pending:
        print(
            f"{arm}: resuming generation for {len(pending)} answers "
            f"with workers={args.workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _generate, row["query"], row["_context"], args.generator
                ): row["qid"]
                for row in pending
            }
            for future in as_completed(futures):
                qid = futures[future]
                answer, error = future.result()
                rows[qid]["answer"] = answer
                rows[qid]["generation_error"] = error
                _save_rows(path, rows, order)
    generation_seconds = time.perf_counter() - generation_started

    judge_started = time.perf_counter()
    if not args.skip_judge:
        pending = [
            row
            for row in rows.values()
            if row.get("answer") and not row.get("judgment")
        ]
        if pending:
            print(
                f"{arm}: resuming judging for {len(pending)} answers "
                f"with workers={args.workers}",
                flush=True,
            )
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(_judge, row, args): row["qid"] for row in pending
                }
                for future in as_completed(futures):
                    qid = futures[future]
                    verdict = future.result()
                    rows[qid]["judgment"] = verdict.judgment
                    rows[qid]["judge_error"] = verdict.error
                    _save_rows(path, rows, order)
    judge_seconds = time.perf_counter() - judge_started

    final = []
    for qid in order:
        row = dict(rows[qid])
        row.pop("_context", None)
        final.append(row)
    _write_json(path, final)

    if not args.skip_judge:
        verdicts = [
            ViDoreVerdict(row["qid"], row.get("judgment", "Incorrect"))
            for row in final
        ]
        scored = score(verdicts)
        summary = {
            "arm": arm,
            "n": scored["n"],
            "correct_only": round(100 * scored["correct_only"], 2),
            "correct_plus_partial": round(100 * scored["correct_plus_partial"], 2),
            "context_pages": round(
                sum(row["n_context_pages"] for row in final) / len(final), 2
            ),
            "context_units": round(
                sum(row["n_context_units"] for row in final) / len(final), 2
            ),
            "context_chars": round(
                sum(row["context_chars"] for row in final) / len(final)
            ),
            "generation_seconds": round(generation_seconds, 3),
            "judge_seconds": round(judge_seconds, 3),
        }
        _write_json(
            args.output_dir
            / f"{args.subset}_{args.language}_discovery_{arm}.summary.json",
            summary,
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def _generate(query: str, context: str, model: str) -> tuple[str, str | None]:
    try:
        answer = complete(
            model,
            ANSWER_PROMPT.format(documents=context, query=query),
            temperature=0.0,
            max_output_tokens=512,
        ).strip()
        return answer, None
    except Exception as exc:  # noqa: BLE001 - persisted for resume
        return "", f"{type(exc).__name__}: {exc}"


def _judge(row: dict[str, Any], args: argparse.Namespace) -> ViDoreVerdict:
    try:
        return judge_answer(
            row["qid"], row["query"], row["gold_answer"], row["answer"],
            model=args.judge, generator_model=args.generator,
        )
    except Exception as exc:  # noqa: BLE001 - persisted for resume
        return ViDoreVerdict(row["qid"], "Incorrect", error=f"{type(exc).__name__}: {exc}")


def _save_rows(path: Path, rows: dict[str, dict[str, Any]], order: list[str]) -> None:
    _write_json(path, [rows[qid] for qid in order])


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            # Windows Defender/indexers can briefly hold the checkpoint after
            # it is written.  Keep the run resumable instead of losing QA
            # progress on a transient replace failure.
            last_error = exc
            time.sleep(0.25 * (attempt + 1))
    if last_error is not None:
        raise last_error


def _require_env(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"{name} is required; set it in .env")


def _arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        default="physics",
        choices=[
            "hr",
            "energy",
            "computer_science",
            "physics",
            "finance_en",
            "finance_fr",
            "industrial",
            "pharmaceuticals",
        ],
    )
    parser.add_argument("--language", default="french", choices=[
        "english", "french", "spanish", "italian", "german", "portuguese",
    ])
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "data/benchmark/vidore_v3")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "data/work/vidore_v3/physics/discovery_bm25")
    parser.add_argument(
        "--parser-config",
        type=Path,
        default=ROOT / "configs/pipeline.data-discovery.yaml",
    )
    parser.add_argument("--chunking-config", type=Path, default=ROOT / "configs/pipeline.yaml")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/work/vidore_v3/physics/discovery_e2e")
    parser.add_argument(
        "--reuse-parser-artifacts",
        type=Path,
        help="Reuse persisted parser result.json files instead of calling the parser",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/benchmark/vidore_v3/results/physics_discovery_e2e")
    parser.add_argument("--top-k-pages", type=int, default=10)
    parser.add_argument("--top-k-chunks", type=int, default=10)
    parser.add_argument("--depth", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--on-demand-per-query",
        action="store_true",
        help=(
            "Run BM25 page discovery and KDL->chunk/embed->hybrid retrieval "
            "per query with persistent caches"
        ),
    )
    parser.add_argument(
        "--query-workers",
        type=int,
        default=4,
        help="Concurrent independent query workers in --on-demand-per-query mode",
    )
    parser.add_argument(
        "--kdl-microbatch-window-seconds",
        type=float,
        default=0.30,
        help="Time window for coalescing missing-page requests across queries",
    )
    parser.add_argument(
        "--kdl-microbatch-max-pages",
        type=int,
        default=32,
        help="Maximum unique pages in one cross-query KDL micro-batch",
    )
    parser.add_argument(
        "--on-demand-cache-dir",
        type=Path,
        help="Persistent per-query parser/chunk cache (defaults below --work-dir)",
    )
    parser.add_argument(
        "--force-reparse-on-demand",
        action="store_true",
        help="Ignore the per-query parser/chunk cache in online mode",
    )
    parser.add_argument("--kdl-max-workers", type=int, default=8)
    parser.add_argument("--kdl-render-processes", type=int, default=8)
    parser.add_argument("--kdl-bbox-max-workers", type=int, default=8)
    parser.add_argument("--kdl-request-workers", type=int, default=24)
    parser.add_argument("--kdl-request-batch-size", type=int, default=8)
    parser.add_argument("--kdl-max-model-sequences", type=int, default=128)
    parser.add_argument("--retrieval-batch-size", type=int, default=64)
    parser.add_argument(
        "--arms",
        default="pages,chunks",
        help="Comma-separated arms: pages, chunks, or both",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--resume-arm",
        choices=["pages", "chunks"],
        help="Resume an existing arm checkpoint without re-running ingestion",
    )
    parser.add_argument("--generator", default="deepseek/deepseek-v4-flash")
    parser.add_argument("--judge", default="openai/gpt-4o")
    parser.add_argument("--embedder", default="", help="Override chunking config embedder")
    parser.add_argument("--embedder-model", default="", help="Override embedder model")
    parser.add_argument("--skip-judge", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
