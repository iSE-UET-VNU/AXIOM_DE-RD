"""Report Physics light-preparation and legacy downstream timing.

This report deliberately separates measurements from estimates:

* BM25 timings are measured locally over the cached PDF-inspector page text.
* V-SPLADE page preparation/query timings are read from its persisted metrics,
  while cached sparse-matrix scoring is measured locally.
* KDL + pdf-inspector parser timing is read from the parser telemetry embedded
  in the cached run.
* The legacy chunk/embed stage did not persist a stage-level wall clock for
  this Physics run.  Its cache reuse cost is measured where possible and its
  page-scaled downstream estimate is labelled as an estimate.

No parser, embedding API, model inference, rendering, or network call is made.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.data_discovery.hierarchical import HierarchyCorpus  # noqa: E402
from research.experiments.evaluate_physics_legacy_second_retrieval import (  # noqa: E402
    _load_cached_query_vectors,
    _load_legacy_chunks,
)
from research.experiments.physics_vsplade_bm25_fusion import (  # noqa: E402
    _load_csr,
    _load_page_texts,
)
from src.chunking_embedding.embedders import sanitize_text  # noqa: E402
from src.evaluation.benchmarks.vidore_v3 import ViDoreV3  # noqa: E402
from src.retrieval.sparse import BM25Index  # noqa: E402


PARSED_RUN = ROOT / "data/output/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
INGESTED_RUN = ROOT / "data/ingested/vidore-v3-physics-kdl-pdf-inspector/32c32a45a92c45bb"
PAGE_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_48q"
FRENCH_QUERY_VECTOR_DIR = ROOT / "data/output/vsplade/vidore_v3_physics_french_302q"
LEGACY_QUERY_CACHE = ROOT / "data/work/vidore_physics_emb"
EMBEDDING_CACHE = ROOT / "data/work/embedding_cache/text-embedding-3-small"
LEGACY_REPORT = ROOT / "data/benchmark/vidore_v3/results/physics_legacy_second_retrieval/report.json"
LEGACY_TRACES = ROOT / "data/benchmark/vidore_v3/results/physics_legacy_second_retrieval/stage_traces.jsonl"
E2E_RETRIEVAL = ROOT / "data/benchmark/vidore_v3/results/physics_e2e/retrieved_kdl_pdf_inspector.retrieval.json"
DISCOVERY_FRESH_MANIFEST = ROOT / "data/benchmark/vidore_v3/results/physics_discovery_e2e_pages_global_two_phase_batch8_workers24_full/manifest.json"
VSPLADE_PAGE_METRICS = ROOT / "data/output/vsplade/vidore_v3_physics_48q/metrics.json"
VSPLADE_FRENCH_METRICS = ROOT / "data/output/vsplade/vidore_v3_physics_french_302q/metrics.json"
OUTPUT_DIR = ROOT / "data/benchmark/vidore_v3/results/physics_pipeline_timing"
MODEL = "openai/text-embedding-3-small"
PAGE_DEPTH = 100
PAGE_COUNTS = (1, 2, 3, 5, 10, 20, 50, 100)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso_seconds(start: str, end: str) -> float | None:
    try:
        return (
            datetime.fromisoformat(end).timestamp()
            - datetime.fromisoformat(start).timestamp()
        )
    except (TypeError, ValueError):
        return None


def _page_units() -> list[str]:
    metadata = _json(PAGE_VECTOR_DIR / "page_metadata.json")
    return [str(row["unit_id"]) for row in metadata]


def _measure_light(questions: list[Any], page_units: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    text_by_unit = _load_page_texts(PARSED_RUN)
    load_text_seconds = time.perf_counter() - started
    page_texts = [text_by_unit.get(unit, "") for unit in page_units]

    started = time.perf_counter()
    bm25 = BM25Index(analyzer_name="plain").build(
        [
            {"chunk_id": unit, "doc_id": unit, "text": text}
            for unit, text in zip(page_units, page_texts)
        ]
    )
    bm25_index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    for question in questions:
        bm25.search(question.query, PAGE_DEPTH)
    bm25_retrieval_seconds = time.perf_counter() - started

    started = time.perf_counter()
    page_matrix = _load_csr(PAGE_VECTOR_DIR / "page_vectors.npz")
    query_matrix = _load_csr(FRENCH_QUERY_VECTOR_DIR / "query_vectors.npz")
    vsplade_load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    scores = query_matrix @ page_matrix.T
    # Include top-k extraction: this is the operation needed by retrieval,
    # not just the sparse matrix multiplication.
    for row in range(scores.shape[0]):
        values = np.asarray(scores.getrow(row).toarray()).ravel()
        count = min(PAGE_DEPTH, values.size)
        positions = np.argpartition(-values, count - 1)[:count]
        positions[np.argsort(-values[positions])]
    vsplade_retrieval_seconds = time.perf_counter() - started

    page_metrics = _json(VSPLADE_PAGE_METRICS)["timing_seconds"]
    french_metrics = _json(VSPLADE_FRENCH_METRICS)["timing_seconds"]
    return {
        "corpus_pages": len(page_units),
        "queries": len(questions),
        "parser_text_pages": sum(bool(text.strip()) for text in page_texts),
        "measured_local": {
            "load_cached_pdf_inspector_page_text_seconds": round(load_text_seconds, 6),
            "bm25_page_index_build_seconds": round(bm25_index_seconds, 6),
            "bm25_retrieval_302q_depth100_seconds": round(bm25_retrieval_seconds, 6),
            "vsplade_sparse_index_load_seconds": round(vsplade_load_seconds, 6),
            "vsplade_retrieval_302q_depth100_seconds": round(vsplade_retrieval_seconds, 6),
        },
        "persisted_vsplade": {
            "page_render_encode_1674_pages_seconds": page_metrics.get("render_encode_pipeline_seconds"),
            "page_render_wall_seconds": page_metrics.get("render_wall_seconds"),
            "page_encode_gpu_sum_seconds": page_metrics.get("encode_gpu_seconds_sum"),
            "page_model_load_seconds": page_metrics.get("load_model"),
            "french_query_encode_302q_seconds": french_metrics.get("encode_queries"),
            "french_query_score_302q_seconds": french_metrics.get("score_all_queries"),
            "french_model_load_seconds": french_metrics.get("load_model"),
            "french_total_including_evaluation_seconds": french_metrics.get("total"),
            "english_query_encode_302q_seconds": _json(
                ROOT / "data/output/vsplade/vidore_v3_physics_english_302q/metrics.json"
            )["timing_seconds"].get("encode_queries"),
            "english_query_score_302q_seconds": _json(
                ROOT / "data/output/vsplade/vidore_v3_physics_english_302q/metrics.json"
            )["timing_seconds"].get("score_all_queries"),
        },
        "interpretation": {
            "bm25_text_source": "KDL + pdf-inspector cached page text; BM25 is not applied to raw PDF bytes",
            "vsplade_indexing": "one-time rendered-page encoding; sparse matrix load is the online index-open cost",
            "vsplade_query_language": "French cache shown here; the main hierarchical experiment used English V-SPLADE queries and must retain that caveat",
        },
    }


def _parser_telemetry() -> dict[str, Any]:
    result_paths = sorted((INGESTED_RUN / "assets").glob("*/result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No parser result files under {INGESTED_RUN}")
    rows = [_json(path) for path in result_paths]
    scheduler = rows[0]["_global_scheduler"]
    classification_sum = sum(float(row["_routing"].get("classification_latency_ms", 0.0)) for row in rows)
    region_sum = sum(float(row["_routing"].get("region_extraction_latency_ms", 0.0)) for row in rows)
    discovery_selected = None
    if DISCOVERY_FRESH_MANIFEST.is_file():
        manifest = _json(DISCOVERY_FRESH_MANIFEST)
        discovery_selected = {
            "selected_unique_pages": manifest.get("selected_unique_pages"),
            "ingested_pages": manifest.get("ingested_pages"),
            "ingestion_cleaning_enrichment_seconds": (manifest.get("timing_seconds") or {}).get("ingestion_cleaning_enrichment"),
            "discovery_retrieval_seconds": (manifest.get("timing_seconds") or {}).get("discovery_retrieval"),
            "parser_artifacts_reused": manifest.get("parser_artifacts_reused", False),
            "meaning": "Observed 302-query top-10 discovery run; ingestion_cleaning_enrichment includes more than parser-only time.",
        }
    return {
        "documents": len(rows),
        "pages": int(scheduler["layout_rendered_pages"]),
        "persisted_full_run_seconds": float(scheduler["end_to_end_latency_ms"]) / 1000.0,
        "persisted_phases_seconds": {
            "layout": float(scheduler["layout_phase_latency_ms"]) / 1000.0,
            "routing": float(scheduler["routing_phase_latency_ms"]) / 1000.0,
            "recognition": float(scheduler["recognition_phase_latency_ms"]) / 1000.0,
            "post_recognition_persistence_drain": float(scheduler.get("post_recognition_persistence_drain_ms", 0.0)) / 1000.0,
        },
        "pdf_inspector_instrumented_component_sum_seconds": {
            "classification": classification_sum / 1000.0,
            "region_extraction": region_sum / 1000.0,
            "classification_plus_region_extraction": (classification_sum + region_sum) / 1000.0,
        },
        "observed_selected_page_discovery_run": discovery_selected,
        "caveat": "The full-run scheduler telemetry is KDL + pdf-inspector, not a standalone light-parser wall clock. The component sum is concurrent request/service time, not wall time.",
    }


def _legacy_cache_and_retrieval(questions: list[Any], page_units: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = HierarchyCorpus.from_parsed_run(PARSED_RUN, subset="physics", page_ids=page_units)
    chunks, legacy_meta = _load_legacy_chunks(PARSED_RUN, corpus)
    cache_materialization_seconds = time.perf_counter() - started

    started = time.perf_counter()
    chunk_index = BM25Index(analyzer_name="auto").build(
        [
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.file_id, "text": chunk.text}
            for chunk in chunks
        ]
    )
    chunk_index_seconds = time.perf_counter() - started

    started = time.perf_counter()
    query_vectors, query_cache = _load_cached_query_vectors(questions, LEGACY_QUERY_CACHE)
    query_cache_load_seconds = time.perf_counter() - started

    chunk_matrix = np.asarray([chunk.vector for chunk in chunks], dtype=np.float32)
    started = time.perf_counter()
    for question in questions:
        chunk_matrix @ query_vectors[question.qid]
    cached_dense_scoring_seconds = time.perf_counter() - started

    cache_presence = 0
    cache_missing = 0
    for chunk in chunks:
        key = hashlib.sha1(f"{MODEL}|{sanitize_text(chunk.text)}".encode("utf-8", "ignore")).hexdigest()
        if (EMBEDDING_CACHE / f"emb_{key}.json").is_file():
            cache_presence += 1
        else:
            cache_missing += 1

    recorded_second_stage_seconds = None
    if LEGACY_TRACES.is_file():
        method_marker = '"method": "legacy-second-page100-max-gamma0.25"'
        pattern = re.compile(r'"second_stage"\s*:\s*([0-9.]+)')
        total = 0.0
        with LEGACY_TRACES.open("r", encoding="utf-8") as handle:
            for line in handle:
                if method_marker not in line:
                    continue
                match = pattern.search(line)
                if match:
                    total += float(match.group(1))
        recorded_second_stage_seconds = total

    legacy_report = _json(LEGACY_REPORT) if LEGACY_REPORT.is_file() else {}
    e2e_retrieval = _json(E2E_RETRIEVAL) if E2E_RETRIEVAL.is_file() else {}
    output_metadata = _json(PARSED_RUN / "metadata.json")
    ingested_metadata = _json(INGESTED_RUN / "metadata.json")
    artifact_gap = _iso_seconds(ingested_metadata.get("created_at"), output_metadata.get("created_at"))
    return {
        "cache_inventory": {
            **legacy_meta,
            "all_records": 5018,
            "record_types": {"text_chunk": 1945, "table": 1792, "image": 1281},
            "embedding_dimension": 1536,
            "embedding_cache_key_hits_for_loaded_text_chunks": cache_presence,
            "embedding_cache_key_misses_for_loaded_text_chunks": cache_missing,
            "query_embedding_cache": query_cache,
        },
        "measured_local_cache_reuse": {
            "parser_output_to_legacy_chunk_materialization_seconds": round(cache_materialization_seconds, 6),
            "legacy_chunk_bm25_index_build_seconds": round(chunk_index_seconds, 6),
            "cached_query_vector_load_302q_seconds": round(query_cache_load_seconds, 6),
            "cached_dense_chunk_scoring_302q_seconds": round(cached_dense_scoring_seconds, 6),
        },
        "persisted_legacy_runs": {
            "legacy_page_hybrid_retrieval_seconds": e2e_retrieval.get("seconds"),
            "legacy_page_hybrid_embedding_api_calls": (e2e_retrieval.get("embedder_stats") or {}).get("api_calls"),
            "legacy_page_hybrid_embedding_cache_hits": (e2e_retrieval.get("embedder_stats") or {}).get("cache_hits"),
            "legacy_second_stage_page100_max_gamma025_302q_seconds": recorded_second_stage_seconds,
            "legacy_second_stage_total_run_seconds": legacy_report.get("timing_seconds", {}).get("total"),
            "legacy_hierarchy_and_chunk_index_build_seconds": legacy_report.get("index_counts", {}).get("build_seconds"),
        },
        "downstream_artifact_gap_seconds": artifact_gap,
        "caveat": "No stage-level chunk_time/embed_time report was persisted for this 42-file Physics output. Cache materialization and BM25/index timings are measured locally; the metadata creation gap includes downstream work beyond chunking/embedding and must not be treated as an isolated stage time.",
    }


def _scaled_estimates(parser: dict[str, Any], legacy: dict[str, Any]) -> list[dict[str, Any]]:
    full_pages = float(parser["pages"])
    full_parse = float(parser["persisted_full_run_seconds"])
    downstream = legacy.get("downstream_artifact_gap_seconds")
    records_per_page = 5018 / full_pages
    text_chunks_per_page = 1945 / full_pages
    table_per_page = 1792 / full_pages
    image_per_page = 1281 / full_pages
    rows: list[dict[str, Any]] = []
    for pages in PAGE_COUNTS:
        ratio = pages / full_pages
        row: dict[str, Any] = {
            "selected_pages": pages,
            "estimated_parser_seconds": round(full_parse * ratio, 3),
            "estimated_records": round(records_per_page * pages),
            "estimated_text_chunks": round(text_chunks_per_page * pages),
            "estimated_table_records": round(table_per_page * pages),
            "estimated_image_records": round(image_per_page * pages),
        }
        if downstream is not None:
            row["estimated_post_ingest_artifact_gap_seconds_not_isolated"] = round(float(downstream) * ratio, 3)
        rows.append(row)
    return rows


def _markdown(report: dict[str, Any]) -> str:
    light = report["light_preparation"]
    parser = report["legacy_downstream"]["parser_telemetry"]
    legacy = report["legacy_downstream"]["cache_and_retrieval"]
    lines = [
        "# Physics pipeline timing report",
        "",
        "All timings are for the 1,674-page / 302-query Physics protocol. Measurements and estimates are kept separate.",
        "",
        "## Light preparation and retrieval",
        "",
        "| Stage | Time | Meaning |",
        "|---|---:|---|",
        f"| Load cached PDF-inspector page text | {light['measured_local']['load_cached_pdf_inspector_page_text_seconds']:.3f}s | Local cache read; not a fresh parse |",
        f"| BM25 page index build | {light['measured_local']['bm25_page_index_build_seconds']:.3f}s | BM25 over PDF-inspector extracted page text |",
        f"| BM25 retrieval, 302 queries, depth 100 | {light['measured_local']['bm25_retrieval_302q_depth100_seconds']:.3f}s | Query scoring only |",
        f"| V-SPLADE sparse index load | {light['measured_local']['vsplade_sparse_index_load_seconds']:.3f}s | Open cached page/query sparse matrices |",
        f"| V-SPLADE retrieval, 302 queries, depth 100 | {light['measured_local']['vsplade_retrieval_302q_depth100_seconds']:.3f}s | Sparse score + top-k extraction |",
        f"| V-SPLADE one-time page render + encode | {light['persisted_vsplade']['page_render_encode_1674_pages_seconds']:.3f}s | Persisted page-vector preparation |",
        f"| V-SPLADE French query encode | {light['persisted_vsplade']['french_query_encode_302q_seconds']:.3f}s | Persisted 302-query run, excluding model load |",
        f"| V-SPLADE French query score | {light['persisted_vsplade']['french_query_score_302q_seconds']:.3f}s | Persisted 302-query run |",
        "",
        "## Legacy downstream cache and retrieval",
        "",
        "| Stage | Time | Status |",
        "|---|---:|---|",
        f"| KDL + pdf-inspector full parser run | {parser['persisted_full_run_seconds']:.3f}s | Direct parser telemetry, 1,674 pages |",
        f"| Observed selected-page ingestion + cleaning + enrichment | {(parser.get('observed_selected_page_discovery_run') or {}).get('ingestion_cleaning_enrichment_seconds', 0):.3f}s | 1,016 selected pages; not parser-only |",
        f"| pdf-inspector routing component sum | {parser['pdf_inspector_instrumented_component_sum_seconds']['classification_plus_region_extraction']:.3f}s | Instrumented service-time sum, not wall time |",
        f"| Legacy chunk/vector cache materialization | {legacy['measured_local_cache_reuse']['parser_output_to_legacy_chunk_materialization_seconds']:.3f}s | Measured local cache read/mapping |",
        f"| Legacy chunk BM25 index build | {legacy['measured_local_cache_reuse']['legacy_chunk_bm25_index_build_seconds']:.3f}s | Measured over cached text chunks |",
        f"| Legacy cached query vector load | {legacy['measured_local_cache_reuse']['cached_query_vector_load_302q_seconds']:.3f}s | 302 cache hits |",
        f"| Legacy second retrieval, page100, 302 queries | {legacy['persisted_legacy_runs']['legacy_second_stage_page100_max_gamma025_302q_seconds']:.3f}s | Persisted stage traces; cache-only |",
        f"| Earlier legacy page hybrid retrieval | {legacy['persisted_legacy_runs']['legacy_page_hybrid_retrieval_seconds']:.3f}s | Includes embedding/API work, BM25 and retrieval; not chunking-only |",
        "",
        "## Page-scaled estimate",
        "",
        "Parser estimate = full parser wall time × selected pages / 1,674. Record counts are scaled from 5,018 cached records. The artifact-gap column is only a sensitivity estimate because it also contains cleaning/enrichment/write work.",
        "",
        "| Pages | Parser estimate | Records | Text chunks | Table | Image | Post-ingest artifact-gap sensitivity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["scaled_estimates"]:
        lines.append(
            f"| {row['selected_pages']} | {row['estimated_parser_seconds']:.3f}s | {row['estimated_records']} | {row['estimated_text_chunks']} | {row['estimated_table_records']} | {row['estimated_image_records']} | {row.get('estimated_post_ingest_artifact_gap_seconds_not_isolated', 0):.3f}s |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "- The standalone full-corpus light pdf-inspector wall time was not persisted; the local text-load time is not a parse time.",
        "- The exact chunking and embedding wall times for the 42-file Physics legacy run were not persisted. Existing vectors make network embedding cost zero when the cache is complete; local cache lookup/indexing remains measurable.",
        "- V-SPLADE page preparation is a one-time cost. Runtime retrieval uses the cached sparse page matrix. The current hierarchical experiment used English V-SPLADE queries; the light timing table also reports French query-cache timings for the frozen French protocol.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmark = ViDoreV3(root=ROOT / "data/benchmark/vidore_v3", subset="physics", language="french")
    questions = sorted(list(benchmark.questions()), key=lambda q: int(q.qid.rsplit("::", 1)[1]))
    page_units = _page_units()
    if len(questions) != 302 or len(page_units) != 1674:
        raise RuntimeError(f"Unexpected Physics dimensions: {len(questions)} queries, {len(page_units)} pages")

    parser = _parser_telemetry()
    legacy = _legacy_cache_and_retrieval(questions, page_units)
    report = {
        "dataset": "vidore_v3/physics",
        "protocol": {"queries": len(questions), "pages": len(page_units), "retrieval_depth": PAGE_DEPTH},
        "light_preparation": _measure_light(questions, page_units),
        "legacy_downstream": {
            "parser_telemetry": parser,
            "cache_and_retrieval": legacy,
        },
        "scaled_estimates": _scaled_estimates(parser, legacy),
        "sources": {
            "parsed_run": str(PARSED_RUN),
            "ingested_run": str(INGESTED_RUN),
            "vsplade_page_metrics": str(VSPLADE_PAGE_METRICS),
            "vsplade_french_metrics": str(VSPLADE_FRENCH_METRICS),
            "legacy_report": str(LEGACY_REPORT),
            "legacy_traces": str(LEGACY_TRACES),
            "e2e_retrieval": str(E2E_RETRIEVAL),
            "discovery_fresh_manifest": str(DISCOVERY_FRESH_MANIFEST),
        },
    }
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_DIR),
        "light": report["light_preparation"]["measured_local"],
        "parser_seconds": parser["persisted_full_run_seconds"],
        "legacy": legacy["measured_local_cache_reuse"],
        "legacy_second_stage_seconds": legacy["persisted_legacy_runs"]["legacy_second_stage_page100_max_gamma025_302q_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
