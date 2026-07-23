"""Stage orchestrator: parsed documents -> chunks -> vectors -> records.

    python -m src.chunking_embedding.runner --config configs/pipeline.yaml \
        [--input DIR --output DIR --chunker NAME --embedder NAME --force]

Reads ``<input>/parsed_data.json`` and writes ``chunk_records.json``,
``vector_records.json`` and ``embedding_report.json``. One bad document never
crashes the batch (it lands in the report's ``skipped_docs``); re-runs skip any
document whose (doc_id, config_hash) already appears in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import argparse
import json
import logging
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .contracts import ChunkEmbedConfig  
from .embedders import create_embedder  
from .fields import FieldContext, Resources, route_document  
from .llm import OpenRouterLLM  
from .registry import chunker_needs  
from .text import SENTENCE_END_CHARS  
from src.utils.config import load_config, resolve_project_path  
from src.utils.env import load_dotenv_file  

logger = logging.getLogger(__name__)

CONFIG_SECTION = "chunking_embedding"
PARSED_DATA_FILENAME = "parsed_data.json"
CHUNK_RECORDS_FILENAME = "chunk_records.json"
VECTOR_RECORDS_FILENAME = "vector_records.json"
REPORT_FILENAME = "embedding_report.json"

RETRIEVAL_PROFILES: dict[str, dict[str, Any]] = {
    "hybrid_default": {
        "name": "hybrid_default",
        "boost_fields": {"title": 2.0, "main_text": 1.0, "tables": 1.0, "figures": 0.7},
        "hybrid_weights": {"dense": 0.6, "sparse": 0.4},
        "rrf_k": 60,
        "suggested_top_k": 10,
        "notes": "Dense+BM25 reciprocal-rank fusion; boost title matches, downweight figure chunks.",
    },
    "dense_only": {
        "name": "dense_only",
        "boost_fields": {"main_text": 1.0, "tables": 1.0, "figures": 1.0},
        "hybrid_weights": {"dense": 1.0, "sparse": 0.0},
        "rrf_k": None,
        "suggested_top_k": 10,
        "notes": "Pure vector search; use when no sparse index is available.",
    },
}


@dataclass
class ChunkEmbedResult:
    chunk_records: list[dict[str, Any]] = field(default_factory=list)
    vector_records: list[dict[str, Any]] = field(default_factory=list)
    skipped_docs: list[dict[str, Any]] = field(default_factory=list)
    embedded_doc_ids: list[str] = field(default_factory=list)
    llm_stats: dict[str, Any] = field(default_factory=dict)
    chunk_time: float = 0.0
    embed_time: float = 0.0


def chunk_and_embed(
    parsed_records: list[dict[str, Any]],
    config: ChunkEmbedConfig,
    embedder: Any | None = None,
    skip_keys: set[tuple[str, str]] | None = None,
) -> ChunkEmbedResult:
    """Chunk and embed parsed records; skip any (doc_id, config_hash) in skip_keys."""
    embedder = embedder or create_embedder(config.embedder, config.embedder_params)
    llm_client = OpenRouterLLM(**config.llm) if "llm" in chunker_needs(config.chunker) else None
    resources = Resources(embedder=embedder, llm=llm_client)
    config_hash = config.config_hash()
    skip_keys = skip_keys or set()
    result = ChunkEmbedResult()

    for record in parsed_records:
        doc_id = str(record.get("object_id") or "")
        source_uri = record.get("source_uri")
        try:
            if not doc_id:
                raise ValueError("record has no object_id")
            if (doc_id, config_hash) in skip_keys:
                continue
            extraction = _extraction_of(record)
            ctx = FieldContext(
                doc_id=doc_id,
                chunker_name=config.chunker,
                chunker_params=config.chunker_params,
                max_rows_per_chunk=config.max_rows_per_chunk,
                language=extraction.get("language"),
            )
            started = time.perf_counter()
            chunks = route_document(extraction, ctx, resources)
            result.chunk_time += time.perf_counter() - started

            started = time.perf_counter()
            vectors = embedder.embed([c.embedding_text() for c in chunks]) if chunks else []
            result.embed_time += time.perf_counter() - started
            if len(vectors) != len(chunks):
                raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

            doc_metadata = {
                "source_uri": source_uri,
                "title": extraction.get("title"),
                "document_type": extraction.get("document_type"),
            }
            for chunk, vector in zip(chunks, vectors):
                chunk_record = chunk.to_record(config_hash)
                chunk_record["metadata"].update(doc_metadata)
                result.chunk_records.append(chunk_record)
                result.vector_records.append(
                    {
                        "id": chunk.chunk_id,
                        "vector": [round(float(x), 6) for x in vector],
                        "model": embedder.model,
                        "dim": len(vector),
                    }
                )
            result.embedded_doc_ids.append(doc_id)
        except Exception as exc:  # one bad document must never crash the batch
            logger.exception("Chunk/embed failed for doc %s (%s)", doc_id or "?", source_uri)
            result.skipped_docs.append(
                {"doc_id": doc_id or None, "source_uri": source_uri, "reason": f"{type(exc).__name__}: {exc}"}
            )
    result.llm_stats = dict(llm_client.stats) if llm_client else {}
    return result


def run(config: dict[str, Any], input_dir: Path, output_dir: Path, force: bool = False) -> dict[str, Any]:
    """Load -> route -> chunk -> embed -> write; return the report."""
    timings: dict[str, float] = {}
    ce_config = ChunkEmbedConfig.from_mapping(config.get(CONFIG_SECTION))
    config_hash = ce_config.config_hash()

    started = time.perf_counter()
    parsed_path = input_dir / PARSED_DATA_FILENAME
    parsed_records = _load_json(parsed_path, None)
    if not isinstance(parsed_records, list):
        raise FileNotFoundError(f"No parsed data list at {parsed_path}")

    existing_chunks = [] if force else _load_json(output_dir / CHUNK_RECORDS_FILENAME, [])
    existing_vectors = [] if force else _load_json(output_dir / VECTOR_RECORDS_FILENAME, [])
    existing_keys = {
        (record["doc_id"], record["metadata"].get("config_hash"))
        for record in existing_chunks
    }
    timings["load"] = time.perf_counter() - started

    embedder = create_embedder(ce_config.embedder, ce_config.embedder_params)
    result = chunk_and_embed(parsed_records, ce_config, embedder=embedder, skip_keys=existing_keys)
    timings["chunk"] = result.chunk_time
    timings["embed"] = result.embed_time

    started = time.perf_counter()
    all_chunk_records = existing_chunks + result.chunk_records
    all_vector_records = existing_vectors + result.vector_records
    skipped_existing = sum(
        1
        for record in parsed_records
        if (str(record.get("object_id") or ""), config_hash) in existing_keys
    )
    report = {
        "contract_version": "chunk-embed-report-v1",
        "config_hash": config_hash,
        "config": ce_config.resolved(),
        "docs": {
            "total": len(parsed_records),
            "indexed": len(result.embedded_doc_ids),
            "skipped_existing": skipped_existing,
            "skipped_failed": len(result.skipped_docs),
        },
        **_chunk_quality_stats(all_chunk_records),
        "token_spend": {embedder.model: embedder.stats["tokens"]},
        "embedder_stats": dict(embedder.stats),
        "llm_stats": result.llm_stats,
        "skipped_docs": result.skipped_docs,
        "retrieval_profile": _retrieval_profile(ce_config, embedder),
    }
    _write_json(output_dir / CHUNK_RECORDS_FILENAME, all_chunk_records)
    _write_json(output_dir / VECTOR_RECORDS_FILENAME, all_vector_records)
    timings["write"] = time.perf_counter() - started
    report["wall_time_seconds"] = {stage: round(seconds, 3) for stage, seconds in timings.items()}
    _write_json(output_dir / REPORT_FILENAME, report)
    return report


def _extraction_of(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("record has no rows")
    extraction = rows[0].get("extraction") if isinstance(rows[0], dict) else None
    if not isinstance(extraction, dict):
        raise ValueError("record has no rows[0].extraction")
    return extraction


def _chunk_quality_stats(chunk_records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    sizes: list[int] = []
    mid_sentence = 0
    text_chunks = 0
    for record in chunk_records:
        counts[record["chunk_type"]] = counts.get(record["chunk_type"], 0) + 1
        sizes.append(len(record["content"]))
        if record["chunk_type"] == "text_chunk":
            text_chunks += 1
            content = record["content"].rstrip()
            field_chars = record["metadata"].get("field_chars")
            ends_mid_field = field_chars is None or record["char_end"] < field_chars
            if content and content[-1] not in SENTENCE_END_CHARS and ends_mid_field:
                mid_sentence += 1
    return {
        "chunk_counts": counts,
        "chars_mean": _mean(sizes),
        "chars_p90": _p90(sizes),
        "mid_sentence_end_rate": round(mid_sentence / text_chunks, 4) if text_chunks else 0.0,
    }


def _retrieval_profile(config: ChunkEmbedConfig, embedder: Any) -> dict[str, Any]:
    profile = dict(RETRIEVAL_PROFILES.get(config.retrieval_profile, RETRIEVAL_PROFILES["hybrid_default"]))
    profile.update({"embedder": embedder.name, "model": embedder.model, "dim": embedder.dim})
    return profile


def _mean(values: list[int | float]) -> float:
    return round(sum(values) / len(values), 1) if values else 0.0


def _p90(values: list[int | float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(0.9 * (len(ordered) - 1) + 0.5))])


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    load_dotenv_file(PROJECT_ROOT)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run AXIOM chunking + embedding.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--input", default=None, help="Dir containing parsed_data.json (default: ingested_dir)")
    parser.add_argument("--output", default=None, help="Output dir for records (default: output_dir)")
    parser.add_argument("--chunker", default=None, help=f"Override {CONFIG_SECTION}.chunker for this run")
    parser.add_argument("--embedder", default=None, help=f"Override {CONFIG_SECTION}.embedder for this run")
    parser.add_argument("--force", action="store_true", help="Re-run even if (doc, config) already in output")
    args = parser.parse_args()

    config = load_config(_resolve_config_path(args.config))
    section = dict(config.get(CONFIG_SECTION) or {})
    if args.chunker:
        section["chunker"] = args.chunker
    if args.embedder:
        section["embedder"] = args.embedder
    config[CONFIG_SECTION] = section

    input_dir = resolve_project_path(PROJECT_ROOT, args.input or config.get("ingested_dir", "data/ingested"))
    output_dir = resolve_project_path(PROJECT_ROOT, args.output or config.get("output_dir", "data/output"))

    report = run(config, input_dir, output_dir, force=args.force)

    docs = report["docs"]
    print(f"Processed {docs['indexed']}/{docs['total']} document(s) "
          f"({docs['skipped_existing']} already done, {docs['skipped_failed']} failed).")
    print(f"Chunks: {report['chunk_counts']}  chars_mean={report['chars_mean']}  "
          f"mid_sentence_end_rate={report['mid_sentence_end_rate']}")
    print(f"Artifacts: {output_dir}")
    for name in (CHUNK_RECORDS_FILENAME, VECTOR_RECORDS_FILENAME, REPORT_FILENAME):
        print(f"- {name}")
    for skipped in report["skipped_docs"]:
        print(f"skipped {skipped['doc_id'] or skipped['source_uri']}: {skipped['reason']}")


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return PROJECT_ROOT / path


if __name__ == "__main__":
    main()
