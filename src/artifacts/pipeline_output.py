"""Per-document writers for cleaning, enrichment, embedding, and final output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chunking_embedding.document_view import DocumentView, document_from_enriched_data
from ..chunking_embedding.lexical import build_corpus_statistics
from ..models import DataObject, EnrichedData, ParsedData, PipelineState
from ..utils.paths import portable_path
from .manifests import build_stage_metadata
from .writer import LocalArtifactWriter


RETRIEVAL_ITEM_TYPES = {
    "text_chunk": "text",
    "table": "table",
    "image": "image",
}


@dataclass
class ArtifactOutput:
    artifact_paths: dict[str, str]


def write_cleaned_artifacts(
    state: PipelineState,
    cleaned_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write one cleaning JSON per document plus common metadata."""
    writer = LocalArtifactWriter(cleaned_dir, project_root=project_root)
    data_by_id = {record.source_object_id: record for record in state.cleaned_data}
    schema_by_id = {record.source_object_id: record for record in state.cleaned_schemas}
    paths: dict[str, Path] = {}

    for data_object in state.data_objects:
        data = data_by_id.get(data_object.object_id)
        schema = schema_by_id.get(data_object.object_id)
        paths[f"cleaned_document:{data_object.object_id}"] = writer.write_json(
            f"documents/{data_object.object_id}.json",
            {
                "contract_version": "cleaned-document-v1",
                "stage": "cleaned",
                "document_id": data_object.object_id,
                "data": data,
                "schema_id": schema.schema_id if schema else None,
            },
            sort_keys=False,
        )

    paths["cleaned_metadata"] = writer.write_json(
        "metadata.json",
        build_stage_metadata(
            state,
            stage="cleaned",
            document_summaries=_document_summaries(state),
            schemas=state.cleaned_schemas,
            document_schema={
                "contract_version": "string",
                "stage": "cleaned",
                "document_id": "string",
                "data": "CleanedData|null",
                "schema_id": "string|null",
            },
            extra={"mode": _stage_mode(state.cleaned_data)},
        ),
        sort_keys=False,
    )
    return paths


def write_enriched_artifacts(
    state: PipelineState,
    enriched_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write one enrichment JSON per document plus common metadata."""
    writer = LocalArtifactWriter(enriched_dir, project_root=project_root)
    data_by_id = {record.source_object_id: record for record in state.enriched_data}
    schema_by_id = {record.source_object_id: record for record in state.enriched_schemas}
    paths: dict[str, Path] = {}

    for data_object in state.data_objects:
        data = data_by_id.get(data_object.object_id)
        schema = schema_by_id.get(data_object.object_id)
        paths[f"enriched_document:{data_object.object_id}"] = writer.write_json(
            f"documents/{data_object.object_id}.json",
            {
                "contract_version": "enriched-document-v1",
                "stage": "enriched",
                "document_id": data_object.object_id,
                "data": data,
                "schema_id": schema.schema_id if schema else None,
            },
            sort_keys=False,
        )

    paths["enriched_metadata"] = writer.write_json(
        "metadata.json",
        build_stage_metadata(
            state,
            stage="enriched",
            document_summaries=_document_summaries(state),
            schemas=state.enriched_schemas,
            document_schema={
                "contract_version": "string",
                "stage": "enriched",
                "document_id": "string",
                "data": "EnrichedData|null",
                "schema_id": "string|null",
            },
            extra={"mode": _stage_mode(state.enriched_data)},
        ),
        sort_keys=False,
    )
    return paths


def write_embedded_artifacts(
    state: PipelineState,
    embedded_dir: str | Path,
    project_root: str | Path | None = None,
) -> ArtifactOutput:
    """Write compact per-document retrieval items plus common metadata."""
    writer = LocalArtifactWriter(embedded_dir, project_root=project_root)
    parsed_by_id = {record.object_id: record for record in state.parsed_data}
    paths: dict[str, Path] = {}

    for data_object in state.data_objects:
        document_id = data_object.object_id
        parsed = parsed_by_id.get(document_id)
        paths[f"embedded_document:{document_id}"] = writer.write_json(
            f"documents/{document_id}.json",
            {
                "contract_version": "embedded-document-v3",
                "stage": "embedded",
                "document_id": document_id,
                "retrieval": _compact_retrieval(
                    state,
                    document_id,
                    parsed.metadata.get("image_files", []) if parsed else [],
                ),
            },
            sort_keys=False,
        )

    paths["embedded_metadata"] = writer.write_json(
        "metadata.json",
        build_stage_metadata(
            state,
            stage="embedded",
            document_summaries=_document_summaries(state),
            schemas={
                "retrieval": _compact_retrieval_schema(),
                "source_reference": _source_reference_schema(),
                "retrieval_record_types": sorted(
                    {record.retrieval_type for record in state.retrieval_records}
                ),
                "vector_dimension": state.embedding_report.get("dimension"),
            },
            document_schema={
                "contract_version": "string",
                "stage": "embedded",
                "document_id": "string",
                "retrieval": _compact_retrieval_schema(),
            },
            extra={"summary": _run_summary(state)},
        ),
        sort_keys=False,
    )
    _record_paths(state, paths, project_root)
    return ArtifactOutput(artifact_paths=state.artifact_paths)


def write_output_artifacts(
    state: PipelineState,
    output_dir: str | Path,
    project_root: str | Path | None = None,
) -> ArtifactOutput:
    """Write one consumer-facing JSON per document plus common metadata.

    Keep the run metadata common, while making every document's stage boundary
    explicit.  Each stage retains its native data contract; retrieval keeps the
    existing compact chunk and embedding contract.
    """
    writer = LocalArtifactWriter(output_dir, project_root=project_root)
    parsed_by_id = {record.object_id: record for record in state.parsed_data}
    cleaned_by_id = {
        record.source_object_id: record for record in state.cleaned_data
    }
    enriched_by_id = {
        record.source_object_id: record for record in state.enriched_data
    }
    paths: dict[str, Path] = {}

    for data_object in state.data_objects:
        document_id = data_object.object_id
        cleaned = cleaned_by_id.get(document_id)
        parsed = parsed_by_id.get(document_id)
        enriched = enriched_by_id.get(document_id)
        document = document_from_enriched_data(enriched) if enriched else None
        paths[f"output_document:{document_id}"] = writer.write_json(
            f"documents/{document_id}.json",
            {
                "document": _output_document(data_object, document, enriched),
                "ingest": _stage_data(
                    parsed,
                    source=data_object,
                ),
                "clean": _stage_data(cleaned),
                "enrich": _stage_data(enriched),
                "retrieval": _compact_retrieval(
                    state,
                    document_id,
                    parsed.metadata.get("image_files", []) if parsed else [],
                ),
            },
            sort_keys=False,
        )

    paths["output_metadata"] = writer.write_json(
        "metadata.json",
        build_stage_metadata(
            state,
            stage="output",
            document_summaries=_document_summaries(state),
            schemas={
                "source_reference": _source_reference_schema(),
                "retrieval_record_types": sorted(
                    {record.retrieval_type for record in state.retrieval_records}
                ),
                "vector_dimension": state.embedding_report.get("dimension"),
            },
            document_schema={
                "document": "consumer-facing document identity and source summary",
                "ingest": "source object and parsed data",
                "clean": "cleaned data",
                "enrich": "enriched data",
                "retrieval": _compact_retrieval_schema(),
            },
            extra={
                "summary": _run_summary(state),
                "completed_modules": state.completed_modules,
                "stage_dirs": {
                    "raw": state.raw_dir,
                    "ingested": state.ingested_dir,
                    "cleaned": state.cleaned_dir,
                    "enriched": state.enriched_dir,
                    "embedded": state.embedded_dir,
                    "output": state.output_dir,
                },
            },
        ),
        sort_keys=False,
    )
    _record_paths(state, paths, project_root)
    return ArtifactOutput(artifact_paths=state.artifact_paths)


def _output_document(
    data_object: DataObject,
    document: DocumentView | None,
    enriched: EnrichedData | None,
) -> dict[str, Any]:
    metadata = data_object.metadata
    payload: dict[str, Any] = {
        "document_id": data_object.object_id,
        "source_uri": data_object.uri,
        "file_name": metadata.get("file_name"),
        "content_type": data_object.content_type,
        "size_bytes": metadata.get("size_bytes"),
        "sha256": metadata.get("sha256"),
        "title": document.title if document else None,
        "document_type": document.document_type if document else None,
        "language": document.language if document else None,
    }
    s3 = {
        "bucket": metadata.get("s3_bucket"),
        "key": metadata.get("s3_key"),
        "etag": metadata.get("s3_etag"),
        "version_id": metadata.get("s3_version_id"),
    }
    if any(value is not None for value in s3.values()):
        payload["s3"] = _without_none(s3)
    source_refs = _extraction_source_refs(
        enriched,
        ("document_type", "language", "title"),
    )
    if source_refs:
        payload["source_refs"] = source_refs
    return _without_none(payload)


def _stage_data(
    data: Any,
    *,
    source: DataObject | None = None,
) -> dict[str, Any]:
    """Wrap one native stage result without reshaping its data payload."""
    payload: dict[str, Any] = {"data": data}
    if source is not None:
        payload["source"] = source
    return payload


def _compact_retrieval(
    state: PipelineState,
    document_id: str,
    image_files: Any,
) -> dict[str, Any]:
    retrieval_records = [
        record
        for record in state.retrieval_records
        if record.source_object_id == document_id
        and record.retrieval_type in RETRIEVAL_ITEM_TYPES
    ]
    vectors_by_record: dict[str, list[dict[str, Any]]] = {}
    for vector in state.vector_records:
        if vector.get("source_object_id") != document_id:
            continue
        record_id = str(vector.get("record_id") or "")
        if record_id:
            vectors_by_record.setdefault(record_id, []).append(vector)

    assets = image_files if isinstance(image_files, list) else []
    items = [
        _compact_retrieval_item(
            record,
            vectors_by_record.get(record.record_id, []),
            assets,
        )
        for record in retrieval_records
    ]
    return {
        "items": items,
        "lexical_stats": build_corpus_statistics(
            (
                item.get("lexical")
                for item in items
                if isinstance(item, dict)
            ),
            scope="document",
        ),
    }


def _compact_retrieval_item(
    record: Any,
    vectors: list[dict[str, Any]],
    image_files: list[Any],
) -> dict[str, Any]:
    content = _strip_parser_audit(_retrieval_item_content(record))
    item: dict[str, Any] = {
        "item_id": _retrieval_item_id(record),
        "type": RETRIEVAL_ITEM_TYPES[record.retrieval_type],
        "position": _without_none(_retrieval_item_position(record)),
        "content": content,
        "embeddings": [_compact_embedding(vector) for vector in vectors],
    }
    component_type = record.metadata.get("component_type")
    if component_type:
        item["component_type"] = component_type
    lexical = record.payload.get("lexical")
    if isinstance(lexical, dict):
        item["lexical"] = dict(lexical)
    if record.retrieval_type == "image":
        image_index = record.payload.get("image_index")
        if isinstance(image_index, int) and 0 <= image_index < len(image_files):
            asset = image_files[image_index]
            if isinstance(asset, dict):
                item["asset"] = _without_none(
                    {
                        "name": asset.get("name"),
                        "path": asset.get("path"),
                    }
                )
    return item


def _compact_embedding(vector: dict[str, Any]) -> dict[str, Any]:
    return _without_none(
        {
            "model": vector.get("embedding_model"),
            "dimension": vector.get("embedding_dimension"),
            "values": vector.get("embedding", []),
        }
    )


def _strip_parser_audit(value: Any) -> Any:
    """Remove provider audit fields while retaining normalized source references."""
    if isinstance(value, list):
        return [_strip_parser_audit(item) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    source_refs: list[dict[str, Any]] = []
    for key, item in value.items():
        if key == "source_refs":
            source_refs.extend(_existing_source_refs(item))
            continue
        if key == "citations":
            source_refs.extend(_normalize_source_refs(item))
            continue
        if key.endswith("_citations"):
            source_refs.extend(
                _normalize_source_refs(item, field=key.removesuffix("_citations"))
            )
            continue
        if key in {"reasoning", "verification", "extraction_status"}:
            continue
        if key.endswith("_meta"):
            continue
        result[key] = _strip_parser_audit(item)
    if source_refs:
        result["source_refs"] = _deduplicate_source_refs(source_refs)
    return result


def _extraction_source_refs(
    enriched: EnrichedData | None,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    if enriched is None:
        return []
    source_refs: list[dict[str, Any]] = []
    for row in enriched.rows:
        extraction = row.get("extraction")
        if not isinstance(extraction, dict):
            continue
        for field in fields:
            source_refs.extend(
                _normalize_source_refs(
                    extraction.get(f"{field}_citations"),
                    field=field,
                )
            )
    return _deduplicate_source_refs(source_refs)


def _normalize_source_refs(
    value: Any,
    *,
    field: str | None = None,
) -> list[dict[str, Any]]:
    citations = value if isinstance(value, list) else [value]
    refs: list[dict[str, Any]] = []
    for citation in citations:
        if not isinstance(citation, str) or not citation.strip():
            continue
        component_id = citation.strip()
        ref: dict[str, Any] = {"component_id": component_id}
        page_match = re.match(r"^/page/(\d+)(?:/|$)", component_id)
        if page_match:
            ref["page"] = int(page_match.group(1))
        if field:
            ref["field"] = field
        refs.append(ref)
    return refs


def _existing_source_refs(value: Any) -> list[dict[str, Any]]:
    candidates = value if isinstance(value, list) else [value]
    refs: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not candidate.get("component_id"):
            continue
        refs.append(
            _without_none(
                {
                    "component_id": str(candidate["component_id"]),
                    "page": candidate.get("page"),
                    "field": candidate.get("field"),
                }
            )
        )
    return refs


def _deduplicate_source_refs(
    source_refs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for ref in source_refs:
        identity = (ref.get("component_id"), ref.get("page"), ref.get("field"))
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(ref)
    return unique


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _source_reference_schema() -> dict[str, str]:
    return {
        "component_id": "string; parser component path",
        "page": "integer; page index parsed from component_id when available",
        "field": "string; semantic field supported by the reference",
    }


def _run_summary(state: PipelineState) -> dict[str, Any]:
    chunk_report = state.retrieval_quality_report
    embedding_report = state.embedding_report
    return {
        "chunking_embedding": _without_none(
            {
                "status": chunk_report.get("status"),
                "document_count": chunk_report.get("document_count"),
                "embedded_document_count": chunk_report.get("embedded_document_count"),
                "skipped_document_count": chunk_report.get("skipped_document_count"),
                "record_count": chunk_report.get("record_count"),
                "counts_by_type": chunk_report.get("counts_by_retrieval_type"),
            }
        ),
        "embedding": _without_none(
            {
                "status": embedding_report.get("status"),
                "provider": embedding_report.get("provider"),
                "model": embedding_report.get("model"),
                "dimension": embedding_report.get("dimension"),
                "generated_count": embedding_report.get("generated_count"),
                "skipped_count": embedding_report.get("skipped_count"),
            }
        ),
        "integration": {
            "schema_match_count": len(state.schema_matches),
            "entity_match_count": len(state.entity_matches),
            "relationship_count": len(state.relationship_records),
        },
    }


def _compact_retrieval_schema() -> dict[str, Any]:
    return {
        "items": [
            {
                "item_id": "string; stable component id",
                "type": "text|table|image",
                "component_type": "main_text|formula|table|figure",
                "position": "source position",
                "content": (
                    "consumer content without parser audit fields; may contain "
                    "normalized source_refs"
                ),
                "asset": "object; optional parsed image asset",
                "lexical": {
                    "contract_version": "lexical-v1",
                    "analyzer": "unicode_word_v1",
                    "tf": "mapping of analyzed term to frequency",
                    "dl": "integer; analyzed token length",
                },
                "embeddings": [
                    {
                        "model": "string",
                        "dimension": "integer",
                        "values": "number[]",
                    }
                ],
            }
        ],
        "lexical_stats": {
            "contract_version": "bm25-corpus-v1",
            "scope": "document",
            "analyzer": "unicode_word_v1",
            "N": "integer; number of lexical retrieval items",
            "avgdl": "number; average analyzed item length",
            "df": "mapping of analyzed term to document frequency",
        },
    }


def _retrieval_item_id(record: Any) -> str:
    payload = record.payload
    id_field = {
        "text_chunk": "chunk_id",
        "table": "table_id",
        "image": "image_id",
    }[record.retrieval_type]
    return str(payload.get(id_field) or record.record_id)


def _retrieval_item_position(record: Any) -> dict[str, Any]:
    payload = record.payload
    index_field = {
        "text_chunk": "chunk_index",
        "table": "table_index",
        "image": "image_index",
    }[record.retrieval_type]
    position = {"index": payload.get(index_field)}
    if record.retrieval_type == "text_chunk":
        position.update(
            {
                "start_char": payload.get("start_char"),
                "end_char": payload.get("end_char"),
            }
        )
    # Which chunking/embedding settings produced this item. It rides in position
    # because the downstream retrieval job persists position verbatim and drops unknown
    # item-level keys; there is no column for it yet. Consumers that do not read
    # it are unaffected. See docs/platform_config_hash_proposal.md.
    if payload.get("config_hash"):
        position["config_hash"] = payload["config_hash"]
    return position


def _retrieval_item_content(record: Any) -> dict[str, Any]:
    payload = record.payload
    if record.retrieval_type == "text_chunk":
        return {"text": payload.get("text", "")}
    if record.retrieval_type == "table":
        table = payload.get("table")
        return dict(table) if isinstance(table, dict) else {"content": table}
    image = payload.get("image")
    return dict(image) if isinstance(image, dict) else {"content": image}


def _document_summaries(state: PipelineState) -> list[dict[str, Any]]:
    return [
        {
            "document_id": record.object_id,
            "source_uri": record.uri,
            "content_type": record.content_type,
            "file_name": record.metadata.get("file_name"),
            "size_bytes": record.metadata.get("size_bytes"),
        }
        for record in state.data_objects
    ]


def _stage_mode(records: list[Any]) -> str:
    return (
        "pass-through"
        if records and all(record.metadata.get("pass_through") for record in records)
        else "implemented"
    )


def _record_paths(
    state: PipelineState,
    paths: dict[str, Path],
    project_root: str | Path | None,
) -> None:
    state.artifact_paths.update(
        {name: portable_path(path, project_root) for name, path in paths.items()}
    )
