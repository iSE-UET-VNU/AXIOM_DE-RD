"""Per-document writers for cleaning, enrichment, embedding, and final output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import PipelineState
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
    """Write each document's indexes and embeddings plus common metadata."""
    writer = LocalArtifactWriter(embedded_dir, project_root=project_root)
    paths: dict[str, Path] = {}

    for data_object in state.data_objects:
        document_id = data_object.object_id
        paths[f"embedded_document:{document_id}"] = writer.write_json(
            f"documents/{document_id}.json",
            {
                "contract_version": "embedded-document-v2",
                "stage": "embedded",
                "document_id": document_id,
                **_retrieval_payload(state, document_id),
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
                "source": state.enriched_schemas,
                "index_record_types": sorted({record.index_type for record in state.index_records}),
                "vector_dimension": state.embedding_report.get("dimension"),
            },
            document_schema={
                "contract_version": "string",
                "stage": "embedded",
                "document_id": "string",
                "retrieval": _retrieval_document_schema(),
            },
            extra=_run_reports(state),
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
    """Write one consolidated end-to-end JSON per document plus common metadata."""
    writer = LocalArtifactWriter(output_dir, project_root=project_root)
    parsed_by_id = {record.object_id: record for record in state.parsed_data}
    initial_schema_by_id = {
        record.source_object_id: record for record in state.initial_schemas
    }
    cleaned_by_id = {record.source_object_id: record for record in state.cleaned_data}
    cleaned_schema_by_id = {
        record.source_object_id: record for record in state.cleaned_schemas
    }
    enriched_by_id = {record.source_object_id: record for record in state.enriched_data}
    enriched_schema_by_id = {
        record.source_object_id: record for record in state.enriched_schemas
    }
    paths: dict[str, Path] = {}

    for data_object in state.data_objects:
        document_id = data_object.object_id
        initial_schema = initial_schema_by_id.get(document_id)
        cleaned_schema = cleaned_schema_by_id.get(document_id)
        enriched_schema = enriched_schema_by_id.get(document_id)
        paths[f"output_document:{document_id}"] = writer.write_json(
            f"documents/{document_id}.json",
            {
                "contract_version": "output-document-v2",
                "document_id": document_id,
                "source": data_object,
                "ingestion": {
                    "parsed": parsed_by_id.get(document_id),
                    "schema_id": initial_schema.schema_id if initial_schema else None,
                },
                "cleaning": {
                    "data": cleaned_by_id.get(document_id),
                    "schema_id": cleaned_schema.schema_id if cleaned_schema else None,
                },
                "enrichment": {
                    "data": enriched_by_id.get(document_id),
                    "schema_id": enriched_schema.schema_id if enriched_schema else None,
                },
                **_retrieval_payload(state, document_id),
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
                "ingested": state.initial_schemas,
                "cleaned": state.cleaned_schemas,
                "enriched": state.enriched_schemas,
                "index_record_types": sorted({record.index_type for record in state.index_records}),
                "vector_dimension": state.embedding_report.get("dimension"),
            },
            document_schema={
                "contract_version": "string",
                "document_id": "string",
                "source": "DataObject",
                "ingestion": "object",
                "cleaning": "object",
                "enrichment": "object",
                "retrieval": _retrieval_document_schema(),
            },
            extra={
                **_run_reports(state),
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


def _retrieval_payload(state: PipelineState, document_id: str) -> dict[str, Any]:
    metadata_record = next(
        (record for record in state.metadata_records if record.source_object_id == document_id),
        None,
    )
    index_records = [
        record for record in state.index_records if record.source_object_id == document_id
    ]
    vectors_by_record: dict[str, list[dict[str, Any]]] = {}
    for vector in state.vector_records:
        if vector.get("source_object_id") != document_id:
            continue
        record_id = str(vector.get("record_id") or "")
        if record_id:
            vectors_by_record.setdefault(record_id, []).append(vector)

    document_record = next(
        (record for record in index_records if record.index_type == "document"),
        None,
    )
    catalog_record = next(
        (record for record in index_records if record.index_type == "catalog"),
        None,
    )
    item_records = [
        record for record in index_records if record.index_type in RETRIEVAL_ITEM_TYPES
    ]
    return {
        "retrieval": {
            "document": (
                _retrieval_context_record(
                    document_record,
                    vectors_by_record.get(document_record.record_id, []),
                )
                if document_record
                else None
            ),
            "catalog": {
                "metadata": metadata_record,
                "index": (
                    _retrieval_context_record(
                        catalog_record,
                        vectors_by_record.get(catalog_record.record_id, []),
                    )
                    if catalog_record
                    else None
                ),
            },
            "items": [
                _retrieval_item(record, vectors_by_record.get(record.record_id, []))
                for record in item_records
            ],
        },
    }


def _retrieval_context_record(
    record: Any,
    vectors: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in record.payload.items()
        if key not in {"embedding", "embedding_model", "embedding_status"}
    }
    return {
        "record_id": record.record_id,
        "index_type": record.index_type,
        "source_object_id": record.source_object_id,
        **payload,
        "embeddings": [_nested_embedding(vector) for vector in vectors],
        "metadata": record.metadata,
    }


def _retrieval_item(record: Any, vectors: list[dict[str, Any]]) -> dict[str, Any]:
    item_type = RETRIEVAL_ITEM_TYPES[record.index_type]
    payload = record.payload
    return {
        "item_id": _retrieval_item_id(record),
        "type": item_type,
        "record_id": record.record_id,
        "document_id": payload.get("document_id") or record.source_object_id,
        "source_object_id": record.source_object_id,
        "position": _retrieval_item_position(record),
        "content": _retrieval_item_content(record),
        "embedding_text": payload.get("embedding_text", ""),
        "embeddings": [_nested_embedding(vector) for vector in vectors],
        "metadata": record.metadata,
    }


def _retrieval_item_id(record: Any) -> str:
    payload = record.payload
    id_field = {
        "text_chunk": "chunk_id",
        "table": "table_id",
        "image": "image_id",
    }[record.index_type]
    return str(payload.get(id_field) or record.record_id)


def _retrieval_item_position(record: Any) -> dict[str, Any]:
    payload = record.payload
    index_field = {
        "text_chunk": "chunk_index",
        "table": "table_index",
        "image": "image_index",
    }[record.index_type]
    position = {"index": payload.get(index_field)}
    if record.index_type == "text_chunk":
        position.update(
            {
                "start_char": payload.get("start_char"),
                "end_char": payload.get("end_char"),
            }
        )
    return position


def _retrieval_item_content(record: Any) -> dict[str, Any]:
    payload = record.payload
    if record.index_type == "text_chunk":
        return {"text": payload.get("text", "")}
    if record.index_type == "table":
        table = payload.get("table")
        return dict(table) if isinstance(table, dict) else {"content": table}
    image = payload.get("image")
    return dict(image) if isinstance(image, dict) else {"content": image}


def _nested_embedding(vector: dict[str, Any]) -> dict[str, Any]:
    return {
        "vector_id": vector.get("vector_id"),
        "model": vector.get("embedding_model"),
        "dimension": vector.get("embedding_dimension"),
        "status": vector.get("embedding_status"),
        "values": vector.get("embedding", []),
        "metadata": vector.get("metadata", {}),
    }


def _retrieval_document_schema() -> dict[str, Any]:
    return {
        "document": "IndexRecord|null; document-level retrieval context",
        "catalog": {
            "metadata": "MetadataRecord|null",
            "index": "IndexRecord|null; catalog-level retrieval context",
        },
        "items": [
            {
                "item_id": "string; stable component id",
                "type": "text|table|image",
                "record_id": "string; source index record id",
                "document_id": "string",
                "source_object_id": "string",
                "position": {
                    "index": "integer",
                    "start_char": "integer; text only",
                    "end_char": "integer; text only",
                },
                "content": "object; shape depends on type",
                "embedding_text": "string; exact text sent to the embedding provider",
                "embeddings": "Embedding[]",
                "metadata": "object",
            }
        ],
    }


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


def _run_reports(state: PipelineState) -> dict[str, Any]:
    return {
        "reports": {
            "index_quality": state.index_quality_report,
            "embedding": state.embedding_report,
            "integration": {
                "mode": "indexing-pass-through",
                "schema_matches": state.schema_matches,
                "entity_matches": state.entity_matches,
                "relationship_records": state.relationship_records,
            },
        }
    }


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
