"""Shared data contracts for the first pipeline scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib


def make_id(*parts: object) -> str:
    """Create a stable short id from deterministic input parts."""
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class DataObject:
    object_id: str
    uri: str
    content_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedData:
    object_id: str
    source_uri: str
    source_format: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialSchema:
    schema_id: str
    source_object_id: str
    fields: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedData:
    source_object_id: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleanedSchema:
    schema_id: str
    source_schema_id: str
    source_object_id: str
    fields: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedData:
    source_object_id: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    annotations: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedSchema:
    schema_id: str
    source_schema_id: str
    source_object_id: str
    fields: dict[str, str] = field(default_factory=dict)
    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    semantic_annotations: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetadataRecord:
    record_id: str
    source_object_id: str
    title: str
    schema_id: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndexRecord:
    record_id: str
    index_type: str
    source_object_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaMatch:
    left_schema_id: str
    right_schema_id: str
    score: float
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityMatch:
    left_entity_id: str
    right_entity_id: str
    score: float
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipRecord:
    record_id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    run_id: str
    input_dir: str
    output_dir: str
    work_dir: str | None = None
    data_objects: list[DataObject] = field(default_factory=list)
    parsed_data: list[ParsedData] = field(default_factory=list)
    initial_schemas: list[InitialSchema] = field(default_factory=list)
    cleaned_data: list[CleanedData] = field(default_factory=list)
    cleaned_schemas: list[CleanedSchema] = field(default_factory=list)
    normalized_texts: list[dict[str, Any]] = field(default_factory=list)
    normalized_images: list[dict[str, Any]] = field(default_factory=list)
    normalized_tables: list[dict[str, Any]] = field(default_factory=list)
    normalized_documents: list[dict[str, Any]] = field(default_factory=list)
    enriched_data: list[EnrichedData] = field(default_factory=list)
    enriched_schemas: list[EnrichedSchema] = field(default_factory=list)
    metadata_records: list[MetadataRecord] = field(default_factory=list)
    index_records: list[IndexRecord] = field(default_factory=list)
    index_quality_report: dict[str, Any] = field(default_factory=dict)
    vector_records: list[dict[str, Any]] = field(default_factory=list)
    embedding_report: dict[str, Any] = field(default_factory=dict)
    schema_matches: list[SchemaMatch] = field(default_factory=list)
    entity_matches: list[EntityMatch] = field(default_factory=list)
    relationship_records: list[RelationshipRecord] = field(default_factory=list)
    completed_modules: list[str] = field(default_factory=list)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    ingestion_config: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
