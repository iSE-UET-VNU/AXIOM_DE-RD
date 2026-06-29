"""Cataloging and indexing module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import EnrichedData, EnrichedSchema, IndexRecord, MetadataRecord
from .documents import build_documents_artifact
from .embeddings import EmbeddingConfig, build_vector_records
from .index_builder import build_index_records
from .metadata_catalog import build_metadata_catalog
from .quality import assert_index_quality, build_index_quality_report


@dataclass
class CatalogingOutput:
    metadata_records: list[MetadataRecord] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    index_records: list[IndexRecord] = field(default_factory=list)
    index_quality_report: dict = field(default_factory=dict)
    vector_records: list[dict[str, Any]] = field(default_factory=list)
    embedding_report: dict[str, Any] = field(default_factory=dict)


def run(
    enriched_data: list[EnrichedData],
    enriched_schemas: list[EnrichedSchema],
    indexing_config: dict[str, Any] | None = None,
) -> CatalogingOutput:
    """Build metadata catalog records and component index records."""
    indexing_config = indexing_config or {}
    metadata_records = build_metadata_catalog(enriched_data, enriched_schemas)
    index_records = build_index_records(enriched_data, metadata_records)
    documents = build_documents_artifact(enriched_data, index_records)
    index_quality_report = build_index_quality_report(index_records)
    assert_index_quality(index_quality_report)
    vector_records, embedding_report = build_vector_records(
        index_records,
        EmbeddingConfig.from_mapping(indexing_config.get("embeddings")),
    )
    return CatalogingOutput(
        metadata_records=metadata_records,
        documents=documents,
        index_records=index_records,
        index_quality_report=index_quality_report,
        vector_records=vector_records,
        embedding_report=embedding_report,
    )
