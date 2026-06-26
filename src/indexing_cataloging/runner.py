"""Cataloging and indexing module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import EnrichedData, EnrichedSchema, IndexRecord, MetadataRecord
from .index_builder import build_db_index_records, build_graph_index_records, build_vector_index_records
from .metadata_catalog import build_metadata_catalog


@dataclass
class CatalogingOutput:
    metadata_records: list[MetadataRecord] = field(default_factory=list)
    index_records: list[IndexRecord] = field(default_factory=list)


def run(enriched_data: list[EnrichedData], enriched_schemas: list[EnrichedSchema]) -> CatalogingOutput:
    """Build metadata catalog records and placeholder target indexes."""
    metadata_records = build_metadata_catalog(enriched_data, enriched_schemas)
    index_records = [
        *build_db_index_records(enriched_data),
        *build_vector_index_records(enriched_data),
        *build_graph_index_records(enriched_data),
    ]
    return CatalogingOutput(metadata_records=metadata_records, index_records=index_records)
