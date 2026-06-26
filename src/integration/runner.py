"""Data integration module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    EnrichedData,
    EnrichedSchema,
    EntityMatch,
    MetadataRecord,
    RelationshipRecord,
    SchemaMatch,
)
from .entity_matching import match_entities
from .relationship_extraction import extract_relationships
from .schema_matching import match_schemas


@dataclass
class IntegrationOutput:
    schema_matches: list[SchemaMatch] = field(default_factory=list)
    entity_matches: list[EntityMatch] = field(default_factory=list)
    relationship_records: list[RelationshipRecord] = field(default_factory=list)


def run(
    enriched_data: list[EnrichedData],
    enriched_schemas: list[EnrichedSchema],
    metadata_records: list[MetadataRecord],
) -> IntegrationOutput:
    """Run placeholder schema/entity matching and relationship extraction."""
    return IntegrationOutput(
        schema_matches=match_schemas(enriched_schemas),
        entity_matches=match_entities(enriched_data),
        relationship_records=extract_relationships(enriched_data, metadata_records),
    )
