"""Data integration module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    EntityMatch,
    IndexRecord,
    RelationshipRecord,
    SchemaMatch,
)


@dataclass
class IntegrationOutput:
    passed_index_records: list[IndexRecord] = field(default_factory=list)
    schema_matches: list[SchemaMatch] = field(default_factory=list)
    entity_matches: list[EntityMatch] = field(default_factory=list)
    relationship_records: list[RelationshipRecord] = field(default_factory=list)


def run(index_records: list[IndexRecord]) -> IntegrationOutput:
    """Pass indexing records through until integration logic is implemented."""
    return IntegrationOutput(passed_index_records=list(index_records))
