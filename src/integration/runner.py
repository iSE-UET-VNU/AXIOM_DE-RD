"""Data integration module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import (
    EntityMatch,
    RelationshipRecord,
    RetrievalRecord,
    SchemaMatch,
)


@dataclass
class IntegrationOutput:
    passed_retrieval_records: list[RetrievalRecord] = field(default_factory=list)
    schema_matches: list[SchemaMatch] = field(default_factory=list)
    entity_matches: list[EntityMatch] = field(default_factory=list)
    relationship_records: list[RelationshipRecord] = field(default_factory=list)


def run(retrieval_records: list[RetrievalRecord]) -> IntegrationOutput:
    """Pass retrieval records through until integration logic is implemented."""
    return IntegrationOutput(passed_retrieval_records=list(retrieval_records))
