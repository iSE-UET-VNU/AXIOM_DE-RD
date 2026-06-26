"""Placeholder schema matching."""

from __future__ import annotations

from itertools import combinations

from ..models import EnrichedSchema, SchemaMatch


def match_schemas(schemas: list[EnrichedSchema]) -> list[SchemaMatch]:
    """Compare schemas with simple field-name overlap.

    TODO: Replace with semantic schema matching.
    """
    matches: list[SchemaMatch] = []
    for left, right in combinations(schemas, 2):
        left_fields = set(left.fields)
        right_fields = set(right.fields)
        union = left_fields | right_fields
        score = len(left_fields & right_fields) / len(union) if union else 0.0
        if score > 0:
            matches.append(
                SchemaMatch(
                    left_schema_id=left.schema_id,
                    right_schema_id=right.schema_id,
                    score=round(score, 4),
                    notes="field-name-overlap-placeholder",
                )
            )
    return matches
