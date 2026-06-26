"""Placeholder entity matching."""

from __future__ import annotations

from ..models import EnrichedData, EntityMatch


def match_entities(enriched_data: list[EnrichedData]) -> list[EntityMatch]:
    """Find obvious entity matches using shared id-like fields.

    TODO: Add fuzzy matching, blocking, and confidence calibration.
    """
    seen: dict[str, str] = {}
    matches: list[EntityMatch] = []

    for data in enriched_data:
        for row_index, row in enumerate(data.rows):
            entity_value = row.get("id") or row.get("entity_id")
            if not entity_value:
                continue

            entity_key = str(entity_value)
            entity_id = f"{data.source_object_id}:{row_index}"
            if entity_key in seen and seen[entity_key] != entity_id:
                matches.append(
                    EntityMatch(
                        left_entity_id=seen[entity_key],
                        right_entity_id=entity_id,
                        score=1.0,
                        evidence={"matched_field": "id/entity_id", "value": entity_key},
                    )
                )
            else:
                seen[entity_key] = entity_id

    return matches
