"""Placeholder relationship extraction."""

from __future__ import annotations

from ..models import EnrichedData, MetadataRecord, RelationshipRecord, make_id


def extract_relationships(
    enriched_data: list[EnrichedData],
    metadata_records: list[MetadataRecord],
) -> list[RelationshipRecord]:
    """Create basic lineage/catalog relationships.

    TODO: Extract domain relationships from content and entity links.
    """
    metadata_by_object = {record.source_object_id: record for record in metadata_records}
    relationships: list[RelationshipRecord] = []

    for data in enriched_data:
        metadata = metadata_by_object.get(data.source_object_id)
        if metadata:
            relationships.append(
                RelationshipRecord(
                    record_id=make_id(data.source_object_id, metadata.record_id, "has-metadata"),
                    source_id=data.source_object_id,
                    target_id=metadata.record_id,
                    relationship_type="has_metadata",
                    properties={"source": "integration-placeholder"},
                )
            )

    return relationships
