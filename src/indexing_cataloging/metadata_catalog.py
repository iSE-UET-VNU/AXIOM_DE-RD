"""Metadata catalog builders."""

from __future__ import annotations

from ..models import EnrichedData, EnrichedSchema, MetadataRecord, make_id


def build_metadata_catalog(
    enriched_data: list[EnrichedData],
    enriched_schemas: list[EnrichedSchema],
) -> list[MetadataRecord]:
    """Build local metadata records for each enriched object."""
    schema_by_object = {schema.source_object_id: schema for schema in enriched_schemas}
    records: list[MetadataRecord] = []

    for data in enriched_data:
        schema = schema_by_object.get(data.source_object_id)
        records.append(
            MetadataRecord(
                record_id=make_id(data.source_object_id, "metadata"),
                source_object_id=data.source_object_id,
                title=f"Data object {data.source_object_id}",
                schema_id=schema.schema_id if schema else None,
                profile=data.profile,
                tags=["auto-cataloged"],
                properties={
                    "annotation_summary": data.annotations,
                    "todo": "Add ownership, lineage, quality, and business metadata.",
                },
            )
        )

    return records
