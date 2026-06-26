"""Data enrichment module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import CleanedData, CleanedSchema, EnrichedData, EnrichedSchema, make_id


@dataclass
class EnrichmentOutput:
    enriched_data: list[EnrichedData] = field(default_factory=list)
    enriched_schemas: list[EnrichedSchema] = field(default_factory=list)


def run(cleaned_data: list[CleanedData], cleaned_schemas: list[CleanedSchema]) -> EnrichmentOutput:
    """Pass cleaned data through until the enrichment module is implemented."""
    output = EnrichmentOutput()
    schema_by_object = {schema.source_object_id: schema for schema in cleaned_schemas}

    for cleaned in cleaned_data:
        source_schema = schema_by_object.get(cleaned.source_object_id)

        enriched = EnrichedData(
            source_object_id=cleaned.source_object_id,
            rows=[dict(row) for row in cleaned.rows],
            annotations={},
            profile={},
            metadata={
                "pass_through": True,
                "todo": "Implement annotation and profiling.",
            },
        )
        enriched_schema = EnrichedSchema(
            schema_id=make_id(cleaned.source_object_id, "enriched-schema"),
            source_schema_id=source_schema.schema_id if source_schema else "",
            source_object_id=cleaned.source_object_id,
            fields=source_schema.fields if source_schema else {},
            semantic_annotations={},
            metadata={
                "pass_through": True,
                "todo": "Add semantic types and ontology mappings.",
            },
        )

        output.enriched_data.append(enriched)
        output.enriched_schemas.append(enriched_schema)

    return output
