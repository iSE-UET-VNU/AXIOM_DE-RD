"""Data enrichment module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import CleanedData, CleanedSchema, EnrichedData, EnrichedSchema, make_id
from .annotation import annotate_data
from .profiling import profile_data


@dataclass
class EnrichmentOutput:
    enriched_data: list[EnrichedData] = field(default_factory=list)
    enriched_schemas: list[EnrichedSchema] = field(default_factory=list)


def run(cleaned_data: list[CleanedData], cleaned_schemas: list[CleanedSchema]) -> EnrichmentOutput:
    """Annotate and profile cleaned data."""
    output = EnrichmentOutput()
    schema_by_object = {schema.source_object_id: schema for schema in cleaned_schemas}

    for cleaned in cleaned_data:
        annotations = annotate_data(cleaned.rows)
        profile = profile_data(cleaned.rows)
        source_schema = schema_by_object.get(cleaned.source_object_id)

        enriched = EnrichedData(
            source_object_id=cleaned.source_object_id,
            rows=cleaned.rows,
            annotations=annotations,
            profile=profile,
            metadata={"todo": "Attach external metadata and richer annotations."},
        )
        enriched_schema = EnrichedSchema(
            schema_id=make_id(cleaned.source_object_id, "enriched-schema"),
            source_schema_id=source_schema.schema_id if source_schema else "",
            source_object_id=cleaned.source_object_id,
            fields=source_schema.fields if source_schema else {},
            semantic_annotations={
                "field_count": annotations["field_count"],
                "has_text": annotations["has_text"],
            },
            metadata={"todo": "Add semantic types and ontology mappings."},
        )

        output.enriched_data.append(enriched)
        output.enriched_schemas.append(enriched_schema)

    return output
