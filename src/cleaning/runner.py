"""Data cleaning module interface."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import CleanedData, CleanedSchema, InitialSchema, ParsedData, make_id


@dataclass
class CleaningOutput:
    cleaned_data: list[CleanedData] = field(default_factory=list)
    cleaned_schemas: list[CleanedSchema] = field(default_factory=list)


def run(parsed_data: list[ParsedData], initial_schemas: list[InitialSchema]) -> CleaningOutput:
    """Pass parsed data through until the cleaning module is implemented."""
    output = CleaningOutput()
    schema_by_object = {schema.source_object_id: schema for schema in initial_schemas}

    for parsed in parsed_data:
        source_schema = schema_by_object.get(parsed.object_id)
        cleaned_schema = CleanedSchema(
            schema_id=make_id(parsed.object_id, "cleaned-schema"),
            source_schema_id=source_schema.schema_id if source_schema else "",
            source_object_id=parsed.object_id,
            fields=source_schema.fields if source_schema else {},
            entities=source_schema.entities if source_schema else [],
            relationships=source_schema.relationships if source_schema else [],
            transformations=["pass_through"],
            metadata={
                "pass_through": True,
                "todo": "Implement standardization, error processing, and imputation.",
            },
        )
        cleaned = CleanedData(
            source_object_id=parsed.object_id,
            rows=[dict(row) for row in parsed.rows],
            issues=[],
            metadata={
                "pass_through": True,
                "source_format": parsed.source_format,
                "source_uri": parsed.source_uri,
                "row_count": len(parsed.rows),
                "source_metadata": parsed.metadata,
            },
        )

        output.cleaned_data.append(cleaned)
        output.cleaned_schemas.append(cleaned_schema)

    return output
