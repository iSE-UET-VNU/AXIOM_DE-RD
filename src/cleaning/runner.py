"""Data cleaning module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import CleanedData, CleanedSchema, InitialSchema, ParsedData, make_id
from .error_processing import process_errors
from .imputation import impute_missing_values
from .standardization import standardize_rows


@dataclass
class CleaningOutput:
    cleaned_data: list[CleanedData] = field(default_factory=list)
    cleaned_schemas: list[CleanedSchema] = field(default_factory=list)


def run(parsed_data: list[ParsedData], initial_schemas: list[InitialSchema]) -> CleaningOutput:
    """Standardize fields, process simple errors, and impute missing values."""
    output = CleaningOutput()
    schema_by_object = {schema.source_object_id: schema for schema in initial_schemas}

    for parsed in parsed_data:
        standardized = standardize_rows(parsed.rows)
        checked_rows, issues = process_errors(standardized)
        imputed_rows = impute_missing_values(checked_rows)

        source_schema = schema_by_object.get(parsed.object_id)
        fields = _infer_fields(imputed_rows)
        cleaned_schema = CleanedSchema(
            schema_id=make_id(parsed.object_id, "cleaned-schema"),
            source_schema_id=source_schema.schema_id if source_schema else "",
            source_object_id=parsed.object_id,
            fields=fields,
            transformations=["standardize_field_names", "process_errors", "impute_missing_values"],
            metadata={
                "todo": "Add validation contracts and domain-specific cleaning rules.",
            },
        )
        cleaned = CleanedData(
            source_object_id=parsed.object_id,
            rows=imputed_rows,
            issues=issues,
            metadata={
                "source_format": parsed.source_format,
                "row_count": len(imputed_rows),
            },
        )

        output.cleaned_data.append(cleaned)
        output.cleaned_schemas.append(cleaned_schema)

    return output


def _infer_fields(rows: list[dict[str, Any]]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            fields.setdefault(key, _type_name(value))
    return fields


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"
