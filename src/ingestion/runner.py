"""Data ingestion module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import DataObject, InitialSchema, ParsedData, make_id
from .parsing import infer_initial_schema, parse_raw_file
from .parsing_formatting import detect_content_type, detect_format
from ..utils.paths import portable_path


@dataclass
class IngestionOutput:
    data_objects: list[DataObject] = field(default_factory=list)
    parsed_data: list[ParsedData] = field(default_factory=list)
    initial_schemas: list[InitialSchema] = field(default_factory=list)


def run(
    input_dir: str | Path,
    parser_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> IngestionOutput:
    """Discover raw files and produce parsed data plus initial schemas."""
    input_path = Path(input_dir)
    output = IngestionOutput()

    if not input_path.exists():
        return output

    for path in sorted(input_path.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        relative_uri = str(path.relative_to(input_path))
        file_format = detect_format(path)
        portable_uri = portable_path(path, project_root)
        data_object = DataObject(
            object_id=make_id("data-object", relative_uri),
            uri=portable_uri,
            content_type=detect_content_type(path),
            metadata={
                "relative_uri": relative_uri,
                "format": file_format,
                "size_bytes": path.stat().st_size,
            },
        )
        parsed = parse_raw_file(path, data_object, parser_config)
        schema = infer_initial_schema(parsed)

        output.data_objects.append(data_object)
        output.parsed_data.append(parsed)
        output.initial_schemas.append(schema)

    return output
