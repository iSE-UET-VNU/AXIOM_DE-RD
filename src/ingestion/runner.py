"""Data ingestion module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import DataObject, InitialSchema, ParsedData, make_id
from .parsing import infer_initial_schema, parse_raw_file
from .normalization import detect_content_type, detect_format, normalize_parsed_data
from ..utils.paths import portable_path


@dataclass
class IngestionOutput:
    data_objects: list[DataObject] = field(default_factory=list)
    parsed_data: list[ParsedData] = field(default_factory=list)
    initial_schemas: list[InitialSchema] = field(default_factory=list)
    normalized_texts: list[dict[str, Any]] = field(default_factory=list)
    normalized_images: list[dict[str, Any]] = field(default_factory=list)
    normalized_tables: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)


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

        relative_uri = path.relative_to(input_path).as_posix()
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

    if output.parsed_data:
        normalized = normalize_parsed_data(output.parsed_data, project_root=project_root)
        output.normalized_texts = normalized.texts
        output.normalized_images = normalized.images
        output.normalized_tables = normalized.tables
        output.documents = enrich_document_records(normalized.documents, output.data_objects)

    return output


def enrich_document_records(
    documents: list[dict[str, Any]],
    data_objects: list[DataObject],
) -> list[dict[str, Any]]:
    """Attach source inventory fields to normalized document records."""
    object_by_id = {item.object_id: item for item in data_objects}
    records: list[dict[str, Any]] = []
    for document in documents:
        record = dict(document)
        data_object = object_by_id.get(str(record.get("document_id")))
        if data_object:
            relative_uri = str(data_object.metadata.get("relative_uri", "")).replace("\\", "/")
            record.update(
                {
                    "file_name": Path(relative_uri or data_object.uri).name,
                    "relative_uri": relative_uri or None,
                    "content_type": data_object.content_type,
                    "size_bytes": data_object.metadata.get("size_bytes"),
                }
            )
        records.append(record)
    return records
