"""Data ingestion module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import DataObject, InitialSchema, ParsedData, QuarantinedDocument, make_id
from .parsing import infer_initial_schema, parse_raw_file
from .detector import detect_content_type, detect_format
from .validation import validate_parsed_document
from ..utils.paths import portable_path


@dataclass
class IngestionOutput:
    data_objects: list[DataObject] = field(default_factory=list)
    parsed_data: list[ParsedData] = field(default_factory=list)
    initial_schemas: list[InitialSchema] = field(default_factory=list)
    quarantined_documents: list[QuarantinedDocument] = field(default_factory=list)

    def extend(self, other: "IngestionOutput") -> None:
        """Append another document result to this batch output."""
        self.data_objects.extend(other.data_objects)
        self.parsed_data.extend(other.parsed_data)
        self.initial_schemas.extend(other.initial_schemas)
        self.quarantined_documents.extend(other.quarantined_documents)


def run(
    input_file: str | Path,
    *,
    source_uri: str | None = None,
    input_metadata: dict[str, Any] | None = None,
    parser_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> IngestionOutput:
    """Parse one local document prepared by an external input adapter."""
    path = Path(input_file)
    output = IngestionOutput()

    if not path.is_file():
        raise FileNotFoundError(f"Staged input document does not exist: {path}")

    metadata = dict(input_metadata or {})
    resolved_source_uri = source_uri or portable_path(path, project_root)
    file_name = str(metadata.get("file_name") or path.name)
    file_format = detect_format(path)
    response_content_type = metadata.get("response_content_type")
    data_object = DataObject(
        object_id=make_id("data-object", resolved_source_uri),
        uri=resolved_source_uri,
        content_type=str(response_content_type or detect_content_type(path)),
        metadata={
            **metadata,
            "relative_uri": file_name,
            "format": file_format,
            "size_bytes": path.stat().st_size,
        },
    )
    parsed = parse_raw_file(path, data_object, parser_config)
    quarantine_reasons = validate_parsed_document(parsed)
    if quarantine_reasons:
        output.quarantined_documents.append(
            QuarantinedDocument(
                document_id=data_object.object_id,
                source=data_object,
                parsed=parsed,
                reasons=quarantine_reasons,
            )
        )
        return output

    schema = infer_initial_schema(parsed)

    output.data_objects.append(data_object)
    output.parsed_data.append(parsed)
    output.initial_schemas.append(schema)

    return output
