"""Data ingestion module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..models import (
    DataObject,
    InitialSchema,
    ParsedData,
    ParseResult,
    ParseStatus,
    QuarantinedDocument,
    make_id,
)
from .parsing import ParsingService, infer_initial_schema
from .detector import detect_content_type, detect_format
from .image_filter import apply_image_filters
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


@dataclass(frozen=True)
class IngestionInput:
    """One prepared input passed to the batch ingestion boundary."""

    path: str | Path
    source_uri: str | None = None
    metadata: dict[str, Any] | None = None


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
    if not path.is_file():
        raise FileNotFoundError(f"Staged input document does not exist: {path}")

    data_object = _build_data_object(
        path,
        source_uri=source_uri,
        input_metadata=input_metadata,
        project_root=project_root,
    )
    parse_result = ParsingService.from_config(parser_config).parse(path, data_object)
    output = IngestionOutput()
    _accept_parse_result(output, data_object, parse_result)
    return output


def run_many(
    inputs: list[IngestionInput],
    *,
    parser_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> IngestionOutput:
    """Parse multiple inputs through one shared parsing service."""

    output = IngestionOutput()
    prepared: list[tuple[Path, DataObject]] = []
    for item in inputs:
        path = Path(item.path)
        if not path.is_file():
            raise FileNotFoundError(f"Staged input document does not exist: {path}")
        prepared.append(
            (
                path,
                _build_data_object(
                    path,
                    source_uri=item.source_uri,
                    input_metadata=item.metadata,
                    project_root=project_root,
                ),
            )
        )

    parse_results = ParsingService.from_config(parser_config).parse_many(prepared)
    for (_, data_object), parse_result in zip(prepared, parse_results):
        _accept_parse_result(output, data_object, parse_result)
    return output


def _build_data_object(
    path: Path,
    *,
    source_uri: str | None,
    input_metadata: dict[str, Any] | None,
    project_root: str | Path | None,
) -> DataObject:
    metadata = dict(input_metadata or {})
    resolved_source_uri = source_uri or portable_path(path, project_root)
    file_name = str(metadata.get("file_name") or path.name)
    file_format = detect_format(path)
    response_content_type = metadata.get("response_content_type")
    return DataObject(
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


def _accept_parse_result(
    output: IngestionOutput,
    data_object: DataObject,
    parse_result: ParseResult,
) -> None:
    parsed = _require_parsed_data(parse_result)
    apply_image_filters(parsed)
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
        return

    schema = infer_initial_schema(parsed)

    output.data_objects.append(data_object)
    output.parsed_data.append(parsed)
    output.initial_schemas.append(schema)


_PARSER_ERROR_TYPES: dict[str, type[Exception]] = {
    "ConnectionError": ConnectionError,
    "FileNotFoundError": FileNotFoundError,
    "PermissionError": PermissionError,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
}


def _require_parsed_data(result: ParseResult) -> ParsedData:
    """Unwrap a successful routed parse while retaining fail-fast behavior."""
    if result.status is ParseStatus.SUCCESS and result.parsed_data is not None:
        return result.parsed_data

    message = str(
        result.error.get("message")
        or result.reason
        or f"{result.backend} parser returned {result.status.value}"
    )
    error_type = str(result.error.get("type") or "")
    exception_type = _PARSER_ERROR_TYPES.get(error_type, RuntimeError)
    raise exception_type(message)
