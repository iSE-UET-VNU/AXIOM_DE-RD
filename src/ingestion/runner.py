"""Data ingestion module interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    _accept_parse_result(
        output,
        data_object,
        parse_result,
        project_root=project_root,
    )
    return output


def run_many(
    inputs: list[IngestionInput],
    *,
    parser_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    on_document_complete: Callable[[IngestionOutput], None] | None = None,
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

    partial_outputs: list[IngestionOutput | None] = [None] * len(prepared)

    def accept_result(index: int, parse_result: ParseResult) -> None:
        if partial_outputs[index] is not None:
            return
        partial = IngestionOutput()
        _, data_object = prepared[index]
        _accept_parse_result(
            partial,
            data_object,
            parse_result,
            project_root=project_root,
        )
        partial_outputs[index] = partial
        if on_document_complete is not None:
            on_document_complete(partial)

    parse_results = ParsingService.from_config(parser_config).parse_many(
        prepared,
        on_result=accept_result,
    )
    for index, parse_result in enumerate(parse_results):
        accept_result(index, parse_result)
    for partial in partial_outputs:
        if partial is None:
            raise RuntimeError("Ingestion did not produce every document result.")
        output.extend(partial)
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
    *,
    project_root: str | Path | None,
) -> None:
    if (
        parse_result.status is not ParseStatus.SUCCESS
        or parse_result.parsed_data is None
    ):
        output.quarantined_documents.append(
            _parse_failure_quarantine(data_object, parse_result)
        )
        return

    parsed = parse_result.parsed_data
    apply_image_filters(parsed, project_root=project_root)
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


def _parse_failure_quarantine(
    data_object: DataObject,
    result: ParseResult,
) -> QuarantinedDocument:
    message = str(
        result.error.get("message")
        or result.reason
        or f"{result.backend} parser returned {result.status.value}."
    )
    parsed = ParsedData(
        object_id=data_object.object_id,
        source_uri=data_object.uri,
        source_format=str(data_object.metadata.get("format") or ""),
        metadata={
            "parser": result.backend,
            "backend": result.backend,
            "status": result.status.value,
            "parse_reason": result.reason,
            "parse_error": dict(result.error),
        },
    )
    return QuarantinedDocument(
        document_id=data_object.object_id,
        source=data_object,
        parsed=parsed,
        reasons=[
            {
                "code": (
                    "parse_failed"
                    if result.status is ParseStatus.FAILED
                    else f"parse_{result.status.value}"
                ),
                "field": "parsing",
                "message": message,
                "backend": result.backend,
                "route": result.route,
                "error": dict(result.error),
            }
        ],
    )
