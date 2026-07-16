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
    make_id,
)
from .parsing import ParsingService
from .normalization import (
    build_initial_schema,
    detect_content_type,
    detect_format,
    normalize_parsed_data,
)
from ..utils.paths import portable_path


@dataclass
class IngestionOutput:
    data_objects: list[DataObject] = field(default_factory=list)
    parse_results: list[ParseResult] = field(default_factory=list)
    parsed_data: list[ParsedData] = field(default_factory=list)
    initial_schemas: list[InitialSchema] = field(default_factory=list)
    normalized_texts: list[dict[str, Any]] = field(default_factory=list)
    normalized_images: list[dict[str, Any]] = field(default_factory=list)
    normalized_tables: list[dict[str, Any]] = field(default_factory=list)
    documents: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


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

    parsing_service = ParsingService.from_config(parser_config)

    for path in sorted(input_path.rglob("*")):
        inventory_error: Exception | None = None
        try:
            if path.is_dir():
                continue
        except Exception as exc:
            inventory_error = exc

        relative_uri = path.relative_to(input_path).as_posix()
        if inventory_error is None:
            try:
                file_format = detect_format(path)
                portable_uri = portable_path(path, project_root)
                content_type = detect_content_type(path)
                size_bytes: int | None = path.stat().st_size
            except Exception as exc:
                inventory_error = exc
        if inventory_error is not None:
            file_format = path.suffix.lower().lstrip(".") or "unknown"
            portable_uri = path.as_posix()
            content_type = "application/octet-stream"
            size_bytes = None
        data_object = DataObject(
            object_id=make_id("data-object", relative_uri),
            uri=portable_uri,
            content_type=content_type,
            metadata={
                "relative_uri": relative_uri,
                "format": file_format,
                "size_bytes": size_bytes,
            },
        )
        output.data_objects.append(data_object)
        if inventory_error is not None:
            routed_backend = parsing_service.router.resolve(path)
            route = str(getattr(routed_backend, "backend_name", "unsupported"))
            parse_result = ParseResult.failed(
                data_object.object_id,
                "inventory",
                inventory_error,
                route=route,
                reason="source_inventory_failed",
            )
            output.parse_results.append(parse_result)
            output.errors.append(_parse_error_record(parse_result))
            continue

        parse_result = parsing_service.parse(path, data_object)
        output.parse_results.append(parse_result)

        if parse_result.status == ParseStatus.FAILED:
            output.errors.append(_parse_error_record(parse_result))
            continue
        if parse_result.status != ParseStatus.SUCCESS:
            continue

        parsed = parse_result.parsed_data
        if parsed is None:  # ParsingService enforces this invariant defensively.
            continue
        output.parsed_data.append(parsed)

        try:
            schema = build_initial_schema(parsed)
        except Exception as exc:
            output.errors.append(
                _stage_error_record(
                    parse_result,
                    stage="ingestion.schema_inference",
                    reason="schema_inference_failed",
                    exc=exc,
                )
            )
        else:
            output.initial_schemas.append(schema)

        try:
            normalized = normalize_parsed_data([parsed], project_root=project_root)
        except Exception as exc:
            output.errors.append(
                _stage_error_record(
                    parse_result,
                    stage="ingestion.normalization",
                    reason="normalization_failed",
                    exc=exc,
                )
            )
        else:
            output.normalized_texts.extend(normalized.texts)
            output.normalized_images.extend(normalized.images)
            output.normalized_tables.extend(normalized.tables)
            output.documents.extend(normalized.documents)

    if output.documents:
        output.documents = enrich_document_records(output.documents, output.data_objects)

    return output


def _stage_error_record(
    result: ParseResult,
    *,
    stage: str,
    reason: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "source_object_id": result.source_object_id,
        "route": result.route,
        "backend": result.backend,
        "reason": reason,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _parse_error_record(result: ParseResult) -> dict[str, Any]:
    """Convert a file-scoped parse/inventory failure to a pipeline error."""
    stage = (
        "ingestion.inventory"
        if result.reason == "source_inventory_failed"
        else "ingestion.parsing"
    )
    return {
        "stage": stage,
        "source_object_id": result.source_object_id,
        "route": result.route,
        "backend": result.backend,
        "reason": result.reason or "parse_failed",
        "error": dict(result.error),
    }


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
