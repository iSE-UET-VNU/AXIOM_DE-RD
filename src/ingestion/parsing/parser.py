"""Minimal local raw file parser.

TODO: Add real parsers for domain documents, images, PDFs, and structured files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

from ...models import DataObject, InitialSchema, ParsedData
from ..parsing_formatting import build_initial_schema
from .lift import LiftAPIConfig, LiftAPIParserClient

TEXT_FORMATS = {"csv", "json", "jsonl", "txt", "md", "yaml", "yml"}
BINARY_FORMATS = {"png", "jpg", "jpeg", "gif", "webp", "pdf", "parquet", "xlsx"}


def parse_raw_file(
    path: str | Path,
    data_object: DataObject,
    parser_config: dict[str, Any] | None = None,
) -> ParsedData:
    """Parse a raw file using the configured parser provider."""
    parser_config = parser_config or {}
    if parser_config.get("provider") == "lift_api":
        return _parse_with_lift_api(path, data_object, parser_config)

    return _parse_locally(path, data_object)


def _parse_with_lift_api(
    path: str | Path,
    data_object: DataObject,
    parser_config: dict[str, Any],
) -> ParsedData:
    lift_config = LiftAPIConfig.from_mapping(parser_config.get("lift_api"))
    client = LiftAPIParserClient(lift_config)

    try:
        return client.parse_file(path, data_object)
    except RuntimeError as exc:
        if not lift_config.fallback_to_local:
            raise
        parsed = _parse_locally(path, data_object)
        parsed.metadata.update(
            {
                "parser": "local-fallback",
                "requested_parser": "lift-api",
                "fallback_reason": str(exc),
            }
        )
        return parsed


def _parse_locally(path: str | Path, data_object: DataObject) -> ParsedData:
    file_path = Path(path)
    file_format = data_object.metadata.get("format", file_path.suffix.lstrip("."))

    if file_format == "csv":
        rows = _parse_csv(file_path)
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=file_format,
            rows=rows,
            metadata={"parser": "csv"},
        )

    if file_format == "json":
        rows, text = _parse_json(file_path)
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=file_format,
            rows=rows,
            text=text,
            metadata={"parser": "json"},
        )

    if file_format in BINARY_FORMATS:
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=file_format,
            metadata={"parser": "binary-placeholder"},
        )

    text = _read_text(file_path) if file_format in TEXT_FORMATS or file_format == "unknown" else None
    rows = [{"text": text}] if text else []
    return ParsedData(
        object_id=data_object.object_id,
        source_uri=data_object.uri,
        source_format=file_format,
        rows=rows,
        text=text,
        metadata={"parser": "text-placeholder"},
    )


def infer_initial_schema(parsed: ParsedData) -> InitialSchema:
    return build_initial_schema(parsed)


def _parse_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_json(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = _read_text(path)
    payload = json.loads(text)

    if isinstance(payload, list):
        rows = [item if isinstance(item, dict) else {"value": item} for item in payload]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = [{"value": payload}]
    return rows, text


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
