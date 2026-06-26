"""Minimal raw file parser.

TODO: Add real parsers for domain documents, images, PDFs, and structured files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

from ...models import DataObject, InitialSchema, ParsedData, make_id

TEXT_FORMATS = {"csv", "json", "jsonl", "txt", "md", "yaml", "yml"}
BINARY_FORMATS = {"png", "jpg", "jpeg", "gif", "webp", "pdf", "parquet", "xlsx"}


def parse_raw_file(path: str | Path, data_object: DataObject) -> ParsedData:
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
    fields: dict[str, str] = {}
    for row in parsed.rows:
        for key, value in row.items():
            fields.setdefault(str(key), _type_name(value))

    if parsed.text is not None and "text" not in fields:
        fields["text"] = "str"

    return InitialSchema(
        schema_id=make_id(parsed.object_id, "initial-schema"),
        source_object_id=parsed.object_id,
        fields=fields,
        metadata={
            "source_format": parsed.source_format,
            "row_count": len(parsed.rows),
            "todo": "Improve type inference and nested schema extraction.",
        },
    )


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
