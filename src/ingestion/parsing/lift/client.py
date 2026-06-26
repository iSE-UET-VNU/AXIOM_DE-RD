"""Lift hosted API parser.

This adapter mirrors the `lift/scripts/run_omnidocbench_datalab.py` pattern:
use `DATALAB_API_KEY`, call `datalab_sdk.DatalabClient.extract`, and map the
schema-aligned response into AXIOM's `ParsedData` contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import time

from ....models import DataObject, ParsedData

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif"}

DEFAULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "description": "Likely document type, such as paper, report, slide, newspaper, note, form, or textbook.",
        },
        "language": {
            "type": "string",
            "description": "Dominant language on the page or document.",
        },
        "title": {
            "type": "string",
            "description": "Main visible title or heading, if any.",
        },
        "main_text": {
            "type": "string",
            "description": "Main readable text in natural reading order. Preserve important formulas, table text, and headings.",
        },
        "tables": {
            "type": "array",
            "description": "Tables visible in the document.",
            "items": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "content": {
                        "type": "string",
                        "description": "Compact plain-text or markdown representation of the table.",
                    },
                },
            },
        },
        "figures": {
            "type": "array",
            "description": "Figures, charts, diagrams, or illustrations.",
            "items": {
                "type": "object",
                "properties": {
                    "caption": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "formulas": {
            "type": "array",
            "description": "Important standalone mathematical formulas.",
            "items": {"type": "string"},
        },
    },
}


@dataclass
class LiftAPIConfig:
    api_key_env: str = "DATALAB_API_KEY"
    mode: str = "balanced"
    schema_path: str | None = None
    output_dir: str | None = "data/processed/lift_outputs"
    fallback_to_local: bool = True

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "LiftAPIConfig":
        config = config or {}
        return cls(
            api_key_env=str(config.get("api_key_env", "DATALAB_API_KEY")),
            mode=str(config.get("mode", "balanced")),
            schema_path=_optional_str(config.get("schema_path")),
            output_dir=_optional_str(config.get("output_dir", "data/processed/lift_outputs")),
            fallback_to_local=bool(config.get("fallback_to_local", True)),
        )

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class LiftAPIParserClient:
    """Small wrapper around the Datalab hosted extraction API."""

    def __init__(self, config: LiftAPIConfig) -> None:
        self.config = config

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        file_path = Path(path)
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise RuntimeError(f"Lift API does not support file type: {file_path.suffix}")

        if not self.config.api_key:
            raise RuntimeError(f"{self.config.api_key_env} is not set.")

        try:
            from datalab_sdk import DatalabClient, ExtractOptions
        except ImportError as exc:
            raise RuntimeError(
                "Missing datalab-python-sdk. Install it with: pip install datalab-python-sdk"
            ) from exc

        schema = _load_schema(self.config.schema_path)
        client = DatalabClient()
        options = ExtractOptions(page_schema=json.dumps(schema), mode=self.config.mode)

        started = time.monotonic()
        result = client.extract(str(file_path), options=options)
        raw_json = _get_attr(result, "extraction_schema_json")
        extraction = _parse_extraction(raw_json)

        payload = {
            "input": {
                "object_id": data_object.object_id,
                "uri": data_object.uri,
                "file_name": file_path.name,
                "content_type": data_object.content_type,
                "metadata": data_object.metadata,
            },
            "status": _get_attr(result, "status"),
            "page_count": _get_attr(result, "page_count"),
            "latency_seconds": round(time.monotonic() - started, 3),
            "extraction": extraction,
        }
        output_path = _write_output(self.config.output_dir, file_path, payload)

        text = _extraction_text(extraction)
        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=str(data_object.metadata.get("format", file_path.suffix.lstrip("."))),
            rows=[{"extraction": extraction, "text": text}],
            text=text,
            metadata={
                "parser": "lift-api",
                "mode": self.config.mode,
                "status": payload["status"],
                "page_count": payload["page_count"],
                "latency_seconds": payload["latency_seconds"],
                "raw_output_path": str(output_path) if output_path else None,
            },
        )


def _load_schema(schema_path: str | None) -> dict[str, Any]:
    if not schema_path:
        return DEFAULT_SCHEMA
    with Path(schema_path).open(encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise RuntimeError(f"Lift schema must be a JSON object: {schema_path}")
    return schema


def _write_output(output_dir: str | None, file_path: Path, payload: dict[str, Any]) -> Path | None:
    if not output_dir:
        return None
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / f"{file_path.stem}.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _parse_extraction(raw_json: Any) -> Any:
    if raw_json is None:
        return None
    if isinstance(raw_json, (dict, list)):
        return raw_json
    try:
        return json.loads(str(raw_json))
    except json.JSONDecodeError:
        return raw_json


def _extraction_text(extraction: Any) -> str | None:
    if not isinstance(extraction, dict):
        return None
    for field in ("markdown", "main_text", "text", "content"):
        value = extraction.get(field)
        if value:
            return str(value)
    return None


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _optional_str(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return str(value)
