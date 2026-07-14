"""Build the self-contained logical-document schema artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ...models import PipelineState
from ...utils.paths import portable_path

_TEMPLATE_PATH = Path(__file__).with_name("logical_document.json")


def build_logical_document_schema(
    state: PipelineState,
    artifact_paths: dict[str, Any],
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON Schema for one logical document plus storage annotations."""
    schema = _load_template()
    all_texts = state.normalized_texts
    formulas = [record for record in all_texts if record.get("role") == "formula"]
    texts = [record for record in all_texts if record.get("role") != "formula"]
    manifest_path = _resolved_path(artifact_paths["manifest"], project_root)

    schema["x-axiom-dataset"] = _dataset_observations(
        state,
        dataset_id=_manifest_dataset_id(manifest_path, state),
        texts=texts,
        formulas=formulas,
    )
    schema["x-axiom-storage"] = {
        "document": _storage_mapping(
            artifact_paths["documents"],
            len(state.normalized_documents),
            project_root,
            record_key="document_id",
        ),
        "texts": _storage_mapping(
            artifact_paths["normalized_texts"],
            len(texts),
            project_root,
            artifact_record_count=len(all_texts),
            filter_rule={"field": "role", "operator": "not_equals", "value": "formula"},
        ),
        "formulas": _storage_mapping(
            artifact_paths["normalized_texts"],
            len(formulas),
            project_root,
            artifact_record_count=len(all_texts),
            filter_rule={"field": "role", "operator": "equals", "value": "formula"},
        ),
        "tables": _storage_mapping(
            artifact_paths["normalized_tables"],
            len(state.normalized_tables),
            project_root,
        ),
        "images": _storage_mapping(
            artifact_paths["normalized_images"],
            len(state.normalized_images),
            project_root,
        ),
    }

    provider_schema = _provider_schema_path(state, project_root)
    schema["x-axiom-provenance"] = {
        "provider_extraction_schema": _artifact_reference(provider_schema, project_root),
        "processed_manifest": _artifact_reference(manifest_path, project_root),
    }
    return schema


def _load_template() -> dict[str, Any]:
    value = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Logical document schema template must be an object: {_TEMPLATE_PATH}")
    return value


def _dataset_observations(
    state: PipelineState,
    *,
    dataset_id: str,
    texts: list[dict[str, Any]],
    formulas: list[dict[str, Any]],
) -> dict[str, Any]:
    documents = state.normalized_documents
    return {
        "dataset_id": dataset_id,
        "run_id": state.run_id,
        "record_counts": {
            "documents": len(documents),
            "texts": len(texts),
            "formulas": len(formulas),
            "tables": len(state.normalized_tables),
            "images": len(state.normalized_images),
        },
        "component_totals": {
            "physical_text_records_including_formulas": sum(
                int(document.get("component_counts", {}).get("texts") or 0)
                for document in documents
            ),
            "formulas": sum(
                int(document.get("component_counts", {}).get("formulas") or 0)
                for document in documents
            ),
            "tables": sum(
                int(document.get("component_counts", {}).get("tables") or 0)
                for document in documents
            ),
            "images": sum(
                int(document.get("component_counts", {}).get("images") or 0)
                for document in documents
            ),
        },
        "document_types": _observed_values(documents, "document_type"),
        "languages": _observed_values(documents, "language"),
        "source_formats": _observed_values(documents, "source_format"),
        "content_types": _observed_values(documents, "content_type"),
        "observed_fields": {
            "document": _observed_fields(documents),
            "text": _observed_fields(texts),
            "formula": _observed_fields(formulas),
            "table": _observed_fields(state.normalized_tables),
            "image": _observed_fields(state.normalized_images),
        },
    }


def _fallback_dataset_id(state: PipelineState) -> str:
    for value in (state.output_dir, state.input_dir, state.work_dir):
        if value:
            name = Path(str(value).replace("\\", "/")).name
            if name:
                return name
    return "unknown"


def _manifest_dataset_id(manifest_path: Path, state: PipelineState) -> str:
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and value.get("dataset_id"):
            return str(value["dataset_id"])
    return _fallback_dataset_id(state)


def _storage_mapping(
    path_value: str | Path,
    record_count: int,
    project_root: str | Path | None,
    *,
    record_key: str | None = None,
    artifact_record_count: int | None = None,
    filter_rule: dict[str, str] | None = None,
) -> dict[str, Any]:
    path = _resolved_path(path_value, project_root)
    mapping: dict[str, Any] = {
        "artifact_path": portable_path(path, project_root),
        "format": "jsonl",
        "record_count": record_count,
        "sha256": _sha256(path) if path.is_file() else None,
    }
    if record_key:
        mapping["record_key"] = record_key
    else:
        mapping["join"] = {
            "parent_artifact": "document",
            "parent_key": "document_id",
            "foreign_key": "document_id",
        }
    if artifact_record_count is not None:
        mapping["artifact_record_count"] = artifact_record_count
    if filter_rule is not None:
        mapping["filter"] = filter_rule
    return mapping


def _observed_fields(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    keys = sorted({str(key) for record in records for key in record})
    for key in keys:
        values = [record[key] for record in records if key in record]
        fields[key] = {
            "types": sorted({_json_type(value) for value in values}),
            "present_count": len(values),
            "null_count": sum(value is None for value in values),
        }
    return fields


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _observed_values(records: list[dict[str, Any]], key: str) -> list[str]:
    values = {
        str(record.get(key))
        for record in records
        if record.get(key) is not None and str(record.get(key)).strip()
    }
    return sorted(values)


def _provider_schema_path(
    state: PipelineState,
    project_root: str | Path | None,
) -> Path | None:
    parser = state.ingestion_config if isinstance(state.ingestion_config, dict) else {}
    lift = parser.get("lift_api") if isinstance(parser.get("lift_api"), dict) else {}
    value = lift.get("schema_path")
    return _resolved_path(value, project_root) if value else None


def _artifact_reference(
    path: Path | None,
    project_root: str | Path | None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "artifact_path": portable_path(path, project_root),
        "sha256": _sha256(path) if path.is_file() else None,
    }


def _resolved_path(
    value: str | Path,
    project_root: str | Path | None,
) -> Path:
    path = Path(value)
    if not path.is_absolute() and project_root:
        path = Path(project_root) / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
