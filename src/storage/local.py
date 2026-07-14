"""Local JSON artifact storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any
import json

from ..models import PipelineState
from .schemas import build_logical_document_schema
from ..utils.paths import portable_path, portable_path_value


@dataclass
class StorageOutput:
    artifact_paths: dict[str, str]


class LocalArtifactStore:
    def __init__(self, root: str | Path, project_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.project_root = Path(project_root).resolve() if project_root else None
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        name: str,
        payload: Any,
        *,
        sort_keys: bool = True,
    ) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            _to_jsonable(payload, self.project_root),
            indent=2,
            sort_keys=sort_keys,
            ensure_ascii=False,
        )
        _atomic_write_text(path, text)
        return path

    def write_jsonl(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = _to_jsonable(payload, self.project_root)
        if not isinstance(rows, list):
            raise TypeError(f"JSONL payload must be a list, got: {type(rows).__name__}")
        lines = [
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        ]
        _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
        return path


def write_processed_artifacts(
    state: PipelineState,
    processed_dir: str | Path,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write the canonical processed document registry and normalized components."""
    store = LocalArtifactStore(processed_dir, project_root=project_root)
    paths = {
        "documents": store.write_jsonl("documents.jsonl", state.normalized_documents),
        "normalized_texts": store.write_jsonl("normalization/texts.jsonl", state.normalized_texts),
        "normalized_tables": store.write_jsonl("normalization/tables.jsonl", state.normalized_tables),
        "normalized_images": store.write_jsonl("normalization/images.jsonl", state.normalized_images),
    }
    manifest = _processed_manifest(state, store.root, paths, project_root)
    paths["manifest"] = store.write_json("manifest.json", manifest)
    return paths


def run(
    state: PipelineState,
    output_dir: str | Path,
    mode: str = "local",
    processed_dir: str | Path | None = None,
    cleaned_dir: str | Path | None = None,
    enriched_dir: str | Path | None = None,
    vector_db_config: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> StorageOutput:
    """Persist local artifacts and optionally upsert text chunk vectors."""
    if mode != "local":
        raise NotImplementedError(f"Only local storage mode is scaffolded, got: {mode}")

    output_data_store = LocalArtifactStore(Path(output_dir) / "data", project_root=project_root)
    output_reports_store = LocalArtifactStore(Path(output_dir) / "reports", project_root=project_root)
    cleaned_store = LocalArtifactStore(cleaned_dir or output_dir, project_root=project_root)
    enriched_store = LocalArtifactStore(enriched_dir or output_dir, project_root=project_root)
    state.vector_db_report = _run_vector_db(state.vector_records, vector_db_config or {})

    paths: dict[str, Path] = {
        **write_processed_artifacts(state, processed_dir or output_dir, project_root),
        "cleaned_data": cleaned_store.write_json("cleaned_data.json", state.cleaned_data),
        "cleaned_schemas": cleaned_store.write_json("cleaned_schemas.json", state.cleaned_schemas),
        "enriched_data": enriched_store.write_json("enriched_data.json", state.enriched_data),
        "enriched_schemas": enriched_store.write_json("enriched_schemas.json", state.enriched_schemas),
        "metadata_catalog": output_data_store.write_json("metadata_catalog.json", state.metadata_records),
        "index_records": output_data_store.write_json("index_records.json", state.index_records),
        "vector_records": output_data_store.write_json("vector_records.json", state.vector_records),
        "index_quality_report": output_reports_store.write_json("index_quality_report.json", state.index_quality_report),
        "embedding_report": output_reports_store.write_json("embedding_report.json", state.embedding_report),
        "vector_db_report": output_reports_store.write_json("vector_db_report.json", state.vector_db_report),
        "integration_updates": output_reports_store.write_json(
            "integration_updates.json",
            {
                "mode": "indexing-pass-through",
                "passed_index_record_count": len(state.index_records),
                "schema_matches": state.schema_matches,
                "entity_matches": state.entity_matches,
                "relationship_records": state.relationship_records,
            },
        ),
    }
    paths["schemas"] = output_data_store.write_json(
        "schemas.json",
        build_logical_document_schema(state, paths, project_root=project_root),
        sort_keys=False,
    )
    pipeline_state_path = output_reports_store.root / "pipeline_state.json"
    state.artifact_paths = {
        **{name: portable_path(path, project_root) for name, path in paths.items()},
        "pipeline_state": portable_path(pipeline_state_path, project_root),
    }
    output_reports_store.write_json("pipeline_state.json", _pipeline_state_manifest(state))
    return StorageOutput(artifact_paths=state.artifact_paths)


def _processed_manifest(
    state: PipelineState,
    processed_dir: Path,
    artifact_paths: dict[str, Path],
    project_root: str | Path | None,
) -> dict[str, Any]:
    parser_config = state.ingestion_config if isinstance(state.ingestion_config, dict) else {}
    lift_config = parser_config.get("lift_api") if isinstance(parser_config.get("lift_api"), dict) else {}
    schema_path = Path(str(lift_config.get("schema_path"))) if lift_config.get("schema_path") else None
    work_root = None
    if state.work_dir:
        work_base = Path(state.work_dir)
        if not work_base.is_absolute() and project_root:
            work_base = Path(project_root) / work_base
        work_root = work_base / state.run_id / "datalab"

    failed_documents = [
        str(document.get("document_id"))
        for document in state.normalized_documents
        if str(document.get("parser", {}).get("status") or "").lower()
        not in {"complete", "completed", "success", "succeeded"}
    ]
    empty_text_documents = [
        str(document.get("document_id"))
        for document in state.normalized_documents
        if not bool(document.get("quality", {}).get("has_text"))
    ]
    missing_assets = sum(
        int(document.get("quality", {}).get("missing_image_assets") or 0)
        for document in state.normalized_documents
    )
    status = "complete"
    if failed_documents or empty_text_documents or missing_assets or state.errors:
        status = "partial"

    artifacts: dict[str, dict[str, Any]] = {}
    records_by_name = {
        "documents": state.normalized_documents,
        "normalized_texts": state.normalized_texts,
        "normalized_tables": state.normalized_tables,
        "normalized_images": state.normalized_images,
    }
    for name, path in artifact_paths.items():
        artifacts[name] = {
            "path": portable_path(path, project_root),
            "format": "jsonl",
            "record_count": len(records_by_name[name]),
            "byte_size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    contracts = {
        "documents": _first_contract_version(state.normalized_documents),
        "texts": _first_contract_version(state.normalized_texts),
        "tables": _first_contract_version(state.normalized_tables),
        "images": _first_contract_version(state.normalized_images),
    }
    return {
        "contract_version": "processed-manifest-v1",
        "dataset_id": processed_dir.name,
        "run_id": state.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_dir": state.input_dir,
        "work_dir": portable_path(work_root, project_root) if work_root else None,
        "parser": {
            "provider": parser_config.get("provider"),
            "mode": lift_config.get("mode"),
            "extract_images": lift_config.get("extract_images"),
            "save_raw_outputs": lift_config.get("save_raw_outputs"),
            "schema_ref": portable_path(schema_path, project_root) if schema_path else None,
            "schema_sha256": _sha256_file(schema_path) if schema_path and schema_path.is_file() else None,
        },
        "contracts": contracts,
        "artifacts": artifacts,
        "quality": {
            "failed_document_ids": failed_documents,
            "empty_text_document_ids": empty_text_documents,
            "missing_image_assets": missing_assets,
            "error_count": len(state.errors),
        },
    }


def _first_contract_version(records: list[dict[str, Any]]) -> str | None:
    for record in records:
        version = record.get("contract_version")
        if version:
            return str(version)
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _run_vector_db(vector_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    provider = str(config.get("provider", "disabled"))
    collection_name = str(config.get("collection_name", "axiom_text_chunks"))

    if not vector_records:
        return {
            "provider": provider,
            "collection_name": collection_name,
            "status": "skipped",
            "upserted": 0,
            "reason": "no_vector_records",
            "errors": [],
            "warnings": [],
        }

    if provider in {"", "disabled", "none"}:
        return {
            "provider": provider,
            "collection_name": collection_name,
            "status": "skipped",
            "upserted": 0,
            "reason": "vector_db_provider_not_configured",
            "errors": [],
            "warnings": [],
        }

    if provider == "mock":
        from .vector_db import VectorDBMock

        return VectorDBMock(collection_name=collection_name).upsert_vectors(vector_records)

    if provider == "milvus":
        from .vector_db import MilvusConfig, MilvusVectorDB

        milvus_config = dict(config)
        milvus_config.setdefault("dimension", _infer_vector_dimension(vector_records))
        return MilvusVectorDB(MilvusConfig.from_mapping(milvus_config)).upsert_vectors(vector_records)

    raise ValueError(f"Unsupported vector_db provider: {provider}")


def _infer_vector_dimension(vector_records: list[dict[str, Any]]) -> int:
    for record in vector_records:
        embedding = record.get("embedding")
        if isinstance(embedding, list):
            return len(embedding)
    raise ValueError("Cannot infer vector dimension from vector records.")


def _pipeline_state_manifest(state: PipelineState) -> dict[str, Any]:
    return {
        "contract_version": "pipeline-state-manifest-v1",
        "run_id": state.run_id,
        "input_dir": state.input_dir,
        "output_dir": state.output_dir,
        "completed_modules": state.completed_modules,
        "artifact_paths": state.artifact_paths,
        "counts": {
            "data_objects": len(state.data_objects),
            "parsed_data": len(state.parsed_data),
            "initial_schemas": len(state.initial_schemas),
            "normalized_texts": len(state.normalized_texts),
            "normalized_images": len(state.normalized_images),
            "normalized_tables": len(state.normalized_tables),
            "normalized_documents": len(state.normalized_documents),
            "cleaned_data": len(state.cleaned_data),
            "cleaned_schemas": len(state.cleaned_schemas),
            "enriched_data": len(state.enriched_data),
            "enriched_schemas": len(state.enriched_schemas),
            "metadata_records": len(state.metadata_records),
            "index_records": len(state.index_records),
            "vector_records": len(state.vector_records),
            "schema_matches": len(state.schema_matches),
            "entity_matches": len(state.entity_matches),
            "relationship_records": len(state.relationship_records),
            "errors": len(state.errors),
        },
        "reports": {
            "index_quality_status": state.index_quality_report.get("status"),
            "embedding_status": state.embedding_report.get("status"),
            "vector_db_status": state.vector_db_report.get("status"),
        },
    }


def _to_jsonable(value: Any, project_root: str | Path | None = None) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value), project_root)
    if isinstance(value, Path):
        return portable_path(value, project_root)
    converted = portable_path_value(value, project_root)
    if converted is not value:
        return converted
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item, project_root) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item, project_root) for item in value]
    return value
