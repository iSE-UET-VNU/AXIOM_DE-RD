"""Local JSON artifact storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
import json

from ..models import PipelineState
from .schemas import build_schema_registry
from ..utils.paths import portable_path, portable_path_value


@dataclass
class StorageOutput:
    artifact_paths: dict[str, str]


class LocalArtifactStore:
    def __init__(self, root: str | Path, project_root: str | Path | None = None) -> None:
        self.root = Path(root)
        self.project_root = Path(project_root).resolve() if project_root else None
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_jsonable(payload, self.project_root), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path


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

    output_store = LocalArtifactStore(output_dir, project_root=project_root)
    output_data_store = LocalArtifactStore(Path(output_dir) / "data", project_root=project_root)
    output_reports_store = LocalArtifactStore(Path(output_dir) / "reports", project_root=project_root)
    processed_store = LocalArtifactStore(processed_dir or output_dir, project_root=project_root)
    cleaned_store = LocalArtifactStore(cleaned_dir or output_dir, project_root=project_root)
    enriched_store = LocalArtifactStore(enriched_dir or output_dir, project_root=project_root)
    state.vector_db_report = _run_vector_db(state.vector_records, vector_db_config or {})

    paths: dict[str, Path] = {
        "data_objects": processed_store.write_json("data_objects.json", state.data_objects),
        "parsed_data": processed_store.write_json("parsed_data.json", state.parsed_data),
        "initial_schemas": processed_store.write_json("initial_schemas.json", state.initial_schemas),
        "cleaned_data": cleaned_store.write_json("cleaned_data.json", state.cleaned_data),
        "cleaned_schemas": cleaned_store.write_json("cleaned_schemas.json", state.cleaned_schemas),
        "enriched_data": enriched_store.write_json("enriched_data.json", state.enriched_data),
        "enriched_schemas": enriched_store.write_json("enriched_schemas.json", state.enriched_schemas),
        "documents": output_data_store.write_json("documents.json", state.documents),
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
        build_schema_registry(state, paths, project_root=project_root),
    )
    pipeline_state_path = output_reports_store.root / "pipeline_state.json"
    state.artifact_paths = {
        **{name: portable_path(path, project_root) for name, path in paths.items()},
        "pipeline_state": portable_path(pipeline_state_path, project_root),
    }
    output_reports_store.write_json("pipeline_state.json", _pipeline_state_manifest(state))
    return StorageOutput(artifact_paths=state.artifact_paths)


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
            "cleaned_data": len(state.cleaned_data),
            "cleaned_schemas": len(state.cleaned_schemas),
            "enriched_data": len(state.enriched_data),
            "enriched_schemas": len(state.enriched_schemas),
            "metadata_records": len(state.metadata_records),
            "documents": len(state.documents),
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
