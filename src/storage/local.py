"""Local JSON artifact storage."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any
import json

from ..models import PipelineState
from .schemas import build_schema_registry


@dataclass
class StorageOutput:
    artifact_paths: dict[str, str]


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_to_jsonable(payload), indent=2, sort_keys=True),
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
) -> StorageOutput:
    """Persist pipeline outputs.

    TODO: Dispatch to real DB, VectorDB, and GraphDB stores when configured.
    """
    if mode != "local":
        raise NotImplementedError(f"Only local storage mode is scaffolded, got: {mode}")

    output_store = LocalArtifactStore(output_dir)
    processed_store = LocalArtifactStore(processed_dir or output_dir)
    cleaned_store = LocalArtifactStore(cleaned_dir or output_dir)
    enriched_store = LocalArtifactStore(enriched_dir or output_dir)

    paths = {
        "data_objects": processed_store.write_json("data_objects.json", state.data_objects),
        "parsed_data": processed_store.write_json("parsed_data.json", state.parsed_data),
        "initial_schemas": processed_store.write_json("initial_schemas.json", state.initial_schemas),
        "cleaned_data": cleaned_store.write_json("cleaned_data.json", state.cleaned_data),
        "cleaned_schemas": cleaned_store.write_json("cleaned_schemas.json", state.cleaned_schemas),
        "enriched_data": enriched_store.write_json("enriched_data.json", state.enriched_data),
        "enriched_schemas": enriched_store.write_json("enriched_schemas.json", state.enriched_schemas),
        "schemas": output_store.write_json("schemas.json", build_schema_registry(state)),
        "metadata_catalog": output_store.write_json("metadata_catalog.json", state.metadata_records),
        "index_records": output_store.write_json("index_records.json", state.index_records),
        "integration_updates": output_store.write_json(
            "integration_updates.json",
            {
                "mode": "indexing-pass-through",
                "passed_index_records": state.index_records,
                "schema_matches": state.schema_matches,
                "entity_matches": state.entity_matches,
                "relationship_records": state.relationship_records,
            },
        ),
    }
    state.artifact_paths = {name: str(path) for name, path in paths.items()}
    state.artifact_paths["pipeline_state"] = str(output_store.write_json("pipeline_state.json", state))
    return StorageOutput(artifact_paths=state.artifact_paths)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value
