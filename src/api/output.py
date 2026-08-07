"""Build the stage-oriented HTTP response from final pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import PipelineState


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_dataeng_output(
    state: PipelineState,
    *,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Return common run metadata and one stage-oriented object per document."""
    root = Path(project_root).resolve()
    output_root = Path(state.output_dir)
    if not output_root.is_absolute():
        output_root = root / output_root
    allowed_roots = (root, output_root.resolve())
    metadata_path = _required_artifact_path(
        state,
        "output_metadata",
        root,
        allowed_roots=allowed_roots,
    )
    documents = [
        _read_json(
            _required_artifact_path(
                state,
                f"output_document:{data_object.object_id}",
                root,
                allowed_roots=allowed_roots,
            )
        )
        for data_object in state.data_objects
    ]
    metadata = _read_json(metadata_path)
    # Versioning is useful for persisted/internal artifacts, but is not part of
    # the public document response contract.
    metadata.pop("contract_version", None)
    metadata.pop("schema", None)
    return public_dataeng_response({
        "metadata": metadata,
        "documents": documents,
    })


def public_dataeng_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove internal envelope fields from the public API response recursively."""
    value = _without_internal_fields(payload)
    if not isinstance(value, dict):
        return {}
    metadata = value.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("schema", None)
    documents = value.get("documents")
    if isinstance(documents, list):
        for document in documents:
            if not isinstance(document, dict):
                continue
            for stage_name in ("ingest", "clean", "enrich"):
                stage = document.get(stage_name)
                if isinstance(stage, dict):
                    stage.pop("schema", None)
                    stage.pop("schema_id", None)
    return value


def _without_internal_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_internal_fields(item)
            for key, item in value.items()
            if key not in {"contract_version", "lineage"}
        }
    if isinstance(value, list):
        return [_without_internal_fields(item) for item in value]
    return value


def _required_artifact_path(
    state: PipelineState,
    name: str,
    project_root: Path,
    *,
    allowed_roots: tuple[Path, ...] | None = None,
) -> Path:
    value = state.artifact_paths.get(name)
    if not value:
        raise RuntimeError(
            f"Pipeline did not produce required API artifact {name!r}. "
            "Keep the artifacts module enabled for the REST API."
        )
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    roots = allowed_roots or (project_root,)
    if not any(_is_within(resolved, root) for root in roots):
        raise RuntimeError(f"Artifact path is outside the project root: {name!r}.") from None
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"Pipeline output artifact does not exist: {path}") from None
    except json.JSONDecodeError:
        raise RuntimeError(f"Pipeline output artifact is invalid JSON: {path}") from None
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pipeline output artifact must contain a JSON object: {path}")
    return payload
