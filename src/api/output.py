"""Build the HTTP response from the pipeline's existing output artifacts."""

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
    """Return the current metadata and document JSON without changing them."""
    root = Path(project_root).resolve()
    metadata_path = _required_artifact_path(state, "output_metadata", root)
    documents = [
        _read_json(
            _required_artifact_path(
                state,
                f"output_document:{data_object.object_id}",
                root,
            )
        )
        for data_object in state.data_objects
    ]
    return {
        "metadata": _read_json(metadata_path),
        "documents": documents,
    }


def _required_artifact_path(
    state: PipelineState,
    name: str,
    project_root: Path,
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
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise RuntimeError(f"Artifact path is outside the project root: {name!r}.") from None
    return resolved


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

