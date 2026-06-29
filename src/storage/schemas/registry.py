"""Schema registry artifact builder."""

from __future__ import annotations

from typing import Any

from ...models import PipelineState
from ...utils.paths import portable_path


def build_schema_registry(
    state: PipelineState,
    artifact_paths: dict[str, Any],
    project_root: Any = None,
) -> dict[str, Any]:
    """Build a lightweight schema registry manifest."""
    return {
        "contract_version": "schema-registry-v1",
        "schemas": {
            "initial_schemas": {
                "artifact_path": portable_path(artifact_paths["initial_schemas"], project_root),
                "count": len(state.initial_schemas),
            },
            "cleaned_schemas": {
                "artifact_path": portable_path(artifact_paths["cleaned_schemas"], project_root),
                "count": len(state.cleaned_schemas),
            },
            "enriched_schemas": {
                "artifact_path": portable_path(artifact_paths["enriched_schemas"], project_root),
                "count": len(state.enriched_schemas),
            },
        },
        "notes": "Schema payloads are stored in their stage artifacts. This registry only points to them.",
    }
