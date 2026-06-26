"""Schema registry artifact builder."""

from __future__ import annotations

from typing import Any

from ...models import PipelineState


def build_schema_registry(state: PipelineState) -> dict[str, Any]:
    """Group all schema artifacts into one local registry payload."""
    return {
        "initial_schemas": state.initial_schemas,
        "cleaned_schemas": state.cleaned_schemas,
        "enriched_schemas": state.enriched_schemas,
    }
