"""Shared metadata builders for per-document stage artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..models import PipelineState


def build_stage_metadata(
    state: PipelineState,
    *,
    stage: str,
    document_summaries: list[dict[str, Any]],
    schemas: Any,
    document_schema: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the single common metadata file for one run stage."""
    payload: dict[str, Any] = {
        "contract_version": "stage-metadata-v1",
        "run_id": state.run_id,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_source": state.input_source,
        "document_count": len(document_summaries),
        "documents": document_summaries,
        "schema": {
            "document_file": document_schema,
            "records": schemas,
        },
        "errors": state.errors,
    }
    if extra:
        payload.update(extra)
    return payload
