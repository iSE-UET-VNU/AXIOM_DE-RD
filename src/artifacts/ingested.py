"""Per-document ingestion artifact writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import PipelineState
from .manifests import build_stage_metadata
from .writer import LocalArtifactWriter


INGESTED_DOCUMENT_SCHEMA = {
    "contract_version": "string",
    "stage": "ingested",
    "status": "succeeded|quarantined",
    "document_id": "string",
    "source": "DataObject",
    "parsed": "ParsedData|null",
    "schema_id": "string|null",
    "failure": "object|null",
}


def write_ingested_artifacts(
    state: PipelineState,
    ingested_dir: str | Path,
    project_root: str | Path | None = None,
    *,
    status: str = "completed",
    expected_document_count: int | None = None,
    document_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Write selected canonical documents and refresh common metadata."""
    writer = LocalArtifactWriter(ingested_dir, project_root=project_root)
    parsed_by_id = {record.object_id: record for record in state.parsed_data}
    schema_by_id = {record.source_object_id: record for record in state.initial_schemas}
    quarantined_by_id = {
        record.document_id: record for record in state.quarantined_documents
    }
    selected_ids = set(document_ids) if document_ids is not None else None
    paths: dict[str, Path] = {}
    summaries: list[dict[str, Any]] = []

    for data_object in state.data_objects:
        parsed = parsed_by_id.get(data_object.object_id)
        schema = schema_by_id.get(data_object.object_id)
        if selected_ids is None or data_object.object_id in selected_ids:
            paths[f"ingested_document:{data_object.object_id}"] = writer.write_json(
                f"documents/{data_object.object_id}.json",
                {
                    "contract_version": "ingested-document-v2",
                    "stage": "ingested",
                    "status": "succeeded",
                    "document_id": data_object.object_id,
                    "source": data_object,
                    "parsed": parsed,
                    "schema_id": schema.schema_id if schema else None,
                    "failure": None,
                },
                sort_keys=False,
            )
        summaries.append(_document_summary(data_object, status="succeeded"))

    for document_id, quarantined in quarantined_by_id.items():
        if selected_ids is None or document_id in selected_ids:
            paths[f"ingested_document:{document_id}"] = writer.write_json(
                f"documents/{document_id}.json",
                {
                    "contract_version": "ingested-document-v2",
                    "stage": "ingested",
                    "status": "quarantined",
                    "document_id": document_id,
                    "source": quarantined.source,
                    "parsed": quarantined.parsed,
                    "schema_id": None,
                    "failure": {
                        "stage": quarantined.stage,
                        "disposition": "quarantined",
                        "reasons": quarantined.reasons,
                    },
                },
                sort_keys=False,
            )
        summaries.append(
            _document_summary(
                quarantined.source,
                status="quarantined",
                reasons=quarantined.reasons,
            )
        )

    expected_count = (
        expected_document_count
        if expected_document_count is not None
        else len(summaries)
    )
    metadata = build_stage_metadata(
        state,
        stage="ingested",
        document_summaries=summaries,
        schemas=state.initial_schemas,
        document_schema=INGESTED_DOCUMENT_SCHEMA,
        extra={
            "progress": {
                "status": status,
                "processed_document_count": len(summaries),
                "expected_document_count": expected_count,
                "succeeded_document_count": len(state.data_objects),
                "quarantined_document_count": len(state.quarantined_documents),
            },
            "parser": _parser_metadata(state.ingestion_config),
        },
    )
    paths["ingested_metadata"] = writer.write_json(
        "metadata.json", metadata, sort_keys=False
    )
    return paths


def _document_summary(
    data_object: Any,
    *,
    status: str,
    reasons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = {
        "document_id": data_object.object_id,
        "status": status,
        "source_uri": data_object.uri,
        "content_type": data_object.content_type,
        "file_name": data_object.metadata.get("file_name"),
        "size_bytes": data_object.metadata.get("size_bytes"),
    }
    if reasons:
        summary["failure_codes"] = [reason.get("code") for reason in reasons]
    return summary


def _parser_metadata(config: dict[str, Any]) -> dict[str, Any]:
    lift = config.get("lift_api") if isinstance(config.get("lift_api"), dict) else {}
    return {
        "provider": config.get("provider"),
        "mode": lift.get("mode"),
        "extract_images": lift.get("extract_images"),
        "save_raw_outputs": lift.get("save_raw_outputs"),
    }
