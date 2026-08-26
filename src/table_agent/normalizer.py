"""Normalize a TableAgent API response into one AXIOM output document."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..chunking_embedding.lexical import (
    build_corpus_statistics,
    build_lexical_payload,
    content_text,
)


_TABLE_FIELDS = {"workbook", "sheet", "status", "structure"}
_METADATA_FIELDS = {"name", "description", "size_bytes"}
_RETRIEVAL_FIELDS = {
    "item_id",
    "id",
    "type",
    "component_type",
    "position",
    "content",
    "lexical",
    "embeddings",
    "embedding",
    "workbook",
    "sheet",
    "retrieval_card",
    "metadata",
    "table_id",
    "table_name",
}
_EMBEDDING_FIELDS = {
    "model",
    "embedding_model",
    "dimension",
    "values",
    "embedding",
}
_RESPONSE_FIELDS = {
    "job_id",
    "stage",
    "workbooks",
    "structures",
    "schema_artifacts",
    "metadata_artifacts",
    "answers",
    "artifacts",
    "retrieval_items",
    "retrieval",
}


def normalize_table_agent_response(
    response: dict[str, Any],
    *,
    source_uri: str,
    source_metadata: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    """Return a stage-oriented API document for a TableAgent workbook."""
    structures = response.get("structures")
    if not isinstance(structures, list):
        raise RuntimeError("TableAgent response is missing the structures array.")

    file_name = str(
        source_metadata.get("file_name")
        or Path(source_uri).name
        or "workbook"
    )
    workbook_name = _workbook_name(response, file_name)
    workbook_metadata = _workbook_metadata(response, workbook_name)
    document_id = _document_id(source_uri)
    tables = [
        _normalize_table(item, workbook_name)
        for item in structures
        if isinstance(item, dict)
    ]
    retrieval_items = _retrieval_items(response, tables)
    embedding_count = sum(
        len(item.get("embeddings", []))
        for item in retrieval_items
        if isinstance(item, dict)
    )

    blocks = [
        {
            "component_id": f"/table-agent/sheet/{index}",
            "block_index": index,
            "type": "table",
            "text": str(table.get("sheet") or table.get("workbook") or ""),
            "source": "table_agent",
        }
        for index, table in enumerate(tables)
    ]
    table_agent_annotations = _table_agent_annotations(
        response,
        workbook_name=workbook_name,
    )
    main_text = str(workbook_metadata.get("description") or "")
    extraction = {
        "main_text": main_text,
        "tables": tables,
        "figures": [],
        "formulas": [],
        "blocks": blocks,
        "reading_order": [block["component_id"] for block in blocks],
        "reading_order_meta": {
            "source": "table_agent",
            "complete": True,
            "block_count": len(blocks),
        },
    }
    stage_row = {"extraction": extraction}

    return {
        "document": _document_identity(
            document_id=document_id,
            source_uri=source_uri,
            source_metadata=source_metadata,
            workbook_name=workbook_name,
            workbook_metadata=workbook_metadata,
        ),
        "ingest": {
            "source": {
                "object_id": document_id,
                "uri": source_uri,
                "content_type": source_metadata.get("content_type")
                or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "metadata": deepcopy(source_metadata),
            },
            "data": {
                "object_id": document_id,
                "source_uri": source_uri,
                "source_format": Path(file_name).suffix.lower().lstrip(".") or "xlsx",
                "rows": [deepcopy(stage_row)],
                "text": main_text,
                "metadata": {"processor": "table_agent"},
                "tables": [],
            },
        },
        "clean": {
            "data": {
                "source_object_id": document_id,
                "rows": [deepcopy(stage_row)],
                "issues": [],
                "metadata": {
                    "pass_through": True,
                    "processor": "table_agent",
                },
            },
        },
        "enrich": {
            "data": {
                "source_object_id": document_id,
                "rows": [stage_row],
                "annotations": {"table_agent": table_agent_annotations},
                "profile": {},
                "metadata": {
                    "pass_through": True,
                    "processor": "table_agent",
                    "job_id": response.get("job_id"),
                    "stage": response.get("stage"),
                    "embedding_count": embedding_count,
                },
            },
        },
        "retrieval": {
            "items": retrieval_items,
            "lexical_stats": build_corpus_statistics(
                (
                    item.get("lexical")
                    for item in retrieval_items
                    if isinstance(item, dict)
                ),
                scope="document",
            ),
        },
    }


def _normalize_table(
    item: dict[str, Any],
    workbook_name: str,
) -> dict[str, Any]:
    table = {
        "workbook": item.get("workbook") or workbook_name,
        "sheet": item.get("sheet"),
        "status": item.get("status"),
        "structure": item.get("structure"),
    }
    extensions = _extensions(item, _TABLE_FIELDS)
    if extensions:
        table["extensions"] = extensions
    return table


def _table_agent_annotations(
    response: dict[str, Any],
    *,
    workbook_name: str,
) -> dict[str, Any]:
    annotations: dict[str, Any] = {
        "workbook_metadata": _workbook_metadata_extensions(
            response,
            workbook_name=workbook_name,
        ),
        "answers": deepcopy(response.get("answers") or []),
        "artifacts": deepcopy(response.get("artifacts") or []),
    }
    extensions = _extensions(response, _RESPONSE_FIELDS)
    workbooks = response.get("workbooks")
    if isinstance(workbooks, list) and len(workbooks) > 1:
        extensions["additional_workbooks"] = deepcopy(workbooks[1:])
    retrieval = response.get("retrieval")
    if isinstance(retrieval, dict):
        retrieval_extensions = _extensions(retrieval, {"items"})
        if retrieval_extensions:
            extensions["retrieval"] = retrieval_extensions
    if extensions:
        annotations["extensions"] = extensions
    return annotations


def _workbook_metadata_extensions(
    response: dict[str, Any],
    *,
    workbook_name: str,
) -> list[dict[str, Any]]:
    artifacts = response.get("metadata_artifacts")
    if not isinstance(artifacts, list):
        return []
    normalized: list[dict[str, Any]] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        entry: dict[str, Any] = {
            "workbook": artifact.get("workbook") or workbook_name,
        }
        metadata = artifact.get("metadata")
        if isinstance(metadata, dict):
            metadata_extensions = _extensions(metadata, _METADATA_FIELDS)
            if metadata_extensions:
                entry["metadata"] = metadata_extensions
        artifact_extensions = _extensions(artifact, {"workbook", "metadata"})
        if artifact_extensions:
            entry["extensions"] = artifact_extensions
        if len(entry) > 1:
            normalized.append(entry)
    return normalized


def _workbook_name(response: dict[str, Any], fallback: str) -> str:
    workbooks = response.get("workbooks")
    if isinstance(workbooks, list):
        for value in workbooks:
            if str(value).strip():
                return str(value).strip()
    return Path(fallback).name


def _workbook_metadata(
    response: dict[str, Any],
    workbook_name: str,
) -> dict[str, Any]:
    artifacts = response.get("metadata_artifacts")
    if not isinstance(artifacts, list):
        return {"name": workbook_name}
    fallback: dict[str, Any] | None = None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        metadata = artifact.get("metadata")
        if not isinstance(metadata, dict):
            continue
        fallback = fallback or metadata
        if str(artifact.get("workbook") or "") == workbook_name:
            return deepcopy(metadata)
    return deepcopy(fallback) if fallback else {"name": workbook_name}


def _retrieval_items(
    response: dict[str, Any],
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    supplied = response.get("retrieval_items")
    if not isinstance(supplied, list):
        retrieval = response.get("retrieval")
        supplied = retrieval.get("items") if isinstance(retrieval, dict) else None
    if isinstance(supplied, list):
        normalized = [
            _with_lexical(_normalize_retrieval_item(item, index))
            for index, item in enumerate(supplied)
            if isinstance(item, dict)
        ]
        if normalized:
            return normalized

    fallback_items = [
        {
            "item_id": ":".join(
                str(value)
                for value in (
                    table.get("workbook"),
                    table.get("sheet"),
                    index,
                )
                if value not in (None, "")
            ),
            "type": "table",
            "component_type": "table",
            "position": {
                "index": index,
                "workbook": table.get("workbook"),
                "sheet": table.get("sheet"),
            },
            "content": deepcopy(table),
            "embeddings": [],
        }
        for index, table in enumerate(tables)
    ]
    return [_with_lexical(item) for item in fallback_items]


def _normalize_retrieval_item(
    item: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    content = item.get("content")
    if not isinstance(content, dict):
        content = {
            key: item.get(key)
            for key in (
                "retrieval_card",
                "metadata",
                "table_id",
                "table_name",
            )
            if item.get(key) is not None
        }
    position = item.get("position")
    if not isinstance(position, dict):
        position = {
            "index": index,
            "workbook": item.get("workbook"),
            "sheet": item.get("sheet"),
        }
    normalized = {
        "item_id": str(item.get("item_id") or item.get("id") or index),
        "type": str(item.get("type") or "table"),
        "component_type": str(
            item.get("component_type") or item.get("type") or "table"
        ),
        "position": deepcopy(position),
        "content": deepcopy(content),
        "embeddings": _normalize_embeddings(item),
    }
    extensions = _extensions(item, _RETRIEVAL_FIELDS)
    if extensions:
        normalized["extensions"] = extensions
    return normalized


def _with_lexical(item: dict[str, Any]) -> dict[str, Any]:
    lexical = build_lexical_payload(content_text(item.get("content")))
    if lexical:
        item["lexical"] = lexical
    return item


def _normalize_embeddings(item: dict[str, Any]) -> list[dict[str, Any]]:
    embeddings = item.get("embeddings")
    if not isinstance(embeddings, list):
        embedding = item.get("embedding")
        embeddings = [embedding] if isinstance(embedding, dict) else []
    result: list[dict[str, Any]] = []
    for embedding in embeddings:
        if not isinstance(embedding, dict):
            continue
        values = embedding.get("values")
        if values is None:
            values = embedding.get("embedding")
        if not isinstance(values, list):
            continue
        normalized = {
            "model": embedding.get("model")
            or embedding.get("embedding_model"),
            "dimension": embedding.get("dimension") or len(values),
            "values": deepcopy(values),
        }
        extensions = _extensions(embedding, _EMBEDDING_FIELDS)
        if extensions:
            normalized["extensions"] = extensions
        result.append(normalized)
    return result


def _document_identity(
    *,
    document_id: str,
    source_uri: str,
    source_metadata: dict[str, Any],
    workbook_name: str,
    workbook_metadata: dict[str, Any],
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "document_id": document_id,
        "source_uri": source_uri,
        "file_name": source_metadata.get("file_name") or workbook_name,
        "content_type": source_metadata.get("response_content_type")
        or "application/octet-stream",
        "size_bytes": source_metadata.get("size_bytes")
        or workbook_metadata.get("size_bytes"),
        "sha256": source_metadata.get("sha256"),
        "title": workbook_metadata.get("name") or workbook_name,
        "document_type": "spreadsheet",
    }
    s3 = {
        "bucket": source_metadata.get("s3_bucket"),
        "key": source_metadata.get("s3_key"),
        "etag": source_metadata.get("s3_etag"),
        "version_id": source_metadata.get("s3_version_id"),
    }
    if any(value is not None for value in s3.values()):
        document["s3"] = {
            key: value for key, value in s3.items() if value is not None
        }
    return {key: value for key, value in document.items() if value is not None}


def _document_id(source_uri: str) -> str:
    import hashlib

    raw = f"data-object::{source_uri}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _extensions(
    value: dict[str, Any],
    consumed_fields: set[str],
) -> dict[str, Any]:
    return {
        str(key): deepcopy(child)
        for key, child in value.items()
        if str(key) not in consumed_fields and child is not None
    }


__all__ = ["normalize_table_agent_response"]
