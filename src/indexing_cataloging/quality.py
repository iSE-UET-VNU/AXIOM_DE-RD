"""Quality checks for indexing contract artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..models import IndexRecord

ALLOWED_INDEX_TYPES = {"document", "text_chunk", "table", "figure", "catalog"}
REQUIRED_RECORD_FIELDS = {"record_id", "source_object_id", "index_type", "payload", "metadata"}
REQUIRED_EMBEDDING_FIELDS = {"embedding_text", "embedding", "embedding_model", "embedding_status"}
EXPECTED_EMBEDDING_STATUS = "pending"


class IndexQualityError(RuntimeError):
    """Raised when index records fail contract validation."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(_error_message(report))


def build_index_quality_report(records: list[IndexRecord]) -> dict[str, Any]:
    """Validate indexing contract records and summarize their shape."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    counts = Counter(record.index_type for record in records)
    records_by_document: dict[str, list[IndexRecord]] = defaultdict(list)
    embedding_pending_count = 0
    records_missing_text_count = 0

    for index, record in enumerate(records):
        record_errors, record_warnings = _validate_record(index, record)
        errors.extend(record_errors)
        warnings.extend(record_warnings)
        records_by_document[record.source_object_id].append(record)

        if record.payload.get("embedding_status") == EXPECTED_EMBEDDING_STATUS:
            embedding_pending_count += 1
        if not _has_text(record.payload.get("embedding_text")):
            records_missing_text_count += 1

    for source_object_id, document_records in sorted(records_by_document.items()):
        document_count = sum(record.index_type == "document" for record in document_records)
        catalog_count = sum(record.index_type == "catalog" for record in document_records)
        if document_count != 1:
            errors.append(
                {
                    "code": "invalid_document_record_count",
                    "source_object_id": source_object_id,
                    "expected": 1,
                    "actual": document_count,
                }
            )
        if catalog_count != 1:
            errors.append(
                {
                    "code": "invalid_catalog_record_count",
                    "source_object_id": source_object_id,
                    "expected": 1,
                    "actual": catalog_count,
                }
            )

    return {
        "contract_version": "indexing-quality-v1",
        "record_count": len(records),
        "document_count": len(records_by_document),
        "counts_by_index_type": {index_type: counts.get(index_type, 0) for index_type in sorted(ALLOWED_INDEX_TYPES)},
        "embedding_pending_count": embedding_pending_count,
        "records_missing_text_count": records_missing_text_count,
        "warnings": warnings,
        "errors": errors,
        "status": "failed" if errors else "passed",
    }


def assert_index_quality(report: dict[str, Any]) -> None:
    """Fail when an index quality report contains contract errors."""
    if report.get("errors"):
        raise IndexQualityError(report)


def _validate_record(index: int, record: IndexRecord) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    record_payload = _record_payload(record)

    for field_name in sorted(REQUIRED_RECORD_FIELDS):
        if not _has_required_value(record_payload.get(field_name)):
            errors.append(_record_error("missing_required_record_field", index, record, field_name))

    if record.index_type not in ALLOWED_INDEX_TYPES:
        errors.append(
            {
                "code": "unknown_index_type",
                "record_index": index,
                "record_id": record.record_id,
                "source_object_id": record.source_object_id,
                "index_type": record.index_type,
            }
        )

    if not isinstance(record.payload, dict):
        errors.append(_record_error("invalid_payload", index, record, "payload"))
        return errors, warnings

    for field_name in sorted(REQUIRED_EMBEDDING_FIELDS):
        if field_name not in record.payload:
            errors.append(_record_error("missing_embedding_field", index, record, field_name))

    if record.payload.get("embedding_status") != EXPECTED_EMBEDDING_STATUS:
        errors.append(
            {
                "code": "invalid_embedding_status",
                "record_index": index,
                "record_id": record.record_id,
                "source_object_id": record.source_object_id,
                "field": "embedding_status",
                "expected": EXPECTED_EMBEDDING_STATUS,
                "actual": record.payload.get("embedding_status"),
            }
        )

    if "embedding_text" in record.payload and not _has_text(record.payload.get("embedding_text")):
        warnings.append(
            {
                "code": "empty_embedding_text",
                "record_index": index,
                "record_id": record.record_id,
                "source_object_id": record.source_object_id,
                "index_type": record.index_type,
            }
        )

    return errors, warnings


def _record_payload(record: IndexRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "source_object_id": record.source_object_id,
        "index_type": record.index_type,
        "payload": record.payload,
        "metadata": record.metadata,
    }


def _record_error(code: str, index: int, record: IndexRecord, field_name: str) -> dict[str, Any]:
    return {
        "code": code,
        "record_index": index,
        "record_id": record.record_id,
        "source_object_id": record.source_object_id,
        "index_type": record.index_type,
        "field": field_name,
    }


def _has_required_value(value: Any) -> bool:
    if isinstance(value, (dict, list)):
        return bool(value)
    return value not in {None, ""}


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _error_message(report: dict[str, Any]) -> str:
    errors = report.get("errors") or []
    if not errors:
        return "Index quality check failed."
    return f"Index quality check failed with {len(errors)} error(s): {errors[0].get('code', 'unknown_error')}"
