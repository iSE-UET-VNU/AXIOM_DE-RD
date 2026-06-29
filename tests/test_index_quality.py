from __future__ import annotations

import unittest

from src.indexing_cataloging.quality import (
    IndexQualityError,
    assert_index_quality,
    build_index_quality_report,
)
from src.models import IndexRecord


def _record(
    index_type: str,
    record_id: str = "record-1",
    source_object_id: str = "doc-1",
    payload: dict | None = None,
) -> IndexRecord:
    return IndexRecord(
        record_id=record_id,
        index_type=index_type,
        source_object_id=source_object_id,
        payload=payload
        or {
            "document_id": source_object_id,
            "embedding_text": "search text",
            "embedding": None,
            "embedding_model": None,
            "embedding_status": "pending",
        },
        metadata={
            "contract_version": "indexing-contract-v1",
            "component_type": index_type,
            "embedding_status": "pending",
        },
    )


class IndexQualityTests(unittest.TestCase):
    def test_valid_records_have_no_errors(self) -> None:
        records = [
            _record("document", "document-1"),
            _record("text_chunk", "chunk-1"),
            _record("catalog", "catalog-1"),
        ]

        report = build_index_quality_report(records)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["counts_by_index_type"]["document"], 1)
        self.assertEqual(report["counts_by_index_type"]["text_chunk"], 1)
        self.assertEqual(report["counts_by_index_type"]["catalog"], 1)
        self.assertEqual(report["document_count"], 1)
        self.assertEqual(report["embedding_pending_count"], 3)

    def test_missing_required_record_field_creates_error(self) -> None:
        records = [
            _record("document", "document-1"),
            _record("catalog", "catalog-1", source_object_id=""),
        ]

        report = build_index_quality_report(records)

        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(error["code"] == "missing_required_record_field" for error in report["errors"]))

    def test_unknown_index_type_creates_error(self) -> None:
        records = [
            _record("document", "document-1"),
            _record("unsupported", "unsupported-1"),
            _record("catalog", "catalog-1"),
        ]

        report = build_index_quality_report(records)

        self.assertTrue(any(error["code"] == "unknown_index_type" for error in report["errors"]))

    def test_document_without_catalog_creates_error(self) -> None:
        report = build_index_quality_report([_record("document", "document-1")])

        self.assertTrue(any(error["code"] == "invalid_catalog_record_count" for error in report["errors"]))

    def test_missing_embedding_text_creates_error(self) -> None:
        payload = {
            "document_id": "doc-1",
            "embedding": None,
            "embedding_model": None,
            "embedding_status": "pending",
        }
        records = [
            _record("document", "document-1"),
            _record("catalog", "catalog-1", payload=payload),
        ]

        report = build_index_quality_report(records)

        self.assertTrue(any(error["code"] == "missing_embedding_field" for error in report["errors"]))

    def test_assert_index_quality_raises_on_errors(self) -> None:
        report = build_index_quality_report([_record("document", "document-1")])

        with self.assertRaises(IndexQualityError):
            assert_index_quality(report)


if __name__ == "__main__":
    unittest.main()
