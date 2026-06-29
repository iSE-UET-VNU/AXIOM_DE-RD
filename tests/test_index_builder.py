from __future__ import annotations

import unittest

from src.indexing_cataloging.index_builder import build_index_records, chunk_text, normalize_document
from src.models import EnrichedData, MetadataRecord


class NormalizeDocumentTests(unittest.TestCase):
    def test_normalizes_lift_document_components(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[
                {
                    "extraction": {
                        "document_type": "newspaper",
                        "language": "en",
                        "title": "Trade update",
                        "main_text": "A long article body.",
                        "tables": [{"caption": "Trade", "content": "| A | B |"}],
                        "figures": [{"caption": "Chart", "description": "A line chart."}],
                        "formulas": ["x = y"],
                    }
                }
            ],
            metadata={"source_uri": "data/raw/doc.png"},
        )

        document = normalize_document(data)

        self.assertEqual(document.document_type, "newspaper")
        self.assertEqual(document.language, "en")
        self.assertEqual(document.title, "Trade update")
        self.assertEqual(document.main_text, "A long article body.")
        self.assertEqual(len(document.tables), 1)
        self.assertEqual(len(document.figures), 1)
        self.assertEqual(len(document.formulas), 1)
        self.assertEqual(document.source_uri, "data/raw/doc.png")

    def test_normalizes_null_component_arrays_to_empty_lists(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[
                {
                    "extraction": {
                        "main_text": "Text",
                        "tables": None,
                        "figures": None,
                        "formulas": None,
                    }
                }
            ],
        )

        document = normalize_document(data)

        self.assertEqual(document.tables, [])
        self.assertEqual(document.figures, [])
        self.assertEqual(document.formulas, [])

    def test_supports_legacy_markdown_extraction(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[{"extraction": {"markdown": "# Title\n\nBody"}}],
        )

        document = normalize_document(data)

        self.assertEqual(document.main_text, "# Title\n\nBody")

    def test_falls_back_to_row_text_when_extraction_text_is_missing(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[{"extraction": {"title": "Untitled"}, "text": "Row text"}],
        )

        document = normalize_document(data)

        self.assertEqual(document.main_text, "Row text")


class ChunkTextTests(unittest.TestCase):
    def test_short_text_creates_one_chunk(self) -> None:
        chunks = chunk_text("short text", chunk_size=1200, overlap=150)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "short text")
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[0].end_char, len("short text"))

    def test_long_text_creates_overlapping_chunks(self) -> None:
        text = "a" * 2500
        chunks = chunk_text(text, chunk_size=1200, overlap=150)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertEqual(chunks[0].end_char, 1200)
        self.assertEqual(chunks[1].start_char, 1050)
        self.assertEqual(chunks[1].end_char, 2250)
        self.assertEqual(chunks[2].start_char, 2100)
        self.assertEqual(chunks[2].end_char, 2500)

    def test_empty_text_creates_no_chunks(self) -> None:
        self.assertEqual(chunk_text(""), [])


class BuildIndexRecordsTests(unittest.TestCase):
    def test_builds_component_index_record_types(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[
                {
                    "extraction": {
                        "document_type": "newspaper",
                        "language": "en",
                        "title": "Trade update",
                        "main_text": "A long article body.",
                        "tables": [{"caption": "Trade", "content": "| A | B |"}],
                        "figures": [{"caption": "Chart", "description": "A line chart."}],
                        "formulas": None,
                    }
                }
            ],
        )
        metadata_record = MetadataRecord(
            record_id="metadata-1",
            source_object_id="doc-1",
            title="Data object doc-1",
            schema_id="schema-1",
            tags=["auto-cataloged"],
        )

        records = build_index_records([data], [metadata_record])
        index_types = [record.index_type for record in records]

        self.assertEqual(index_types, ["document", "text_chunk", "table", "figure", "catalog"])
        self.assertTrue(all(record.record_id for record in records))
        self.assertTrue(all(record.source_object_id == "doc-1" for record in records))
        self.assertTrue(all(record.payload for record in records))
        self.assertTrue(all("embedding_text" in record.payload for record in records))
        self.assertTrue(all("embedding" in record.payload for record in records))
        self.assertTrue(all("embedding_model" in record.payload for record in records))
        self.assertTrue(all(record.payload["embedding_status"] == "pending" for record in records))
        self.assertTrue(all(record.metadata["contract_version"] == "indexing-contract-v1" for record in records))
        self.assertEqual(records[-1].payload["metadata_record_id"], "metadata-1")

    def test_document_without_text_skips_text_chunks(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[{"extraction": {"title": "No text", "tables": [], "figures": [], "formulas": []}}],
        )

        records = build_index_records([data], [])

        self.assertEqual([record.index_type for record in records], ["document", "catalog"])


if __name__ == "__main__":
    unittest.main()
