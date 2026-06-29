from __future__ import annotations

import unittest

from src.ingestion.parsing_formatting import build_initial_schema, normalize_parsed_document
from src.models import ParsedData


class ParsingFormattingTests(unittest.TestCase):
    def test_builds_canonical_initial_schema_from_lift_output(self) -> None:
        parsed = ParsedData(
            object_id="doc-1",
            source_uri="data/raw/doc.png",
            source_format="png",
            rows=[
                {
                    "extraction": {
                        "document_type": "newspaper",
                        "document_type_citations": ["/page/0/Text/1"],
                        "document_type_meta": {"verification": {"status": "PASS"}},
                        "language": "en",
                        "title": "Trade update",
                        "main_text": "Article body",
                        "tables": [{"caption": "Trade", "content": "| A | B |"}],
                        "figures": [{"caption": "Chart", "description": "Line chart"}],
                        "formulas": None,
                    },
                    "text": "Article body",
                }
            ],
            text="Article body",
            metadata={"parser": "lift-api"},
        )

        schema = build_initial_schema(parsed)

        self.assertEqual(schema.metadata["schema_source"], "format_parsing")
        self.assertEqual(schema.metadata["contract_version"], "initial-document-schema-v1")
        self.assertIn("document.document_type", schema.fields)
        self.assertIn("table.caption", schema.fields)
        self.assertNotIn("extraction.document_type_meta", schema.fields)
        self.assertNotIn("extraction.document_type_citations", schema.fields)
        self.assertEqual(schema.metadata["component_counts"]["tables"], 1)
        self.assertEqual(schema.metadata["component_counts"]["figures"], 1)
        self.assertIn("table", {entity["entity_type"] for entity in schema.entities})
        self.assertIn("has_figure", {rel["relationship_type"] for rel in schema.relationships})

    def test_normalizes_null_component_arrays_to_empty_lists(self) -> None:
        parsed = ParsedData(
            object_id="doc-1",
            source_uri="data/raw/doc.png",
            source_format="png",
            rows=[{"extraction": {"main_text": "Body", "tables": None, "figures": None, "formulas": None}}],
            metadata={"parser": "lift-api"},
        )

        document = normalize_parsed_document(parsed)

        self.assertEqual(document["tables"], [])
        self.assertEqual(document["figures"], [])
        self.assertEqual(document["formulas"], [])
        self.assertEqual(document["main_text"], "Body")


if __name__ == "__main__":
    unittest.main()
