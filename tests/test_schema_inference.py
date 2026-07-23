from __future__ import annotations

import unittest

from src.ingestion.schema_inference import build_initial_schema
from src.models import ParsedData, ParsedTable


class SchemaInferenceTests(unittest.TestCase):
    def test_builds_main_document_schema_without_parser_tables(self) -> None:
        parsed = ParsedData(
            object_id="document-1",
            source_uri="sample.pdf",
            source_format="pdf",
            text="Document body",
            metadata={"parser": "lift-api"},
        )

        schema = build_initial_schema(parsed)

        self.assertEqual(schema.source_object_id, parsed.object_id)
        self.assertEqual(schema.fields["document.main_text"], "string")
        self.assertEqual(schema.metadata["component_counts"]["tables"], 0)
        self.assertNotIn("table.Customer", schema.fields)

    def test_adds_fields_observed_in_parsed_tables(self) -> None:
        parsed = ParsedData(
            object_id="document-2",
            source_uri="sample.xlsx",
            source_format="xlsx",
            tables=[
                ParsedTable(
                    name="Customers",
                    source_ref="sheet-1",
                    headers=["Customer", "Amount", ""],
                    rows=[["A", "10", "ignored"]],
                )
            ],
            metadata={"backend": "table"},
        )

        schema = build_initial_schema(parsed)

        self.assertEqual(schema.fields["table.Customer"], "string")
        self.assertEqual(schema.fields["table.Amount"], "string")
        self.assertNotIn("table.", schema.fields)
        self.assertEqual(schema.metadata["component_counts"]["tables"], 1)
        table_entity = next(
            entity for entity in schema.entities if entity["entity_type"] == "table"
        )
        self.assertEqual(table_entity["fields"]["table.Customer"], "string")
        self.assertEqual(table_entity["fields"]["table.Amount"], "string")


if __name__ == "__main__":
    unittest.main()
