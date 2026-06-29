from __future__ import annotations

import unittest

from src.indexing_cataloging.documents import build_documents_artifact
from src.indexing_cataloging.index_builder import build_index_records
from src.indexing_cataloging.metadata_catalog import build_metadata_catalog
from src.models import EnrichedData, EnrichedSchema


class DocumentsArtifactTests(unittest.TestCase):
    def test_builds_document_payload_with_elements_and_chunks(self) -> None:
        data = EnrichedData(
            source_object_id="doc-1",
            rows=[
                {
                    "extraction": {
                        "document_type": "report",
                        "language": "en",
                        "title": "Revenue",
                        "main_text": "hello world",
                        "tables": [{"caption": "T1", "content": "| a |"}],
                        "figures": [{"caption": "F1", "description": "chart"}],
                        "formulas": ["x = y"],
                    }
                }
            ],
            metadata={"source_uri": "data/raw/doc.png"},
        )
        enriched_schema = EnrichedSchema(
            schema_id="schema-1",
            source_schema_id="clean-schema-1",
            source_object_id="doc-1",
        )
        metadata_records = build_metadata_catalog([data], [enriched_schema])
        index_records = build_index_records([data], metadata_records)

        documents = build_documents_artifact([data], index_records)

        self.assertEqual(len(documents), 1)
        document = documents[0]
        self.assertEqual(document["contract_version"], "documents-v1")
        self.assertEqual(document["document_id"], "doc-1")
        self.assertEqual(document["main_text"], "hello world")
        self.assertEqual(len(document["elements"]["tables"]), 1)
        self.assertEqual(len(document["elements"]["figures"]), 1)
        self.assertEqual(len(document["elements"]["formulas"]), 1)
        self.assertEqual(len(document["text_chunks"]), 1)
        self.assertIn("document", document["index_record_ids"])
        self.assertIn("text_chunk", document["index_record_ids"])


if __name__ == "__main__":
    unittest.main()
