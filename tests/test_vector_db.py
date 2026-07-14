from __future__ import annotations

import unittest

from src.storage.vector_db import VectorDBMock, milvus_rows_from_vector_records


class VectorDBTests(unittest.TestCase):
    def test_mock_upsert_returns_count(self) -> None:
        report = VectorDBMock(collection_name="test_collection").upsert_vectors(
            [
                {"vector_id": "v1", "embedding": [0.1, 0.2]},
                {"vector_id": "v2", "embedding": [0.2, 0.3]},
            ]
        )

        self.assertEqual(report["provider"], "mock")
        self.assertEqual(report["collection_name"], "test_collection")
        self.assertEqual(report["upserted"], 2)
        self.assertEqual(report["status"], "passed")

    def test_milvus_row_mapping_flattens_vector_record(self) -> None:
        rows = milvus_rows_from_vector_records(
            [
                {
                    "vector_id": "v1",
                    "record_id": "r1",
                    "index_type": "table",
                    "source_object_id": "doc-1",
                    "document_id": "doc-1",
                    "chunk_id": "chunk-1",
                    "chunk_index": 2,
                    "table_id": "table-1",
                    "image_id": "",
                    "source_block_id": "/page/0/Table/6",
                    "page": 0,
                    "source_uri": "data/raw/doc.png",
                    "title": "Title",
                    "document_type": "newspaper",
                    "language": "en",
                    "text": "chunk text",
                    "start_char": 10,
                    "end_char": 20,
                    "embedding": [0.1, 0.2],
                    "embedding_model": "local-hash-embedding-v1",
                }
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vector_id"], "v1")
        self.assertEqual(rows[0]["embedding"], [0.1, 0.2])
        self.assertEqual(rows[0]["index_type"], "table")
        self.assertEqual(rows[0]["chunk_index"], 2)
        self.assertEqual(rows[0]["table_id"], "table-1")
        self.assertEqual(rows[0]["source_block_id"], "/page/0/Table/6")
        self.assertEqual(rows[0]["source_uri"], "data/raw/doc.png")
        self.assertEqual(rows[0]["embedding_model"], "local-hash-embedding-v1")


if __name__ == "__main__":
    unittest.main()
