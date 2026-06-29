from __future__ import annotations

import unittest

from src.indexing_cataloging.embeddings import EmbeddingConfig, build_vector_records
from src.models import IndexRecord


def _record(index_type: str, record_id: str, payload: dict) -> IndexRecord:
    return IndexRecord(
        record_id=record_id,
        index_type=index_type,
        source_object_id="doc-1",
        payload=payload,
        metadata={
            "contract_version": "indexing-contract-v1",
            "component_type": index_type,
            "embedding_status": "pending",
        },
    )


class EmbeddingTests(unittest.TestCase):
    def test_openrouter_config_uses_openrouter_defaults(self) -> None:
        config = EmbeddingConfig.from_mapping({"enabled": True, "provider": "openrouter"})

        self.assertTrue(config.enabled)
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.model, "openai/text-embedding-3-small")
        self.assertEqual(config.dimension, 1536)
        self.assertEqual(config.api_key_env, "OPENROUTER_API_KEY")

    def test_local_hash_embedding_is_deterministic_and_correct_dimension(self) -> None:
        index_records = [
            _record(
                "document",
                "document-1",
                {
                    "document_id": "doc-1",
                    "source_uri": "data/raw/doc.png",
                    "title": "Title",
                    "document_type": "newspaper",
                    "language": "en",
                    "embedding_text": "document text",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
            _record(
                "text_chunk",
                "chunk-1",
                {
                    "document_id": "doc-1",
                    "chunk_id": "chunk-a",
                    "chunk_index": 0,
                    "text": "hello world",
                    "start_char": 0,
                    "end_char": 11,
                    "embedding_text": "hello world",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
            _record(
                "catalog",
                "catalog-1",
                {
                    "document_id": "doc-1",
                    "embedding_text": "catalog text",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
        ]
        config = EmbeddingConfig.from_mapping(
            {
                "enabled": True,
                "provider": "local_hash",
                "dimension": 8,
                "target_index_types": ["text_chunk"],
            }
        )

        first, first_report = build_vector_records(index_records, config)
        second, second_report = build_vector_records(index_records, config)

        self.assertEqual(first_report["status"], "passed")
        self.assertEqual(second_report["status"], "passed")
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["embedding"], second[0]["embedding"])
        self.assertEqual(first[0]["embedding_dimension"], 8)
        self.assertEqual(len(first[0]["embedding"]), 8)

    def test_only_target_index_types_are_embedded(self) -> None:
        index_records = [
            _record(
                "document",
                "document-1",
                {
                    "document_id": "doc-1",
                    "embedding_text": "document text",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
            _record(
                "text_chunk",
                "chunk-1",
                {
                    "document_id": "doc-1",
                    "chunk_id": "chunk-a",
                    "chunk_index": 0,
                    "text": "hello",
                    "embedding_text": "hello",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
        ]
        config = EmbeddingConfig.from_mapping({"enabled": True, "provider": "local_hash", "dimension": 4})

        vector_records, report = build_vector_records(index_records, config)

        self.assertEqual(len(vector_records), 1)
        self.assertEqual(vector_records[0]["record_id"], "chunk-1")
        self.assertEqual(report["eligible_count"], 1)

    def test_empty_embedding_text_is_skipped_and_reported(self) -> None:
        index_records = [
            _record(
                "text_chunk",
                "chunk-1",
                {
                    "document_id": "doc-1",
                    "chunk_id": "chunk-a",
                    "chunk_index": 0,
                    "text": "",
                    "embedding_text": "",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            )
        ]
        config = EmbeddingConfig.from_mapping({"enabled": True, "provider": "local_hash", "dimension": 4})

        vector_records, report = build_vector_records(index_records, config)

        self.assertEqual(vector_records, [])
        self.assertEqual(report["skipped_count"], 1)
        self.assertEqual(report["warnings"][0]["code"], "empty_embedding_text")

    def test_vector_record_preserves_chunk_lineage(self) -> None:
        index_records = [
            _record(
                "document",
                "document-1",
                {
                    "document_id": "doc-1",
                    "source_uri": "data/raw/doc.png",
                    "title": "Doc title",
                    "document_type": "newspaper",
                    "language": "en",
                    "embedding_text": "document text",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
            _record(
                "text_chunk",
                "chunk-1",
                {
                    "document_id": "doc-1",
                    "chunk_id": "chunk-a",
                    "chunk_index": 3,
                    "text": "chunk text",
                    "start_char": 10,
                    "end_char": 20,
                    "embedding_text": "chunk text",
                    "embedding": None,
                    "embedding_model": None,
                    "embedding_status": "pending",
                },
            ),
        ]
        config = EmbeddingConfig.from_mapping({"enabled": True, "provider": "local_hash", "dimension": 4})

        vector_records, _ = build_vector_records(index_records, config)
        vector = vector_records[0]

        self.assertEqual(vector["vector_id"], "chunk-1")
        self.assertEqual(vector["document_id"], "doc-1")
        self.assertEqual(vector["chunk_id"], "chunk-a")
        self.assertEqual(vector["chunk_index"], 3)
        self.assertEqual(vector["source_uri"], "data/raw/doc.png")
        self.assertEqual(vector["start_char"], 10)
        self.assertEqual(vector["end_char"], 20)
        self.assertEqual(vector["title"], "Doc title")
        self.assertEqual(vector["embedding_status"], "embedded")


if __name__ == "__main__":
    unittest.main()
