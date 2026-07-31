from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import cleaning, enrichment, indexing_cataloging
from src.artifacts import write_ingested_artifacts
from src.ingestion.parsing.chandra2 import Chandra2Provider
from src.ingestion.parsing.lift import LiftAPIParserClient
from src.ingestion.parsing.service import ParsingService
from src.ingestion.runner import run as run_ingestion
from src.models import ParsedData, PipelineState


class IngestionRouterIntegrationTests(unittest.TestCase):
    def test_lift_provider_routes_markdown_to_local_text_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "README.md"
            path.write_text("# Routed locally\n\nContent.", encoding="utf-8")

            with patch.object(
                LiftAPIParserClient,
                "parse_file",
                side_effect=AssertionError("Lift must not receive Markdown"),
            ):
                output = run_ingestion(
                    path,
                    parser_config={"provider": "lift_api"},
                    project_root=root,
                )

        self.assertEqual(len(output.parsed_data), 1)
        parsed = output.parsed_data[0]
        self.assertEqual(parsed.metadata["parser"], "text")
        self.assertEqual(parsed.metadata["backend"], "text")
        self.assertEqual(parsed.text, "# Routed locally\n\nContent.")

    def test_table_formats_are_not_registered_in_main_router(self) -> None:
        router = ParsingService.from_config({"provider": "chandra2"}).router

        for file_name in ("records.csv", "legacy.xls", "records.xlsx"):
            with self.subTest(file_name=file_name):
                self.assertIsNone(router.resolve(Path(file_name)))

    def test_top_level_lift_provider_still_handles_visual_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "sample.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            def fake_parse(
                _client: LiftAPIParserClient,
                source: str | Path,
                data_object: object,
            ) -> ParsedData:
                return ParsedData(
                    object_id=data_object.object_id,
                    source_uri=data_object.uri,
                    source_format="pdf",
                    rows=[{"extraction": {"main_text": "Lift output"}}],
                    text="Lift output",
                    metadata={"parser": "lift-api"},
                )

            with patch.object(
                LiftAPIParserClient,
                "parse_file",
                autospec=True,
                side_effect=fake_parse,
            ) as parse_file:
                output = run_ingestion(
                    path,
                    parser_config={"provider": "lift_api"},
                    project_root=root,
                )

        parse_file.assert_called_once()
        self.assertEqual(output.parsed_data[0].text, "Lift output")
        self.assertEqual(output.parsed_data[0].metadata["backend"], "lift_api")

    def test_provider_failure_is_quarantined_without_crashing_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "sample.pdf"
            path.write_bytes(b"%PDF-1.4\n")

            with patch.object(
                Chandra2Provider,
                "parse_file",
                side_effect=RuntimeError("runtime unavailable"),
            ):
                output = run_ingestion(
                    path,
                    parser_config={"provider": "chandra2"},
                    project_root=root,
                )

        self.assertEqual(output.data_objects, [])
        self.assertEqual(output.parsed_data, [])
        self.assertEqual(len(output.quarantined_documents), 1)
        quarantined = output.quarantined_documents[0]
        self.assertEqual(quarantined.reasons[0]["code"], "parse_failed")
        self.assertEqual(quarantined.reasons[0]["backend"], "chandra2")
        self.assertEqual(quarantined.parsed.metadata["status"], "failed")

    def test_ingested_document_contract_is_unchanged_after_routing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "note.txt"
            path.write_text("Stable ingestion output.", encoding="utf-8")
            output = run_ingestion(
                path,
                parser_config={"provider": "lift_api"},
                project_root=root,
            )
            state = PipelineState(
                run_id="router-test",
                input_source=path.as_posix(),
                embedded_dir="embedded",
                output_dir="output",
                data_objects=output.data_objects,
                parsed_data=output.parsed_data,
                initial_schemas=output.initial_schemas,
                ingestion_config={"provider": "lift_api"},
            )
            artifact_dir = root / "ingested"
            write_ingested_artifacts(state, artifact_dir, project_root=root)
            document_path = next((artifact_dir / "documents").glob("*.json"))
            payload = json.loads(document_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["contract_version"], "ingested-document-v2")
        self.assertEqual(payload["stage"], "ingested")
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(
            set(payload),
            {
                "contract_version",
                "stage",
                "status",
                "document_id",
                "source",
                "parsed",
                "schema_id",
                "failure",
            },
        )
        self.assertEqual(
            set(payload["parsed"]),
            {
                "object_id",
                "source_uri",
                "source_format",
                "rows",
                "text",
                "metadata",
                "tables",
            },
        )

    def test_json_text_reaches_chunks_and_local_hash_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "states.json"
            raw = '{"VN": "Vietnam", "JP": "Japan"}'
            path.write_text(raw, encoding="utf-8")

            ingested = run_ingestion(
                path,
                parser_config={"provider": "chandra2"},
                project_root=root,
            )
            cleaned = cleaning.run(
                ingested.parsed_data,
                ingested.initial_schemas,
            )
            enriched = enrichment.run(
                cleaned.cleaned_data,
                cleaned.cleaned_schemas,
            )
            indexed = indexing_cataloging.run(
                enriched.enriched_data,
                enriched.enriched_schemas,
                indexing_config={
                    "embeddings": {
                        "enabled": True,
                        "provider": "local_hash",
                        "dimension": 16,
                        "target_index_types": ["text_chunk"],
                    }
                },
            )

        chunks = [
            record
            for record in indexed.index_records
            if record.index_type == "text_chunk"
        ]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].payload["text"], raw)
        self.assertEqual(len(indexed.vector_records), 1)
        self.assertEqual(indexed.embedding_report["status"], "passed")


class DocumentProviderConfigTests(unittest.TestCase):
    def test_top_level_chandra_provider_only_changes_document_route(self) -> None:
        router = ParsingService.from_config({"provider": "chandra2"}).router

        self.assertEqual(router.resolve(Path("scan.pdf")).provider_name, "chandra2")
        self.assertEqual(router.resolve(Path("note.md")).backend_name, "text")
        self.assertIsNone(router.resolve(Path("book.xlsx")))

    def test_legacy_router_config_remains_supported(self) -> None:
        router = ParsingService.from_config(
            {
                "provider": "router",
                "document": {"provider": "chandra2"},
            }
        ).router

        self.assertEqual(router.resolve(Path("scan.pdf")).provider_name, "chandra2")


if __name__ == "__main__":
    unittest.main()
