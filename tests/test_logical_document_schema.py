from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.models import PipelineState
from src.storage.schemas import build_logical_document_schema


class LogicalDocumentSchemaTests(unittest.TestCase):
    def test_builds_logical_schema_from_canonical_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._artifact_paths(root)
            provider_schema = root / "src" / "provider.json"
            provider_schema.parent.mkdir(parents=True)
            provider_schema.write_text('{"type":"object"}', encoding="utf-8")
            state = self._state(provider_schema)

            schema = build_logical_document_schema(state, paths, project_root=root)
            expected_manifest_sha = hashlib.sha256(paths["manifest"].read_bytes()).hexdigest()

        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["type"], "object")
        self.assertNotIn("entities", schema)
        self.assertEqual(
            {"texts", "tables", "images", "formulas"},
            {name for name in schema["properties"] if name in {"texts", "tables", "images", "formulas"}},
        )
        self.assertEqual(schema["properties"]["texts"]["items"]["$ref"], "#/$defs/text")
        self.assertEqual(schema["properties"]["formulas"]["items"]["$ref"], "#/$defs/formula")
        self.assertEqual(
            schema["properties"]["contract_version"]["const"],
            "normalized-document-v1",
        )
        self.assertEqual(schema["$defs"]["formula"]["properties"]["role"], {"const": "formula"})
        self.assertEqual(
            schema["$defs"]["text"]["properties"]["role"]["not"],
            {"const": "formula"},
        )

        observed = schema["x-axiom-dataset"]
        self.assertEqual(observed["dataset_id"], "sample")
        self.assertEqual(
            observed["record_counts"],
            {"documents": 1, "texts": 1, "formulas": 1, "tables": 1, "images": 1},
        )
        self.assertEqual(observed["languages"], ["en"])
        self.assertEqual(observed["observed_fields"]["formula"]["role"]["types"], ["string"])

        storage = schema["x-axiom-storage"]
        self.assertEqual(storage["document"]["artifact_path"], "data/processed/sample/documents.jsonl")
        self.assertEqual(storage["texts"]["artifact_record_count"], 2)
        self.assertEqual(storage["texts"]["record_count"], 1)
        self.assertEqual(storage["texts"]["filter"]["operator"], "not_equals")
        self.assertEqual(storage["formulas"]["filter"]["operator"], "equals")
        self.assertEqual(storage["tables"]["join"]["foreign_key"], "document_id")

        provenance = schema["x-axiom-provenance"]
        self.assertEqual(provenance["provider_extraction_schema"]["artifact_path"], "src/provider.json")
        self.assertEqual(
            provenance["processed_manifest"]["sha256"],
            expected_manifest_sha,
        )
        serialized = json.dumps(schema, ensure_ascii=False)
        self.assertNotIn("SECRET_BODY", serialized)
        self.assertNotIn("SECRET_EMBEDDING", serialized)
        self.assertNotIn(str(root), serialized)
        self.assertNotIn("DATALAB_API_KEY", serialized)

    def test_schema_fields_cover_every_physical_record_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._artifact_paths(root)
            provider_schema = root / "src" / "provider.json"
            provider_schema.parent.mkdir(parents=True)
            provider_schema.write_text("{}", encoding="utf-8")
            state = self._state(provider_schema)
            schema = build_logical_document_schema(state, paths, project_root=root)

        document_fields = set(schema["properties"])
        self.assertTrue(set(state.normalized_documents[0]).issubset(document_fields))
        self.assertEqual(set(state.normalized_texts[0]), set(schema["$defs"]["text"]["properties"]))
        self.assertEqual(set(state.normalized_texts[1]), set(schema["$defs"]["formula"]["properties"]))
        self.assertEqual(set(state.normalized_tables[0]), set(schema["$defs"]["table"]["properties"]))
        self.assertEqual(set(state.normalized_images[0]), set(schema["$defs"]["image"]["properties"]))

    def test_empty_components_keep_complete_contracts_and_zero_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = self._artifact_paths(root)
            state = PipelineState(
                run_id="run-empty",
                input_dir="data/raw/empty",
                output_dir="data/output/empty",
            )

            schema = build_logical_document_schema(state, paths, project_root=root)

        self.assertEqual(
            schema["x-axiom-dataset"]["record_counts"],
            {"documents": 0, "texts": 0, "formulas": 0, "tables": 0, "images": 0},
        )
        self.assertEqual(set(schema["$defs"]), {"text", "formula", "table", "image"})
        self.assertEqual(schema["x-axiom-storage"]["images"]["record_count"], 0)

    def _artifact_paths(self, root: Path) -> dict[str, Path]:
        processed = root / "data" / "processed" / "sample"
        paths = {
            "documents": processed / "documents.jsonl",
            "normalized_texts": processed / "normalization" / "texts.jsonl",
            "normalized_tables": processed / "normalization" / "tables.jsonl",
            "normalized_images": processed / "normalization" / "images.jsonl",
            "manifest": processed / "manifest.json",
        }
        for name, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f'{{"artifact":"{name}"}}\n', encoding="utf-8")
        return paths

    def _state(self, provider_schema: Path) -> PipelineState:
        common_text = {
            "contract_version": "normalized-text-v1",
            "document_id": "doc-1",
            "source_uri": "data/raw/sample/report.pdf",
            "source_block_id": None,
            "page": 0,
            "text": "SECRET_BODY",
            "embedding_text": "SECRET_EMBEDDING",
            "section_path": [],
            "source_artifact": "extract.document.json",
        }
        return PipelineState(
            run_id="run-1",
            input_dir="data/raw/sample",
            output_dir="data/output/sample",
            work_dir="data/work/sample",
            normalized_documents=[
                {
                    "contract_version": "normalized-document-v1",
                    "document_id": "doc-1",
                    "source_uri": "data/raw/sample/report.pdf",
                    "source_format": "pdf",
                    "document_type": "report",
                    "language": "en",
                    "title": "Annual report",
                    "component_counts": {"texts": 2, "tables": 1, "images": 1, "formulas": 1},
                    "parser": {"provider": "lift-api", "mode": "accurate", "status": "complete"},
                    "page_count": 2,
                    "work_artifact_uri": "data/work/sample/run-1/datalab/report--doc-1",
                    "quality": {"has_text": True, "missing_image_assets": 0},
                    "file_name": "report.pdf",
                    "relative_uri": "report.pdf",
                    "content_type": "application/pdf",
                    "size_bytes": 10,
                }
            ],
            normalized_texts=[
                {**common_text, "text_id": "text-1", "role": "paragraph"},
                {**common_text, "text_id": "formula-1", "role": "formula", "text": "x = y"},
            ],
            normalized_tables=[
                {
                    "contract_version": "normalized-table-v1",
                    "table_id": "table-1",
                    "document_id": "doc-1",
                    "source_uri": "data/raw/sample/report.pdf",
                    "source_block_id": "table-block",
                    "page": 0,
                    "caption": "Table",
                    "html": "<table></table>",
                    "rows": [["A"], ["1"]],
                    "markdown": "| A |",
                    "embedding_text": "Table A",
                    "row_count": 2,
                    "column_count": 1,
                    "headers": ["A"],
                    "semantic": {},
                    "source_artifact": "extract.document.json",
                }
            ],
            normalized_images=[
                {
                    "contract_version": "normalized-image-v1",
                    "image_id": "image-1",
                    "document_id": "doc-1",
                    "source_uri": "data/raw/sample/report.pdf",
                    "source_block_id": "image-block",
                    "page": 0,
                    "image_name": "image.jpg",
                    "image_path": "data/work/sample/run-1/datalab/report--doc-1/images/image.jpg",
                    "visible_caption": "",
                    "generated_description": "Diagram",
                    "caption_is_visible": False,
                    "description_source": "vlm_generated",
                    "embedding_text": "Diagram",
                    "semantic": {},
                    "source_artifact": "extract.document.json",
                }
            ],
            ingestion_config={
                "provider": "lift_api",
                "lift_api": {
                    "api_key_env": "DATALAB_API_KEY",
                    "schema_path": str(provider_schema),
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
