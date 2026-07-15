from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from src.ingestion.runner import enrich_document_records
from src.models import DataObject, PipelineState
from src.storage.local import run as run_storage
from src.utils.paths import portable_path, portable_path_value


class PortablePathTests(unittest.TestCase):
    def test_portable_path_makes_project_paths_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "data" / "output" / "artifact.json"

            self.assertEqual(portable_path(path, root), "data/output/artifact.json")

    def test_portable_path_value_leaves_non_paths_unchanged(self) -> None:
        self.assertEqual(portable_path_value("openrouter", Path.cwd()), "openrouter")

    def test_document_relative_uri_uses_posix_separators(self) -> None:
        records = enrich_document_records(
            [{"document_id": "doc-1"}],
            [
                DataObject(
                    object_id="doc-1",
                    uri="data/raw/sample/nested/file.pdf",
                    metadata={"relative_uri": "nested\\file.pdf"},
                )
            ],
        )

        self.assertEqual(records[0]["relative_uri"], "nested/file.pdf")

    def test_storage_pipeline_state_uses_relative_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_path = root / "src" / "schema.json"
            schema_path.parent.mkdir()
            schema_path.write_text("{}", encoding="utf-8")
            state = PipelineState(
                run_id="run-1",
                input_dir="data/raw/final_test",
                output_dir="data/output/final_test",
                work_dir="data/work/final_test",
                normalized_documents=[
                    {
                        "contract_version": "normalized-document-v1",
                        "document_id": "doc-1",
                        "quality": {"has_text": True, "missing_image_assets": 0},
                        "parser": {"status": "complete"},
                    }
                ],
                normalized_texts=[
                    {"contract_version": "normalized-text-v1", "document_id": "doc-1", "text": "xin chào"}
                ],
                vector_records=[
                    {
                        "record_id": "chunk-1",
                        "embedding": [0.1, 0.2],
                        "embedding_dimension": 2,
                    }
                ],
            )
            state.ingestion_config = {
                "provider": "lift_api",
                "lift_api": {
                    "api_key_env": "DATALAB_API_KEY",
                    "mode": "accurate",
                    "schema_path": str(schema_path),
                    "extract_images": True,
                    "save_raw_outputs": True,
                },
            }

            run_storage(
                state,
                root / "data" / "output" / "final_test",
                processed_dir=root / "data" / "processed" / "final_test",
                cleaned_dir=root / "data" / "cleaned" / "final_test",
                enriched_dir=root / "data" / "enriched" / "final_test",
                project_root=root,
            )

            self.assertEqual(
                state.artifact_paths["pipeline_state"],
                "data/output/final_test/reports/pipeline_state.json",
            )
            self.assertEqual(
                state.artifact_paths["documents"],
                "data/processed/final_test/documents.jsonl",
            )
            self.assertEqual(
                state.artifact_paths["index_records"],
                "data/output/final_test/data/index_records.json",
            )
            self.assertEqual(
                state.artifact_paths["vector_records"],
                "data/output/final_test/data/vector_records.json",
            )
            self.assertEqual(
                state.artifact_paths["normalized_texts"],
                "data/processed/final_test/normalization/texts.jsonl",
            )
            self.assertEqual(
                state.artifact_paths["normalized_images"],
                "data/processed/final_test/normalization/images.jsonl",
            )
            self.assertEqual(
                state.artifact_paths["normalized_tables"],
                "data/processed/final_test/normalization/tables.jsonl",
            )
            self.assertEqual(
                state.artifact_paths["manifest"],
                "data/processed/final_test/manifest.json",
            )
            manifest = json.loads(
                (root / "data" / "processed" / "final_test" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["work_dir"],
                "data/work/final_test/run-1/datalab",
            )
            self.assertEqual(manifest["parser"]["schema_ref"], "src/schema.json")
            self.assertEqual(manifest["artifacts"]["documents"]["record_count"], 1)
            documents_path = root / manifest["artifacts"]["documents"]["path"]
            self.assertEqual(
                manifest["artifacts"]["documents"]["sha256"],
                hashlib.sha256(documents_path.read_bytes()).hexdigest(),
            )
            manifest_text = json.dumps(manifest)
            self.assertNotIn(str(root), manifest_text)
            self.assertNotIn("DATALAB_API_KEY", manifest_text)
            output_schema_path = (
                root / "data" / "output" / "final_test" / "data" / "schemas.json"
            )
            output_schema_text = output_schema_path.read_text(encoding="utf-8")
            output_schema = json.loads(output_schema_text)
            self.assertEqual(output_schema["$id"], "axiom://schemas/logical-document-v1")
            self.assertEqual(output_schema["type"], "object")
            self.assertNotIn("entities", output_schema)
            self.assertLess(output_schema_text.index('"$schema"'), output_schema_text.index('"$id"'))
            self.assertLess(
                output_schema_text.index('"properties"'),
                output_schema_text.index('"$defs"'),
            )
            self.assertEqual(
                output_schema["x-axiom-storage"]["document"]["artifact_path"],
                "data/processed/final_test/documents.jsonl",
            )
            text_path = root / "data" / "processed" / "final_test" / "normalization" / "texts.jsonl"
            self.assertEqual(text_path.read_text(encoding="utf-8").count("\n"), 1)
            self.assertIn("xin chào", text_path.read_text(encoding="utf-8"))
            self.assertFalse((root / "data" / "output" / "final_test" / "data" / "documents.json").exists())
            vector_path = root / state.artifact_paths["vector_records"]
            self.assertEqual(
                json.loads(vector_path.read_text(encoding="utf-8")),
                state.vector_records,
            )
            self.assertNotIn("vector_db_report", state.artifact_paths)
            self.assertFalse(
                (root / "data/output/final_test/reports/vector_db_report.json").exists()
            )
            self.assertFalse(any(str(path).startswith(str(root)) for path in state.artifact_paths.values()))


if __name__ == "__main__":
    unittest.main()
