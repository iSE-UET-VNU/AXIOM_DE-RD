from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.api.output import build_dataeng_output
from src.models import DataObject, PipelineState


class ExternalOutputArtifactTests(unittest.TestCase):
    def _state(
        self,
        output_root: Path,
        metadata_path: Path,
        document_path: Path,
    ) -> PipelineState:
        state = PipelineState(
            run_id="run-1",
            input_source="benchmark",
            embedded_dir=str(output_root.parent / "embedded"),
            output_dir=str(output_root),
            data_objects=[DataObject("doc-1", "benchmark/doc.pdf")],
        )
        state.artifact_paths = {
            "output_metadata": str(metadata_path),
            "output_document:doc-1": str(document_path),
        }
        return state

    def test_builds_response_from_configured_external_output_root(self) -> None:
        workspace_tmp = Path.cwd() / ".tmp"
        workspace_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as temp_dir:
            root = Path(temp_dir)
            project_root = root / "repo"
            output_root = root / "drive" / "output" / "run-1"
            project_root.mkdir()
            (output_root / "documents").mkdir(parents=True)
            metadata_path = output_root / "metadata.json"
            document_path = output_root / "documents" / "doc-1.json"
            metadata_path.write_text(
                json.dumps({"run_id": "run-1", "contract_version": "internal"}),
                encoding="utf-8",
            )
            document_path.write_text(
                json.dumps({"document": {"document_id": "doc-1"}}),
                encoding="utf-8",
            )

            response = build_dataeng_output(
                self._state(
                    output_root,
                    metadata_path,
                    document_path,
                ),
                project_root=project_root,
            )

        self.assertEqual(response["metadata"]["run_id"], "run-1")
        self.assertEqual(response["documents"][0]["document"]["document_id"], "doc-1")
        self.assertNotIn("contract_version", response["metadata"])

    def test_rejects_artifact_outside_repo_and_configured_output_root(self) -> None:
        workspace_tmp = Path.cwd() / ".tmp"
        workspace_tmp.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace_tmp) as temp_dir:
            root = Path(temp_dir)
            project_root = root / "repo"
            output_root = root / "drive" / "output" / "run-1"
            outside_root = root / "drive" / "output-other"
            project_root.mkdir()
            output_root.mkdir(parents=True)
            outside_root.mkdir(parents=True)
            metadata_path = outside_root / "metadata.json"
            document_path = output_root / "doc-1.json"
            metadata_path.write_text("{}", encoding="utf-8")
            document_path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "outside the project root"):
                build_dataeng_output(
                    self._state(
                        output_root,
                        metadata_path,
                        document_path,
                    ),
                    project_root=project_root,
                )


if __name__ == "__main__":
    unittest.main()
