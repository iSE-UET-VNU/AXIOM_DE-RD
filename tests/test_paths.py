from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.models import PipelineState
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

    def test_storage_pipeline_state_uses_relative_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = PipelineState(
                run_id="run-1",
                input_dir="data/raw/final_test",
                output_dir="data/output/final_test",
            )

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
                "data/output/final_test/data/documents.json",
            )
            self.assertEqual(
                state.artifact_paths["index_records"],
                "data/output/final_test/data/index_records.json",
            )
            self.assertEqual(
                state.artifact_paths["vector_records"],
                "data/output/final_test/data/vector_records.json",
            )
            self.assertFalse(any(str(path).startswith(str(root)) for path in state.artifact_paths.values()))


if __name__ == "__main__":
    unittest.main()
