from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import parser_comparison


class ParserComparisonMarkdownTests(unittest.TestCase):
    def _experiment(self, root: Path) -> Path:
        experiment = root / "experiment"
        documents = []
        for index in range(5):
            payload = f"source-{index}".encode("utf-8")
            checksum = hashlib.sha256(payload).hexdigest()
            document_id = checksum[:16]
            filename = f"source-{index}.png"
            corpus_path = experiment / "corpus" / document_id / filename
            corpus_path.parent.mkdir(parents=True, exist_ok=True)
            corpus_path.write_bytes(payload)
            documents.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "relative_path": f"fixtures/{filename}",
                    "mime_type": "image/png",
                    "size_bytes": len(payload),
                    "sha256": checksum,
                }
            )
            for provider, markdown_name in (
                ("chandra2", "result.md"),
                ("datalab", "convert.md"),
            ):
                document_dir = experiment / provider / "documents" / document_id
                image_path = document_dir / "images" / "asset.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(b"image")
                markdown = "# Title\n\n![Chart](asset.png)\n\nBody\n"
                if provider == "datalab":
                    markdown = "# Title\n\n![Chart](asset.png)\n\nChart\n\nBody\n"
                (document_dir / markdown_name).write_text(markdown, encoding="utf-8")
                (document_dir / "metadata.json").write_text(
                    json.dumps({"input_sha256": checksum, "page_count": 1}),
                    encoding="utf-8",
                )

        (experiment / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "fixture",
                    "corpus_fingerprint": "fixture-fingerprint",
                    "documents": documents,
                }
            ),
            encoding="utf-8",
        )
        return experiment

    def test_compare_markdown_pairs_by_checksum_and_labels_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = self._experiment(Path(temporary_directory))
            result = parser_comparison.compare_markdown(
                argparse.Namespace(experiment=experiment, output_dir=None)
            )

            self.assertEqual(result, 0)
            metrics = json.loads(
                (experiment / "comparison" / "markdown_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metrics["document_count"], 5)
            self.assertEqual(metrics["scope"], "markdown_only")
            self.assertIn("not OCR accuracy", metrics["agreement_warning"])
            first = metrics["documents"][0]
            self.assertEqual(first["providers"]["chandra2"]["page_count"], 1)
            self.assertEqual(
                first["providers"]["chandra2"][
                    "basename_resolved_image_reference_count"
                ],
                1,
            )
            self.assertEqual(
                first["providers"]["chandra2"][
                    "directly_resolved_image_reference_count"
                ],
                0,
            )
            self.assertEqual(
                first["providers"]["datalab"]["image_alt_repeated_in_body_count"],
                1,
            )
            self.assertTrue(
                (experiment / "comparison" / "markdown_metrics.csv").is_file()
            )

    def test_compare_markdown_rejects_provider_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = self._experiment(Path(temporary_directory))
            manifest = json.loads(
                (experiment / "manifest.json").read_text(encoding="utf-8")
            )
            document_id = manifest["documents"][0]["document_id"]
            metadata_path = (
                experiment / "chandra2" / "documents" / document_id / "metadata.json"
            )
            metadata_path.write_text(
                json.dumps({"input_sha256": "wrong", "page_count": 1}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                parser_comparison.compare_markdown(
                    argparse.Namespace(experiment=experiment, output_dir=None)
                )


if __name__ == "__main__":
    unittest.main()
