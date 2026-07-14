from __future__ import annotations

import base64
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.parsing.lift.client import (
    _bundle_dir,
    _decode_base64_image,
    _rewrite_markdown_asset_refs,
    _safe_image_name,
    _write_images,
    _write_lift_raw_outputs,
)


@dataclass
class FakeLiftResult:
    status: str = "complete"
    markdown: str | None = None
    extraction_schema_json: str | None = None
    json: dict | None = None
    chunks: dict | None = None
    images: dict | None = None


class LiftClientImageTests(unittest.TestCase):
    def test_decodes_plain_and_data_uri_base64_images(self) -> None:
        content = b"image-bytes"
        encoded = base64.b64encode(content).decode("ascii")

        self.assertEqual(_decode_base64_image(encoded), content)
        self.assertEqual(_decode_base64_image(f"data:image/png;base64,{encoded}"), content)

    def test_writes_lift_images_to_output_directory(self) -> None:
        content = b"image-bytes"
        encoded = base64.b64encode(content).decode("ascii")

        with tempfile.TemporaryDirectory() as output_dir:
            bundle_dir = Path(output_dir) / "source-page--doc-1"
            bundle_dir.mkdir()
            image_files = _write_images(
                bundle_dir,
                {"nested/path/figure.png": encoded},
            )

            self.assertEqual(len(image_files), 1)
            self.assertEqual(image_files[0]["status"], "saved")
            image_path = Path(image_files[0]["path"])
            self.assertEqual(image_path.name, "figure.png")
            self.assertEqual(image_path.read_bytes(), content)
            self.assertEqual(image_path.parent.name, "images")

    def test_duplicate_image_basenames_get_stable_hash_suffixes(self) -> None:
        encoded = base64.b64encode(b"image").decode("ascii")
        images = {"first/figure.png": encoded, "second/figure.png": encoded}

        with tempfile.TemporaryDirectory() as output_dir:
            first_bundle = Path(output_dir) / "first"
            second_bundle = Path(output_dir) / "second"
            first_bundle.mkdir()
            second_bundle.mkdir()

            first_names = [Path(item["path"]).name for item in _write_images(first_bundle, images)]
            second_names = [Path(item["path"]).name for item in _write_images(second_bundle, images)]

            self.assertEqual(first_names, second_names)
            self.assertEqual(len(set(first_names)), 2)
            self.assertTrue(any("--" in name for name in first_names))

    def test_bundle_name_includes_object_id(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            first = _bundle_dir(output_dir, Path("same name.png"), "doc-1")
            second = _bundle_dir(output_dir, Path("same name.png"), "doc-2")

            self.assertNotEqual(first, second)
            self.assertEqual(first.name, "same-name--doc-1")

    def test_assigns_extension_from_data_uri_when_name_has_no_suffix(self) -> None:
        name = _safe_image_name("figure", 1, "data:image/jpeg;base64,abcd")

        self.assertEqual(name, "figure.jpg")

    def test_writes_raw_lift_outputs_for_extract_and_convert_results(self) -> None:
        extract_result = FakeLiftResult(
            markdown="extract markdown",
            extraction_schema_json='{"field": "value"}',
        )
        convert_result = FakeLiftResult(
            markdown="convert markdown",
            json={"pages": []},
            chunks={"chunks": []},
        )

        with tempfile.TemporaryDirectory() as output_dir:
            bundle_dir = Path(output_dir) / "source-page--doc-1"
            bundle_dir.mkdir()
            image_dir = bundle_dir / "images"
            image_dir.mkdir()
            image_path = image_dir / "figure.png"
            image_path.write_bytes(b"image")
            paths = _write_lift_raw_outputs(
                bundle_dir,
                extract_result=extract_result,
                convert_result=convert_result,
                image_files=[
                    {"name": "figure.png", "path": str(image_path), "status": "saved"}
                ],
            )

            self.assertIn("extract_raw_json", paths)
            self.assertIn("extract_markdown", paths)
            self.assertIn("extract_extraction_schema_json", paths)
            self.assertIn("convert_raw_json", paths)
            self.assertIn("convert_markdown", paths)
            self.assertIn("convert_json", paths)
            self.assertIn("convert_chunks", paths)
            self.assertEqual(Path(paths["convert_markdown"]).read_text(encoding="utf-8"), "convert markdown")

    def test_writes_raw_and_rendered_markdown_when_image_links_change(self) -> None:
        result = FakeLiftResult(
            markdown=(
                "![figure](figure.png)\n"
                "![web](https://example.com/x.png)\n"
                "![inline](data:image/png;base64,AAAA)"
            )
        )
        with tempfile.TemporaryDirectory() as output_dir:
            bundle_dir = Path(output_dir) / "source--doc-1"
            bundle_dir.mkdir()
            image_path = bundle_dir / "images" / "figure.png"
            image_path.parent.mkdir()
            image_path.write_bytes(b"image")

            paths = _write_lift_raw_outputs(
                bundle_dir,
                extract_result=result,
                image_files=[{"name": "figure.png", "path": str(image_path), "status": "saved"}],
            )

            self.assertEqual(Path(paths["extract_markdown"]).read_text(encoding="utf-8"), result.markdown)
            rendered = Path(paths["extract_rendered_markdown"]).read_text(encoding="utf-8")
            self.assertIn("](images/figure.png)", rendered)
            self.assertIn("](https://example.com/x.png)", rendered)
            self.assertIn("](data:image/png;base64,AAAA)", rendered)

    def test_does_not_write_partial_rendered_markdown_for_missing_local_asset(self) -> None:
        result = FakeLiftResult(markdown="![known](known.png)\n![missing](missing.png)")
        with tempfile.TemporaryDirectory() as output_dir:
            bundle_dir = Path(output_dir) / "source--doc-1"
            bundle_dir.mkdir()
            image_path = bundle_dir / "images" / "known.png"
            image_path.parent.mkdir()
            image_path.write_bytes(b"image")

            paths = _write_lift_raw_outputs(
                bundle_dir,
                extract_result=result,
                image_files=[{"name": "known.png", "path": str(image_path), "status": "saved"}],
            )

            self.assertNotIn("extract_rendered_markdown", paths)
            self.assertFalse((bundle_dir / "extract.rendered.md").exists())
            self.assertEqual((bundle_dir / "extract.md").read_text(encoding="utf-8"), result.markdown)

    def test_rewrite_removes_a_stale_rendered_derivative(self) -> None:
        result = FakeLiftResult(markdown="![missing](missing.png)")
        with tempfile.TemporaryDirectory() as output_dir:
            bundle_dir = Path(output_dir) / "source--doc-1"
            bundle_dir.mkdir()
            stale_path = bundle_dir / "extract.rendered.md"
            stale_path.write_text("stale", encoding="utf-8")

            paths = _write_lift_raw_outputs(bundle_dir, extract_result=result)

            self.assertNotIn("extract_rendered_markdown", paths)
            self.assertFalse(stale_path.exists())

    def test_rewrite_is_idempotent(self) -> None:
        markdown = "![figure](images/figure.png)"
        self.assertEqual(
            _rewrite_markdown_asset_refs(markdown, {"figure.png": "figure.png"}),
            markdown,
        )


if __name__ == "__main__":
    unittest.main()
