from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ingestion.image_filter import apply_image_filters
from src.ingestion.parsing import ParsingService
from src.ingestion.runner import run as run_ingestion
from src.models import DataObject, ParsedData, ParseResult


def _figure(component_id: str, description: str) -> dict[str, object]:
    return {
        "caption": "",
        "description": description,
        "description_citations": [component_id],
    }


def _block(
    component_id: str,
    bbox: object,
    *,
    page_box: object | None = (0, 0, 1000, 1000),
    block_type: str = "Image",
) -> dict[str, object]:
    block: dict[str, object] = {
        "component_id": component_id,
        "page": 0,
        "type": block_type,
        "bbox": bbox,
    }
    if page_box is not None:
        block["page_box"] = page_box
    return block


def _parsed(
    source_format: str,
    figures: list[object],
    blocks: list[dict[str, object]],
    assets: list[dict[str, object]] | None = None,
) -> ParsedData:
    return ParsedData(
        object_id="document-1",
        source_uri=f"source.{source_format}",
        source_format=source_format,
        rows=[
            {
                "text": "Document body",
                "extraction": {
                    "main_text": "Document body",
                    "figures": figures,
                    "tables": [],
                    "formulas": [],
                },
                "source_blocks": blocks,
            }
        ],
        text="Document body",
        metadata={
            "parser": "test",
            "figure_count": len(figures),
            "image_count": len(assets or []),
            "image_files": assets or [],
        },
    )


class ImageFilterTests(unittest.TestCase):
    def test_drops_below_one_percent_but_keeps_exact_threshold(self) -> None:
        small_ref = "/page/0/Picture/0"
        threshold_ref = "/page/0/Picture/1"
        parsed = _parsed(
            "pptx",
            [
                _figure(small_ref, "Small decoration"),
                _figure(threshold_ref, "Useful image"),
            ],
            [
                _block(small_ref, [0, 0, 99, 100]),
                _block(threshold_ref, [0, 0, 100, 100]),
            ],
            [
                {
                    "name": "small.png",
                    "path": "assets/small.png",
                    "status": "saved",
                    "source_ref": small_ref,
                },
                {
                    "name": "kept.png",
                    "path": "assets/kept.png",
                    "status": "saved",
                    "source_ref": threshold_ref,
                },
            ],
        )

        report = apply_image_filters(parsed)

        figures = parsed.rows[0]["extraction"]["figures"]
        self.assertEqual([item["description"] for item in figures], ["Useful image"])
        self.assertEqual(report["before_count"], 2)
        self.assertEqual(report["kept_count"], 1)
        self.assertEqual(report["dropped_count"], 1)
        self.assertAlmostEqual(report["dropped"][0]["area_ratio"], 0.0099)
        self.assertEqual(
            report["dropped"][0]["reasons"],
            ["below_min_page_area_ratio"],
        )
        self.assertEqual(
            [item["name"] for item in parsed.metadata["image_files"]],
            ["kept.png"],
        )
        self.assertEqual(parsed.metadata["figure_count"], 1)
        self.assertEqual(parsed.metadata["image_count"], 1)

    def test_ocr_logo_rule_uses_standalone_word_and_120_character_limit(self) -> None:
        logo_ref = "/page/0/Image/0"
        long_ref = "/page/0/Image/1"
        logotype_ref = "/page/0/Image/2"
        long_description = f"{'x' * 116} logo"
        self.assertEqual(len(long_description), 121)
        parsed = _parsed(
            "pdf",
            [
                _figure(logo_ref, "Corporate LOGO"),
                _figure(long_ref, long_description),
                _figure(logotype_ref, "A logotype"),
            ],
            [
                _block(logo_ref, [0, 0, 500, 500]),
                _block(long_ref, [0, 0, 500, 500]),
                _block(logotype_ref, [0, 0, 500, 500]),
            ],
        )

        report = apply_image_filters(parsed)

        figures = parsed.rows[0]["extraction"]["figures"]
        self.assertEqual(
            [item["description"] for item in figures],
            [long_description, "A logotype"],
        )
        self.assertEqual(
            report["dropped"][0]["reasons"],
            ["short_ocr_logo_description"],
        )
        self.assertTrue(report["ocr_input"])

    def test_ocr_icon_rule_uses_standalone_word(self) -> None:
        icon_ref = "/page/0/Image/0"
        iconic_ref = "/page/0/Image/1"
        parsed = _parsed(
            "pdf",
            [
                _figure(icon_ref, "Blue clipboard ICON"),
                _figure(iconic_ref, "An iconic illustration"),
            ],
            [
                _block(icon_ref, [0, 0, 500, 500]),
                _block(iconic_ref, [0, 0, 500, 500]),
            ],
        )

        report = apply_image_filters(parsed)

        self.assertEqual(
            [item["description"] for item in parsed.rows[0]["extraction"]["figures"]],
            ["An iconic illustration"],
        )
        self.assertEqual(
            report["dropped"][0]["reasons"],
            ["short_ocr_icon_description"],
        )
        self.assertEqual(
            report["rules"]["short_description_keywords"],
            ["logo", "icon"],
        )

    def test_office_logo_does_not_use_ocr_specific_rule(self) -> None:
        source_ref = "/page/0/Picture/0"
        parsed = _parsed(
            "docx",
            [_figure(source_ref, "Company logo")],
            [_block(source_ref, [0, 0, 500, 500])],
        )

        report = apply_image_filters(parsed)

        self.assertEqual(report["dropped_count"], 0)
        self.assertEqual(len(parsed.rows[0]["extraction"]["figures"]), 1)
        self.assertFalse(report["ocr_input"])

    def test_infers_page_area_and_keeps_unknown_geometry(self) -> None:
        small_ref = "/page/0/Image/0"
        unknown_ref = "/page/0/Image/1"
        parsed = _parsed(
            "pdf",
            [
                _figure(small_ref, "Tiny image"),
                _figure(unknown_ref, "Unknown geometry"),
            ],
            [
                _block(small_ref, [0, 0, 5, 10], page_box=None),
                _block(
                    "/page/0/Text/2",
                    [0, 0, 100, 100],
                    page_box=None,
                    block_type="Text",
                ),
                _block(unknown_ref, "not-a-box", page_box=None),
            ],
        )

        report = apply_image_filters(parsed)

        figures = parsed.rows[0]["extraction"]["figures"]
        self.assertEqual(
            [item["description"] for item in figures],
            ["Unknown geometry"],
        )
        self.assertEqual(
            report["dropped"][0]["page_area_source"],
            "inferred_block_union",
        )
        self.assertEqual(report["geometry_unavailable_count"], 1)
        self.assertEqual(
            report["geometry_unavailable"][0]["reason"],
            "missing_or_invalid_bbox",
        )

    def test_keeps_raw_images_and_copies_retained_file_to_filtered_images(self) -> None:
        drop_ref = "/page/0/Image/0"
        keep_ref = "/page/0/Image/1"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "assets" / "document-1" / "images"
            image_dir.mkdir(parents=True)
            dropped_path = image_dir / "dropped.png"
            kept_path = image_dir / "kept.png"
            dropped_path.write_bytes(b"dropped")
            kept_path.write_bytes(b"kept")
            parsed = _parsed(
                "pdf",
                [
                    _figure(drop_ref, "Tiny"),
                    _figure(keep_ref, "Chart"),
                ],
                [
                    _block(drop_ref, [0, 0, 10, 10]),
                    _block(keep_ref, [0, 0, 500, 500]),
                ],
                [
                    {
                        "name": "kept.png",
                        "path": str(kept_path),
                        "status": "saved",
                        "source_ref": keep_ref,
                    },
                    {
                        "name": "dropped.png",
                        "path": str(dropped_path),
                        "status": "saved",
                        "source_ref": drop_ref,
                    },
                ],
            )

            report = apply_image_filters(parsed)

            self.assertEqual(
                [item["name"] for item in parsed.metadata["image_files"]],
                ["kept.png"],
            )
            self.assertEqual(report["dropped"][0]["asset"]["name"], "dropped.png")
            filtered_path = (
                root / "assets" / "document-1" / "filtered_images" / "kept.png"
            )
            self.assertEqual(dropped_path.read_bytes(), b"dropped")
            self.assertTrue(kept_path.is_file())
            self.assertEqual(filtered_path.read_bytes(), b"kept")
            self.assertEqual(
                parsed.metadata["image_files"][0]["path"],
                str(filtered_path),
            )
            self.assertEqual(
                parsed.metadata["image_files"][0]["filter_copy_status"],
                "copied",
            )
            self.assertEqual(report["copied_count"], 1)
            self.assertEqual(report["copy_failed_count"], 0)

    def test_mismatched_asset_count_does_not_steal_a_source_ref_match(self) -> None:
        drop_ref = "/page/0/Image/0"
        keep_ref = "/page/0/Image/1"
        parsed = _parsed(
            "pdf",
            [
                _figure(drop_ref, "Tiny"),
                _figure(keep_ref, "Chart"),
            ],
            [
                _block(drop_ref, [0, 0, 10, 10]),
                _block(keep_ref, [0, 0, 500, 500]),
            ],
            [
                {
                    "name": "chart.png",
                    "path": "assets/chart.png",
                    "status": "saved",
                    "source_ref": keep_ref,
                }
            ],
        )

        apply_image_filters(parsed)

        self.assertEqual(
            [item["name"] for item in parsed.metadata["image_files"]],
            ["chart.png"],
        )

    def test_runner_filters_before_schema_inference(self) -> None:
        small_ref = "/page/0/Image/0"
        keep_ref = "/page/0/Figure/1"
        parsed = _parsed(
            "pdf",
            [
                _figure(small_ref, "Tiny image"),
                _figure(keep_ref, "Large chart"),
            ],
            [
                _block(small_ref, [0, 0, 10, 10]),
                _block(keep_ref, [0, 0, 500, 500], block_type="Figure"),
            ],
        )

        class _FakeParsingService:
            def parse(
                self,
                path: Path,
                data_object: DataObject,
            ) -> ParseResult:
                parsed.object_id = data_object.object_id
                parsed.source_uri = data_object.uri
                return ParseResult.success(
                    data_object.object_id,
                    "test",
                    parsed,
                    route="document",
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "document.pdf"
            path.write_bytes(b"%PDF-1.4\n")
            with patch.object(
                ParsingService,
                "from_config",
                return_value=_FakeParsingService(),
            ):
                output = run_ingestion(path, project_root=root)

        self.assertEqual(len(output.parsed_data), 1)
        self.assertEqual(
            output.initial_schemas[0].metadata["component_counts"]["figures"],
            1,
        )
        self.assertEqual(
            output.parsed_data[0].metadata["image_filtering"]["dropped_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
