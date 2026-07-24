from __future__ import annotations

import unittest

from src.ingestion.image_contract import normalize_markdown_images
from src.ingestion.image_filter import apply_image_filters
from src.models import ParsedData
from src.reading_order import source_blocks_from_parser_json


class ImageContractTests(unittest.TestCase):
    def test_markdown_assets_become_filterable_ingestion_figures(self) -> None:
        document_json = {
            "children": [
                {
                    "id": "/page/0/Page/0",
                    "block_type": "Page",
                    "bbox": [0, 0, 1000, 1000],
                    "children": [
                        {
                            "id": "/page/0/Picture/1",
                            "block_type": "Picture",
                            "bbox": [100, 100, 700, 700],
                            "html": (
                                '<img src="chart.png" '
                                'alt="Quarterly performance chart">'
                            ),
                        }
                    ],
                }
            ]
        }
        source_blocks = source_blocks_from_parser_json(document_json)
        self.assertEqual(source_blocks[0]["page_box"], [0, 0, 1000, 1000])
        image_files = [
            {"name": "chart.png", "path": "images/chart.png", "status": "saved"},
            {"name": "logo.png", "path": "images/logo.png", "status": "saved"},
        ]
        markdown = (
            "![Company logo](logo.png)\n\n"
            "![Quarterly performance chart](chart.png)"
        )

        extraction, added = normalize_markdown_images(
            None,
            markdown,
            image_files,
            source_blocks,
        )

        self.assertEqual(added, 2)
        self.assertEqual(extraction["main_text"], markdown)
        self.assertEqual(
            extraction["figures"][1]["description_citations"],
            ["/page/0/Picture/1"],
        )
        self.assertEqual(
            image_files[0]["source_ref"],
            "/page/0/Picture/1",
        )

        parsed = ParsedData(
            object_id="doc-1",
            source_uri="source.pdf",
            source_format="pdf",
            rows=[
                {
                    "extraction": extraction,
                    "text": markdown,
                    "source_blocks": source_blocks,
                }
            ],
            text=markdown,
            metadata={"image_files": image_files},
        )
        report = apply_image_filters(parsed)

        self.assertEqual(report["before_count"], 2)
        self.assertEqual(report["dropped_count"], 1)
        self.assertEqual(
            report["dropped"][0]["reasons"],
            ["short_ocr_logo_description"],
        )
        self.assertEqual(
            parsed.rows[0]["extraction"]["figures"][0]["description"],
            "Quarterly performance chart",
        )
        self.assertEqual(
            parsed.metadata["image_files"][0]["name"],
            "chart.png",
        )

    def test_existing_figures_are_not_duplicated(self) -> None:
        source_ref = "/page/0/Figure/0"
        extraction = {
            "main_text": "Body",
            "figures": [
                {
                    "description": "Existing chart",
                    "description_citations": [source_ref],
                }
            ],
        }
        image_files = [
            {"name": "chart.png", "path": "images/chart.png", "status": "saved"}
        ]

        normalized, added = normalize_markdown_images(
            extraction,
            "![Existing chart](chart.png)",
            image_files,
            [],
        )

        self.assertEqual(added, 0)
        self.assertEqual(len(normalized["figures"]), 1)
        self.assertEqual(image_files[0]["source_ref"], source_ref)


if __name__ == "__main__":
    unittest.main()
