from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.normalization import normalize_parsed_data
from src.models import ParsedData


class NormalizationArtifactTests(unittest.TestCase):
    def test_normalizes_lift_text_image_and_table_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "lift_outputs" / "sample_raw_lift"
            raw_dir.mkdir(parents=True)
            image_path = root / "lift_outputs" / "sample_images" / "street.jpg"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")

            document_path = raw_dir / "extract.document.json"
            extraction_path = raw_dir / "extract.extraction_schema.json"
            document_path.write_text(
                json.dumps(
                    {
                        "children": [
                            {
                                "id": "/page/0/Document/0",
                                "block_type": "Document",
                                "children": [
                                    {
                                        "id": "/page/0/SectionHeader/5",
                                        "block_type": "SectionHeader",
                                        "html": "<h2><p>INDEX TO PEOPLE</p></h2>",
                                        "page": 0,
                                    },
                                    {
                                        "id": "/page/0/Table/6",
                                        "block_type": "Table",
                                        "html": (
                                            "<table><tr><th>A</th><td>Dovigi, Patrick.....B2</td>"
                                            "<th>Q</th><td>Qureshi, Abid.....B2</td></tr>"
                                            "<tr><td>Adler, Jonathan.....B6</td><th>G</th>"
                                            "<th>R</th><td>Ramaswamy, Vivek.....A5</td></tr></table>"
                                        ),
                                        "page": 0,
                                    },
                                    {
                                        "id": "/page/0/Picture/8",
                                        "block_type": "Picture",
                                        "html": (
                                            '<img alt="A busy city street in Manhattan with heavy-duty trucks '
                                            'and pedestrians." src="street.jpg"/>'
                                            '<div class="img-description"><div class="img-alt">'
                                            "A busy city street in Manhattan with heavy-duty trucks and pedestrians."
                                            "</div></div>"
                                        ),
                                        "page": 0,
                                    },
                                ],
                            }
                        ],
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            extraction_path.write_text(
                json.dumps(
                    {
                        "document_type": "newspaper page",
                        "language": "en",
                        "title": "INDEX TO BUSINESSES",
                        "main_text": "INDEX TO PEOPLE\nA busy city street in Manhattan.",
                        "tables": [
                            {
                                "caption": "INDEX TO PEOPLE",
                                "content": "| A | Q |\n|---|---|\n| Adler, Jonathan.....B6 | Qureshi, Abid.....B2 |",
                                "content_citations": ["/page/0/Table/6"],
                            }
                        ],
                        "figures": [
                            {
                                "caption": "",
                                "caption_meta": {"extraction_status": "NOT_RESOLVABLE"},
                                "description": "A busy city street in Manhattan with heavy-duty trucks and pedestrians.",
                                "description_citations": ["/page/0/Picture/8"],
                            }
                        ],
                        "formulas": [],
                    }
                ),
                encoding="utf-8",
            )

            parsed = ParsedData(
                object_id="doc-1",
                source_uri="data/raw/sample.png",
                source_format="png",
                rows=[],
                metadata={
                    "raw_lift_outputs": {
                        "extract_json": str(document_path),
                        "extract_extraction_schema_json": str(extraction_path),
                    },
                    "image_files": [
                        {"name": "street.jpg", "path": str(image_path), "status": "saved"},
                    ],
                },
            )

            output = normalize_parsed_data([parsed], project_root=root)

        self.assertEqual(len(output.tables), 1)
        self.assertEqual(output.tables[0]["caption"], "INDEX TO PEOPLE")
        self.assertEqual(output.tables[0]["source_block_id"], "/page/0/Table/6")
        self.assertIn(["A", "Dovigi, Patrick.....B2", "Q", "Qureshi, Abid.....B2"], output.tables[0]["rows"])
        self.assertEqual(len(output.images), 1)
        self.assertEqual(output.images[0]["visible_caption"], "")
        self.assertEqual(output.images[0]["embedding_text"].count("A busy city street"), 1)
        self.assertNotIn("bbox", output.texts[0])
        self.assertNotIn("polygon", output.texts[0])
        self.assertNotIn("bbox", output.tables[0])
        self.assertNotIn("polygon", output.tables[0])
        self.assertNotIn("bbox", output.images[0])
        self.assertNotIn("polygon", output.images[0])
        self.assertNotIn("![", "\n".join(item["text"] for item in output.texts))
        self.assertNotIn("|---|", "\n".join(item["text"] for item in output.texts))
        self.assertEqual(output.documents[0]["component_counts"]["tables"], 1)
        self.assertEqual(output.documents[0]["component_counts"]["images"], 1)

    def test_falls_back_to_schema_extraction_without_raw_lift_files(self) -> None:
        parsed = ParsedData(
            object_id="doc-1",
            source_uri="data/raw/sample.txt",
            source_format="txt",
            rows=[
                {
                    "extraction": {
                        "main_text": "Fallback body",
                        "tables": [{"caption": "Fallback table", "content": "| A | B |\n|---|---|\n| 1 | 2 |"}],
                        "figures": [],
                    }
                }
            ],
        )

        output = normalize_parsed_data([parsed])

        self.assertEqual(output.texts[0]["text"], "Fallback body")
        self.assertEqual(output.tables[0]["caption"], "Fallback table")
        self.assertEqual(output.tables[0]["rows"], [["A", "B"], ["1", "2"]])

    def test_falls_back_to_convert_markdown_and_all_saved_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = root / "data" / "work" / "sample" / "run-1" / "datalab" / "doc-1"
            bundle.mkdir(parents=True)
            markdown_path = bundle / "convert.md"
            markdown_path.write_text(
                "# Report\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                encoding="utf-8",
            )
            image_paths = []
            for name in ("one.jpg", "two.jpg"):
                path = bundle / "images" / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(b"image")
                image_paths.append(path)

            parsed = ParsedData(
                object_id="doc-1",
                source_uri="data/raw/sample.pdf",
                source_format="pdf",
                rows=[{"extraction": {"formulas": ["x = y"]}}],
                metadata={
                    "parser": "lift-api",
                    "status": "complete",
                    "raw_lift_outputs": {"convert_markdown": str(markdown_path)},
                    "image_files": [
                        {"name": path.name, "path": str(path), "status": "saved"}
                        for path in image_paths
                    ],
                },
            )

            output = normalize_parsed_data([parsed], project_root=root)

        self.assertTrue(any(item["role"] == "formula" for item in output.texts))
        self.assertEqual(len(output.tables), 1)
        self.assertEqual(output.tables[0]["source_artifact"], "convert.md")
        self.assertEqual(len(output.images), 2)
        self.assertEqual({item["image_name"] for item in output.images}, {"one.jpg", "two.jpg"})
        self.assertTrue(output.documents[0]["quality"]["has_text"])

    def test_picture_without_decoded_asset_is_not_an_image_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            document_path = root / "extract.document.json"
            document_path.write_text(
                json.dumps(
                    {
                        "children": [
                            {
                                "id": "/page/0/Picture/1",
                                "block_type": "Picture",
                                "html": '<img alt="missing" src="missing.jpg"/>',
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            parsed = ParsedData(
                object_id="doc-1",
                source_uri="data/raw/sample.pdf",
                source_format="pdf",
                text="body",
                metadata={"raw_lift_outputs": {"extract_json": str(document_path)}},
            )

            output = normalize_parsed_data([parsed], project_root=root)

        self.assertEqual(output.images, [])
        self.assertEqual(output.documents[0]["quality"]["missing_image_assets"], 0)


if __name__ == "__main__":
    unittest.main()
