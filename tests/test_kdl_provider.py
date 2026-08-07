from __future__ import annotations

import tempfile
import asyncio
import unittest
from pathlib import Path

from PIL import Image

from src.ingestion.parsing.kdl import KDLConfig, KDLProvider
from src.ingestion.parsing.backends import DocumentParser
from src.ingestion.parsing.service import ParsingService
from src.models import DataObject


class _FakeNanoEngine:
    def __init__(self):
        self.active_pages = 0
        self.max_active_pages = 0
        self.active_bbox_requests = 0
        self.max_active_bbox_requests = 0

    async def _parse_page(
        self,
        client,
        layout_semaphore,
        bbox_semaphore,
        image,
        page_number,
    ):
        self.active_pages += 1
        self.max_active_pages = max(self.max_active_pages, self.active_pages)
        async with layout_semaphore:
            await asyncio.sleep(0.01)

        async def recognize_bbox():
            async with bbox_semaphore:
                self.active_bbox_requests += 1
                self.max_active_bbox_requests = max(
                    self.max_active_bbox_requests,
                    self.active_bbox_requests,
                )
                await asyncio.sleep(0.01)
                self.active_bbox_requests -= 1

        await asyncio.gather(*(recognize_bbox() for _ in range(4)))
        self.active_pages -= 1
        return [
            {
                "category": "Text",
                "bbox": [0.1, 0.1, 0.9, 0.9],
                "content": f"page {page_number}",
                "layout_order": 0,
                "page_number": page_number,
            }
        ]

    def finalize_elements(self, elements):
        by_page = {}
        for element in elements:
            by_page.setdefault(element["page_number"], []).append(element)
        return {
            "markdown": "\n\n".join(element["content"] for element in elements),
            "markdown_pages": [
                {"page_number": page, "content": "\n\n".join(item["content"] for item in page_elements)}
                for page, page_elements in sorted(by_page.items())
            ],
            "pages": [
                {"page_number": page, "elements": page_elements}
                for page, page_elements in sorted(by_page.items())
            ],
        }


class _FakeKDLProvider(KDLProvider):
    def __init__(self, config):
        super().__init__(config)
        self.fake_engine = _FakeNanoEngine()

    def _engine(self):
        return self.fake_engine


class KDLProviderTests(unittest.TestCase):
    def test_config_rejects_non_vllm_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "method"):
            KDLConfig.from_mapping({"method": "hf"})

    def test_config_uses_chandra_style_defaults(self) -> None:
        config = KDLConfig.from_mapping(
            {"method": "vllm"}
        )
        self.assertEqual(config.max_workers, 32)
        self.assertEqual(config.render_processes, 32)
        self.assertEqual(config.bbox_max_workers, 32)
        self.assertEqual(config.request_timeout_seconds, 3600)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.layout_max_output_tokens, 6000)
        self.assertEqual(config.table_max_output_tokens, 5500)

    def test_service_routes_pdf_to_kdl_provider(self) -> None:
        service = ParsingService.from_config(
            {
                "provider": "kdl_frontier_nano",
                "kdl": {"endpoint_url": "http://127.0.0.1:8000/v1"},
            }
        )
        backend = service.router.resolve("sample.pdf")
        self.assertIsInstance(backend, DocumentParser)
        self.assertIsInstance(backend.provider, KDLProvider)

    def test_adapter_preserves_pages_layout_and_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = KDLProvider(
                KDLConfig(
                    endpoint_url="http://127.0.0.1:8000/v1",
                    save_raw_outputs=False,
                )
            )
            source = Path(temp_dir) / "sample.pdf"
            raw = {
                "markdown": "# A title\n\nA paragraph\n\n<table><tr><td>x</td></tr></table>",
                "markdown_pages": [
                    {"page_number": 1, "content": "# A title\n\nA paragraph"},
                    {"page_number": 2, "content": "<table><tr><td>x</td></tr></table>"},
                ],
                "pages": [
                    {
                        "page_number": 1,
                        "elements": [
                            {"category": "Title", "bbox": [0.1, 0.1, 0.9, 0.2], "content": "A title", "layout_order": 0},
                            {"category": "Text", "bbox": [0.1, 0.3, 0.9, 0.4], "content": "A paragraph", "layout_order": 1},
                        ],
                    },
                    {
                        "page_number": 2,
                        "elements": [
                            {"category": "Table", "bbox": [0.1, 0.1, 0.9, 0.5], "content": "<table><tr><td>x</td></tr></table>", "layout_order": 0},
                            {"category": "Picture", "bbox": [0.1, 0.55, 0.4, 0.8], "content": "a chart", "layout_order": 1},
                            {"category": "Formula", "bbox": [0.5, 0.55, 0.9, 0.8], "content": "E=mc^2", "layout_order": 2},
                        ],
                    },
                ],
            }
            parsed = provider._to_parsed_data(
                source,
                DataObject("doc-1", "benchmark/sample.pdf", metadata={"format": "pdf"}),
                raw,
                0.0,
                2,
            )

        self.assertEqual(parsed.metadata["parser"], "kdl")
        self.assertEqual(parsed.metadata["page_count"], 2)
        self.assertEqual(parsed.metadata["table_count"], 1)
        self.assertEqual(parsed.metadata["figure_count"], 1)
        self.assertEqual(parsed.metadata["formula_count"], 1)
        self.assertEqual(
            parsed.rows[0]["reading_order"],
            [
                "/page/0/SectionHeader/0",
                "/page/0/Text/1",
                "/page/1/Table/0",
                "/page/1/Figure/1",
                "/page/1/EquationBlock/2",
            ],
        )
        self.assertEqual(
            parsed.rows[0]["extraction"]["tables"][0]["content_format"],
            "html",
        )

    def test_continuous_queue_returns_each_completed_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / "a.png", root / "b.png"]
            for path in paths:
                Image.new("RGB", (64, 64), "white").save(path)
            provider = _FakeKDLProvider(
                KDLConfig(
                    endpoint_url="http://127.0.0.1:8000/v1",
                    max_workers=2,
                    render_processes=1,
                    bbox_max_workers=3,
                    save_raw_outputs=False,
                )
            )
            completed = []
            outcomes = provider.parse_files_with_errors(
                [
                    (path, DataObject(f"doc-{index}", str(path)))
                    for index, path in enumerate(paths)
                ],
                on_document_complete=lambda index, result: completed.append(index),
            )

        self.assertTrue(all(not isinstance(item, Exception) for item in outcomes))
        self.assertCountEqual(completed, [0, 1])
        self.assertEqual([item.text for item in outcomes], ["page 1", "page 1"])
        self.assertLessEqual(provider.fake_engine.max_active_pages, 2)
        self.assertLessEqual(provider.fake_engine.max_active_bbox_requests, 3)

    def test_document_failure_does_not_discard_another_document(self) -> None:
        class FailingEngine(_FakeNanoEngine):
            async def _parse_page(
                self, client, layout_semaphore, bbox_semaphore, image, page_number
            ):
                if image.getpixel((0, 0)) == (0, 0, 0):
                    raise RuntimeError("synthetic page failure")
                return await super()._parse_page(
                    client,
                    layout_semaphore,
                    bbox_semaphore,
                    image,
                    page_number,
                )

        class FailingProvider(_FakeKDLProvider):
            def __init__(self, config):
                KDLProvider.__init__(self, config)
                self.fake_engine = FailingEngine()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good = root / "good.png"
            bad = root / "bad.png"
            Image.new("RGB", (64, 64), "white").save(good)
            Image.new("RGB", (64, 64), "black").save(bad)
            provider = FailingProvider(
                KDLConfig(
                    endpoint_url="http://127.0.0.1:8000/v1",
                    max_workers=2,
                    render_processes=1,
                    bbox_max_workers=2,
                    save_raw_outputs=False,
                )
            )
            outcomes = provider.parse_files_with_errors(
                [
                    (bad, DataObject("bad", str(bad))),
                    (good, DataObject("good", str(good))),
                ]
            )

        self.assertIsInstance(outcomes[0], Exception)
        self.assertNotIsInstance(outcomes[1], Exception)


if __name__ == "__main__":
    unittest.main()
