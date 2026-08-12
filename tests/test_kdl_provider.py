from __future__ import annotations

import tempfile
import asyncio
import threading
import time
import unittest
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

from src.ingestion.parsing.kdl import KDLConfig, KDLProvider, _KDLDocument
from src.ingestion.parsing.backends import DocumentParser
from src.ingestion.parsing.service import ParsingService
from src.models import DataObject
from src.pipeline import _continuous_document_queue


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
        **kwargs,
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
        self.assertEqual(config.request_workers, 8)
        self.assertEqual(config.request_batch_size, 1)
        self.assertEqual(config.max_model_sequences, 32)
        self.assertEqual(config.scheduler, "parsebench_document")
        self.assertEqual(config.request_timeout_seconds, 3600)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.layout_max_output_tokens, 6000)
        self.assertEqual(config.table_max_output_tokens, 5500)

    def test_config_rejects_batch_larger_than_model_sequence_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "request_batch_size"):
            KDLConfig.from_mapping(
                {"request_batch_size": 8, "max_model_sequences": 4}
            )

    def test_config_rejects_unknown_scheduler(self) -> None:
        with self.assertRaisesRegex(ValueError, "scheduler"):
            KDLConfig.from_mapping({"scheduler": "unknown"})

    def test_config_rejects_removed_page_scheduler(self) -> None:
        with self.assertRaisesRegex(ValueError, "scheduler"):
            KDLConfig.from_mapping({"scheduler": "page"})

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

    def test_hybrid_provider_uses_top_level_continuous_document_queue(self) -> None:
        resolved = _continuous_document_queue(
            {
                "provider": "kdl_pdf_inspector",
                "kdl": {
                    "continuous_page_queue": True,
                    "request_batch_size": 4,
                },
            }
        )

        self.assertIsNotNone(resolved)
        provider, config = resolved
        self.assertEqual(provider, "kdl")
        self.assertEqual(config["request_batch_size"], 4)

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

    def test_parsebench_scheduler_serializes_requests_per_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [root / "a.png", root / "b.png"]
            for path in paths:
                Image.new("RGB", (64, 64), "white").save(path)
            provider = _FakeKDLProvider(
                KDLConfig(
                    endpoint_url="http://127.0.0.1:8000/v1",
                    scheduler="parsebench_document",
                    max_workers=2,
                    render_processes=1,
                    bbox_max_workers=2,
                    request_batch_size=4,
                    max_model_sequences=8,
                    save_raw_outputs=False,
                )
            )
            outcomes = provider.parse_files_with_errors(
                [
                    (path, DataObject(f"doc-{index}", str(path)))
                    for index, path in enumerate(paths)
                ]
            )

        self.assertTrue(all(not isinstance(item, Exception) for item in outcomes))
        self.assertEqual(provider.fake_engine.max_active_pages, 2)
        self.assertEqual(provider.fake_engine.max_active_bbox_requests, 2)

    def test_global_two_phase_batches_across_documents_after_layout_barrier(self) -> None:
        class GlobalEngine:
            def __init__(self) -> None:
                self.layout_batches = []
                self.recognition_batches = []
                self.layout_pages = 0

            async def layout_batch(
                self,
                client,
                pages,
                request_semaphore,
                **kwargs,
            ):
                self.layout_batches.append(len(pages))
                self.layout_pages += len(pages)
                return [
                    {
                        "text": [
                            {
                                "category": "Text",
                                "bbox": [0.05, 0.05, 0.95, 0.95],
                                "layout_order": 0,
                                "page_number": page_number,
                            }
                        ],
                        "table": [],
                        "picture": [],
                        "formula": [],
                    }
                    for _, page_number in pages
                ]

            async def recognize_prepared_batch(
                self,
                client,
                stage,
                elements,
                request_semaphore,
                **kwargs,
            ):
                self.recognition_batches.append(
                    (stage, [element["job_id"] for element in elements])
                )
                if self.layout_pages != 5:
                    raise AssertionError("recognition started before layout barrier")
                for element in elements:
                    element["content"] = element["job_id"]
                    element["recognition_source"] = "kdl_text"
                return []

            def finalize_elements(self, elements):
                ordered = sorted(elements, key=lambda item: item["layout_order"])
                text = "\n".join(str(item.get("content") or "") for item in ordered)
                return {
                    "markdown": text,
                    "markdown_pages": [{"page_number": 1, "content": text}],
                    "pages": [{"page_number": 1, "elements": ordered}],
                }

        class GlobalProvider(KDLProvider):
            def __init__(self, config):
                super().__init__(config)
                self.global_engine = GlobalEngine()

            def _engine(self):
                return self.global_engine

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(5):
                path = root / f"{index}.png"
                image = Image.new("RGB", (96, 96), "white")
                draw = ImageDraw.Draw(image)
                draw.rectangle((8, 8, 88, 88), outline="black", width=4)
                draw.line((8, 20 + index, 88, 70), fill="black", width=3)
                image.save(path)
                paths.append(path)
            provider = GlobalProvider(
                KDLConfig(
                    endpoint_url="http://127.0.0.1:8000/v1",
                    scheduler="global_two_phase",
                    render_processes=2,
                    request_workers=2,
                    request_batch_size=4,
                    max_model_sequences=8,
                    save_raw_outputs=False,
                )
            )
            outcomes = provider.parse_files_with_errors(
                [
                    (path, DataObject(f"doc-{index}", str(path)))
                    for index, path in enumerate(paths)
                ]
            )

        self.assertTrue(all(not isinstance(item, Exception) for item in outcomes))
        self.assertEqual(sorted(provider.global_engine.layout_batches), [1, 4])
        recognition_sizes = sorted(
            len(job_ids)
            for stage, job_ids in provider.global_engine.recognition_batches
            if stage == "text"
        )
        self.assertEqual(recognition_sizes, [1, 4])
        full_batch = next(
            job_ids
            for stage, job_ids in provider.global_engine.recognition_batches
            if stage == "text" and len(job_ids) == 4
        )
        self.assertEqual(len({job_id.split(":", 1)[0] for job_id in full_batch}), 4)

    def test_global_two_phase_skips_second_render_for_fully_native_text(self) -> None:
        class NativeRouter:
            async def route_document_text_regions(self, context, page_buckets):
                routed = {}
                for page_number, bucket in page_buckets.items():
                    for element in bucket:
                        element["content"] = "native"
                        element["recognition_source"] = "pdf_inspector_region"
                    routed[page_number] = set(range(len(bucket)))
                return routed

        provider = KDLProvider(
            KDLConfig(
                scheduler="global_two_phase",
                save_raw_outputs=False,
            ),
            text_router=NativeRouter(),
        )
        document = _KDLDocument(
            Path("native.pdf"),
            DataObject("native", "native.pdf"),
            1,
            pages=[None],
            layout_pages=[
                {
                    "text": [
                        {
                            "category": "Text",
                            "bbox": [0.1, 0.1, 0.9, 0.2],
                            "layout_order": 0,
                            "page_number": 1,
                        }
                    ],
                    "table": [],
                    "picture": [],
                    "formula": [],
                }
            ],
            routing_context=object(),
        )

        job_map, crop_tasks = asyncio.run(
            provider._prepare_global_recognition([document], [0])
        )

        self.assertEqual(job_map, {})
        self.assertEqual(crop_tasks, {})
        self.assertEqual(document.pages[0][0]["content"], "native")

    def test_global_two_phase_persists_completed_documents_during_recognition(self) -> None:
        class OverlapEngine:
            def __init__(self) -> None:
                self.call_count = 0
                self.active_requests = 0
                self.persistence_started = threading.Event()
                self.persistence_overlapped = False

            async def layout_batch(
                self,
                client,
                pages,
                request_semaphore,
                **kwargs,
            ):
                return [
                    {
                        "text": [
                            {
                                "category": "Text",
                                "bbox": [0.05, 0.05, 0.95, 0.95],
                                "layout_order": 0,
                                "page_number": page_number,
                            }
                        ],
                        "table": [],
                        "picture": [],
                        "formula": [],
                    }
                    for _, page_number in pages
                ]

            async def recognize_prepared_batch(
                self,
                client,
                stage,
                elements,
                request_semaphore,
                **kwargs,
            ):
                self.call_count += 1
                call_number = self.call_count
                self.active_requests += 1
                try:
                    if call_number == 1:
                        while self.call_count < 2:
                            await asyncio.sleep(0.005)
                        await asyncio.sleep(0.02)
                    else:
                        for _ in range(100):
                            if self.persistence_started.is_set():
                                self.persistence_overlapped = True
                                break
                            await asyncio.sleep(0.01)
                    for element in elements:
                        element["content"] = element["job_id"]
                        element["recognition_source"] = "kdl_text"
                    return []
                finally:
                    self.active_requests -= 1

            def finalize_elements(self, elements):
                text = "\n".join(str(item.get("content") or "") for item in elements)
                return {
                    "markdown": text,
                    "markdown_pages": [{"page_number": 1, "content": text}],
                    "pages": [{"page_number": 1, "elements": elements}],
                }

        class OverlapProvider(KDLProvider):
            def __init__(self, config):
                super().__init__(config)
                self.overlap_engine = OverlapEngine()

            def _engine(self):
                return self.overlap_engine

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = []
            for index in range(9):
                path = root / f"{index}.png"
                image = Image.new("RGB", (96, 96), "white")
                draw = ImageDraw.Draw(image)
                draw.rectangle((8, 8, 88, 88), outline="black", width=4)
                draw.line((8, 16 + index, 88, 76), fill="black", width=3)
                image.save(path)
                paths.append(path)
            provider = OverlapProvider(
                KDLConfig(
                    scheduler="global_two_phase",
                    render_processes=2,
                    request_workers=2,
                    request_batch_size=4,
                    max_model_sequences=8,
                    save_raw_outputs=False,
                )
            )

            def persist(index, outcome):
                provider.overlap_engine.persistence_started.set()
                time.sleep(0.03)

            outcomes = provider.parse_files_with_errors(
                [
                    (path, DataObject(f"doc-{index}", str(path)))
                    for index, path in enumerate(paths)
                ],
                on_document_complete=persist,
            )

        self.assertTrue(all(not isinstance(item, Exception) for item in outcomes))
        self.assertTrue(provider.overlap_engine.persistence_overlapped)
        telemetry = outcomes[0].metadata["kdl_global_scheduler"]
        self.assertGreater(telemetry["documents_finalized_during_recognition"], 0)
        self.assertGreater(telemetry["persistence_queue_peak"], 0)

    def test_document_failure_does_not_discard_another_document(self) -> None:
        class FailingEngine(_FakeNanoEngine):
            async def _parse_page(
                self,
                client,
                layout_semaphore,
                bbox_semaphore,
                image,
                page_number,
                **kwargs,
            ):
                if image.getpixel((0, 0)) == (0, 0, 0):
                    raise RuntimeError("synthetic page failure")
                return await super()._parse_page(
                    client,
                    layout_semaphore,
                    bbox_semaphore,
                    image,
                    page_number,
                    **kwargs,
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
