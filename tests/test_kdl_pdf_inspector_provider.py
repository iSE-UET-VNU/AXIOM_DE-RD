from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from src.ingestion.parsing.backends import DocumentParser
from src.ingestion.parsing.kdl import KDLConfig, KDLProvider
from src.ingestion.parsing.kdl_frontier_engine import (
    NanoEngine,
    NanoUsage,
    SequenceLimiter,
    _nano_chat_batch,
)
from src.ingestion.parsing.kdl_pdf_inspector import (
    KdlPdfInspectorProvider,
    _PdfInspectorTextRouter,
)
from src.ingestion.parsing.pdf_inspector import (
    PdfInspectorClassification,
    PdfInspectorRegionText,
)
from src.ingestion.parsing.service import ParsingService


class _FakeClassifier:
    def __init__(
        self,
        pdf_type: str = "text_based",
        pages_needing_ocr: frozenset[int] = frozenset(),
    ) -> None:
        self.pdf_type = pdf_type
        self.pages_needing_ocr = pages_needing_ocr
        self.calls: list[Path] = []

    def classify(self, path: str | Path) -> PdfInspectorClassification:
        self.calls.append(Path(path))
        return PdfInspectorClassification(
            pdf_type=self.pdf_type,
            confidence=0.95,
            page_count=2,
            pages_needing_ocr=self.pages_needing_ocr,
            latency_ms=12.5,
        )


class _FakeExtractor:
    def __init__(self, results: list[PdfInspectorRegionText]) -> None:
        self.results = results
        self.calls: list[tuple[Path, int, list[list[float]]]] = []

    def extract(self, path, page_index, boxes):
        self.calls.append(
            (
                Path(path),
                int(page_index),
                [[float(value) for value in box] for box in boxes],
            )
        )
        return list(self.results)

    def extract_pages(self, path, page_regions):
        return [
            self.extract(path, page_index, boxes)
            for page_index, boxes in page_regions
        ]


class KdlPdfInspectorRoutingTests(unittest.TestCase):
    def test_batch_transport_maps_choices_and_records_sequence_usage(self) -> None:
        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {
                    "choices": [
                        {"index": 1, "message": {"content": "second"}},
                        {"index": 0, "message": {"content": "first"}},
                    ]
                }

        class Client:
            def __init__(self) -> None:
                self.calls = []

            async def post(self, url, *, json, headers=None):
                self.calls.append((url, json, headers))
                return Response()

        client = Client()
        usage = NanoUsage()
        results = asyncio.run(
            _nano_chat_batch(
                client,
                "https://example.test/v1/chat/completions/batch",
                [
                    {"model": "model", "messages": [[{"content": "one"}]]},
                    {"model": "model", "messages": [[{"content": "two"}]]},
                ],
                asyncio.Semaphore(1),
                stage="text",
                usage=usage,
                sequence_limiter=SequenceLimiter(2),
            )
        )

        self.assertEqual(results, ["first", "second"])
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(client.calls[0][1]["messages"]), 2)
        snapshot = usage.snapshot()
        self.assertEqual(snapshot["stage_calls"], {"text": 1})
        self.assertEqual(snapshot["stage_sequences"], {"text": 2})
        self.assertEqual(snapshot["batch_size_counts"], {"2": 1})

    def test_engine_chunks_recognition_by_stage_batch_size(self) -> None:
        image = Image.new("RGB", (64, 64), "white")
        buckets = {
            "text": [
                {
                    "category": "Text",
                    "bbox": [0.1, 0.1, 0.9, 0.2],
                    "layout_order": index,
                    "page_number": 1,
                    "preprocessed_image": image,
                }
                for index in range(5)
            ],
            "table": [],
            "picture": [],
            "formula": [],
        }
        chat = AsyncMock(return_value="layout tokens")
        batch_chat = AsyncMock(side_effect=[["a", "b", "c", "d"], ["e"]])
        engine = NanoEngine(
            "http://localhost:8000/v1",
            "model",
            2,
            10,
            request_batch_size=4,
            max_model_sequences=8,
        )

        with (
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.analyze_page_content",
                return_value=SimpleNamespace(is_blank=False),
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.prepare_native_layout_image",
                side_effect=lambda value: value,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_chat",
                chat,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_chat_batch",
                batch_chat,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.is_native_layout_response",
                return_value=True,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.parse_native_layout_tokens",
                return_value=[],
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_group_by_bucket",
                return_value=buckets,
            ),
        ):
            elements = asyncio.run(
                engine._parse_page(
                    SimpleNamespace(),
                    asyncio.Semaphore(1),
                    asyncio.Semaphore(2),
                    image,
                    1,
                )
            )

        self.assertEqual(chat.await_count, 1)
        self.assertEqual(batch_chat.await_count, 2)
        batch_lengths = sorted(
            len(call.args[2]) for call in batch_chat.await_args_list
        )
        self.assertEqual(batch_lengths, [1, 4])
        self.assertEqual([element["content"] for element in elements], ["a", "b", "c", "d", "e"])

    def test_engine_falls_back_missing_batch_choices_to_single_requests(self) -> None:
        image = Image.new("RGB", (64, 64), "white")
        buckets = {
            "text": [
                {
                    "category": "Text",
                    "bbox": [0.1, 0.1, 0.9, 0.2],
                    "layout_order": index,
                    "page_number": 1,
                    "preprocessed_image": image,
                }
                for index in range(2)
            ],
            "table": [],
            "picture": [],
            "formula": [],
        }
        chat = AsyncMock(side_effect=["layout tokens", "recovered"])
        batch_chat = AsyncMock(return_value=["first", None])
        usage = NanoUsage()
        engine = NanoEngine(
            "http://localhost:8000/v1",
            "model",
            2,
            10,
            request_batch_size=4,
            max_model_sequences=8,
        )

        with (
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.analyze_page_content",
                return_value=SimpleNamespace(is_blank=False),
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.prepare_native_layout_image",
                side_effect=lambda value: value,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_chat",
                chat,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_chat_batch",
                batch_chat,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.is_native_layout_response",
                return_value=True,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.parse_native_layout_tokens",
                return_value=[],
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_group_by_bucket",
                return_value=buckets,
            ),
        ):
            elements = asyncio.run(
                engine._parse_page(
                    SimpleNamespace(),
                    asyncio.Semaphore(1),
                    asyncio.Semaphore(2),
                    image,
                    1,
                    usage=usage,
                )
            )

        self.assertEqual(chat.await_count, 2)
        self.assertEqual([element["content"] for element in elements], ["first", "recovered"])
        self.assertEqual(
            usage.snapshot()["batch_fallback_sequences"],
            {"text": 1},
        )
        self.assertEqual(
            usage.snapshot()["batch_fallback_recovered_sequences"],
            {"text": 1},
        )
        self.assertEqual(usage.snapshot()["unrecovered_sequences"], {})

    def test_service_registers_hybrid_without_changing_pure_kdl(self) -> None:
        fake_api = SimpleNamespace()
        with patch(
            "src.ingestion.parsing.pdf_inspector.import_module",
            return_value=fake_api,
        ):
            hybrid = ParsingService.from_config(
                {"provider": "kdl_pdf_inspector", "kdl": {"save_raw_outputs": False}}
            ).router.resolve("sample.pdf")
        pure = ParsingService.from_config(
            {"provider": "kdl", "kdl": {"save_raw_outputs": False}}
        ).router.resolve("sample.pdf")

        self.assertIsInstance(hybrid, DocumentParser)
        self.assertIsInstance(hybrid.provider, KdlPdfInspectorProvider)
        self.assertIsInstance(hybrid.provider, KDLProvider)
        self.assertIsInstance(pure.provider, KDLProvider)
        self.assertNotIsInstance(pure.provider, KdlPdfInspectorProvider)
        self.assertIsNone(pure.provider._text_router)

    def test_mixed_pdf_routes_native_page_and_falls_back_for_ocr_page(self) -> None:
        classifier = _FakeClassifier("mixed", frozenset({1}))
        extractor = _FakeExtractor(
            [PdfInspectorRegionText("Native heading", False)]
        )
        router = _PdfInspectorTextRouter(classifier, extractor)

        with patch(
            "src.ingestion.parsing.kdl_pdf_inspector._pdf_page_dimensions",
            return_value=((100.0, 200.0), (300.0, 400.0)),
        ):
            context = router.prepare_document(Path("mixed.pdf"))

        first = [{"category": "Title", "bbox": [0.1, 0.2, 0.5, 0.6]}]
        second = [{"category": "Text", "bbox": [0.0, 0.0, 1.0, 1.0]}]
        routed_first = asyncio.run(router.route_text_regions(context, 1, first))
        routed_second = asyncio.run(router.route_text_regions(context, 2, second))

        self.assertEqual(routed_first, {0})
        self.assertEqual(routed_second, set())
        self.assertEqual(first[0]["content"], "Native heading")
        self.assertEqual(first[0]["recognition_source"], "pdf_inspector_region")
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual(extractor.calls[0][1], 0)
        self.assertEqual(extractor.calls[0][2], [[10.0, 40.0, 50.0, 120.0]])
        metadata = router.routing_metadata(context)
        self.assertEqual(metadata["text_candidates"], 2)
        self.assertEqual(metadata["native_text_regions"], 1)
        self.assertEqual(metadata["kdl_text_fallback_regions"], 1)
        self.assertEqual(metadata["pages_needing_ocr"], [1])

    def test_document_router_extracts_all_native_pages_in_one_call(self) -> None:
        class DocumentExtractor:
            def __init__(self) -> None:
                self.calls = []

            def extract_pages(self, path, page_regions):
                self.calls.append((Path(path), list(page_regions)))
                return [
                    [PdfInspectorRegionText(f"page-{page_index}", False)]
                    for page_index, _ in page_regions
                ]

        extractor = DocumentExtractor()
        router = _PdfInspectorTextRouter(_FakeClassifier(), extractor)
        with patch(
            "src.ingestion.parsing.kdl_pdf_inspector._pdf_page_dimensions",
            return_value=((100.0, 200.0), (300.0, 400.0)),
        ):
            context = router.prepare_document(Path("text.pdf"))
        buckets = {
            1: [{"category": "Text", "bbox": [0.1, 0.1, 0.9, 0.2]}],
            2: [{"category": "Title", "bbox": [0.2, 0.2, 0.8, 0.3]}],
        }

        routed = asyncio.run(
            router.route_document_text_regions(context, buckets)
        )

        self.assertEqual(routed, {1: {0}, 2: {0}})
        self.assertEqual(len(extractor.calls), 1)
        self.assertEqual([page for page, _ in extractor.calls[0][1]], [0, 1])
        self.assertEqual(buckets[1][0]["content"], "page-0")
        self.assertEqual(buckets[2][0]["content"], "page-1")

    def test_region_level_empty_and_needs_ocr_results_fall_back(self) -> None:
        extractor = _FakeExtractor(
            [
                PdfInspectorRegionText("Native text", False),
                PdfInspectorRegionText("Unreadable", True, "broken_encoding"),
                PdfInspectorRegionText("", False),
            ]
        )
        router = _PdfInspectorTextRouter(_FakeClassifier(), extractor)
        with patch(
            "src.ingestion.parsing.kdl_pdf_inspector._pdf_page_dimensions",
            return_value=((612.0, 792.0),),
        ):
            context = router.prepare_document(Path("text.pdf"))
        bucket = [
            {"category": category, "bbox": [0.1, 0.1, 0.9, 0.2]}
            for category in ("Text", "Section-header", "Caption")
        ]

        routed = asyncio.run(router.route_text_regions(context, 1, bucket))

        self.assertEqual(routed, {0})
        metadata = router.routing_metadata(context)
        self.assertEqual(metadata["native_text_regions"], 1)
        self.assertEqual(metadata["kdl_text_fallback_regions"], 2)
        self.assertEqual(
            metadata["text_routing_fallback_reasons"],
            {"empty_region": 1, "region_needs_ocr": 1},
        )

    def test_region_error_and_count_mismatch_fall_back_without_raising(self) -> None:
        class FailingExtractor:
            def extract(self, path, page_index, boxes):
                raise RuntimeError("synthetic extraction error")

        class MismatchExtractor:
            def extract(self, path, page_index, boxes):
                return []

        for extractor, reason in (
            (FailingExtractor(), "region_extraction_error"),
            (MismatchExtractor(), "region_count_mismatch"),
        ):
            with self.subTest(reason=reason):
                router = _PdfInspectorTextRouter(_FakeClassifier(), extractor)
                with patch(
                    "src.ingestion.parsing.kdl_pdf_inspector._pdf_page_dimensions",
                    return_value=((100.0, 200.0),),
                ):
                    context = router.prepare_document(Path("text.pdf"))
                bucket = [
                    {"category": "Text", "bbox": [0.1, 0.1, 0.9, 0.2]}
                ]

                routed = asyncio.run(
                    router.route_text_regions(context, 1, bucket)
                )

                self.assertEqual(routed, set())
                metadata = router.routing_metadata(context)
                self.assertEqual(metadata["kdl_text_fallback_regions"], 1)
                self.assertEqual(
                    metadata["text_routing_fallback_reasons"][reason], 1
                )

    def test_concurrent_pages_keep_document_contexts_isolated(self) -> None:
        class PathExtractor:
            def extract(self, path, page_index, boxes):
                return [PdfInspectorRegionText(Path(path).stem, False)]

        router = _PdfInspectorTextRouter(_FakeClassifier(), PathExtractor())
        with patch(
            "src.ingestion.parsing.kdl_pdf_inspector._pdf_page_dimensions",
            return_value=((100.0, 100.0), (100.0, 100.0)),
        ):
            first_context = router.prepare_document(Path("first.pdf"))
            second_context = router.prepare_document(Path("second.pdf"))
        first = [{"category": "Text", "bbox": [0.0, 0.0, 1.0, 1.0]}]
        second = [{"category": "Text", "bbox": [0.0, 0.0, 1.0, 1.0]}]

        async def route_both():
            return await asyncio.gather(
                router.route_text_regions(first_context, 1, first),
                router.route_text_regions(second_context, 1, second),
            )

        asyncio.run(route_both())

        self.assertEqual(first[0]["content"], "first")
        self.assertEqual(second[0]["content"], "second")

    def test_scanned_and_non_pdf_inputs_do_not_call_region_extractor(self) -> None:
        extractor = _FakeExtractor([PdfInspectorRegionText("unused", False)])
        scanned_router = _PdfInspectorTextRouter(
            _FakeClassifier("scanned", frozenset({0})), extractor
        )
        scanned = scanned_router.prepare_document(Path("scan.pdf"))
        image = scanned_router.prepare_document(Path("image.png"))
        element = [{"category": "Text", "bbox": [0.0, 0.0, 1.0, 1.0]}]

        self.assertEqual(
            asyncio.run(scanned_router.route_text_regions(scanned, 1, element)),
            set(),
        )
        self.assertEqual(
            asyncio.run(scanned_router.route_text_regions(image, 1, element)),
            set(),
        )
        self.assertEqual(extractor.calls, [])

    def test_classification_error_falls_back_to_full_kdl(self) -> None:
        class FailingClassifier:
            def classify(self, path):
                raise RuntimeError("synthetic classification error")

        extractor = _FakeExtractor([PdfInspectorRegionText("unused", False)])
        router = _PdfInspectorTextRouter(FailingClassifier(), extractor)
        context = router.prepare_document(Path("broken.pdf"))
        bucket = [{"category": "Text", "bbox": [0.0, 0.0, 1.0, 1.0]}]

        routed = asyncio.run(router.route_text_regions(context, 1, bucket))

        self.assertEqual(routed, set())
        self.assertEqual(extractor.calls, [])
        metadata = router.routing_metadata(context)
        self.assertEqual(metadata["pdf_type"], "classification_error")
        self.assertIn("synthetic classification error", metadata["pdf_inspector_error"])
        self.assertEqual(metadata["kdl_text_fallback_regions"], 1)

    def test_missing_optional_dependency_is_a_clear_configuration_error(self) -> None:
        missing = ModuleNotFoundError("No module named 'pdf_inspector'")
        missing.name = "pdf_inspector"
        with patch(
            "src.ingestion.parsing.pdf_inspector.import_module",
            side_effect=missing,
        ):
            with self.assertRaisesRegex(RuntimeError, "pdf-inspector"):
                ParsingService.from_config({"provider": "kdl_pdf_inspector"})

    def test_engine_skips_kdl_text_calls_but_formats_native_categories(self) -> None:
        class NativeRouter:
            async def route_text_regions(self, context, page_number, bucket):
                values = ["Report", "Overview", "First item"]
                for element, value in zip(bucket, values, strict=True):
                    element["content"] = value
                    element["recognition_source"] = "pdf_inspector_region"
                return set(range(len(bucket)))

        image = Image.new("RGB", (64, 64), "white")
        text_elements = [
            {
                "category": category,
                "bbox": [0.1, 0.1, 0.9, 0.2],
                "layout_order": index,
                "page_number": 1,
                "preprocessed_image": image,
            }
            for index, category in enumerate(
                ("Title", "Section-header", "List-item")
            )
        ]
        formula = {
            "category": "Formula",
            "bbox": [0.1, 0.3, 0.9, 0.4],
            "layout_order": 3,
            "page_number": 1,
            "preprocessed_image": image,
        }
        buckets = {
            "text": text_elements,
            "table": [],
            "picture": [],
            "formula": [formula],
        }
        chat = AsyncMock(side_effect=["layout tokens", "E=mc^2"])
        engine = NanoEngine("http://localhost:8000/v1", "model", 4, 10, text_router=NativeRouter())

        with (
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.analyze_page_content",
                return_value=SimpleNamespace(is_blank=False),
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.prepare_native_layout_image",
                side_effect=lambda value: value,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_chat",
                chat,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.is_native_layout_response",
                return_value=True,
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine.parse_native_layout_tokens",
                return_value=[],
            ),
            patch(
                "src.ingestion.parsing.kdl_frontier_engine._nano_group_by_bucket",
                return_value=buckets,
            ),
        ):
            elements = asyncio.run(
                engine._parse_page(
                    SimpleNamespace(),
                    asyncio.Semaphore(1),
                    asyncio.Semaphore(4),
                    image,
                    1,
                    routing_context=object(),
                )
            )

        self.assertEqual(chat.await_count, 2)  # layout + formula, no text OCR
        raw = engine.finalize_elements(elements)
        self.assertIn("# Report", raw["markdown"])
        self.assertIn("## Overview", raw["markdown"])
        self.assertIn("- First item", raw["markdown"])
        self.assertEqual(
            raw["pages"][0]["elements"][0]["recognition_source"],
            "pdf_inspector_region",
        )


if __name__ == "__main__":
    unittest.main()
