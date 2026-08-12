"""Hybrid KDL layout/recognition with native PDF text region extraction."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from .kdl import KDLConfig, KDLProvider
from .pdf_inspector import (
    PdfInspectorClassifier,
    PdfInspectorRegionExtractor,
)

logger = logging.getLogger(__name__)


@dataclass
class _RoutingStats:
    text_candidates: int = 0
    native_text_regions: int = 0
    kdl_text_fallback_regions: int = 0
    region_extraction_latency_ms: float = 0.0
    fallback_reasons: Counter[str] = field(default_factory=Counter)


@dataclass
class _DocumentRoutingContext:
    source_path: Path
    pdf_type: str
    confidence: float
    pages_needing_ocr: frozenset[int]
    page_dimensions: tuple[tuple[float, float], ...] = ()
    native_routing_enabled: bool = False
    classification_latency_ms: float = 0.0
    classification_error: str | None = None
    stats: _RoutingStats = field(default_factory=_RoutingStats)


class _PdfInspectorTextRouter:
    """Adapt independent pdf-inspector components to KDL's text bucket hook."""

    def __init__(
        self,
        classifier: PdfInspectorClassifier,
        extractor: PdfInspectorRegionExtractor,
    ) -> None:
        self._classifier = classifier
        self._extractor = extractor

    def prepare_document(self, source_path: Path) -> _DocumentRoutingContext:
        if source_path.suffix.lower() != ".pdf":
            return _DocumentRoutingContext(
                source_path=source_path,
                pdf_type="non_pdf",
                confidence=0.0,
                pages_needing_ocr=frozenset({0}),
            )

        classify_started = perf_counter()
        try:
            classification = self._classifier.classify(source_path)
        except Exception as exc:
            latency_ms = (perf_counter() - classify_started) * 1000.0
            logger.warning(
                "pdf-inspector classification failed for %s; using full KDL: %s",
                source_path,
                exc,
            )
            return _DocumentRoutingContext(
                source_path=source_path,
                pdf_type="classification_error",
                confidence=0.0,
                pages_needing_ocr=frozenset(),
                classification_latency_ms=latency_ms,
                classification_error=f"{type(exc).__name__}: {exc}",
            )

        enabled = classification.pdf_type in {"text_based", "mixed"}
        dimensions: tuple[tuple[float, float], ...] = ()
        dimension_error: str | None = None
        if enabled:
            try:
                dimensions = _pdf_page_dimensions(source_path)
            except Exception as exc:
                enabled = False
                dimension_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Could not read PDF page dimensions for %s; using full KDL: %s",
                    source_path,
                    exc,
                )

        return _DocumentRoutingContext(
            source_path=source_path,
            pdf_type=classification.pdf_type,
            confidence=classification.confidence,
            pages_needing_ocr=classification.pages_needing_ocr,
            page_dimensions=dimensions,
            native_routing_enabled=enabled,
            classification_latency_ms=classification.latency_ms,
            classification_error=dimension_error,
        )

    async def route_text_regions(
        self,
        context: _DocumentRoutingContext | None,
        page_number: int,
        text_bucket: list[dict[str, Any]],
    ) -> set[int]:
        if context is None:
            return set()

        stats = context.stats
        candidate_count = len(text_bucket)
        stats.text_candidates += candidate_count
        page_index = page_number - 1

        if not context.native_routing_enabled:
            stats.kdl_text_fallback_regions += candidate_count
            stats.fallback_reasons["native_routing_disabled"] += candidate_count
            return set()
        if page_index in context.pages_needing_ocr:
            stats.kdl_text_fallback_regions += candidate_count
            stats.fallback_reasons["page_needs_ocr"] += candidate_count
            return set()
        if page_index < 0 or page_index >= len(context.page_dimensions):
            stats.kdl_text_fallback_regions += candidate_count
            stats.fallback_reasons["missing_page_dimensions"] += candidate_count
            return set()

        page_width, page_height = context.page_dimensions[page_index]
        valid_candidates: list[tuple[int, dict[str, Any]]] = []
        boxes: list[list[float]] = []
        for bucket_index, element in enumerate(text_bucket):
            bbox = element.get("bbox")
            if not _valid_normalized_bbox(bbox):
                stats.kdl_text_fallback_regions += 1
                stats.fallback_reasons["invalid_bbox"] += 1
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            boxes.append(
                [
                    x1 * page_width,
                    y1 * page_height,
                    x2 * page_width,
                    y2 * page_height,
                ]
            )
            valid_candidates.append((bucket_index, element))

        if not boxes:
            return set()

        extract_started = perf_counter()
        try:
            regions = await asyncio.to_thread(
                self._extractor.extract,
                context.source_path,
                page_index,
                boxes,
            )
        except Exception as exc:
            stats.region_extraction_latency_ms += (
                perf_counter() - extract_started
            ) * 1000.0
            stats.kdl_text_fallback_regions += len(valid_candidates)
            stats.fallback_reasons["region_extraction_error"] += len(valid_candidates)
            logger.warning(
                "pdf-inspector region extraction failed for %s page %d; "
                "using KDL: %s",
                context.source_path,
                page_number,
                exc,
            )
            return set()
        stats.region_extraction_latency_ms += (
            perf_counter() - extract_started
        ) * 1000.0

        if len(regions) != len(valid_candidates):
            stats.kdl_text_fallback_regions += len(valid_candidates)
            stats.fallback_reasons["region_count_mismatch"] += len(valid_candidates)
            return set()

        routed: set[int] = set()
        for (bucket_index, element), region in zip(
            valid_candidates,
            regions,
            strict=True,
        ):
            text = region.text.strip()
            if region.needs_ocr:
                stats.kdl_text_fallback_regions += 1
                stats.fallback_reasons["region_needs_ocr"] += 1
                continue
            if not text:
                stats.kdl_text_fallback_regions += 1
                stats.fallback_reasons["empty_region"] += 1
                continue
            element["content"] = text
            element["recognition_source"] = "pdf_inspector_region"
            routed.add(bucket_index)

        stats.native_text_regions += len(routed)
        return routed

    async def route_document_text_regions(
        self,
        context: _DocumentRoutingContext | None,
        page_buckets: dict[int, list[dict[str, Any]]],
    ) -> dict[int, set[int]]:
        """Route all native-text pages through one document-level API call."""

        routed_by_page = {page: set() for page in page_buckets}
        if context is None:
            return routed_by_page
        stats = context.stats
        pending: list[
            tuple[int, list[tuple[int, dict[str, Any]]], list[list[float]]]
        ] = []
        for page_number, bucket in sorted(page_buckets.items()):
            stats.text_candidates += len(bucket)
            page_index = page_number - 1
            if not context.native_routing_enabled:
                stats.kdl_text_fallback_regions += len(bucket)
                stats.fallback_reasons["native_routing_disabled"] += len(bucket)
                continue
            if page_index in context.pages_needing_ocr:
                stats.kdl_text_fallback_regions += len(bucket)
                stats.fallback_reasons["page_needs_ocr"] += len(bucket)
                continue
            if page_index < 0 or page_index >= len(context.page_dimensions):
                stats.kdl_text_fallback_regions += len(bucket)
                stats.fallback_reasons["missing_page_dimensions"] += len(bucket)
                continue
            page_width, page_height = context.page_dimensions[page_index]
            candidates: list[tuple[int, dict[str, Any]]] = []
            boxes: list[list[float]] = []
            for bucket_index, element in enumerate(bucket):
                bbox = element.get("bbox")
                if not _valid_normalized_bbox(bbox):
                    stats.kdl_text_fallback_regions += 1
                    stats.fallback_reasons["invalid_bbox"] += 1
                    continue
                x1, y1, x2, y2 = (float(value) for value in bbox)
                boxes.append(
                    [
                        x1 * page_width,
                        y1 * page_height,
                        x2 * page_width,
                        y2 * page_height,
                    ]
                )
                candidates.append((bucket_index, element))
            if boxes:
                pending.append((page_number, candidates, boxes))
        if not pending:
            return routed_by_page

        extract_started = perf_counter()
        try:
            if hasattr(self._extractor, "extract_pages"):
                page_results = await asyncio.to_thread(
                    self._extractor.extract_pages,
                    context.source_path,
                    [(page - 1, boxes) for page, _, boxes in pending],
                )
            else:
                page_results = await asyncio.to_thread(
                    lambda: [
                        self._extractor.extract(
                            context.source_path, page - 1, boxes
                        )
                        for page, _, boxes in pending
                    ]
                )
        except Exception as exc:
            stats.region_extraction_latency_ms += (
                perf_counter() - extract_started
            ) * 1000.0
            failed = sum(len(candidates) for _, candidates, _ in pending)
            stats.kdl_text_fallback_regions += failed
            stats.fallback_reasons["region_extraction_error"] += failed
            logger.warning(
                "pdf-inspector document region extraction failed for %s; "
                "using KDL: %s",
                context.source_path,
                exc,
            )
            return routed_by_page
        stats.region_extraction_latency_ms += (
            perf_counter() - extract_started
        ) * 1000.0

        if len(page_results) != len(pending):
            failed = sum(len(candidates) for _, candidates, _ in pending)
            stats.kdl_text_fallback_regions += failed
            stats.fallback_reasons["region_count_mismatch"] += failed
            return routed_by_page
        for (page_number, candidates, _), regions in zip(
            pending, page_results, strict=True
        ):
            if len(regions) != len(candidates):
                stats.kdl_text_fallback_regions += len(candidates)
                stats.fallback_reasons["region_count_mismatch"] += len(candidates)
                continue
            routed = routed_by_page[page_number]
            for (bucket_index, element), region in zip(
                candidates, regions, strict=True
            ):
                text = str(region.text or "").strip()
                if bool(region.needs_ocr):
                    stats.kdl_text_fallback_regions += 1
                    stats.fallback_reasons["region_needs_ocr"] += 1
                elif not text:
                    stats.kdl_text_fallback_regions += 1
                    stats.fallback_reasons["empty_region"] += 1
                else:
                    element["content"] = text
                    element["recognition_source"] = "pdf_inspector_region"
                    routed.add(bucket_index)
            stats.native_text_regions += len(routed)
        return routed_by_page

    def routing_metadata(
        self,
        context: _DocumentRoutingContext | None,
    ) -> dict[str, Any]:
        if context is None:
            return {
                "pdf_type": "routing_context_missing",
                "classification_confidence": 0.0,
                "pages_needing_ocr": [],
                "text_candidates": 0,
                "native_text_regions": 0,
                "kdl_text_fallback_regions": 0,
                "classification_latency_ms": 0.0,
                "region_extraction_latency_ms": 0.0,
            }
        stats = context.stats
        metadata: dict[str, Any] = {
            "pdf_type": context.pdf_type,
            "classification_confidence": context.confidence,
            "pages_needing_ocr": sorted(context.pages_needing_ocr),
            "text_candidates": stats.text_candidates,
            "native_text_regions": stats.native_text_regions,
            "kdl_text_fallback_regions": stats.kdl_text_fallback_regions,
            "classification_latency_ms": round(
                context.classification_latency_ms, 3
            ),
            "region_extraction_latency_ms": round(
                stats.region_extraction_latency_ms, 3
            ),
        }
        if stats.fallback_reasons:
            metadata["text_routing_fallback_reasons"] = dict(
                sorted(stats.fallback_reasons.items())
            )
        if context.classification_error:
            metadata["pdf_inspector_error"] = context.classification_error
        return metadata


class KdlPdfInspectorProvider(KDLProvider):
    """Option 3: KDL layout, native PDF text, KDL structured recognition."""

    provider_name = "kdl_pdf_inspector"
    inference_mode = "kdl_layout_pdf_inspector_text_kdl_structured"

    def __init__(
        self,
        config: KDLConfig,
        *,
        classifier: PdfInspectorClassifier | None = None,
        extractor: PdfInspectorRegionExtractor | None = None,
    ) -> None:
        resolved_classifier = classifier or PdfInspectorClassifier()
        resolved_extractor = extractor or PdfInspectorRegionExtractor()
        super().__init__(
            config,
            text_router=_PdfInspectorTextRouter(
                resolved_classifier,
                resolved_extractor,
            ),
        )


def _pdf_page_dimensions(path: Path) -> tuple[tuple[float, float], ...]:
    import fitz

    with fitz.open(str(path)) as document:
        return tuple(
            (float(page.rect.width), float(page.rect.height)) for page in document
        )


def _valid_normalized_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


__all__ = ["KdlPdfInspectorProvider"]
