"""Small, provider-neutral wrappers around the pdf-inspector Python API."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


@dataclass(frozen=True)
class PdfInspectorClassification:
    """Document-level routing facts returned by pdf-inspector."""

    pdf_type: str
    confidence: float
    page_count: int
    pages_needing_ocr: frozenset[int]
    latency_ms: float


@dataclass(frozen=True)
class PdfInspectorRegionText:
    """Normalized text result for one requested PDF region."""

    text: str
    needs_ocr: bool
    ocr_reason: str | None = None


class PdfInspectorClassifier:
    """Classify PDFs without coupling the result to a recognition model."""

    def __init__(self, api: Any | None = None) -> None:
        self._api = api if api is not None else _load_pdf_inspector()

    def classify(self, path: str | Path) -> PdfInspectorClassification:
        started = perf_counter()
        result = self._api.classify_pdf(str(path))
        return PdfInspectorClassification(
            pdf_type=str(result.pdf_type),
            confidence=float(result.confidence),
            page_count=int(result.page_count),
            pages_needing_ocr=frozenset(
                int(page) for page in result.pages_needing_ocr
            ),
            latency_ms=(perf_counter() - started) * 1000.0,
        )


class PdfInspectorRegionExtractor:
    """Extract native text from a batch of regions on one PDF page."""

    def __init__(self, api: Any | None = None) -> None:
        self._api = api if api is not None else _load_pdf_inspector()

    def extract(
        self,
        path: str | Path,
        page_index: int,
        boxes: Sequence[Sequence[float]],
    ) -> list[PdfInspectorRegionText]:
        page_results = self._api.extract_text_in_regions(
            str(path),
            [(int(page_index), [[float(value) for value in box] for box in boxes])],
        )
        if len(page_results) != 1:
            raise ValueError(
                "pdf-inspector returned a different number of pages than requested"
            )
        return [
            PdfInspectorRegionText(
                text=str(region.text or ""),
                needs_ocr=bool(region.needs_ocr),
                ocr_reason=(
                    str(region.ocr_reason)
                    if getattr(region, "ocr_reason", None) is not None
                    else None
                ),
            )
            for region in page_results[0].regions
        ]


def ensure_pdf_inspector_available() -> None:
    """Fail early with an actionable message when the optional extra is absent."""

    _load_pdf_inspector()


def _load_pdf_inspector() -> Any:
    try:
        return import_module("pdf_inspector")
    except ModuleNotFoundError as exc:
        if exc.name != "pdf_inspector":
            raise
        raise RuntimeError(
            "The kdl_pdf_inspector provider requires the optional pdf-inspector "
            "package. Install it with `pip install -e .[pdf-inspector]`."
        ) from exc


__all__ = [
    "PdfInspectorClassification",
    "PdfInspectorClassifier",
    "PdfInspectorRegionExtractor",
    "PdfInspectorRegionText",
    "ensure_pdf_inspector_available",
]
