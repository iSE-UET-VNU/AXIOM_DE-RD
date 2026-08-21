"""On-demand, page-level data discovery for PDF data lakes."""

from .pipeline import (
    DiscoveryHit,
    OnDemandResult,
    PageEvidence,
    PageIndex,
    PdfInspectorPageParser,
    build_page_index,
    run_on_demand,
)

__all__ = [
    "DiscoveryHit",
    "OnDemandResult",
    "PageEvidence",
    "PageIndex",
    "PdfInspectorPageParser",
    "build_page_index",
    "run_on_demand",
]
