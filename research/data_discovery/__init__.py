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
from .on_demand_per_query import (
    OnDemandPerQueryRunner,
    OnDemandQueryResult,
    PIPELINE_VERSION as ON_DEMAND_PER_QUERY_PIPELINE_VERSION,
    PreparedChunk,
)

__all__ = [
    "DiscoveryHit",
    "OnDemandResult",
    "PageEvidence",
    "PageIndex",
    "PdfInspectorPageParser",
    "build_page_index",
    "run_on_demand",
    "OnDemandPerQueryRunner",
    "OnDemandQueryResult",
    "ON_DEMAND_PER_QUERY_PIPELINE_VERSION",
    "PreparedChunk",
]
