"""Public interface for writing stage artifacts to local files."""

from .ingested import write_ingested_artifacts
from .pipeline_output import (
    ArtifactOutput,
    write_cleaned_artifacts,
    write_embedded_artifacts,
    write_enriched_artifacts,
    write_output_artifacts,
)

__all__ = [
    "ArtifactOutput",
    "write_cleaned_artifacts",
    "write_embedded_artifacts",
    "write_enriched_artifacts",
    "write_ingested_artifacts",
    "write_output_artifacts",
]
