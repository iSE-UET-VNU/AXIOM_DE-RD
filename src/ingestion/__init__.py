"""Data ingestion public interface."""

from .runner import IngestionInput, IngestionOutput, run, run_many

__all__ = ["IngestionInput", "IngestionOutput", "run", "run_many"]
