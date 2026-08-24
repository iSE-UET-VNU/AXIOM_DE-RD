"""Data ingestion public interface."""

from .runner import IngestionInput, IngestionOutput, object_id_for, run, run_many

__all__ = ["IngestionInput", "IngestionOutput", "object_id_for", "run", "run_many"]
