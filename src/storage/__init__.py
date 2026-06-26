"""Centralized persistence public interface."""

from .local import StorageOutput, run

__all__ = ["StorageOutput", "run"]
