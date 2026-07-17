"""Minimal ingestion file format detection.

TODO: Replace extension-based detection with content sniffing when formats grow.
"""

from __future__ import annotations

from pathlib import Path
import mimetypes


def detect_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    return suffix or "unknown"


def detect_content_type(path: str | Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"
