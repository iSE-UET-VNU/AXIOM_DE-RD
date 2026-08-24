"""Run the page-level data-discovery baseline from the repository root."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.data_discovery.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
