"""Main entrypoint for the AXIOM_DE-RD pipeline."""

from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.pipeline import cli, run_pipeline
else:
    from .pipeline import cli, run_pipeline

__all__ = ["cli", "run_pipeline"]


if __name__ == "__main__":
    cli()
