"""Export enriched_data.json from a completed pipeline run.

The chunking stage consumes enrichment output, so this reads
``data/enriched/<run-id>/`` and flattens it into the single file
``scripts/run_chunking_embedding.py`` expects.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_trace import export_enriched_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Export enriched_data.json from a pipeline run.")
    parser.add_argument("--run-id", required=True, help="Pipeline run id under data/enriched/<run-id>.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for enriched_data.json (default: data/work/chunking_embedding_<run-id>).",
    )
    args = parser.parse_args()

    out_dir = export_enriched_data(args.run_id, project_root=PROJECT_ROOT, output_dir=args.out)
    print(f"enriched_data.json written to {out_dir}")


if __name__ == "__main__":
    main()
