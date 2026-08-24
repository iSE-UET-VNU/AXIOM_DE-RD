"""Render an HTML trace report for a pipeline run and optional chunking run."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_trace import write_trace_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a pipeline trace report.")
    parser.add_argument("--run-id", required=True, help="Pipeline run id under data/<stage>/<run-id>.")
    parser.add_argument(
        "--chunk-dir",
        default=None,
        help="Optional chunking_embedding output directory to visualize alongside the pipeline run.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory or .html file path (default: data/output/traces/<run-id>).",
    )
    args = parser.parse_args()

    html_path, json_path = write_trace_report(
        args.run_id,
        project_root=PROJECT_ROOT,
        chunk_dir=args.chunk_dir,
        out=args.out,
    )
    print(f"trace report: {html_path}")
    print(f"trace data:   {json_path}")


if __name__ == "__main__":
    main()
