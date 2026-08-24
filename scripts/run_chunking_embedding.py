"""Run the chunking + embedding stage with explicit local-friendly overrides."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking_embedding import run_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chunking + embedding stage.")
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing enriched_data.json (or documents/*.json).",
    )
    parser.add_argument("--output", required=True, help="Directory for chunk/vector/report artifacts.")
    parser.add_argument("--chunker", default="recursive", help="Chunker name.")
    parser.add_argument(
        "--chunker-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Chunker parameter; repeat for multiple values. VALUE is parsed as JSON when possible.",
    )
    parser.add_argument("--embedder", default="local_hash", help="Embedder name.")
    parser.add_argument("--dimension", type=int, default=256, help="Embedder vector dimension.")
    parser.add_argument(
        "--embedder-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Embedder parameter; repeat for multiple values. VALUE is parsed as JSON when possible.",
    )
    parser.add_argument(
        "--retrieval-profile",
        default="hybrid_default",
        help="Retrieval profile name for the report.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild even if records already exist.")
    args = parser.parse_args()

    config = {
        "chunking_embedding": {
            "chunker": args.chunker,
            "chunker_params": _parse_params(args.chunker_param),
            "max_rows_per_chunk": 20,
            "embedder": args.embedder,
            "embedder_params": {"dimension": args.dimension, **_parse_params(args.embedder_param)},
            "retrieval_profile": args.retrieval_profile,
        }
    }

    report = run_cli(config, Path(args.input), Path(args.output), force=args.force)
    docs = report["docs"]
    print(
        f"Processed {docs['indexed']}/{docs['total']} document(s) "
        f"({docs['skipped_existing']} already done, {docs['skipped_failed']} failed)."
    )
    print(f"Artifacts: {Path(args.output).resolve()}")
    print(f"Chunk counts: {report['chunk_counts']}")
    print(f"Report: {Path(args.output) / 'embedding_report.json'}")


def _parse_params(values: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Expected KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty parameter name in {raw!r}")
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


if __name__ == "__main__":
    main()
