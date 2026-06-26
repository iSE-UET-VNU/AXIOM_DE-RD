"""Run only the ingestion/parsing stage."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import run as run_ingestion  # noqa: E402
from src.storage.local import LocalArtifactStore  # noqa: E402
from src.utils.config import load_config, resolve_project_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AXIOM parsing only.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = load_config(config_path)

    input_dir = resolve_project_path(
        PROJECT_ROOT,
        args.input_dir or config.get("input_dir", "data/raw"),
    )
    output_dir = resolve_project_path(
        PROJECT_ROOT,
        args.output_dir or config.get("processed_dir", "data/processed"),
    )

    result = run_ingestion(input_dir, parser_config=_resolve_parser_config(config.get("parsing", {})))

    store = LocalArtifactStore(output_dir)
    paths = {
        "data_objects": store.write_json("data_objects.json", result.data_objects),
        "parsed_data": store.write_json("parsed_data.json", result.parsed_data),
        "initial_schemas": store.write_json("initial_schemas.json", result.initial_schemas),
    }

    print(f"Parsed {len(result.parsed_data)} object(s).")
    print(f"Artifacts: {output_dir}")
    for name, path in paths.items():
        print(f"- {name}: {path}")


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    return PROJECT_ROOT / path


def _resolve_parser_config(value: object) -> dict:
    if not isinstance(value, dict):
        return {}

    parser_config = dict(value)
    lift_config = parser_config.get("lift_api")
    if isinstance(lift_config, dict):
        lift_config = dict(lift_config)
        for key in ("output_dir", "schema_path"):
            path_value = lift_config.get(key)
            if path_value:
                lift_config[key] = str(resolve_project_path(PROJECT_ROOT, path_value))
        parser_config["lift_api"] = lift_config
    return parser_config


if __name__ == "__main__":
    main()
