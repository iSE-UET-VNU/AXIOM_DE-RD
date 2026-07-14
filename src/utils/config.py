"""Configuration loading helpers.

The sample config is JSON-compatible YAML so the project can run without
requiring PyYAML. If PyYAML is installed, regular YAML files are also accepted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is not installed, so pipeline.yaml must be JSON-compatible."
            ) from exc
    else:
        data = yaml.safe_load(text) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return data


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def resolve_parser_config(
    project_root: Path,
    value: Any,
    work_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Resolve parser paths and route provider artifacts into this run's work directory."""
    if not isinstance(value, dict):
        return {}

    parser_config = dict(value)
    lift_config = parser_config.get("lift_api")
    if isinstance(lift_config, dict):
        lift_config = dict(lift_config)
        schema_path = lift_config.get("schema_path")
        if schema_path:
            lift_config["schema_path"] = str(resolve_project_path(project_root, schema_path))
        lift_config["output_dir"] = str(Path(work_dir) / run_id / "datalab")
        lift_config["project_root"] = str(project_root)
        parser_config["lift_api"] = lift_config
    return parser_config
