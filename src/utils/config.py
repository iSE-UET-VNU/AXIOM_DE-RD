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
