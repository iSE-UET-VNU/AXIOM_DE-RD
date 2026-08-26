"""Configuration loading helpers.

The sample config is JSON-compatible YAML so the project can run without
requiring PyYAML. If PyYAML is installed, regular YAML files are also accepted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import json


DEFAULT_PIPELINE_CONFIG_PATH = "configs/pipeline.kdl-pdf-inspector.yaml"


def default_pipeline_config_path() -> str:
    """Return the shared deployment/CLI config, allowing an environment override."""

    return os.getenv("AXIOM_PIPELINE_CONFIG", DEFAULT_PIPELINE_CONFIG_PATH)


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
                "PyYAML is not installed, so the pipeline config must be "
                "JSON-compatible."
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
    parser_assets_dir: str | Path,
) -> dict[str, Any]:
    """Resolve parser paths and route parsed assets into the ingested run."""
    if not isinstance(value, dict):
        return {}

    parser_config = dict(value)
    lift_config = parser_config.get("lift_api")
    if isinstance(lift_config, dict):
        lift_config = dict(lift_config)
        schema_path = lift_config.get("schema_path")
        if schema_path:
            lift_config["schema_path"] = str(resolve_project_path(project_root, schema_path))
        lift_config["output_dir"] = str(Path(parser_assets_dir))
        lift_config["project_root"] = str(project_root)
        parser_config["lift_api"] = lift_config

    chandra_config = parser_config.get("chandra2")
    if isinstance(chandra_config, dict):
        chandra_config = dict(chandra_config)
        chandra_config["output_dir"] = str(Path(parser_assets_dir))
        chandra_config["project_root"] = str(project_root)
        parser_config["chandra2"] = chandra_config

    kdl_config = parser_config.get("kdl")
    if isinstance(kdl_config, dict):
        kdl_config = dict(kdl_config)
        kdl_config["output_dir"] = str(Path(parser_assets_dir))
        kdl_config["project_root"] = str(project_root)
        parser_config["kdl"] = kdl_config

    pptx_config = parser_config.get("pptx")
    pptx_config = dict(pptx_config) if isinstance(pptx_config, dict) else {}
    pptx_config["output_dir"] = str(Path(parser_assets_dir))
    pptx_config["project_root"] = str(project_root)
    parser_config["pptx"] = pptx_config

    word_config = parser_config.get("word")
    word_config = dict(word_config) if isinstance(word_config, dict) else {}
    word_config["output_dir"] = str(Path(parser_assets_dir))
    word_config["project_root"] = str(project_root)
    soffice_path = word_config.get("soffice_path")
    if soffice_path:
        word_config["soffice_path"] = str(
            resolve_project_path(project_root, soffice_path)
        )
    parser_config["word"] = word_config
    return parser_config
