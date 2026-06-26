"""Pipeline orchestrator for the AXIOM_DE-RD scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import logging
import time

from . import cleaning, enrichment, indexing_cataloging, ingestion, integration, storage
from .models import PipelineState, make_id
from .utils.config import load_config, resolve_project_path

MODULE_ORDER = [
    "ingestion",
    "cleaning",
    "enrichment",
    "indexing_cataloging",
    "integration",
    "storage",
]


def run_pipeline(config_path: str | Path = "configs/pipeline.yaml") -> PipelineState:
    project_root = Path(__file__).resolve().parents[1]
    config_file = _resolve_config_path(project_root, config_path)
    config = load_config(config_file)

    _configure_logging(config)
    logger = logging.getLogger(__name__)

    input_dir = resolve_project_path(project_root, config.get("input_dir", "data/raw"))
    processed_dir = resolve_project_path(project_root, config.get("processed_dir", config.get("output_dir", "data/processed")))
    cleaned_dir = resolve_project_path(project_root, config.get("cleaned_dir", "data/cleaned"))
    enriched_dir = resolve_project_path(project_root, config.get("enriched_dir", "data/enriched"))
    output_dir = resolve_project_path(project_root, config.get("output_dir", "data/output"))
    artifact_dir = _artifact_dir(project_root, output_dir, config)
    enabled_modules = set(config.get("enabled_modules", MODULE_ORDER))

    state = PipelineState(
        run_id=make_id("pipeline-run", time.time_ns()),
        input_dir=str(input_dir),
        output_dir=str(output_dir),
    )

    logger.info("Starting pipeline run %s", state.run_id)

    if "ingestion" in enabled_modules:
        result = ingestion.run(input_dir, parser_config=_resolve_parser_config(project_root, config.get("parsing", {})))
        state.data_objects = result.data_objects
        state.parsed_data = result.parsed_data
        state.initial_schemas = result.initial_schemas
        state.completed_modules.append("ingestion")
        logger.info("Ingested %s data object(s)", len(state.data_objects))

    if "cleaning" in enabled_modules:
        result = cleaning.run(state.parsed_data, state.initial_schemas)
        state.cleaned_data = result.cleaned_data
        state.cleaned_schemas = result.cleaned_schemas
        state.completed_modules.append("cleaning")
        logger.info("Cleaning pass-through produced %s dataset(s)", len(state.cleaned_data))

    if "enrichment" in enabled_modules:
        result = enrichment.run(state.cleaned_data, state.cleaned_schemas)
        state.enriched_data = result.enriched_data
        state.enriched_schemas = result.enriched_schemas
        state.completed_modules.append("enrichment")
        logger.info("Enrichment pass-through produced %s dataset(s)", len(state.enriched_data))

    if "indexing_cataloging" in enabled_modules:
        result = indexing_cataloging.run(state.enriched_data, state.enriched_schemas)
        state.metadata_records = result.metadata_records
        state.index_records = result.index_records
        state.completed_modules.append("indexing_cataloging")
        logger.info(
            "Built %s metadata record(s) and %s index record(s)",
            len(state.metadata_records),
            len(state.index_records),
        )

    if "integration" in enabled_modules:
        result = integration.run(state.index_records)
        state.schema_matches = result.schema_matches
        state.entity_matches = result.entity_matches
        state.relationship_records = result.relationship_records
        state.completed_modules.append("integration")
        logger.info(
            "Integration pass-through accepted %s index record(s)",
            len(result.passed_index_records),
        )

    if "storage" in enabled_modules:
        mode = str(config.get("storage", {}).get("mode", "local"))
        storage.run(
            state,
            artifact_dir,
            mode=mode,
            processed_dir=processed_dir,
            cleaned_dir=cleaned_dir,
            enriched_dir=enriched_dir,
        )
        state.completed_modules.append("storage")
        logger.info("Wrote artifacts to %s", artifact_dir)

    logger.info("Finished pipeline run %s", state.run_id)
    return state


def cli(argv: list[str] | None = None) -> PipelineState:
    parser = argparse.ArgumentParser(description="Run the AXIOM_DE-RD pipeline scaffold.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config.")
    args = parser.parse_args(argv)

    state = run_pipeline(args.config)
    print(f"Pipeline run {state.run_id} completed.")
    print(f"Modules: {', '.join(state.completed_modules) or 'none'}")
    if state.artifact_paths:
        print(f"Artifacts: {state.artifact_paths.get('pipeline_state')}")
    return state


def _resolve_config_path(project_root: Path, config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    return project_root / path


def _artifact_dir(project_root: Path, output_dir: Path, config: dict[str, Any]) -> Path:
    storage_config = config.get("storage", {})
    local_config = storage_config.get("local", {}) if isinstance(storage_config, dict) else {}
    configured = local_config.get("artifacts_dir") if isinstance(local_config, dict) else None
    if configured:
        return resolve_project_path(project_root, configured)
    return output_dir / "artifacts"


def _resolve_parser_config(project_root: Path, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    parser_config = dict(value)
    lift_config = parser_config.get("lift_api")
    if isinstance(lift_config, dict):
        lift_config = dict(lift_config)
        for key in ("output_dir", "schema_path"):
            path_value = lift_config.get(key)
            if path_value:
                lift_config[key] = str(resolve_project_path(project_root, path_value))
        parser_config["lift_api"] = lift_config
    return parser_config


def _configure_logging(config: dict[str, Any]) -> None:
    logging_config = config.get("logging", {})
    level_name = "INFO"
    if isinstance(logging_config, dict):
        level_name = str(logging_config.get("level", level_name))
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s %(name)s - %(message)s",
    )


if __name__ == "__main__":
    cli()
