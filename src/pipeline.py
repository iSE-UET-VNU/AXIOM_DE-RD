"""Pipeline orchestrator for the AXIOM_DE-RD scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import logging
import time

from . import (
    artifacts,
    cleaning,
    enrichment,
    indexing_cataloging,
    ingestion,
    integration,
    local_reader,
    s3_reader,
)
from .models import PipelineState, make_id
from .utils.config import load_config, resolve_parser_config, resolve_project_path
from .utils.env import load_dotenv_file
from .utils.paths import portable_path

MODULE_ORDER = [
    "ingestion",
    "cleaning",
    "enrichment",
    "indexing_cataloging",
    "integration",
    "artifacts",
]


def run_pipeline(
    config_path: str | Path = "configs/pipeline.yaml",
    *,
    presigned_inventory: dict[str, Any] | None = None,
    s3_uri: str | None = None,
    s3_info_file: str | Path | None = None,
    s3_object_key: str | None = None,
    s3_all_objects: bool = False,
    local_raw: str | Path | None = None,
) -> PipelineState:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv_file(project_root)
    config_file = _resolve_config_path(project_root, config_path)
    config = load_config(config_file)

    _configure_logging(config)
    logger = logging.getLogger(__name__)

    raw_root = resolve_project_path(project_root, config.get("raw_dir", "data/raw"))
    ingested_root = resolve_project_path(project_root, config.get("ingested_dir", "data/ingested"))
    cleaned_root = resolve_project_path(project_root, config.get("cleaned_dir", "data/cleaned"))
    enriched_root = resolve_project_path(project_root, config.get("enriched_dir", "data/enriched"))
    embedded_root = resolve_project_path(project_root, config.get("embedded_dir", "data/embedded"))
    output_root = resolve_project_path(project_root, config.get("output_dir", "data/output"))
    enabled_modules = set(config.get("enabled_modules", MODULE_ORDER))
    persist_artifacts = "artifacts" in enabled_modules
    run_id = make_id("pipeline-run", time.time_ns())
    raw_dir = raw_root / run_id
    ingested_dir = ingested_root / run_id
    cleaned_dir = cleaned_root / run_id
    enriched_dir = enriched_root / run_id
    embedded_dir = embedded_root / run_id
    output_dir = output_root / run_id

    state = PipelineState(
        run_id=run_id,
        input_source=None,
        raw_dir=portable_path(raw_dir, project_root),
        ingested_dir=portable_path(ingested_dir, project_root),
        cleaned_dir=portable_path(cleaned_dir, project_root),
        enriched_dir=portable_path(enriched_dir, project_root),
        embedded_dir=portable_path(embedded_dir, project_root),
        output_dir=portable_path(output_dir, project_root),
    )

    logger.info("Starting pipeline run %s", state.run_id)

    if "ingestion" in enabled_modules:
        input_config = config.get("input", {})
        if not isinstance(input_config, dict):
            raise ValueError("input configuration must be a mapping.")
        s3_config = config.get("s3_input", {})
        if not isinstance(s3_config, dict):
            raise ValueError("s3_input configuration must be a mapping.")
        input_mode = _resolve_input_mode(
            input_config,
            s3_config,
            presigned_inventory=presigned_inventory,
            s3_uri=s3_uri,
            s3_info_file=s3_info_file,
            local_raw=local_raw,
        )
        process_all_objects = (
            presigned_inventory is not None
            or s3_all_objects
            or bool(s3_config.get("all_objects", False))
        )
        if process_all_objects and input_mode != "presigned_info":
            raise ValueError("s3_all_objects requires presigned_info input.")
        if process_all_objects and s3_object_key:
            raise ValueError("Use either s3_all_objects or s3_object_key, not both.")
        parser_config = resolve_parser_config(
            project_root,
            config.get("parsing", {}),
            ingested_dir / "assets",
        )
        state.ingestion_config = parser_config
        expected_document_count = 1

        if input_mode == "local_raw":
            if s3_object_key:
                raise ValueError("s3_object_key cannot be used with local_raw input.")
            local_config = config.get("local_input", {})
            if not isinstance(local_config, dict):
                raise ValueError("local_input configuration must be a mapping.")
            batch = local_reader.read_local_inputs(
                local_config,
                local_raw=local_raw,
                project_root=project_root,
            )
            state.input_source = batch.source
            state.raw_dir = batch.source
            expected_document_count = len(batch.inputs)
            if _uses_continuous_chandra_queue(parser_config):
                logger.info(
                    "Submitting %s local document(s) to the continuous Chandra2 page queue",
                    len(batch.inputs),
                )
                partial_result = ingestion.run_many(
                    [
                        ingestion.IngestionInput(
                            path=item.path,
                            source_uri=item.source_uri,
                            metadata=item.metadata,
                        )
                        for item in batch.inputs
                    ],
                    parser_config=parser_config,
                    project_root=project_root,
                )
                _accept_ingestion_result(
                    state,
                    partial_result,
                    ingested_dir,
                    project_root,
                    expected_document_count,
                    persist_artifacts,
                )
            else:
                for item in batch.inputs:
                    partial_result = ingestion.run(
                        item.path,
                        source_uri=item.source_uri,
                        input_metadata=item.metadata,
                        parser_config=parser_config,
                        project_root=project_root,
                    )
                    _accept_ingestion_result(
                        state,
                        partial_result,
                        ingested_dir,
                        project_root,
                        expected_document_count,
                        persist_artifacts,
                    )
        else:
            resolved_s3_config = {**s3_config, "mode": input_mode}
            raw_objects_dir = raw_dir / "objects"
            if process_all_objects:
                batch = s3_reader.read_all_presigned_s3_inputs(
                    resolved_s3_config,
                    raw_objects_dir,
                    s3_info_file=s3_info_file,
                    inventory=presigned_inventory,
                    project_root=project_root,
                )
                state.input_source = batch.source
                expected_document_count = len(batch.objects)
                for index, item in enumerate(batch.objects):
                    logger.info(
                        "Ingesting presigned S3 object %s/%s: %s",
                        index + 1,
                        len(batch.objects),
                        item.source_uri,
                    )
                    partial_result = ingestion.run(
                        item.path,
                        source_uri=item.source_uri,
                        input_metadata=item.metadata,
                        parser_config=parser_config,
                        project_root=project_root,
                    )
                    _accept_ingestion_result(
                        state,
                        partial_result,
                        ingested_dir,
                        project_root,
                        expected_document_count,
                        persist_artifacts,
                    )
            else:
                downloaded = s3_reader.read_s3_input(
                    resolved_s3_config,
                    raw_objects_dir / "000000",
                    s3_uri=s3_uri,
                    s3_info_file=s3_info_file,
                    s3_object_key=s3_object_key,
                    project_root=project_root,
                )
                state.input_source = downloaded.source_uri
                partial_result = ingestion.run(
                    downloaded.path,
                    source_uri=downloaded.source_uri,
                    input_metadata=downloaded.metadata,
                    parser_config=parser_config,
                    project_root=project_root,
                )
                _accept_ingestion_result(
                    state,
                    partial_result,
                    ingested_dir,
                    project_root,
                    expected_document_count,
                    persist_artifacts,
                )
        state.completed_modules.append("ingestion")
        if persist_artifacts:
            _record_artifact_paths(
                state,
                artifacts.write_ingested_artifacts(
                    state,
                    ingested_dir,
                    project_root,
                    status=(
                        "completed_with_errors"
                        if state.quarantined_documents
                        else "completed"
                    ),
                    expected_document_count=expected_document_count,
                    document_ids=[],
                ),
                project_root,
            )
            logger.info("Wrote ingestion artifacts to %s", ingested_dir)
        logger.info(
            "Ingestion completed with %s succeeded and %s quarantined document(s)",
            len(state.data_objects),
            len(state.quarantined_documents),
        )

    if "cleaning" in enabled_modules:
        result = cleaning.run(state.parsed_data, state.initial_schemas)
        state.cleaned_data = result.cleaned_data
        state.cleaned_schemas = result.cleaned_schemas
        state.completed_modules.append("cleaning")
        if persist_artifacts:
            _record_artifact_paths(
                state,
                artifacts.write_cleaned_artifacts(state, cleaned_dir, project_root),
                project_root,
            )
            logger.info("Wrote cleaning artifacts to %s", cleaned_dir)
        logger.info(
            "Cleaning pass-through produced %s dataset(s)",
            len(state.cleaned_data),
        )

    if "enrichment" in enabled_modules:
        result = enrichment.run(state.cleaned_data, state.cleaned_schemas)
        state.enriched_data = result.enriched_data
        state.enriched_schemas = result.enriched_schemas
        state.completed_modules.append("enrichment")
        if persist_artifacts:
            _record_artifact_paths(
                state,
                artifacts.write_enriched_artifacts(state, enriched_dir, project_root),
                project_root,
            )
            logger.info("Wrote enrichment artifacts to %s", enriched_dir)
        logger.info(
            "Enrichment pass-through produced %s dataset(s)",
            len(state.enriched_data),
        )

    if "indexing_cataloging" in enabled_modules:
        result = indexing_cataloging.run(
            state.enriched_data,
            state.enriched_schemas,
            indexing_config=config.get("indexing", {}),
        )
        state.metadata_records = result.metadata_records
        state.index_records = result.index_records
        state.index_quality_report = result.index_quality_report
        state.vector_records = result.vector_records
        state.embedding_report = result.embedding_report
        state.completed_modules.append("indexing_cataloging")
        logger.info(
            "Built %s metadata record(s), %s index record(s), and %s vector record(s); quality status: %s",
            len(state.metadata_records),
            len(state.index_records),
            len(state.vector_records),
            state.index_quality_report.get("status", "unknown"),
        )

    if "integration" in enabled_modules:
        result = integration.run(state.index_records)
        state.index_records = result.passed_index_records
        state.schema_matches = result.schema_matches
        state.entity_matches = result.entity_matches
        state.relationship_records = result.relationship_records
        state.completed_modules.append("integration")
        logger.info(
            "Integration pass-through accepted %s index record(s)",
            len(result.passed_index_records),
        )

    if persist_artifacts:
        state.completed_modules.append("artifacts")
        artifacts.write_embedded_artifacts(
            state,
            embedded_dir,
            project_root=project_root,
        )
        artifacts.write_output_artifacts(
            state,
            output_dir,
            project_root=project_root,
        )
        logger.info("Wrote embedding and index artifacts to %s", embedded_dir)
        logger.info("Wrote consolidated output artifacts to %s", output_dir)

    if state.errors:
        logger.warning(
            "Finished pipeline run %s with %s error(s), including %s quarantined document(s)",
            state.run_id,
            len(state.errors),
            len(state.quarantined_documents),
        )
    else:
        logger.info("Finished pipeline run %s", state.run_id)
    return state


def _accept_ingestion_result(
    state: PipelineState,
    result: ingestion.IngestionOutput,
    ingested_dir: Path,
    project_root: Path,
    expected_document_count: int,
    persist_artifacts: bool,
) -> None:
    """Add one parsed result to state and persist it before continuing."""
    state.data_objects.extend(result.data_objects)
    state.parsed_data.extend(result.parsed_data)
    state.initial_schemas.extend(result.initial_schemas)
    state.quarantined_documents.extend(result.quarantined_documents)
    state.errors.extend(
        {
            "code": "document_quarantined",
            "stage": "ingestion",
            "status": "quarantined",
            "document_id": record.document_id,
            "source_uri": record.source.uri,
            "reasons": record.reasons,
        }
        for record in result.quarantined_documents
    )

    if not persist_artifacts:
        return

    document_ids = [record.object_id for record in result.data_objects]
    document_ids.extend(record.document_id for record in result.quarantined_documents)
    _record_artifact_paths(
        state,
        artifacts.write_ingested_artifacts(
            state,
            ingested_dir,
            project_root,
            status="in_progress",
            expected_document_count=expected_document_count,
            document_ids=document_ids,
        ),
        project_root,
    )
    logging.getLogger(__name__).info(
        "Persisted ingested document %s/%s: %s",
        len(state.data_objects) + len(state.quarantined_documents),
        expected_document_count,
        ", ".join(document_ids),
    )


def cli(argv: list[str] | None = None) -> PipelineState:
    parser = argparse.ArgumentParser(description="Run the AXIOM_DE-RD pipeline scaffold.")
    parser.add_argument("--config", default="configs/pipeline.yaml", help="Path to pipeline config.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--s3-uri",
        default=None,
        help="S3 URI override in s3://bucket/key form. Defaults to S3_URI.",
    )
    input_group.add_argument(
        "--s3-info-file",
        default=None,
        help="JSON inventory containing object keys and presigned URLs.",
    )
    input_group.add_argument(
        "--local-raw",
        default=None,
        help="Local raw file or directory. Uses local_input discovery settings.",
    )
    parser.add_argument(
        "--s3-object-key",
        default=None,
        help="Object key to select from --s3-info-file. Defaults to S3_OBJECT_KEY.",
    )
    parser.add_argument(
        "--s3-all-objects",
        action="store_true",
        help="Process every presigned object in the selected S3 info file.",
    )
    args = parser.parse_args(argv)

    state = run_pipeline(
        args.config,
        s3_uri=args.s3_uri,
        s3_info_file=args.s3_info_file,
        s3_object_key=args.s3_object_key,
        s3_all_objects=args.s3_all_objects,
        local_raw=args.local_raw,
    )
    run_status = "completed_with_errors" if state.errors else "completed"
    print(f"Pipeline run {state.run_id} {run_status}.")
    print(f"Modules: {', '.join(state.completed_modules) or 'none'}")
    if state.quarantined_documents:
        print(f"Quarantined documents: {len(state.quarantined_documents)}")
    if state.artifact_paths:
        print(f"Artifacts: {state.artifact_paths.get('output_metadata')}")
    return state


def _resolve_config_path(project_root: Path, config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate

    return project_root / path


def _resolve_input_mode(
    input_config: dict[str, Any],
    s3_config: dict[str, Any],
    *,
    presigned_inventory: dict[str, Any] | None,
    s3_uri: str | None,
    s3_info_file: str | Path | None,
    local_raw: str | Path | None,
) -> str:
    explicit_modes = [
        mode
        for mode, value in (
            ("presigned_info", presigned_inventory),
            ("s3_uri", s3_uri),
            ("presigned_info", s3_info_file),
            ("local_raw", local_raw),
        )
        if value is not None
    ]
    if len(explicit_modes) > 1:
        raise ValueError("Use only one input source per pipeline run.")
    configured_mode = input_config.get("mode", s3_config.get("mode", "s3_uri"))
    mode = explicit_modes[0] if explicit_modes else str(configured_mode)
    mode = mode.strip().lower()
    if mode not in {"s3_uri", "presigned_info", "local_raw"}:
        raise ValueError(
            "input.mode must be 's3_uri', 'presigned_info', or 'local_raw'."
        )
    return mode


def _configure_logging(config: dict[str, Any]) -> None:
    logging_config = config.get("logging", {})
    level_name = "INFO"
    if isinstance(logging_config, dict):
        level_name = str(logging_config.get("level", level_name))
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s %(name)s - %(message)s",
    )


def _uses_continuous_chandra_queue(parser_config: dict[str, Any]) -> bool:
    provider = str(parser_config.get("provider") or "").strip().lower()
    document_config = parser_config.get("document")
    if provider in {"", "router"} and isinstance(document_config, dict):
        provider = str(document_config.get("provider") or "").strip().lower()
    provider = provider.replace("-", "_")
    chandra_config = parser_config.get("chandra2")
    return (
        provider in {"chandra", "chandra2", "chandra_2"}
        and isinstance(chandra_config, dict)
        and bool(chandra_config.get("continuous_page_queue", False))
    )


def _record_artifact_paths(
    state: PipelineState,
    paths: dict[str, Path],
    project_root: Path,
) -> None:
    state.artifact_paths.update(
        {name: portable_path(path, project_root) for name, path in paths.items()}
    )


if __name__ == "__main__":
    cli()
