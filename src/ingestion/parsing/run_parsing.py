"""Run only the ingestion/parsing stage."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import run as run_ingestion  # noqa: E402
from src.models import PipelineState, make_id  # noqa: E402
from src.storage.local import write_processed_artifacts  # noqa: E402
from src.utils.config import load_config, resolve_parser_config, resolve_project_path  # noqa: E402
from src.utils.env import load_dotenv_file  # noqa: E402
from src.utils.paths import portable_path  # noqa: E402


def main() -> None:
    load_dotenv_file(PROJECT_ROOT)

    parser = argparse.ArgumentParser(description="Run AXIOM parsing only.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--work-dir", default=None)
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
    work_dir = resolve_project_path(
        PROJECT_ROOT,
        args.work_dir or config.get("work_dir", "data/work"),
    )
    run_id = make_id("pipeline-run", time.time_ns())
    parser_config = resolve_parser_config(
        PROJECT_ROOT,
        config.get("parsing", {}),
        work_dir,
        run_id,
    )

    result = run_ingestion(
        input_dir,
        parser_config=parser_config,
        project_root=PROJECT_ROOT,
    )

    state = PipelineState(
        run_id=run_id,
        input_dir=portable_path(input_dir, PROJECT_ROOT),
        output_dir=portable_path(output_dir, PROJECT_ROOT),
        work_dir=portable_path(work_dir, PROJECT_ROOT),
        data_objects=result.data_objects,
        parsed_data=result.parsed_data,
        initial_schemas=result.initial_schemas,
        normalized_texts=result.normalized_texts,
        normalized_images=result.normalized_images,
        normalized_tables=result.normalized_tables,
        normalized_documents=result.documents,
        ingestion_config=parser_config,
        completed_modules=["ingestion"],
    )
    paths = write_processed_artifacts(state, output_dir, PROJECT_ROOT)

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


if __name__ == "__main__":
    main()
