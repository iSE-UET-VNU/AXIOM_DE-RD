"""Environment loading helpers for local scripts."""

from __future__ import annotations

from pathlib import Path
import os

from dotenv import dotenv_values, load_dotenv


def load_dotenv_file(project_root: str | Path, file_name: str = ".env", override: bool = False) -> dict[str, str]:
    """Load a project .env file into process env using python-dotenv."""
    env_path = Path(project_root) / file_name
    if not env_path.exists():
        return {}

    values = dotenv_values(env_path)
    load_dotenv(env_path, override=override)

    return {
        key: os.environ[key]
        for key, value in values.items()
        if value is not None and key in os.environ
    }
