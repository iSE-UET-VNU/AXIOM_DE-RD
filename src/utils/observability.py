"""Small, dependency-free helpers for durable pipeline observability."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return an explicit UTC timestamp suitable for JSONL checkpoints."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def configure_run_logging(
    log_path: str | Path | None = None, *, level: str | int | None = None
) -> None:
    """Configure console logging and, when requested, a durable log file."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(
        level=(
            level
            if level is not None
            else os.getenv("AXIOM_LOG_LEVEL", "INFO").upper()
        ),
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
        force=True,
    )


class JsonEventLogger:
    """Append timestamped, machine-readable run events.

    Logging must never make a benchmark fail, so an unavailable event path is
    reported through the normal logger and otherwise ignored.
    """

    def __init__(self, path: str | Path | None, *, run_name: str) -> None:
        self.path = Path(path) if path else None
        self.run_name = run_name
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        if self.path is None:
            return
        record = {
            "timestamp_utc": utc_now_iso(),
            "event": event,
            "run": self.run_name,
            "pid": os.getpid(),
            **fields,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                handle.flush()
        except Exception:  # pragma: no cover - diagnostics must be best effort
            logging.getLogger(__name__).debug(
                "Could not append run event to %s", self.path, exc_info=True
            )
