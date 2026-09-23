import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


class Timings:
    def __init__(self, path, **context):
        self.path, self.context = Path(path), context
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stage, unit, n, seconds, **extra):
        row = {"stage": stage, "unit": unit, "n": n, "seconds": round(seconds, 6),
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **self.context, **extra}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @contextmanager
    def timed(self, stage, unit, n, **extra):
        started = perf_counter()
        yield
        self.record(stage, unit, n, perf_counter() - started, **extra)


def rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
