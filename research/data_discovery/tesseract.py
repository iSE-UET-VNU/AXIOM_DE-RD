"""Small, cache-friendly Tesseract adapter for page-level preparation.

The adapter deliberately uses the Tesseract executable instead of pytesseract
so that the light-preparation runner can record the exact command, timings and
page-level failures without adding a Python OCR dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import csv
import shutil
import statistics
import subprocess
import time
from typing import Any


@dataclass(frozen=True)
class OCRResult:
    """Result and timing information for one source page."""

    text: str
    word_count: int
    mean_confidence: float
    render_seconds: float
    ocr_seconds: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "mean_confidence": round(float(self.mean_confidence), 4),
            "render_seconds": round(float(self.render_seconds), 6),
            "ocr_seconds": round(float(self.ocr_seconds), 6),
            "error": self.error,
        }


def resolve_tesseract(command: str | Path) -> str:
    """Resolve a configured executable or fail with an actionable message."""

    value = str(command)
    candidate = Path(value)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(value)
    if found:
        return found
    if value == "tesseract" and Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").is_file():
        return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    raise FileNotFoundError(
        f"Tesseract executable not found: {command!r}. "
        "Install Tesseract or pass --tesseract explicitly."
    )


def check_tesseract(command: str | Path, language: str) -> dict[str, Any]:
    """Validate the executable and requested traineddata before OCR starts."""

    binary = resolve_tesseract(command)
    version = subprocess.run(
        [binary, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )
    if version.returncode != 0:
        raise RuntimeError(
            f"Could not execute Tesseract {binary}: {version.stdout.strip()}"
        )
    languages = subprocess.run(
        [binary, "--list-langs"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=30,
    )
    available = {
        line.strip()
        for line in languages.stdout.splitlines()
        if line.strip() and not line.lower().startswith("list of available")
    }
    requested = {part.strip() for part in language.split("+") if part.strip()}
    missing = sorted(requested - available)
    if missing:
        raise RuntimeError(
            f"Tesseract language data missing: {missing}. "
            f"Available languages: {sorted(available)}"
        )
    first_line = version.stdout.splitlines()[0] if version.stdout else "unknown"
    return {
        "executable": binary,
        "version": first_line.strip(),
        "language": language,
        "available_languages": sorted(available),
    }


def _render_input(path: Path, page_index: int, dpi: int) -> tuple[bytes, float]:
    started = time.perf_counter()
    if path.suffix.lower() == ".pdf":
        import fitz

        with fitz.open(str(path)) as document:
            page = document.load_page(int(page_index))
            pixmap = page.get_pixmap(dpi=int(dpi), alpha=False)
            image_bytes = pixmap.tobytes("png")
    else:
        from PIL import Image

        with Image.open(path) as image:
            output = BytesIO()
            image.convert("RGB").save(output, format="PNG")
            image_bytes = output.getvalue()
    return image_bytes, time.perf_counter() - started


def _parse_tsv(payload: bytes) -> tuple[str, int, float]:
    words: list[str] = []
    confidences: list[float] = []
    text = payload.decode("utf-8", errors="replace")
    for row in csv.DictReader(StringIO(text), delimiter="\t"):
        token = (row.get("text") or "").strip()
        if not token:
            continue
        words.append(token)
        try:
            confidence = float(row.get("conf", "-1"))
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence >= 0:
            confidences.append(confidence)
    return (
        " ".join(words),
        len(words),
        statistics.mean(confidences) if confidences else 0.0,
    )


def ocr_page(
    path: str | Path,
    page_index: int,
    *,
    tesseract: str | Path,
    language: str,
    dpi: int = 144,
    psm: int = 3,
    tessdata_dir: str | Path | None = None,
    timeout_seconds: float = 120.0,
) -> OCRResult:
    """Render and OCR one zero-based PDF page or one image input."""

    source = Path(path)
    started = time.perf_counter()
    try:
        image_bytes, render_seconds = _render_input(source, page_index, dpi)
        command = [
            resolve_tesseract(tesseract),
            "stdin",
            "stdout",
            "-l",
            language,
        ]
        if tessdata_dir is not None:
            command.extend(["--tessdata-dir", str(tessdata_dir)])
        command.extend(["--psm", str(int(psm)), "tsv"])
        ocr_started = time.perf_counter()
        completed = subprocess.run(
            command,
            input=image_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=float(timeout_seconds),
        )
        ocr_seconds = time.perf_counter() - ocr_started
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", errors="replace").strip()
            return OCRResult(
                "",
                0,
                0.0,
                render_seconds,
                ocr_seconds,
                error or f"tesseract exit={completed.returncode}",
            )
        text, word_count, confidence = _parse_tsv(completed.stdout)
        return OCRResult(text, word_count, confidence, render_seconds, ocr_seconds)
    except Exception as exc:  # noqa: BLE001 - persist page-level failures
        elapsed = time.perf_counter() - started
        return OCRResult(
            "",
            0,
            0.0,
            elapsed,
            0.0,
            f"{type(exc).__name__}: {exc}",
        )
