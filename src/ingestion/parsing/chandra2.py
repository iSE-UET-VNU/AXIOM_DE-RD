"""Chandra2 document provider backed by a self-hosted vLLM server."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ...models import DataObject, ParsedData
from ...utils.paths import portable_path

CHANDRA2_EXTENSIONS = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"}
)


@dataclass(frozen=True)
class Chandra2Config:
    """Runtime and artifact settings for Chandra2 vLLM inference."""

    batch_size: int = 28
    max_workers: int = 4
    max_output_tokens: int = 12384
    max_retries: int = 6
    include_headers_footers: bool = False
    save_raw_outputs: bool = True
    output_dir: str | None = "data/work/chandra2"
    project_root: str | None = None

    @classmethod
    def from_mapping(cls, config: dict[str, Any] | None) -> "Chandra2Config":
        config = config or {}
        return cls(
            batch_size=_positive_int(config, "batch_size", 28),
            max_workers=_positive_int(config, "max_workers", 4),
            max_output_tokens=_positive_int(config, "max_output_tokens", 12384),
            max_retries=_non_negative_int(config, "max_retries", 6),
            include_headers_footers=bool(
                config.get("include_headers_footers", False)
            ),
            save_raw_outputs=bool(config.get("save_raw_outputs", True)),
            output_dir=_optional_str(config.get("output_dir", "data/work/chandra2")),
            project_root=_optional_str(config.get("project_root")),
        )


@dataclass(frozen=True)
class _ChandraRuntime:
    load_file: Callable[[str, dict[str, Any]], list[Any]]
    inference_manager: Callable[..., Any]
    batch_input_item: Callable[..., Any]


class Chandra2Provider:
    """Convert PDF and image pages to ordered Markdown with Chandra2."""

    provider_name = "chandra2"
    supported_extensions = CHANDRA2_EXTENSIONS

    def __init__(
        self,
        config: Chandra2Config,
        *,
        _runtime_loader: Callable[[], _ChandraRuntime] | None = None,
    ) -> None:
        self.config = config
        self._runtime_loader = _runtime_loader or _load_runtime
        self._runtime: _ChandraRuntime | None = None
        self._manager: Any = None

    def parse_file(self, path: str | Path, data_object: DataObject) -> ParsedData:
        file_path = Path(path)
        if file_path.suffix.lower() not in self.supported_extensions:
            raise RuntimeError(
                f"Chandra2 does not support file type: {file_path.suffix}"
            )

        started = time.monotonic()
        runtime = self._get_runtime()
        pages = runtime.load_file(str(file_path), {})
        if not pages:
            raise RuntimeError("Chandra2 could not load any pages from the document.")

        results: list[Any] = []
        for start in range(0, len(pages), self.config.batch_size):
            page_batch = pages[start : start + self.config.batch_size]
            batch = [
                runtime.batch_input_item(image=image, prompt_type="ocr_layout")
                for image in page_batch
            ]
            generated = list(
                self._get_manager().generate(
                    batch,
                    include_images=False,
                    include_headers_footers=self.config.include_headers_footers,
                    max_output_tokens=self.config.max_output_tokens,
                    max_workers=self.config.max_workers,
                    max_retries=self.config.max_retries,
                )
            )
            if len(generated) != len(batch):
                raise RuntimeError(
                    "Chandra2 returned a different number of results than input pages."
                )
            failed_pages = [
                start + index + 1
                for index, result in enumerate(generated)
                if bool(getattr(result, "error", False))
            ]
            if failed_pages:
                pages_text = ", ".join(str(page) for page in failed_pages)
                raise RuntimeError(f"Chandra2 failed to parse page(s): {pages_text}")
            results.extend(generated)

        markdown = "\n\n".join(
            str(getattr(result, "markdown", "") or "") for result in results
        )
        latency_seconds = round(time.monotonic() - started, 3)
        token_count = sum(
            int(getattr(result, "token_count", 0) or 0) for result in results
        )
        raw_output_path, raw_metadata_path = _write_raw_outputs(
            self.config,
            file_path,
            data_object,
            markdown,
            results,
            latency_seconds,
        )

        return ParsedData(
            object_id=data_object.object_id,
            source_uri=data_object.uri,
            source_format=str(
                data_object.metadata.get("format", file_path.suffix.lstrip("."))
            ),
            rows=[{"text": markdown}],
            text=markdown,
            metadata={
                "parser": "chandra2",
                "method": "vllm",
                "model_name": os.getenv("VLLM_MODEL_NAME", "chandra"),
                "page_count": len(results),
                "token_count": token_count,
                "latency_seconds": latency_seconds,
                "raw_output_path": raw_output_path,
                "raw_metadata_path": raw_metadata_path,
            },
        )

    def _get_runtime(self) -> _ChandraRuntime:
        if self._runtime is None:
            self._runtime = self._runtime_loader()
        return self._runtime

    def _get_manager(self) -> Any:
        if self._manager is None:
            self._manager = self._get_runtime().inference_manager(method="vllm")
        return self._manager


def _load_runtime() -> _ChandraRuntime:
    try:
        from chandra.input import load_file
        from chandra.model import InferenceManager
        from chandra.model.schema import BatchInputItem
    except ImportError as exc:
        raise RuntimeError(
            "Missing Chandra2 dependency. Install it with: pip install -e .[chandra2]"
        ) from exc
    return _ChandraRuntime(load_file, InferenceManager, BatchInputItem)


def _write_raw_outputs(
    config: Chandra2Config,
    file_path: Path,
    data_object: DataObject,
    markdown: str,
    results: list[Any],
    latency_seconds: float,
) -> tuple[str | None, str | None]:
    if not config.save_raw_outputs or not config.output_dir:
        return None, None

    bundle_dir = Path(config.output_dir) / (
        f"{_safe_slug(file_path.stem)}--{data_object.object_id}"
    )
    bundle_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = bundle_dir / "result.md"
    metadata_path = bundle_dir / "metadata.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "source_object_id": data_object.object_id,
                "file_name": file_path.name,
                "page_count": len(results),
                "token_count": sum(
                    int(getattr(result, "token_count", 0) or 0)
                    for result in results
                ),
                "latency_seconds": latency_seconds,
                "pages": [
                    {
                        "page_number": index + 1,
                        "token_count": int(
                            getattr(result, "token_count", 0) or 0
                        ),
                    }
                    for index, result in enumerate(results)
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return (
        portable_path(markdown_path, config.project_root),
        portable_path(metadata_path, config.project_root),
    )


def _positive_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value <= 0:
        raise ValueError(f"chandra2.{name} must be greater than zero")
    return value


def _non_negative_int(config: dict[str, Any], name: str, default: int) -> int:
    value = int(config.get(name, default))
    if value < 0:
        raise ValueError(f"chandra2.{name} must be zero or greater")
    return value


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in {None, ""} else None


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._") or "document"
