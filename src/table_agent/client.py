"""Synchronous HTTP client for the existing TableAgent upload API."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import httpx


DEFAULT_WORKBOOK_EXTENSIONS = (
    ".xls",
    ".xlsm",
    ".xlsx",
    ".xltm",
    ".xltx",
)


@dataclass(frozen=True)
class TableAgentClientConfig:
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8001"
    upload_path: str = "/v1/jobs/upload"
    api_key: str | None = None
    timeout_seconds: float = 1800.0
    verify_ssl: bool = True
    stage: str = "structure"
    embed: bool = True
    sheets: tuple[str, ...] = ()
    supported_extensions: tuple[str, ...] = DEFAULT_WORKBOOK_EXTENSIONS

    @classmethod
    def from_mapping(cls, value: Any) -> "TableAgentClientConfig":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("table_agent configuration must be a mapping.")

        stage = str(value.get("stage", "structure")).strip().lower()
        if stage != "structure":
            raise ValueError(
                "table_agent.stage must be 'structure' for ingestion-only integration."
            )

        timeout_seconds = float(value.get("timeout_seconds", 1800.0))
        if timeout_seconds <= 0:
            raise ValueError("table_agent.timeout_seconds must be greater than zero.")

        extensions_value = value.get(
            "supported_extensions",
            list(DEFAULT_WORKBOOK_EXTENSIONS),
        )
        if not isinstance(extensions_value, list):
            raise ValueError("table_agent.supported_extensions must be a list.")
        extensions = tuple(_normalize_extension(item) for item in extensions_value)
        if not extensions:
            raise ValueError("table_agent.supported_extensions must not be empty.")

        sheets_value = value.get("sheets", [])
        if not isinstance(sheets_value, list):
            raise ValueError("table_agent.sheets must be a list.")

        base_url_env = str(value.get("base_url_env", "TABLE_AGENT_BASE_URL"))
        configured_base_url = str(
            os.getenv(base_url_env)
            or value.get("base_url")
            or "http://127.0.0.1:8001"
        ).strip()
        if not configured_base_url:
            raise ValueError("table_agent.base_url must not be blank.")

        api_key_env = str(
            value.get("api_key_env", "TABLE_AGENT_SERVICE_API_KEY")
        )
        api_key_value = os.getenv(api_key_env) or value.get("api_key")
        api_key = str(api_key_value) if api_key_value else None

        upload_path = str(value.get("upload_path", "/v1/jobs/upload")).strip()
        if not upload_path.startswith("/"):
            upload_path = f"/{upload_path}"

        return cls(
            enabled=bool(value.get("enabled", False)),
            base_url=configured_base_url.rstrip("/"),
            upload_path=upload_path,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            verify_ssl=bool(value.get("verify_ssl", True)),
            stage=stage,
            embed=bool(value.get("embed", True)),
            sheets=tuple(
                str(sheet).strip()
                for sheet in sheets_value
                if str(sheet).strip()
            ),
            supported_extensions=extensions,
        )


class TableAgentClient:
    """Upload one workbook to TableAgent and return its JSON response."""

    def __init__(
        self,
        config: TableAgentClientConfig,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client

    def accepts(self, path: str | Path) -> bool:
        return (
            self.config.enabled
            and Path(path).suffix.lower() in self.config.supported_extensions
        )

    def process_workbook(self, path: str | Path) -> dict[str, Any]:
        workbook = Path(path)
        if not workbook.is_file():
            raise FileNotFoundError(f"TableAgent workbook does not exist: {workbook}")
        if not self.accepts(workbook):
            raise ValueError(
                f"TableAgent does not accept workbook extension {workbook.suffix!r}."
            )

        payload = json.dumps(
            {
                "stage": self.config.stage,
                "queries": [],
                "embed": self.config.embed,
                "sheets": list(self.config.sheets),
            },
            ensure_ascii=False,
        )
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key

        url = f"{self.config.base_url}{self.config.upload_path}"
        try:
            with workbook.open("rb") as handle:
                files = {
                    "files": (
                        workbook.name,
                        handle,
                        _workbook_content_type(workbook),
                    )
                }
                if self._http_client is not None:
                    response = self._http_client.post(
                        url,
                        data={"payload": payload},
                        files=files,
                        headers=headers,
                    )
                else:
                    with httpx.Client(
                        timeout=self.config.timeout_seconds,
                        verify=self.config.verify_ssl,
                    ) as client:
                        response = client.post(
                            url,
                            data={"payload": payload},
                            files=files,
                            headers=headers,
                        )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"TableAgent request timed out after {self.config.timeout_seconds:g} seconds."
            ) from None
        except httpx.RequestError as exc:
            raise RuntimeError(f"Could not reach TableAgent at {url}: {exc}.") from None

        if 400 <= response.status_code < 500:
            raise ValueError(
                f"TableAgent rejected workbook {workbook.name!r}: "
                f"HTTP {response.status_code}: {_response_detail(response)}"
            )
        if response.status_code >= 500:
            raise RuntimeError(
                f"TableAgent failed for workbook {workbook.name!r}: "
                f"HTTP {response.status_code}: {_response_detail(response)}"
            )

        try:
            response.raise_for_status()
            result = response.json()
        except (httpx.HTTPStatusError, ValueError):
            raise RuntimeError(
                f"TableAgent returned an invalid response for workbook {workbook.name!r}."
            ) from None
        if not isinstance(result, dict):
            raise RuntimeError("TableAgent response must be a JSON object.")
        if not isinstance(result.get("structures"), list):
            raise RuntimeError("TableAgent response is missing the structures array.")
        return result


def _normalize_extension(value: Any) -> str:
    extension = str(value).strip().lower()
    if not extension:
        raise ValueError("table_agent.supported_extensions cannot contain blanks.")
    return extension if extension.startswith(".") else f".{extension}"


def _workbook_content_type(path: Path) -> str:
    return {
        ".xls": "application/vnd.ms-excel",
        ".xlsm": "application/vnd.ms-excel.sheet.macroenabled.12",
        ".xlsx": (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        ".xltm": "application/vnd.ms-excel.template.macroenabled.12",
        ".xltx": (
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.template"
        ),
    }.get(path.suffix.lower(), "application/octet-stream")


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, dict) and payload.get("detail") is not None:
        return str(payload["detail"])[:500]
    return str(payload)[:500]
