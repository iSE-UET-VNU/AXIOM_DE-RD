"""KDL endpoint health state and circuit-breaker primitives."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import httpx

from ...utils.observability import JsonEventLogger, utc_now_iso

logger = logging.getLogger("kdl_frontier_nano.health")


class KDLHostUnavailableError(RuntimeError):
    """Raised when the KDL endpoint is no longer safe to keep calling."""

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        status_code: int | None = None,
        failure_kind: str = "unknown",
        stage: str | None = None,
        consecutive_failures: int = 0,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.failure_kind = failure_kind
        self.stage = stage
        self.consecutive_failures = consecutive_failures


def error_status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return int(status_code) if status_code is not None else None


def is_host_failure(error: BaseException) -> bool:
    """Classify failures that indicate the remote KDL host is unhealthy."""

    status_code = error_status_code(error)
    if status_code is not None:
        return status_code >= 500 or status_code in {408, 429}
    return isinstance(
        error,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.NetworkError,
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    )


class KDLHostHealth:
    """A per-run circuit breaker shared by all concurrent KDL requests."""

    def __init__(
        self,
        endpoint: str,
        *,
        failure_threshold: int = 3,
        abort_on_open: bool = True,
        event_logger: JsonEventLogger | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.failure_threshold = max(1, int(failure_threshold))
        self.abort_on_open = bool(abort_on_open)
        self.event_logger = event_logger
        self._lock = threading.Lock()
        self._open = False
        self._consecutive_failures = 0
        self._total_host_failures = 0
        self._total_host_successes = 0
        self._last_status_code: int | None = None
        self._last_failure: str | None = None
        self._last_failure_at: str | None = None
        self._circuit_opened_at: str | None = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def before_request(self, *, stage: str | None = None) -> None:
        with self._lock:
            if not self._open:
                return
            count = self._consecutive_failures
        raise KDLHostUnavailableError(
            "KDL host circuit is open; refusing another request",
            endpoint=self.endpoint,
            status_code=self._last_status_code,
            failure_kind="circuit_open",
            stage=stage,
            consecutive_failures=count,
        )

    def record_success(self, *, stage: str | None = None, latency_ms: float = 0.0) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._total_host_successes += 1
        if self.event_logger is not None:
            self.event_logger.emit(
                "kdl_request_success",
                endpoint=self.endpoint,
                stage=stage,
                latency_ms=round(latency_ms, 3),
            )

    def record_failure(
        self,
        error: BaseException,
        *,
        stage: str | None = None,
        attempt: int | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        if not is_host_failure(error):
            return
        status_code = error_status_code(error)
        now = utc_now_iso()
        with self._lock:
            self._consecutive_failures += 1
            self._total_host_failures += 1
            self._last_status_code = status_code
            self._last_failure = f"{type(error).__name__}: {error}"
            self._last_failure_at = now
            count = self._consecutive_failures
            should_open = self.abort_on_open and count >= self.failure_threshold
            if should_open and not self._open:
                self._open = True
                self._circuit_opened_at = now

        logger.warning(
            "KDL host failure endpoint=%s stage=%s status=%s attempt=%s "
            "consecutive=%d latency_ms=%.1f error=%s",
            self.endpoint,
            stage or "unknown",
            status_code or "n/a",
            attempt if attempt is not None else "n/a",
            count,
            latency_ms,
            error,
        )
        if self.event_logger is not None:
            self.event_logger.emit(
                "kdl_host_failure",
                endpoint=self.endpoint,
                stage=stage,
                status_code=status_code,
                attempt=attempt,
                consecutive_failures=count,
                latency_ms=round(latency_ms, 3),
                error_type=type(error).__name__,
                error=str(error),
            )
        if should_open:
            logger.error(
                "KDL host circuit OPEN endpoint=%s threshold=%d; stopping this run",
                self.endpoint,
                self.failure_threshold,
            )
            if self.event_logger is not None:
                self.event_logger.emit(
                    "kdl_circuit_open",
                    endpoint=self.endpoint,
                    stage=stage,
                    status_code=status_code,
                    consecutive_failures=count,
                    threshold=self.failure_threshold,
                )
            raise KDLHostUnavailableError(
                "KDL host unavailable after "
                f"{count} consecutive host failures; stopping this run",
                endpoint=self.endpoint,
                status_code=status_code,
                failure_kind="host_unavailable",
                stage=stage,
                consecutive_failures=count,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "endpoint": self.endpoint,
                "open": self._open,
                "abort_on_open": self.abort_on_open,
                "failure_threshold": self.failure_threshold,
                "consecutive_failures": self._consecutive_failures,
                "total_host_failures": self._total_host_failures,
                "total_host_successes": self._total_host_successes,
                "last_status_code": self._last_status_code,
                "last_failure": self._last_failure,
                "last_failure_at": self._last_failure_at,
                "circuit_opened_at": self._circuit_opened_at,
            }
