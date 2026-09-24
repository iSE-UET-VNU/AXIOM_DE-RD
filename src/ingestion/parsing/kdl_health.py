"""KDL endpoint health state and circuit-breaker primitives."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
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
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.status_code = status_code
        self.failure_kind = failure_kind
        self.stage = stage
        self.consecutive_failures = consecutive_failures
        self.retry_after_seconds = retry_after_seconds


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
        recovery_cooldown_seconds: float = 30.0,
        recovery_max_attempts: int = 5,
        event_logger: JsonEventLogger | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.failure_threshold = max(1, int(failure_threshold))
        self.abort_on_open = bool(abort_on_open)
        self.recovery_cooldown_seconds = max(0.0, float(recovery_cooldown_seconds))
        self.recovery_max_attempts = max(0, int(recovery_max_attempts))
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
        self._circuit_opened_monotonic: float | None = None
        self._recovery_attempts = 0
        self._probe_in_flight = False

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def before_request(self, *, stage: str | None = None) -> None:
        with self._lock:
            if not self._open:
                return
            if self._recovery_attempts >= self.recovery_max_attempts:
                count = self._consecutive_failures
                raise KDLHostUnavailableError(
                    "KDL host circuit recovery attempts exhausted",
                    endpoint=self.endpoint,
                    status_code=self._last_status_code,
                    failure_kind="circuit_open_exhausted",
                    stage=stage,
                    consecutive_failures=count,
                )
            now = time.monotonic()
            opened_at = self._circuit_opened_monotonic or now
            elapsed = max(0.0, now - opened_at)
            if not self._probe_in_flight and elapsed >= self.recovery_cooldown_seconds:
                self._probe_in_flight = True
                self._recovery_attempts += 1
                return
            retry_after = (
                0.25
                if self._probe_in_flight
                else max(0.05, self.recovery_cooldown_seconds - elapsed)
            )
            count = self._consecutive_failures
        raise KDLHostUnavailableError(
            "KDL host circuit is open; waiting for recovery cooldown",
            endpoint=self.endpoint,
            status_code=self._last_status_code,
            failure_kind="circuit_open",
            stage=stage,
            consecutive_failures=count,
            retry_after_seconds=retry_after,
        )

    async def wait_until_ready(self, *, stage: str | None = None) -> None:
        """Wait asynchronously until a normal request or one recovery probe is allowed."""

        while True:
            try:
                self.before_request(stage=stage)
                return
            except KDLHostUnavailableError as error:
                if error.failure_kind == "circuit_open_exhausted":
                    raise
                delay = error.retry_after_seconds
                if delay is None:
                    delay = self.recovery_cooldown_seconds
                await asyncio.sleep(max(0.05, delay))

    def record_success(self, *, stage: str | None = None, latency_ms: float = 0.0) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._total_host_successes += 1
            self._open = False
            self._probe_in_flight = False
            self._recovery_attempts = 0
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
            was_open = self._open
            probe_in_flight = self._probe_in_flight
            self._total_host_failures += 1
            if not was_open:
                self._consecutive_failures += 1
            self._last_status_code = status_code
            self._last_failure = f"{type(error).__name__}: {error}"
            self._last_failure_at = now
            count = self._consecutive_failures
            should_open = self.abort_on_open and count >= self.failure_threshold
            opened_now = should_open and not was_open
            if opened_now:
                self._open = True
                self._circuit_opened_at = now
                self._circuit_opened_monotonic = time.monotonic()
                self._recovery_attempts = 0
                self._probe_in_flight = False
            elif was_open and probe_in_flight:
                # A failed half-open probe restarts the cooldown window.
                self._circuit_opened_at = now
                self._circuit_opened_monotonic = time.monotonic()
                self._probe_in_flight = False

        # Requests that were already in flight when the circuit opened are
        # expected to fail together. They must not extend the failure count or
        # produce hundreds of duplicate circuit-open messages.
        if was_open and not probe_in_flight:
            return

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
        if opened_now or (was_open and probe_in_flight):
            if self.recovery_max_attempts > 0:
                logger.error(
                    "KDL host circuit OPEN endpoint=%s threshold=%d; "
                    "cooldown=%.1fs recovery_attempts=%d",
                    self.endpoint,
                    self.failure_threshold,
                    self.recovery_cooldown_seconds,
                    self.recovery_max_attempts,
                )
            else:
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
                    recovery_cooldown_seconds=self.recovery_cooldown_seconds,
                    recovery_max_attempts=self.recovery_max_attempts,
                )
            raise KDLHostUnavailableError(
                "KDL host unavailable after "
                f"{count} consecutive host failures; stopping this run",
                endpoint=self.endpoint,
                status_code=status_code,
                failure_kind="host_unavailable",
                stage=stage,
                consecutive_failures=count,
                retry_after_seconds=self.recovery_cooldown_seconds,
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "endpoint": self.endpoint,
                "open": self._open,
                "abort_on_open": self.abort_on_open,
                "failure_threshold": self.failure_threshold,
                "recovery_cooldown_seconds": self.recovery_cooldown_seconds,
                "recovery_max_attempts": self.recovery_max_attempts,
                "recovery_attempts": self._recovery_attempts,
                "consecutive_failures": self._consecutive_failures,
                "total_host_failures": self._total_host_failures,
                "total_host_successes": self._total_host_successes,
                "last_status_code": self._last_status_code,
                "last_failure": self._last_failure,
                "last_failure_at": self._last_failure_at,
                "circuit_opened_at": self._circuit_opened_at,
            }
