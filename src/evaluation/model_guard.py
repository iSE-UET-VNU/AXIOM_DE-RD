"""Refuse to benchmark against a mock model, at resolution time.

Five of the gateway's nine aliases are backed by ``local-fake``.
``embedding-default`` returns 8-dimensional vectors with HTTP 200 and
``AxiomGatewayEmbedder`` adopts that width, so an arm using it indexes,
retrieves, scores, and raises nothing. Recorded as bug #6 in docs/testing_rules.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import json
import os
import urllib.request

MODEL_SERVICE_URL = "http://localhost:8006/api/v1"

# Matched on the registry's ``adapter_type``: a second mock would be named differently.
MOCK_ADAPTERS = frozenset({"fake", "mock", "stub", "dummy"})


class UnknownAlias(RuntimeError):
    """The alias is not registered, or is registered but inactive."""


class MockProvider(RuntimeError):
    """The alias resolves to a mock provider. Never acceptable in a benchmark."""


@dataclass(frozen=True)
class Resolution:
    alias: str
    capability: str
    provider: str
    adapter_type: str
    upstream_model_id: str

    @property
    def is_mock(self) -> bool:
        return self.adapter_type.lower() in MOCK_ADAPTERS


def _get(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("items", "data", "models", "providers", "deployments"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def resolve(alias: str, *, base_url: str = "", timeout: float = 15.0) -> Resolution:
    """Alias -> provider and upstream model. Gateway-routed aliases only."""
    root = (base_url or os.getenv("AXIOM_MODEL_SERVICE_URL") or MODEL_SERVICE_URL).rstrip("/")
    registry = f"{root}/model-registry"

    models = _items(_get(f"{registry}/models", timeout))
    match = next((m for m in models if m.get("alias") == alias), None)
    if match is None:
        raise UnknownAlias(
            f"model alias {alias!r} is not registered. Active aliases: "
            f"{sorted(m.get('alias', '') for m in models)}. Register it or pass one "
            "of these; a fallback would measure a different model than it reports."
        )
    if str(match.get("status")).lower() != "active":
        raise UnknownAlias(f"model alias {alias!r} is registered but {match.get('status')!r}")

    deployments = [
        d for d in _items(_get(f"{registry}/models/{match['id']}/deployments", timeout))
        if str(d.get("status")).lower() == "active"
    ]
    if not deployments:
        raise UnknownAlias(f"model alias {alias!r} has no active deployment")

    providers = {p.get("id"): p for p in _items(_get(f"{registry}/providers", timeout))}
    # Highest priority wins, matching the gateway's own routing order.
    deployment = sorted(deployments, key=lambda d: d.get("priority", 0), reverse=True)[0]
    provider = providers.get(deployment.get("provider_id")) or {}

    return Resolution(
        alias=alias,
        capability=str(match.get("capability") or ""),
        provider=str(provider.get("name") or "unknown"),
        adapter_type=str(provider.get("adapter_type") or "unknown"),
        upstream_model_id=str(deployment.get("upstream_model_id") or ""),
    )


def assert_real(aliases: Iterable[str], **kwargs: Any) -> dict[str, Resolution]:
    """Resolve every alias a run will touch, refuse any mock, return the manifest."""
    resolved = {alias: resolve(alias, **kwargs) for alias in dict.fromkeys(aliases) if alias}
    mocks = [r for r in resolved.values() if r.is_mock]
    if mocks:
        detail = ", ".join(
            f"{r.alias} -> {r.provider}/{r.upstream_model_id} (adapter {r.adapter_type})"
            for r in mocks
        )
        raise MockProvider(
            f"refusing to benchmark against a mock provider: {detail}. These answer "
            "HTTP 200 with synthetic output -- embedding-default returns 8-dimensional "
            "vectors -- so an arm using one produces a plausible, meaningless score."
        )
    return resolved
