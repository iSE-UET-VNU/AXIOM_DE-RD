"""The guard between a benchmark arm and the model it actually reaches."""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.evaluation.model_guard import (
    MockProvider,
    Resolution,
    UnknownAlias,
    assert_real,
    resolve,
)

FAKE_PROVIDER = "cbd2e336-3dc8-40ea-b3a4-802eb2569c6b"
REAL_PROVIDER = "ceaf6315-3ec9-4073-a753-88ecd3d7111e"

MODELS = [
    {"id": "m-embed-fake", "alias": "embedding-default", "capability": "embedding", "status": "active"},
    {"id": "m-embed-real", "alias": "openrouter-embedding", "capability": "embedding", "status": "active"},
    {"id": "m-rerank", "alias": "llm-rerank", "capability": "llm", "status": "active"},
    {"id": "m-strong", "alias": "llm-rerank-strong", "capability": "llm", "status": "active"},
    {"id": "m-retired", "alias": "llm-retired", "capability": "llm", "status": "inactive"},
    {"id": "m-orphan", "alias": "llm-orphan", "capability": "llm", "status": "active"},
]

DEPLOYMENTS = {
    "m-embed-fake": [{"provider_id": FAKE_PROVIDER, "upstream_model_id": "fake/embedding",
                      "status": "active", "priority": 100}],
    "m-embed-real": [{"provider_id": REAL_PROVIDER, "upstream_model_id": "openai/text-embedding-3-small",
                      "status": "active", "priority": 10}],
    "m-rerank": [{"provider_id": REAL_PROVIDER, "upstream_model_id": "openai/gpt-4o-mini",
                  "status": "active", "priority": 10}],
    "m-strong": [{"provider_id": REAL_PROVIDER, "upstream_model_id": "openai/gpt-4o",
                  "status": "active", "priority": 10}],
    "m-retired": [],
    # Active alias whose only deployment is off; the gateway 404s at call time.
    "m-orphan": [{"provider_id": REAL_PROVIDER, "upstream_model_id": "openai/gpt-4o",
                  "status": "inactive", "priority": 10}],
}

PROVIDERS = [
    {"id": FAKE_PROVIDER, "name": "local-fake", "adapter_type": "fake", "base_url": "fake://local"},
    {"id": REAL_PROVIDER, "name": "openrouter-main", "adapter_type": "openrouter",
     "base_url": "https://openrouter.ai/api/v1"},
]


@pytest.fixture
def registry(monkeypatch) -> None:
    def fake_get(url: str, timeout: float) -> Any:
        if url.endswith("/models"):
            return MODELS
        if url.endswith("/providers"):
            return PROVIDERS
        if "/deployments" in url:
            return DEPLOYMENTS[url.rsplit("/models/", 1)[1].split("/")[0]]
        raise AssertionError(f"unexpected registry call {url}")

    monkeypatch.setattr("src.evaluation.model_guard._get", fake_get)


# -- seam: benchmark arm vs the provider it actually reaches --------------------


def test_mock_backed_alias_is_refused(registry):
    """``embedding-default`` answers 200 with 8-dim vectors. That must not run."""
    with pytest.raises(MockProvider, match="embedding-default"):
        assert_real(["embedding-default"])


def test_the_refusal_names_the_provider_and_upstream(registry):
    """An error that does not say which alias is wrong costs a debugging cycle."""
    with pytest.raises(MockProvider) as error:
        assert_real(["openrouter-embedding", "embedding-default"])
    message = str(error.value)
    assert "local-fake" in message and "fake/embedding" in message
    assert "adapter fake" in message


def test_real_providers_resolve_to_their_upstream_model(registry):
    """The manifest records the upstream string, not the alias."""
    resolved = assert_real(["llm-rerank", "llm-rerank-strong", "openrouter-embedding"])
    assert resolved["llm-rerank"].upstream_model_id == "openai/gpt-4o-mini"
    assert resolved["llm-rerank-strong"].upstream_model_id == "openai/gpt-4o"
    assert resolved["openrouter-embedding"].upstream_model_id == "openai/text-embedding-3-small"
    assert all(not r.is_mock for r in resolved.values())


def test_one_mock_among_real_ones_still_raises(registry):
    """The guard is all-or-nothing; a single mock poisons the arm."""
    with pytest.raises(MockProvider):
        assert_real(["llm-rerank", "embedding-default", "openrouter-embedding"])


def test_unregistered_alias_raises_and_lists_what_exists(registry):
    """``llm-default`` and ``llm-judge`` are named in the harness but unregistered."""
    with pytest.raises(UnknownAlias, match="llm-default"):
        assert_real(["llm-default"])
    with pytest.raises(UnknownAlias) as error:
        assert_real(["llm-judge"])
    assert "llm-rerank" in str(error.value)


def test_inactive_alias_and_orphaned_deployment_raise(registry):
    with pytest.raises(UnknownAlias, match="inactive"):
        assert_real(["llm-retired"])
    with pytest.raises(UnknownAlias, match="no active deployment"):
        assert_real(["llm-orphan"])


def test_mock_detection_is_by_adapter_type_not_provider_name(registry, monkeypatch):
    """A second mock would not be called ``local-fake``."""
    renamed = [dict(p, name="perfectly-normal-provider") if p["adapter_type"] == "fake" else p
               for p in PROVIDERS]
    monkeypatch.setattr(
        "src.evaluation.model_guard._get",
        lambda url, timeout: (
            MODELS if url.endswith("/models")
            else renamed if url.endswith("/providers")
            else DEPLOYMENTS[url.rsplit("/models/", 1)[1].split("/")[0]]
        ),
    )
    with pytest.raises(MockProvider):
        assert_real(["embedding-default"])


def test_resolution_is_recorded_shape(registry):
    """What goes in the run manifest."""
    resolution = resolve("llm-rerank")
    assert isinstance(resolution, Resolution)
    assert json.dumps(resolution.__dict__)  # serialisable into the manifest
    assert resolution.provider == "openrouter-main"
    assert resolution.capability == "llm"
