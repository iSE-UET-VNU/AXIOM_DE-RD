"""A transient network error must be retried, not abort the corpus."""

from __future__ import annotations

import pytest
import requests

from src.chunking_embedding.embedders.axiom_gateway import AxiomGatewayEmbedder
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder


class Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    @property
    def text(self):
        return ""


def openrouter_ok(batch_size):
    return Response({"data": [{"index": i, "embedding": [0.1] * 1536}
                              for i in range(batch_size)]})


def gateway_ok(batch_size):
    return Response({"dimensions": 1536,
                     "data": [{"id": str(i), "index": i, "embedding": [0.1] * 1536}
                              for i in range(batch_size)]})


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


@pytest.mark.parametrize(
    "make,ok",
    [
        (lambda d: OpenRouterEmbedder(cache_dir=d), openrouter_ok),
        (lambda d: AxiomGatewayEmbedder(cache_dir=d), gateway_ok),
    ],
    ids=["openrouter", "axiom_gateway"],
)
def test_read_timeout_is_retried(monkeypatch, tmp_path, make, ok):
    """The failure that killed the ViDoRe dense arms mid-run."""
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("read timed out")
        return ok(len(kwargs["json"]["input"]))

    embedder = make(tmp_path)
    embedder._sleep = lambda _: None
    monkeypatch.setattr(requests, "post", flaky)

    vectors = embedder.embed(["one page of text"])
    assert calls["n"] == 2
    assert len(vectors) == 1 and len(vectors[0]) == 1536


@pytest.mark.parametrize(
    "make",
    [lambda d: OpenRouterEmbedder(cache_dir=d), lambda d: AxiomGatewayEmbedder(cache_dir=d)],
    ids=["openrouter", "axiom_gateway"],
)
def test_persistent_failure_still_raises_with_the_cause(monkeypatch, tmp_path, make):
    """Retrying must not turn a real outage into silence."""
    def always(*args, **kwargs):
        raise requests.exceptions.ConnectionError("no route to host")

    embedder = make(tmp_path)
    embedder._sleep = lambda _: None
    monkeypatch.setattr(requests, "post", always)

    with pytest.raises(RuntimeError, match="ConnectionError"):
        embedder.embed(["one page of text"])
