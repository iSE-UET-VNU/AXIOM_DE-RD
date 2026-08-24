from __future__ import annotations

from typing import Any

import pytest

from src.chunking_embedding.embedders import openrouter as openrouter_module
from src.chunking_embedding.embedders import (
    CJK_CHAR_LIMIT,
    DEFAULT_CHAR_LIMIT,
    sanitize_text,
)
from src.chunking_embedding.embedders.local import LocalHashEmbedder
from src.chunking_embedding.embedders.openrouter import OpenRouterEmbedder


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("" if payload is None else str(payload))

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok_payload(batch_size: int, dim: int = 4) -> dict:
    return {
        "data": [
            {"index": i, "embedding": [float(i)] * dim}
            for i in reversed(range(batch_size))  # out of order on purpose
        ]
    }


def _client(tmp_path, **kwargs) -> OpenRouterEmbedder:
    client = OpenRouterEmbedder(cache_dir=tmp_path / "cache", **kwargs)
    client._sleep = lambda seconds: None
    return client


# ── sanitization ──────────────────────────────────────────────────────────────

def test_sanitize_empty_becomes_single_space() -> None:
    assert sanitize_text("") == " "
    assert sanitize_text("   \n\t") == " "


def test_sanitize_truncates_long_ascii_at_default_limit() -> None:
    assert len(sanitize_text("a" * (DEFAULT_CHAR_LIMIT + 500))) == DEFAULT_CHAR_LIMIT


def test_sanitize_cjk_heavy_text_uses_tight_limit() -> None:
    """CJK-heavy text takes the tight char cap; ASCII-heavy text takes the default.

    The char cap is an upper bound, not an exact length: token-aware truncation may
    cut further. A 7,000-char CJK input used to clip to exactly the char limit and
    still reach 8,213 tokens, which the API rejects with HTTP 400.
    """
    text = "漢" * 10_000
    assert 0 < len(sanitize_text(text)) <= CJK_CHAR_LIMIT
    mixed = ("漢" * 3 + "a" * 7) * 5_000  # 30% CJK
    assert 0 < len(sanitize_text(mixed)) <= CJK_CHAR_LIMIT
    mostly_ascii = ("漢" + "a" * 9) * 5_000  # 10% CJK
    assert CJK_CHAR_LIMIT < len(sanitize_text(mostly_ascii)) <= DEFAULT_CHAR_LIMIT


def test_sanitize_keeps_every_input_under_the_api_token_cap() -> None:
    """The invariant that matters: no sanitized input exceeds the 8191-token limit."""
    tiktoken = pytest.importorskip("tiktoken")
    encoder = tiktoken.get_encoding("cl100k_base")
    for text in ("漢" * 10_000, ("漢" * 3 + "a" * 7) * 5_000, "a" * 40_000):
        assert len(encoder.encode(sanitize_text(text))) <= 8_191


# ── hardened HTTP client (mocked) ─────────────────────────────────────────────

def test_embed_success_orders_by_index_and_caches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls: list[dict] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        calls.append(kwargs["json"])
        return _FakeResponse(200, _ok_payload(len(kwargs["json"]["input"])))

    monkeypatch.setattr(openrouter_module.requests, "post", fake_post)
    client = _client(tmp_path)
    vectors = client.embed(["alpha", "beta", ""])
    assert len(vectors) == 3
    assert vectors[0] == [0.0, 0.0, 0.0, 0.0]
    assert vectors[2] == [2.0, 2.0, 2.0, 2.0]
    assert calls[0]["input"] == ["alpha", "beta", " "]

    vectors_again = client.embed(["alpha", "beta", ""])
    assert vectors_again == vectors
    assert len(calls) == 1
    assert client.stats["cache_hits"] == 3


def test_http_200_with_error_body_retries_then_raises_server_message(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    attempts = {"n": 0}
    sleeps: list[float] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        attempts["n"] += 1
        return _FakeResponse(200, {"error": {"message": "upstream quota exceeded"}})

    monkeypatch.setattr(openrouter_module.requests, "post", fake_post)
    client = _client(tmp_path)
    client._sleep = sleeps.append
    with pytest.raises(RuntimeError) as excinfo:
        client.embed(["hello"])
    assert attempts["n"] == 4
    assert sleeps == [2.0, 4.0, 8.0]
    assert "upstream quota exceeded" in str(excinfo.value)


def test_recovers_after_transient_error_body(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    responses = [
        _FakeResponse(200, {"error": {"message": "flaky"}}),
        _FakeResponse(500, None, text="internal error"),
        _FakeResponse(200, _ok_payload(1)),
    ]

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        return responses.pop(0)

    monkeypatch.setattr(openrouter_module.requests, "post", fake_post)
    client = _client(tmp_path)
    assert client.embed(["hello"]) == [[0.0, 0.0, 0.0, 0.0]]
    assert client.stats["retries"] == 2


def test_batches_of_64(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    batch_sizes: list[int] = []

    def fake_post(url: str, **kwargs: Any) -> _FakeResponse:
        batch_sizes.append(len(kwargs["json"]["input"]))
        return _FakeResponse(200, _ok_payload(len(kwargs["json"]["input"])))

    monkeypatch.setattr(openrouter_module.requests, "post", fake_post)
    client = _client(tmp_path)
    client.embed([f"text {i}" for i in range(150)])
    assert batch_sizes == [64, 64, 22]


def test_missing_api_key_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = _client(tmp_path)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        client.embed(["hello"])


# ── local hash embedder ───────────────────────────────────────────────────────

def test_local_hash_is_deterministic_and_normalized() -> None:
    embedder = LocalHashEmbedder(dimension=32)
    [a], [b] = embedder.embed(["same text"]), embedder.embed(["same text"])
    assert a == b
    assert len(a) == 32
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    [c] = embedder.embed(["different text"])
    assert c != a
