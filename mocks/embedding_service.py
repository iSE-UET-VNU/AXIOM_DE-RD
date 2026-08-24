"""Mock embedding service — the contract we ask the model-service team to meet.

OpenAI-wire-compatible, so the same client code points at this stub, at
OpenRouter, or at their real service by changing one base URL. Vectors are
deterministic hashes: useless for retrieval quality, correct for wiring tests.

    uvicorn mocks.embedding_service:app --port 8002
"""

from __future__ import annotations

from typing import Any
import hashlib
import os

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

MODEL = os.getenv("MOCK_EMBEDDING_MODEL", "openai/text-embedding-3-small")
DIMENSION = int(os.getenv("MOCK_EMBEDDING_DIMENSION", "1536"))
API_KEY = os.getenv("MOCK_EMBEDDING_API_KEY", "")

app = FastAPI(title="Mock Embedding Service", version="1.0.0")


class EmbeddingRequest(BaseModel):
    model: str = MODEL
    input: list[str] | str
    dimensions: int | None = Field(default=None)


def _vector(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte / 127.5) - 1.0 for byte in digest)
        counter += 1
    values = values[:dimension]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return [value / norm for value in values]


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid api key")
    texts = [request.input] if isinstance(request.input, str) else list(request.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")
    dimension = request.dimensions or DIMENSION
    return {
        "object": "list",
        "model": request.model,
        "data": [
            {"object": "embedding", "index": index, "embedding": _vector(text, dimension)}
            for index, text in enumerate(texts)
        ],
        "usage": {"prompt_tokens": sum(len(t) // 4 for t in texts), "total_tokens": sum(len(t) // 4 for t in texts)},
    }


@app.get("/health/ready")
def ready() -> dict[str, str]:
    return {"status": "ok", "model": MODEL, "dimension": str(DIMENSION)}
