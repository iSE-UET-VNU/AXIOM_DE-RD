"""Local BAAI/bge-m3 embedder via sentence-transformers (optional dependency)."""

from __future__ import annotations

from typing import Any
import logging

from ..registry import embedder
from ._sanitize import sanitize_text

logger = logging.getLogger(__name__)


@embedder("bge_m3_local")
class BgeM3LocalEmbedder:
    name = "bge_m3_local"

    def __init__(self, model: str = "BAAI/bge-m3", dimension: int = 1024, **_ignored: Any) -> None:
        self.model = model
        self.dim = dimension
        self.stats: dict[str, int] = {"tokens": 0, "api_calls": 0, "cache_hits": 0, "retries": 0}
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "bge_m3_local requires sentence-transformers: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model)
            try:
                self._model.max_seq_length = min(int(self._model.max_seq_length or 1024), 1024)
            except Exception:
                pass
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        sanitized = [sanitize_text(t) for t in texts]
        longest = max((len(t) for t in sanitized), default=0) // 4
        batch_size = 32 if longest <= 512 else (8 if longest <= 2048 else 2)
        try:
            encoded = model.encode(sanitized, batch_size=batch_size, show_progress_bar=False)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                encoded = model.encode(sanitized, batch_size=1, show_progress_bar=False)
            except Exception:
                logger.warning("%s: GPU OOM even at batch 1 -> falling back to CPU", self.name)
                model.to("cpu")
                encoded = model.encode(sanitized, batch_size=8, show_progress_bar=False, device="cpu")
        return [[float(x) for x in row] for row in encoded]
