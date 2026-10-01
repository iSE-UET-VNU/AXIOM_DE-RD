import hashlib
import json

import numpy as np

from ..bench import BENCH, work
from ..openrouter import embed

MODEL = "openai/text-embedding-3-small"
DIM, SLICE = 1536, 256


def load_store(name, bench=BENCH):
    folder = work("embeddings", name, bench=bench)
    texts, vectors = folder / "chunk_texts.json", folder / "chunk_vectors.npy"
    if not (texts.exists() and vectors.exists()):
        raise SystemExit(f"{folder} has no chunk_texts.json + chunk_vectors.npy to embed from")
    store = dict(zip(json.loads(texts.read_text()), np.load(vectors)))
    queries, qids = folder / "query_vectors.npy", folder / "query_ids.json"
    if queries.exists() and qids.exists():
        store.update(zip(json.loads(qids.read_text()), np.load(queries)))
    return store


class Embedder:
    def __init__(self, bench=BENCH, cache_name="te3s_ppocr", allow_api=False, model=MODEL, store=None,
                 query_ids=None):
        self.cache = work("embedding_cache", cache_name, bench=bench)
        self.allow_api = allow_api
        self.model = model
        self.store = load_store(store, bench) if store else {}
        self.query_ids = query_ids or {}
        self.api_calls = self.store_hits = self.cache_hits = 0

    def _key(self, text):
        return self.cache / (hashlib.sha256((self.model + "\n" + text).encode()).hexdigest() + ".json")

    def _lookup(self, text):
        vector = self.store.get(text)
        if vector is None and text in self.query_ids:
            vector = self.store.get(self.query_ids[text])
        return vector

    def __call__(self, texts):
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        todo = []
        for i, text in enumerate(texts):
            vector = self._lookup(text)
            if vector is not None:
                out[i] = vector
                self.store_hits += 1
            else:
                todo.append(i)
        missing = [i for i in todo if not self._key(texts[i]).exists()]
        self.cache_hits += len(todo) - len(missing)
        if missing and not self.allow_api:
            raise SystemExit(f"{len(missing)} of {len(texts)} texts are in neither the embedding store nor "
                             f"{self.cache}; rerun with --allow-api to embed them")
        self.api_calls += len(missing)
        for start in range(0, len(todo), SLICE):
            batch = todo[start:start + SLICE]
            out[batch] = embed([texts[i] for i in batch], self.model, self.cache)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
