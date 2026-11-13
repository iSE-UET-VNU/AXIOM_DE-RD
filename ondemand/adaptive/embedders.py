import hashlib
import json
import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import requests

from ..bench import BENCH, queries as load_queries, work
from ..light_retrieval import DENSE_MODEL, cache_misses, chunk_vectors, page_scores
from ..openrouter import _post, embed
from ..text import real_text
from .modes import refuse
from .registry import register

QWEN = "qwen/qwen3-embedding-8b"
QWEN_PREFIX = "Instruct: Given a question, retrieve document pages that answer the question\nQuery: "
QWEN_CARD_PREFIX = "Instruct: Given a question, retrieve document pages that answer the question\nQuery:"
QWEN_DIM, QWEN_MAX_TOKENS, QWEN_CARD_MAX_TOKENS = 4096, 6000, 8192
KNOWN_SOURCES = {"dense_qwen8b_ppocr": "pages_ocr_ppocr.jsonl", "dense_qwen8b": "pages_ocr_sparse.jsonl"}
PROVIDER = {"order": ["nebius"], "quantizations": ["bf16"], "allow_fallbacks": False}
PROVIDER_TAG = "nebius_bf16"
DEFAULT_STEM = "dense_qwen8b_nebius_bf16_pages_ppocrv5"
WORKERS = 8
log = logging.getLogger("ondemand.embedder")


def unit(vectors):
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


@register("embedder", "te3s")
class TE3S:
    name = "te3s"

    def __init__(self, argument=None, bench=BENCH, online=False, **_):
        self.cache = work("embedding_cache", argument or "te3s_ppocr", bench=bench)
        self.online = online
        self.text_source = None

    def prepare(self, rows, pages_name):
        self.n_pages, self.text_source = len(rows), pages_name
        self.owners, self.vectors = chunk_vectors(rows, self.cache, self.online)

    def scores(self, text, qid=None, raw=None):
        if not self.online and cache_misses([text], self.cache):
            refuse("embedder", "the te3s query embedding", "--stage-mode embedder=online")
        qvec = unit(embed([text], DENSE_MODEL, self.cache)[0])
        return page_scores(self.owners, self.vectors, qvec, self.n_pages)

    def settings(self):
        return {"embedder": self.name, "model": DENSE_MODEL, "level": "chunk, max over a page's chunks",
                "cache": self.cache.name, "text_source": self.text_source}


@register("embedder", "qwen3_8b")
class Qwen3Embedding8B:
    name = "qwen3_8b"

    def __init__(self, argument=None, bench=BENCH, online=False, **_):
        self.bench, self.online = bench, online
        self.stem = argument or DEFAULT_STEM
        self.endpoint = os.getenv("QWEN_EMBED_ENDPOINT") or None
        self.legacy = self.stem in KNOWN_SOURCES
        self.backend = "openrouter-unpinned" if self.legacy else "local" if self.endpoint else PROVIDER_TAG
        self.cache = work("embedding_cache", "qwen8b" if self.legacy else f"qwen8b_{self.backend}", bench=bench)
        self.served = Counter()
        self.prefix = QWEN_PREFIX if self.legacy else QWEN_CARD_PREFIX
        self.max_tokens = QWEN_MAX_TOKENS if self.legacy else QWEN_CARD_MAX_TOKENS
        self.tokenizer = "tiktoken cl100k_base" if self.legacy else "Qwen/Qwen3-Embedding-8B"

    def files(self):
        folder = work("light_v2", bench=self.bench)
        return (folder / f"{self.stem}_pages.npy", folder / f"{self.stem}_queries.npy",
                folder / f"{self.stem}_meta.json")

    def prepare(self, rows, pages_name):
        units = [r["page_id"] for r in rows]
        pages, queries, meta = self.files()
        if not pages.exists():
            if not self.online or self.legacy:
                refuse("embedder", f"the Qwen3-Embedding-8B page matrix {pages.name}",
                       "--stage-mode embedder=online, or colab/qwen3_embed_pages.sh")
            self.build(rows, pages_name)
        info = json.loads(meta.read_text())
        missing = sorted(set(units) - set(info["units"]))
        if missing:
            raise SystemExit(f"{pages.name} has no vector for {len(missing)} pages of {pages_name}, e.g. {missing[:2]}")
        where = {u: i for i, u in enumerate(info["units"])}
        self.pages = unit(np.load(pages)[[where[u] for u in units]])
        self.text_source = info.get("pages_source") or KNOWN_SOURCES.get(self.stem, "unknown")
        if self.text_source != pages_name:
            log.warning("%s was embedded from %s, but BM25 reads %s; dense and BM25 see different text on pages "
                        "whose OCR differs", pages.name, self.text_source, pages_name)
        self.query_rows = {}
        if queries.exists():
            stored = np.load(queries)
            self.query_rows = {q: unit(stored[i]) for i, q in enumerate(info.get("qids", []))}

    def page_texts(self, rows):
        if self.legacy:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return [enc.decode(enc.encode(real_text(r["text"]))[:QWEN_MAX_TOKENS]) for r in rows]
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained(self.tokenizer)
        out = []
        for r in rows:
            text = real_text(r["text"])
            ids = tok.encode(text, add_special_tokens=False).ids
            out.append(tok.decode(ids[:self.max_tokens - 1]) if len(ids) >= self.max_tokens else text)
        return out

    def build(self, rows, pages_name):
        texts = self.page_texts(rows)
        keep = [i for i, t in enumerate(texts) if t.strip()]
        queries = load_queries(self.bench)
        qids = sorted(queries)
        started = perf_counter()
        vectors = np.zeros((len(rows), QWEN_DIM), dtype=np.float32)
        vectors[keep] = self.embed([texts[i] for i in keep])
        page_seconds = perf_counter() - started
        query_vectors = self.embed([self.prefix + queries[q]["query"] for q in qids])
        pages, query_path, meta = self.files()
        np.save(pages, vectors)
        np.save(query_path, query_vectors)
        meta.write_text(json.dumps({"model": QWEN, "backend": self.backend, "provider": PROVIDER,
                                    "served_providers": dict(self.served), "texts_per_request": 1,
                                    "pages_source": pages_name, "max_tokens": self.max_tokens,
                                    "truncation_tokenizer": self.tokenizer, "query_prefix": self.prefix,
                                    "empty_pages": len(rows) - len(keep),
                                    "page_seconds": page_seconds, "qids": qids,
                                    "units": [r["page_id"] for r in rows]}))
        log.info("built %s: %d pages, %d queries, served by %s", pages.name, len(keep), len(qids), dict(self.served))

    def key(self, text):
        return self.cache / (hashlib.sha256(f"{QWEN}|{self.backend}\n{text}".encode()).hexdigest() + ".json")

    def request(self, text):
        if self.endpoint:
            response = requests.post(f"{self.endpoint.rstrip('/')}/embeddings", timeout=600,
                                     json={"model": os.getenv("QWEN_EMBED_SERVED") or QWEN, "input": [text]})
            response.raise_for_status()
            return response.json()["data"][0]["embedding"], "local"
        payload = _post("embeddings", {"model": QWEN, "input": [text], "provider": PROVIDER})
        served = payload.get("provider") or "unreported"
        if served != "unreported" and PROVIDER["order"][0] not in served.lower():
            raise RuntimeError(f"pinned {PROVIDER['order'][0]} but {served} answered")
        return payload["data"][0]["embedding"], served

    def embed(self, texts, fresh=False):
        def one(text):
            path = self.key(text)
            if not fresh and path.exists():
                return json.loads(path.read_text())["embedding"], None
            vector, served = self.request(text)
            if not fresh:
                path.write_text(json.dumps({"embedding": vector, "served_provider": served}))
            return vector, served

        out = [None] * len(texts)
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(one, t): i for i, t in enumerate(texts)}
            for done, future in enumerate(as_completed(futures), 1):
                out[futures[future]], served = future.result()
                if served:
                    self.served[served] += 1
                if done % 250 == 0 or done == len(texts):
                    log.info("qwen3_8b embedded %d/%d texts", done, len(texts))
        return np.asarray(out, dtype=np.float32)

    def legacy_query(self, text):
        path = self.cache / (hashlib.sha256((QWEN + "\n" + QWEN_PREFIX + text).encode()).hexdigest() + ".json")
        return np.asarray(json.loads(path.read_text()), dtype=np.float32) if path.exists() else None

    def scores(self, text, qid=None, raw=None):
        qvec = self.query_rows.get(qid) if text == raw else None
        if qvec is None and self.legacy:
            qvec = self.legacy_query(text)
        if qvec is None and not self.legacy:
            path = self.key(self.prefix + text)
            if path.exists() or self.online:
                qvec = self.embed([self.prefix + text])[0]
        if qvec is None:
            refuse("embedder", "the Qwen3-Embedding-8B query vector", "--stage-mode embedder=online")
        return self.pages @ unit(qvec)

    def repeat_check(self, rows, n_pages=50, n_queries=20):
        texts = self.page_texts(rows)
        keep = [i for i, t in enumerate(texts) if t.strip()]
        picked = keep[::max(1, len(keep) // n_pages)][:n_pages]
        pages, query_path, meta = self.files()
        stored_pages, stored_queries = np.load(pages), np.load(query_path)
        info = json.loads(meta.read_text())
        where = {u: i for i, u in enumerate(info["units"])}
        queries = load_queries(self.bench)
        qids = info["qids"][:n_queries]
        fresh = self.embed([texts[i] for i in picked] + [self.prefix + queries[q]["query"] for q in qids], fresh=True)
        old = np.vstack([stored_pages[[where[rows[i]["page_id"]] for i in picked]],
                         stored_queries[[info["qids"].index(q) for q in qids]]])
        cos = np.sum(unit(fresh) * unit(old), axis=1)
        return {"n": len(cos), "identical": int(np.sum(np.all(fresh == old, axis=1))),
                "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
                "max_abs_diff": float(np.abs(fresh - old).max()), "served": dict(self.served)}

    def settings(self):
        return {"embedder": self.name, "model": QWEN, "level": "page", "matrix": self.stem,
                "backend": self.backend, "provider": None if self.legacy else PROVIDER,
                "query_prefix": self.prefix, "max_tokens": self.max_tokens, "truncation_tokenizer": self.tokenizer,
                "text_source": getattr(self, "text_source", None)}
