"""Stage 1 - light preparation: data preparation (corpus) + query preparation.

Data preparation
- Page text: common.PAGES_PATH, the repo's pages_ppocrv5.jsonl (pdf-inspector text; PP-OCRv5 for scanned / image
  pages; produced upstream, not copied into the bundle).
  real_text() removes "[Image: ...]" placeholders. BM25 index: enfr analyzer (English + French stemming), PageBM25.
- Page vectors: Qwen3-Embedding-8B, ONE vector per page (page text clipped to 6,000 cl100k tokens), cached in
  5,091 x 4096, rows in cache/vectors_meta.json page_ids order. --live re-embeds only
  when the cache is missing. The vectors are read from common.PAGE_VECTORS (repo cache, not copied into the bundle).

Query preparation
- Three-layer decomposition by Qwen3-32B (OpenRouter, no thinking, prompt PROMPT_V3): SCOPE (document title), TOPIC
  (section heading), ANCHOR (detail in a sentence), DISCARD (question form, abstract categories, collection boilerplate).
  BM25 text = SCOPE + TOPIC + ANCHOR joined by " ; " (falls back to the full question when all three are empty).
  Raw responses cached in cache/llm_decomposition/<sha256(model + "\\n" + prompt)>.txt.
- Query vectors: Qwen3-Embedding-8B of QWEN_PREFIX + full question (cache/query_vectors_qwen3-embedding-8b.npy).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import common as C
from .text import PageBM25, enfr, real_text

LAYERS = ("scope", "topic", "anchor", "discard")
BOILER = re.compile(r"^(?:as per |according to |in )?(?:the )?(?:united states )?(?:air force )?(?:usaf )?"
                    r"(?:technical )?(?:manuals?|orders?|documents?)(?: and (?:technical )?orders?)?$")


@dataclass
class Corpus:
    page_ids: list[str]
    texts: list[str]
    files: list[str]
    page_file: np.ndarray
    page_lang: np.ndarray
    bm25: PageBM25
    vectors: np.ndarray          # L2-normalised, rows aligned with page_ids
    timing: dict


def prepare_data(live: bool = False) -> Corpus:
    t0 = time.perf_counter()
    if not C.PAGES_PATH.exists():
        raise FileNotFoundError(f"page text not found: {C.PAGES_PATH} (pass --pages or set DD_PAGES; see README)")
    rows = C.read_jsonl(C.PAGES_PATH)
    page_ids = [r["page_id"] for r in rows]
    meta = json.loads(C.VECTOR_META.read_text())
    assert page_ids == meta["page_ids"], "page file is not the 5,091-page corpus of this bundle (order or ids differ)"
    texts = [real_text(str(r.get("text") or "")) for r in rows]
    docs = [p.split("#", 1)[0] for p in page_ids]
    files = sorted(set(docs)); fidx = {d: i for i, d in enumerate(files)}
    t1 = time.perf_counter()
    bm25 = PageBM25([enfr(t) for t in texts])
    t2 = time.perf_counter()
    found = next((p for p in (C.LOCAL_PAGE_VECTORS, C.PAGE_VECTORS) if p.exists()), None)
    if found is not None:
        V = np.load(found)          # rows follow vectors_meta.json page_ids, checked above
        assert V.shape == (len(page_ids), 4096), f"unexpected page-vector shape {V.shape}"
        source = "cache"
    elif live:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        clipped = [enc.decode(enc.encode(t)[:6000]) for t in texts]
        keep = [i for i, t in enumerate(clipped) if t.strip()]
        V = np.zeros((len(texts), 4096), np.float32)
        V[keep] = C.embed([clipped[i] for i in keep])
        np.save(C.LOCAL_PAGE_VECTORS, V)
        source = "api"
    else:
        raise FileNotFoundError(f"page vectors not found: {C.PAGE_VECTORS} (pass --page-vectors / DD_PAGE_VECTORS, "
                                "or --live to re-embed the corpus with Qwen3-Embedding-8B)")
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    t3 = time.perf_counter()
    page_lang = np.array([C.lang(t) for t in texts])
    timing = {"load_pages_s": round(t1 - t0, 3), "bm25_build_s": round(t2 - t1, 3),
              f"page_vectors_{source}_s": round(t3 - t2, 3), "language_detection_s": round(time.perf_counter() - t3, 3)}
    return Corpus(page_ids, texts, files, np.array([fidx[d] for d in docs]), page_lang, bm25, V, timing)


# ----------------------------------------------------------------------------- query preparation

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w°%.-]+", " ", text.lower())).strip()


def post_process(row: dict) -> dict:
    """Collection boilerplate leaves SCOPE/TOPIC/ANCHOR; a piece kept in an earlier layer (scope > topic > anchor) is
    removed from later layers and from DISCARD."""
    seen = set()
    for layer in ("scope", "topic", "anchor"):
        kept = []
        for piece in row[layer]:
            key = _norm(piece)
            stripped = re.sub(r"^(?:the |as per |according to )", "", key)
            if BOILER.match(stripped) or stripped in ("usaf", "air force", "united states air force usaf"):
                row["discard"].append(piece)
                continue
            if key and key not in seen:
                kept.append(piece); seen.add(key)
        row[layer] = kept
    row["discard"] = [p for p in dict.fromkeys(row["discard"]) if _norm(p) not in seen]
    return row


def _chat(prompt: str, live: bool, provider: str | None, timeout: float) -> tuple[str, dict]:
    """Cache: cache/llm_decomposition/<provider or openrouter_default>/<sha256(model + "\\n" + prompt)>.txt, with the
    call's latency / provider / cost in a .meta.json next to it (so a replay still reports the original timings)."""
    folder = C.LLM_CACHE / (provider or "openrouter_default")
    path = folder / (hashlib.sha256((C.LLM_MODEL + "\n" + prompt).encode()).hexdigest() + ".txt")
    meta_path = path.with_suffix(".meta.json")
    if path.exists() and path.read_text(encoding="utf-8").strip():
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return path.read_text(encoding="utf-8"), {"cache": "hit", "original_call": meta}
    if not live:
        raise FileNotFoundError(f"no cached decomposition for this prompt ({folder.name}/{path.name}); rerun with --live")
    body = {"model": C.LLM_MODEL, "temperature": 0.0, "max_tokens": 800, "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False}, "messages": [{"role": "user", "content": prompt}]}
    if provider:   # pinned: no silent fallback to another provider
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    t0 = time.perf_counter(); text, info = "", {}
    for attempt in range(3):
        try:
            p = C.post("chat/completions", body, retries=2, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            p = {"error": str(exc)}
        text = (((p.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        info = {"cache": "live", "latency_s": round(time.perf_counter() - t0, 2), "provider": p.get("provider"),
                "cost_usd": (p.get("usage") or {}).get("cost"), "prompt_tokens": (p.get("usage") or {}).get("prompt_tokens"),
                "completion_tokens": (p.get("usage") or {}).get("completion_tokens"),
                "reasoning_tokens": ((p.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "attempt": attempt, "error": p.get("error")}
        if text.strip() and "{" in text:
            folder.mkdir(parents=True, exist_ok=True); path.write_text(text, encoding="utf-8")
            meta_path.write_text(json.dumps({k: v for k, v in info.items() if k != "cache"}), encoding="utf-8")
            break
    return text, info


def prepare_queries(live: bool = False, provider: str | None = None, workers: int = 16, timeout: float = 60):
    queries = C.load_queries(); qids = sorted(queries)

    def one(q):
        t0 = time.perf_counter()
        text, info = _chat(C.PROMPT_V3.format(query=queries[q]), live, provider, timeout)
        try:
            parsed, failed = C.parse_json_object(text), False
        except ValueError:
            parsed, failed = {}, True          # no usable JSON: BM25 falls back to the full question
        row = {"qid": q, "query": queries[q]}
        for layer in LAYERS:
            row[layer] = [str(x).strip() for x in parsed.get(layer) or [] if str(x).strip()]
        row = post_process(row)
        row["answer_shape"] = str(parsed.get("answer_shape") or "")
        row["bm25_text"] = " ; ".join(x for l in ("scope", "topic", "anchor") for x in row[l]) or queries[q]
        row["bm25_fallback_to_question"] = row["bm25_text"] == queries[q]
        row["llm_failed"] = failed
        row["llm"] = {**info, "wall_s": round(time.perf_counter() - t0, 3)}
        return row

    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, qids))
    wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    meta = json.loads(C.VECTOR_META.read_text()) if C.VECTOR_META.exists() else {}
    if C.QUERY_VECTORS.exists() and sorted(meta.get("qids", [])) == qids:
        Qv = np.load(C.QUERY_VECTORS); qrow = {q: i for i, q in enumerate(meta["qids"])}
        Qv = np.stack([Qv[qrow[q]] for q in qids]); qsrc = "cache"
    elif live:
        Qv = C.embed([C.QWEN_PREFIX + queries[q] for q in qids]); qsrc = "api"
    else:
        raise FileNotFoundError("query vectors missing; rerun with --live")
    Qv = Qv / (np.linalg.norm(Qv, axis=1, keepdims=True) + 1e-9)
    timing = {"decomposition_wall_s": round(wall, 3), "decomposition_workers": workers,
              "decomposition_cache_hits": sum(r["llm"].get("cache") == "hit" for r in rows),
              "decomposition_live_calls": sum(r["llm"].get("cache") == "live" for r in rows),
              f"query_vectors_{qsrc}_s": round(time.perf_counter() - t1, 3)}
    calls = [r["llm"] if r["llm"].get("cache") == "live" else r["llm"].get("original_call") or {} for r in rows]
    lat = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    timing["decomposition_provider"] = provider or "openrouter_default (not pinned)"
    if lat:   # per-call API latency, from this run or from the run that filled the cache
        timing.update(decomposition_latency_from=("this run" if timing["decomposition_live_calls"] else "cached original calls"),
                      decomposition_latency_median_s=round(float(np.median(lat)), 3),
                      decomposition_latency_mean_s=round(float(np.mean(lat)), 3),
                      decomposition_latency_p90_s=round(float(np.percentile(lat, 90)), 3),
                      decomposition_latency_max_s=round(float(np.max(lat)), 3),
                      decomposition_cost_usd=round(sum(c.get("cost_usd") or 0 for c in calls), 5),
                      decomposition_providers_seen=sorted({str(c.get("provider")) for c in calls}),
                      decomposition_reasoning_tokens_total=sum(c.get("reasoning_tokens") or 0 for c in calls))
    return rows, {q: Qv[i] for i, q in enumerate(qids)}, timing
