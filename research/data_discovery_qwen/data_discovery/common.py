"""Shared paths, constants, I/O, OpenRouter client and metric helpers of the data-discovery bundle."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

BUNDLE = Path(__file__).resolve().parents[1]
INPUTS, CACHE, OUTPUTS = BUNDLE / "inputs", BUNDLE / "cache", BUNDLE / "outputs"
# The 5,091-page text and page vectors are NOT copied into the bundle (size); they are read from the repo caches.
# Override with --pages / --page-vectors (run_all.py) or the env vars DD_PAGES / DD_PAGE_VECTORS.
# Repo root: env DD_REPO, else the first parent folder that has data/work/ (the bundle sits in <repo>/research/data_discovery_qwen/).
REPO = Path(os.environ["DD_REPO"]) if os.environ.get("DD_REPO") else next(
    (p for p in BUNDLE.parents if (p / "data/work").is_dir()), BUNDLE.parent)
FINGERPRINT = "ef7f11811803"
PAGES_PATH = Path(os.environ.get("DD_PAGES", REPO / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work"
                                 / "ondemand_v2" / FINGERPRINT / "light_prep/pages_ppocrv5.jsonl"))
PAGE_VECTORS = Path(os.environ.get("DD_PAGE_VECTORS", REPO / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2"
                                   / FINGERPRINT / "light_v2/dense_qwen8b_ppocr_pages.npy"))
LOCAL_PAGE_VECTORS = CACHE / "page_vectors_qwen3-embedding-8b.npy"   # written only by --live re-embedding
EXCLUDED = INPUTS / "excluded_questions.json"
QUERY_VECTORS = CACHE / "query_vectors_qwen3-embedding-8b.npy"
VECTOR_META = CACHE / "vectors_meta.json"
LLM_CACHE = CACHE / "llm_decomposition"
EMBED_CACHE = CACHE / "embedding_api"
SOURCES = ("vidore_physics", "vidore_industrial", "ohrbench")

EMBED_MODEL = "qwen/qwen3-embedding-8b"
QWEN_PREFIX = "Instruct: Given a question, retrieve document pages that answer the question\nQuery: "
LLM_MODEL = "qwen/qwen3-32b"
API_URL = "https://openrouter.ai/api/v1"

PROMPT_V3 = """A question was written about one document of a mixed collection (technical manuals and checklists,
course slides, reports, papers and forms; some documents are in French). Split the question into short pieces and
give each piece ONE type by asking where the document would print it.

SCOPE  = the document's cover / title. Which document or setting the WHOLE document is about: a document name, number
         or acronym; the organisation, product, facility, vehicle or equipment family it covers; the course or chapter
         subject; the general domain (e.g. hot refueling, corrosion prevention, quantum mechanics, annual report).
         NEVER scope: generic references to the collection ("USAF technical manuals and orders", "the manual",
         "technical manual", "the document", "the course", "the report") -> DISCARD.
TOPIC  = a section heading or slide title: the procedure, subject or system the answer is about (a noun phrase a
         table of contents would list).
         NEVER topic: abstract categories invented by the question writer ("safety risks", "safety requirements",
         "durability", "factors affecting", "key components", "standards", "accountability", "hazard mitigation",
         "considerations", "measures", "implications", "significance") -> DISCARD.
ANCHOR = inside a sentence, table or figure of the answer page: a specific detail such as a named part or equipment,
         a substance or material, a person or organisation, a value with its unit, a date, a specific action or
         condition, a role, a code, a formula or symbol, or a document cited inside the text.
DISCARD = question form and answer type ("explain", "describe", "list", "compare", "sequence of actions",
         "similarities", "purpose of", "how it is constructed"), the abstract categories above, collection references.

Rules:
- Copy pieces exactly from the question; drop leading articles and function words. Each word goes to ONE piece only.
- Keep a multi-word name together ("R-12 HSV", "globe valve", "Jet B"). Do not repeat a piece in two types.
- A paraphrase of a whole-document subject is SCOPE even if long; you may join two spans with " ... ".
- answer_shape: "single_fact" (one specific detail), "every_instance" (an item that recurs in many places of the
  document), "document_summary" (an overview of the whole document).

Examples (not from the collection):
Q: Describe the steps for draining condensate from the R-11 refueler sump before the first fueling of the day, as per USAF technical orders.
A: {{"scope": ["R-11 refueler"], "topic": ["draining condensate"], "anchor": ["sump", "before the first fueling of the day"], "discard": ["Describe the steps for", "as per USAF technical orders"], "answer_shape": "single_fact", "relations": [["draining condensate", "from", "sump"]]}}
Q: How does the manual address the hazards of servicing aircraft while passengers remain on board?
A: {{"scope": ["servicing aircraft ... while passengers remain on board"], "topic": [], "anchor": [], "discard": ["How does the manual address the hazards of"], "answer_shape": "document_summary", "relations": []}}
Q: Why does an electron passing through two slits not follow a single trajectory, according to the course?
A: {{"scope": [], "topic": ["two slits"], "anchor": ["electron", "single trajectory"], "discard": ["Why does", "according to the course"], "answer_shape": "single_fact", "relations": [["electron", "passing through", "two slits"]]}}
Q: What was the net income of the Northwind Group in 2021 as stated in its annual report?
A: {{"scope": ["Northwind Group", "annual report"], "topic": ["net income"], "anchor": ["2021"], "discard": ["What was", "as stated in its"], "answer_shape": "single_fact", "relations": [["net income", "in", "2021"]]}}

Question: {query}
Return JSON only:
{{"scope": [...], "topic": [...], "anchor": [...], "discard": [...], "answer_shape": "...", "relations": [[...]]}}"""


# ----------------------------------------------------------------------------- I/O

def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_queries() -> dict[str, str]:
    return {r["query_id"]: r["query"] for r in read_jsonl(INPUTS / "queries.jsonl")}


def load_qrels() -> dict[str, dict[str, int]]:
    """query_id -> {page_id: grade}; grade 2 = fully relevant, 1 = partly relevant (OHR: 1)."""
    out: dict[str, dict[str, int]] = {}
    for r in read_jsonl(INPUTS / "qrels.jsonl"):
        out.setdefault(r["query_id"], {})[r["page_id"]] = int(r["relevance"])
    return out


# ----------------------------------------------------------------------------- OpenRouter

def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        for env in [BUNDLE / ".env"] + [p / ".env" for p in BUNDLE.parents[:3]]:
            if env.exists():
                for line in env.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("OPENROUTER_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (environment or .env); needed only for --live")
    return key


def post(path: str, body: dict, retries: int = 4, timeout: float = 180) -> dict:
    import requests
    last = ""
    for attempt in range(retries):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            r = requests.post(f"{API_URL}/{path}", json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {api_key()}"})
            payload = r.json()
        except Exception as exc:  # noqa: BLE001  network / JSON error
            last = str(exc); continue
        if r.status_code == 200 and not payload.get("error"):
            return payload
        last = str(payload.get("error") or payload)[:300]
    raise RuntimeError(last)


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    """Qwen3-Embedding-8B vectors, cached per text in cache/embedding_api/ (same key scheme as the original run)."""
    EMBED_CACHE.mkdir(parents=True, exist_ok=True)
    key = lambda t: EMBED_CACHE / (hashlib.sha256((EMBED_MODEL + "\n" + t).encode()).hexdigest() + ".json")
    out, todo = [None] * len(texts), []
    for i, t in enumerate(texts):
        if key(t).exists():
            out[i] = json.loads(key(t).read_text())
        else:
            todo.append(i)
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        data = post("embeddings", {"model": EMBED_MODEL, "input": [texts[i] for i in idx]})["data"]
        for i, item in zip(idx, sorted(data, key=lambda d: d["index"])):
            out[i] = item["embedding"]; key(texts[i]).write_text(json.dumps(item["embedding"]))
    return np.asarray(out, dtype=np.float32)


def parse_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
    return json.loads(match.group(0))


# ----------------------------------------------------------------------------- scoring helpers

def max_norm(v: np.ndarray) -> np.ndarray:
    top = float(v.max()) if v.size else 0.0
    return np.clip(v, 0, None) / top if top > 0 else np.zeros_like(v, dtype=np.float64)


def minmax(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


FR = re.compile(r"\b(le|la|les|des|du|une|est|et|dans|pour|qui|que|sur|par|au|aux|ce|cette)\b", re.I)
EN = re.compile(r"\b(the|and|of|to|is|for|with|shall|will|be|on|by|this|that|are)\b", re.I)


def lang(text: str) -> str:
    """French vs English by function-word counts; 'unknown' below 5 hits. No labels are used."""
    f, e = len(FR.findall(text)), len(EN.findall(text))
    if f + e < 5:
        return "unknown"
    return "fr" if f > e else "en"


def rank_matched_hybrid(bm: np.ndarray, dn: np.ndarray, same: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """alpha * BM25 + (1 - alpha) * dense on every page. A page in another language than the question has no usable
    BM25, so it borrows one by rank: the r-th cross-language page by dense gets the r-th highest BM25 among the
    same-language pages (ranks scaled by the two group sizes)."""
    out = alpha * bm + (1 - alpha) * dn
    cross = np.flatnonzero(~same)
    if len(cross) and same.any():
        sb = np.sort(bm[same])[::-1]
        r = np.argsort(np.argsort(-dn[cross], kind="stable"), kind="stable")
        idx = np.minimum((r * same.sum() / len(cross)).astype(int), len(sb) - 1)
        out[cross] = alpha * sb[idx] + (1 - alpha) * dn[cross]
    return out


def page_metrics(ranked: list[str], gold: dict[str, int], ks=(10, 20, 100)) -> dict[str, float]:
    """R@K over all gold pages (grades 1 and 2); nDCG@K with linear gain = grade."""
    out = {}
    for k in ks:
        top = ranked[:k]
        out[f"R@{k}"] = sum(1 for p in top if p in gold) / len(gold)
        dcg = sum(gold.get(p, 0) / np.log2(r + 2) for r, p in enumerate(top))
        ideal = sorted(gold.values(), reverse=True)[:k]
        out[f"nDCG@{k}"] = dcg / sum(v / np.log2(r + 2) for r, v in enumerate(ideal))
    return out


def by_source(per_query: dict[str, dict[str, float]], scale: float = 100.0, digits: int = 2) -> dict:
    out = {}
    for src in ("all",) + SOURCES:
        rows = [v for q, v in per_query.items() if src == "all" or q.startswith(src)]
        out[src] = {k: round(scale * float(np.mean([r[k] for r in rows])), digits) for k in rows[0]}
    return out
