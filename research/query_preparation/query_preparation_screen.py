"""Query-preparation screens on the Industrial PP-OCRv5 page corpus.

Rebuilds the one-off drivers behind ``data/work/query_preparation/`` (see the
"Query preparation / pain-point decomposition screen" section of AGENTS.md):

* ``keywords``  - pain-point keyword extraction for all 75 Industrial queries;
  BM25 on the full query vs. the keyword query, keyword BM25 fused with the
  original-query embedding, and the saved hierarchical hybrid as context.
* ``decompose`` - keep/decompose decision for low-recall Industrial queries
  (saved hybrid Discovery Recall@100 < 0.60); per-subquery BM25 unions, with and
  without the saved hybrid top-20, reranked by cached EVIE-4.5B scores.
* ``focus``     - single-query diagnostic (default ``vidore_industrial::192``),
  including a compact-sentence embedding arm.
* ``verify``    - compare a rebuilt output directory with an original one.

LLM steps have two modes:

* ``--llm-mode replay`` (default) reuses the decisions saved by the original
  experiment (``keyword_queries.jsonl`` / ``decompositions.jsonl``). No API call.
  The original prompts were not preserved, so this is the only exact
  reproduction path.
* ``--llm-mode live`` calls the model through OpenRouter with the prompts in
  this file. These prompts are reconstructions, not the original prompts; write
  live runs to a new output directory and label them as a new prompt version.

Qrels are used only for subset selection (``decompose``) and evaluation; they
are never sent to an LLM. Retrieval reuses frozen artifacts:

* pages:  ``light_prep/pages_ppocrv5.jsonl`` (5,091 pages), ``[Image: ...]``
  placeholders removed before BM25 and chunking;
* BM25:   ``enfr`` analyzer + ``PageBM25`` from ``research/baselines/hierarchical_light/ondemand``;
* dense:  ``openai/text-embedding-3-small`` cached chunk/query vectors
  (512-word windows, 128 overlap), cosine, max-pooled per page;
* fusion: ``0.30 * minmax(BM25) + 0.70 * minmax(dense)``, each signal min-max
  normalized over its own top 100 and zero elsewhere (page-level, no file boost);
* saved hybrid: ``light_run_top100`` from ``gate_ppocrv5_ungated.json``;
* EVIE: cached ``evie_scores_2048.npy`` (220 x 5,091) for the original query.

Examples::

    python research/query_preparation/query_preparation_screen.py keywords
    python research/query_preparation/query_preparation_screen.py decompose
    python research/query_preparation/query_preparation_screen.py focus --qid vidore_industrial::192
    python research/query_preparation/query_preparation_screen.py verify \
        --original data/work/query_preparation/industrial_all_keyword_fusion \
        --rebuilt research/query_preparation/results/rebuild/keywords
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/baselines/hierarchical_light"))
from ondemand.text import PageBM25, enfr, real_text, windows  # noqa: E402

FINGERPRINT = "ef7f11811803"
HANDOFF = ROOT / "data/work/adaptive_routers_handoff/adaptive_routers_handoff/data/work/ondemand_v2" / FINGERPRINT
EVAL = ROOT / "data/work/Cached-Ondemand-Eval/data/work/ondemand_v2" / FINGERPRINT
PAGES_PATH = HANDOFF / "light_prep/pages_ppocrv5.jsonl"
GATE_PATH = HANDOFF / "light_prep/gate_ppocrv5_ungated.json"
EMBED_CACHE = HANDOFF / "embedding_cache/te3s_ppocr"
EVIE_PATH = HANDOFF / "routing_axes/k3/k3_out_shard0/evie_scores_2048.npy"
EVIE_QIDS = EVAL / "colvec/colvec_qids.json"
EVIE_KEYS = EVAL / "colvec/colvec_keys.json"
QUERIES_PATH = ROOT / "data/raw/0.1. BENCHMARK/queries.jsonl"
QRELS_PATH = (
    ROOT / "data/work/light_debug_bundle_benchmark_default_bm25_tesseract_kf3_top20"
    / "bundle_debug_light-retrieval/input/qrels_page_level.jsonl"
)
PREP = ROOT / "data/work/query_preparation"
ORIGINAL_KEYWORDS = PREP / "industrial_all_keyword_fusion/keyword_queries.jsonl"
ORIGINAL_DECOMPOSITIONS = PREP / "industrial_low_recall_decomposition/strict_prompt_v2/decompositions.jsonl"
ORIGINAL_FOCUS = PREP / "industrial_192_painpoint_fusion/experiment.json"
ORIGINAL_FOCUS_EMBEDDING = PREP / "industrial_192_painpoint_fusion/embedding_cache/emb_9a98e529e9958ab2d0f189ae70c65de44a04862b.json"
DEFAULT_OUT = ROOT / "research/query_preparation/results/rebuild"

SOURCE = "vidore_industrial"
EMBED_MODEL = "openai/text-embedding-3-small"
LLM_MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1"
W_BM25, W_DENSE, NORM_DEPTH, DEPTH = 0.30, 0.70, 100, 100
CHUNK_WORDS, CHUNK_OVERLAP = 512, 128
RECALL_KS = (10, 50, 100)
LOW_RECALL_THRESHOLD = 0.60
SUBQUERY_KS = (5, 10, 20)
HYBRID_PREFIX = 20

KEYWORD_PROMPT = """You prepare search queries for a lexical (BM25) retriever over technical manuals.
Select the pain-point phrases of the question: the specific entities, procedures,
equipment, quantities and conditions that a relevant page must mention.

Rules:
- Copy phrases verbatim from the question; do not add synonyms or inferred terms.
- Drop question framing and generic source references such as "describe",
  "explain", "according to the manual", "as per the USAF technical manuals and orders".
- Keep identifiers, model numbers, acronyms and numbers exactly as written.
- Return 1 to 5 short phrases.

Example
Question: Describe the key safety steps and servicing procedures for concurrent servicing of the R-12 HSV as per the USAF technical manuals and orders.
Answer: {{"keywords": ["safety", "servicing procedures", "concurrent servicing", "R-12 HSV"]}}

Question: {query}
Answer with JSON only: {{"keywords": [...]}}"""

DECOMPOSE_PROMPT = """You decide whether a search question should be split into subqueries for a
lexical (BM25) retriever over technical manuals.

Decompose only when the question asks about distinct answer facets that are
likely to be described on different pages. Otherwise keep one concise query.

Rules:
- Each subquery must be grounded in words of the question; do not invent synonyms.
- Do not repeat the same anchor phrase in every subquery unless needed to disambiguate.
- Drop generic source references such as "as per the USAF technical manuals and orders".
- Use 2 to 4 subqueries when decomposing; exactly 1 concise query otherwise.

Question: {query}
Answer with JSON only:
{{"decompose": true|false, "reason": "<short reason>", "subqueries": ["...", "..."]}}"""


# --------------------------------------------------------------------------- io

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def doc_of(page_id: str) -> str:
    return page_id.split("#", 1)[0]


# ---------------------------------------------------------------------- corpus

class Corpus:
    """Frozen pages, BM25, dense chunk vectors, saved hybrid run and EVIE scores."""

    def __init__(self) -> None:
        for path in (PAGES_PATH, GATE_PATH, EMBED_CACHE, EVIE_PATH, EVIE_QIDS, EVIE_KEYS, QUERIES_PATH, QRELS_PATH):
            if not path.exists():
                raise FileNotFoundError(path)
        pages = read_jsonl(PAGES_PATH)
        self.page_ids = [row["page_id"] for row in pages]
        self.texts = [real_text(row.get("text") or "") for row in pages]
        if len(self.page_ids) != 5091 or len(set(self.page_ids)) != 5091:
            raise ValueError(f"expected 5,091 unique pages, found {len(set(self.page_ids))}")
        self.index = {page_id: i for i, page_id in enumerate(self.page_ids)}

        self.queries = {row["query_id"]: row for row in read_jsonl(QUERIES_PATH) if row["query_id"].startswith(SOURCE + "::")}
        self.gold: dict[str, list[str]] = {}
        for row in read_jsonl(QRELS_PATH):
            qid = row["query_id"]
            if qid in self.queries and row["page_id"] not in self.gold.setdefault(qid, []):
                self.gold[qid].append(row["page_id"])
        missing = [p for pages_ in self.gold.values() for p in pages_ if p not in self.index]
        if len(self.queries) != 75 or set(self.gold) != set(self.queries) or missing:
            raise ValueError(f"Industrial qrel audit failed: {len(self.queries)} queries, {len(self.gold)} with gold, {len(missing)} unmapped")
        self.qids = sorted(self.queries, key=lambda q: int(q.split("::")[1]))

    def query(self, qid: str) -> str:
        return self.queries[qid]["query"]

    @cached_property
    def bm25(self) -> PageBM25:
        return PageBM25([enfr(text) for text in self.texts])

    @cached_property
    def chunks(self) -> tuple[np.ndarray, np.ndarray]:
        owners, vectors = [], []
        for i, text in enumerate(self.texts):
            for start, end in windows(text, CHUNK_WORDS, CHUNK_OVERLAP):
                chunk = text[start:end]
                if chunk.strip():
                    owners.append(i)
                    vectors.append(self._cached_vector(chunk, required=True))
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.asarray(owners), matrix

    @cached_property
    def saved_hybrid(self) -> dict[str, list[str]]:
        gate = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        run = gate["light_run_top100"]
        return {qid: run[qid] for qid in self.qids}

    @cached_property
    def evie(self) -> tuple[np.ndarray, dict[str, int], dict[str, int]]:
        scores = np.load(EVIE_PATH, mmap_mode="r")
        qids = json.loads(EVIE_QIDS.read_text(encoding="utf-8"))
        keys = json.loads(EVIE_KEYS.read_text(encoding="utf-8"))
        if scores.shape != (len(qids), len(keys)) or scores.shape != (220, 5091):
            raise ValueError(f"unexpected EVIE matrix {scores.shape}")
        return scores, {q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(keys)}

    # -- dense

    @staticmethod
    def _cache_path(text: str) -> Path:
        return EMBED_CACHE / (hashlib.sha256((EMBED_MODEL + "\n" + text).encode()).hexdigest() + ".json")

    def _cached_vector(self, text: str, required: bool = False) -> list[float] | None:
        path = self._cache_path(text)
        if path.exists():
            return json.loads(path.read_text())
        if required:
            raise FileNotFoundError(f"missing cached embedding for chunk/query: {text[:80]!r}")
        return None

    def query_vector(self, text: str, allow_api: bool = False) -> np.ndarray:
        vector = self._cached_vector(text)
        if vector is None:
            if not allow_api:
                raise FileNotFoundError(f"no cached embedding for query {text!r}; pass --allow-embedding-api")
            vector = embed_api(text)
            self._cache_path(text).write_text(json.dumps(vector))
        return np.asarray(vector, dtype=np.float32)

    def dense_scores(self, vector: np.ndarray) -> np.ndarray:
        owners, matrix = self.chunks
        vector = vector / np.linalg.norm(vector)
        scores = np.full(len(self.page_ids), -1.0, dtype=np.float32)
        np.maximum.at(scores, owners, matrix @ vector)
        return scores

    # -- ranking

    def bm25_scores(self, text: str) -> np.ndarray:
        return self.bm25.scores(enfr(text))

    def top(self, scores: np.ndarray, k: int = DEPTH) -> list[str]:
        return [self.page_ids[i] for i in np.argsort(-scores, kind="stable")[:k]]

    def fused(self, bm25: np.ndarray, dense: np.ndarray) -> np.ndarray:
        return W_BM25 * minmax_top(bm25) + W_DENSE * minmax_top(dense)

    def evie_rank(self, qid: str, pool: list[str]) -> list[str]:
        scores, q_index, p_index = self.evie
        row = q_index[qid]
        return sorted(pool, key=lambda p: float(scores[row, p_index[p]]), reverse=True)


def minmax_top(scores: np.ndarray, k: int = NORM_DEPTH) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")[:k]
    values = scores[order].astype(np.float64)
    out = np.zeros(len(scores), dtype=np.float64)
    lo, hi = values.min(), values.max()
    out[order] = (values - lo) / (hi - lo) if hi > lo else 1.0
    return out


# --------------------------------------------------------------------- metrics

def ranked_metrics(ranking: list[str], gold: list[str]) -> dict[str, Any]:
    position = {page: i + 1 for i, page in enumerate(ranking)}
    ranks = {page: position.get(page) for page in gold}
    metrics = {f"recall@{k}": sum(1 for r in ranks.values() if r and r <= k) / len(gold) for k in RECALL_KS}
    return {"metrics": metrics, "gold_ranks": ranks, "top20": ranking[:20]}


def pool_metrics(corpus: Corpus, qid: str, pool: list[str]) -> dict[str, Any]:
    gold = corpus.gold[qid]
    gold_files = {doc_of(p) for p in gold}
    pool_set = set(pool)
    files = {doc_of(p) for p in pool}
    top10 = corpus.evie_rank(qid, pool)[:10]
    return {
        "candidate_pages": len(pool),
        "candidate_files": len(files),
        "candidate_page_recall": len(pool_set & set(gold)) / len(gold),
        "candidate_gold_file_recall": len(files & gold_files) / len(gold_files),
        "evie_recall@10": len(set(top10) & set(gold)) / len(gold),
        "evie_top10": top10,
    }


def macro(rows: list[dict[str, Any]], arm_key: str = "arms", metric_key: str | None = "metrics") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for arm in rows[0][arm_key]:
        values = [row[arm_key][arm][metric_key] if metric_key else row[arm_key][arm] for row in rows]
        names = [k for k, v in values[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        out[arm] = {name: float(np.mean([v[name] for v in values])) for name in names}
    return out


def paired(rows: list[dict[str, Any]], a: str, b: str, metric: str) -> dict[str, int]:
    count = Counter()
    for row in rows:
        x, y = row["arms"][a]["metrics"][metric], row["arms"][b]["metrics"][metric]
        count["better" if x > y + 1e-12 else "worse" if x < y - 1e-12 else "same"] += 1
    return {k: count[k] for k in ("better", "same", "worse")}


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


# ------------------------------------------------------------------------- llm

def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (environment or repo .env)")
    return key


def post(path: str, body: dict[str, Any], retries: int = 4) -> dict[str, Any]:
    import requests

    last = ""
    for attempt in range(retries):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            response = requests.post(f"{API_URL}/{path}", json=body, timeout=180,
                                     headers={"Authorization": f"Bearer {api_key()}"})
            payload = response.json()
        except Exception as error:  # network or JSON decode error
            last = str(error)
            continue
        if response.status_code == 200 and not payload.get("error"):
            return payload
        last = str(payload.get("error") or payload)[:300]
    raise RuntimeError(last)


def embed_api(text: str) -> list[float]:
    return post("embeddings", {"model": EMBED_MODEL, "input": [text]})["data"][0]["embedding"]


def chat_cached(prompt: str, cache_dir: Path, model: str = LLM_MODEL) -> tuple[str, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ("llm_" + hashlib.sha256((model + "\n" + prompt).encode()).hexdigest() + ".txt")
    if path.exists():
        return path.read_text(encoding="utf-8"), True
    payload = post("chat/completions", {"model": model, "temperature": 0.0, "max_tokens": 400,
                                        "messages": [{"role": "user", "content": prompt}]})
    text = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    if not text.strip():
        raise RuntimeError(f"{model}: empty response")
    path.write_text(text, encoding="utf-8")
    return text, False


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
    return json.loads(match.group(0))


def run_live(prompts: dict[str, str], cache_dir: Path, workers: int) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    def one(qid: str) -> tuple[str, dict[str, Any], bool]:
        text, hit = chat_cached(prompts[qid], cache_dir)
        return qid, parse_json_object(text), hit

    results, stats = {}, Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for qid, parsed, hit in pool.map(one, list(prompts)):
            results[qid] = parsed
            stats["cache_hits" if hit else "api_calls"] += 1
    return results, {"api_calls": stats["api_calls"], "cache_hits": stats["cache_hits"]}


def validate_keywords(query: str, phrases: list[str]) -> tuple[list[str], list[str]]:
    """Keep phrases whose analyzed tokens all occur in the analyzed query."""
    query_terms = set(enfr(query))
    kept, dropped = [], []
    for phrase in phrases:
        terms = enfr(phrase)
        (kept if terms and all(t in query_terms for t in terms) else dropped).append(phrase)
    return kept, dropped


# -------------------------------------------------------------------- keywords

def get_keywords(corpus: Corpus, qids: list[str], args: argparse.Namespace, out: Path) -> tuple[dict[str, list[str]], dict[str, Any]]:
    if args.llm_mode == "replay":
        rows = {row["qid"]: row for row in read_jsonl(args.replay_from or ORIGINAL_KEYWORDS)}
        missing = [q for q in qids if q not in rows]
        if missing:
            raise ValueError(f"replay file lacks {len(missing)} qids, e.g. {missing[:3]}")
        return {q: rows[q]["keywords"] for q in qids}, {"mode": "replay", "source": str(args.replay_from or ORIGINAL_KEYWORDS),
                                                         "api_calls": 0, "prompt": "original prompt not preserved"}
    prompts = {q: KEYWORD_PROMPT.format(query=corpus.query(q)) for q in qids}
    parsed, stats = run_live(prompts, out / "llm_cache", args.workers)
    keywords = {}
    for q in qids:
        kept, _ = validate_keywords(corpus.query(q), [str(p) for p in parsed[q].get("keywords", [])])
        keywords[q] = kept or [corpus.query(q)]
    return keywords, {"mode": "live", "model": LLM_MODEL, "prompt_version": "reconstructed_keyword_v1", **stats}


def cmd_keywords(args: argparse.Namespace) -> Path:
    out = args.output_dir or DEFAULT_OUT / "keywords"
    corpus = Corpus()
    qids = corpus.qids
    started = time.perf_counter()
    keywords, llm_info = get_keywords(corpus, qids, args, out)
    replayed_queries: dict[str, str] = {}
    if args.llm_mode == "replay":  # keep the exact saved BM25 string
        replayed_queries = {r["qid"]: r["keyword_query"] for r in read_jsonl(args.replay_from or ORIGINAL_KEYWORDS) if r.get("keyword_query")}

    keyword_rows, rows, dropped_total = [], [], 0
    for qid in qids:
        query = corpus.query(qid)
        kept, dropped = validate_keywords(query, keywords[qid])
        dropped_total += len(dropped)
        keyword_query = replayed_queries.get(qid) or " ".join(kept or keywords[qid])
        keyword_rows.append({"qid": qid, "query": query, "keywords": keywords[qid], "keyword_query": keyword_query,
                             **({"dropped_by_validation": dropped} if dropped else {})})
        bm_full = corpus.bm25_scores(query)
        bm_kw = corpus.bm25_scores(keyword_query)
        dense = corpus.dense_scores(corpus.query_vector(query, args.allow_embedding_api))
        gold = corpus.gold[qid]
        rows.append({
            "qid": qid, "query": query, "keywords": keywords[qid], "keyword_query": keyword_query, "gold_count": len(gold),
            "arms": {
                "bm25_full": ranked_metrics(corpus.top(bm_full), gold),
                "bm25_keywords": ranked_metrics(corpus.top(bm_kw), gold),
                "bm25_keywords_plus_original_embedding": ranked_metrics(corpus.top(corpus.fused(bm_kw, dense)), gold),
                "saved_original_hybrid": ranked_metrics(corpus.saved_hybrid[qid], gold),
            },
        })

    metrics = {
        "queries": len(rows), "pages": len(corpus.page_ids), "gold_queries": sum(1 for q in qids if corpus.gold[q]),
        "keyword_extractor": {**llm_info, "qrels_or_answers_provided": False,
                              "validation": "phrases kept only if every enfr token occurs in the enfr-analyzed query",
                              "phrases_dropped_by_validation": dropped_total},
        "semantic_model": f"{EMBED_MODEL} cached embeddings of original full queries",
        "fusion": {"weights": f"{W_BM25:.2f} BM25 keyword + {W_DENSE:.2f} dense original query",
                   "normalization": f"per-signal min-max over top {NORM_DEPTH} page scores",
                   "page_pooling": "maximum chunk cosine similarity per page",
                   "parent_file_boost": "omitted for the newly constructed page-level arms"},
        "metrics_macro": macro(rows),
        "per_query_comparisons": {
            "keywords_vs_full_bm25_recall100": paired(rows, "bm25_keywords", "bm25_full", "recall@100"),
            "fused_vs_keyword_bm25_recall100": paired(rows, "bm25_keywords_plus_original_embedding", "bm25_keywords", "recall@100"),
        },
        "wall_seconds": round(time.perf_counter() - started, 2),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "driver": "research/query_preparation/query_preparation_screen.py keywords",
    }
    write_jsonl(out / "keyword_queries.jsonl", keyword_rows)
    write_jsonl(out / "per_query.jsonl", rows)
    write_json(out / "metrics.json", metrics)
    write_keyword_report(out, metrics)
    print(f"wrote {out}")
    return out


def write_keyword_report(out: Path, metrics: dict[str, Any]) -> None:
    m = metrics["metrics_macro"]
    labels = [("bm25_full", "BM25, full query"), ("bm25_keywords", "BM25, extracted keywords"),
              ("bm25_keywords_plus_original_embedding", "Keyword BM25 + original-query embedding"),
              ("saved_original_hybrid", "Saved original hybrid (full query)")]
    kw, fu = metrics["per_query_comparisons"]["keywords_vs_full_bm25_recall100"], metrics["per_query_comparisons"]["fused_vs_keyword_bm25_recall100"]
    lines = [
        "# Industrial keyword-query retrieval experiment", "",
        f"- Queries: {metrics['queries']} Industrial questions; corpus: {metrics['pages']:,} PP-OCRv5 pages.",
        f"- Keyword extraction: `{metrics['keyword_extractor']['mode']}` mode "
        f"({metrics['keyword_extractor'].get('source') or metrics['keyword_extractor'].get('prompt_version')}); qrels/answers never sent.",
        "- BM25: frozen `enfr` analyzer and PageBM25 over placeholder-stripped page text.",
        f"- Fusion: {metrics['fusion']['weights']}, {metrics['fusion']['normalization']}; page-level, no parent/file boost.", "",
        "| Arm | Recall@10 | Recall@50 | Recall@100 |", "|---|---:|---:|---:|",
        *[f"| {label} | {pct(m[a]['recall@10'])} | {pct(m[a]['recall@50'])} | {pct(m[a]['recall@100'])} |" for a, label in labels], "",
        f"At Recall@100, keyword BM25 beats/ties/loses to full-query BM25 on {kw['better']}/{kw['same']}/{kw['worse']} queries; "
        f"adding the original-query embedding beats/ties/loses to keyword BM25 on {fu['better']}/{fu['same']}/{fu['worse']}.", "",
        "Exploratory page-level comparison; the saved hybrid is the hierarchical baseline and is not an apples-to-apples arm.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------- decompose

def get_decompositions(corpus: Corpus, qids: list[str], args: argparse.Namespace, out: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if args.llm_mode == "replay":
        source = args.replay_from or ORIGINAL_DECOMPOSITIONS
        rows = {row["qid"]: row for row in read_jsonl(source)}
        missing = [q for q in qids if q not in rows]
        if missing:
            raise ValueError(f"replay file lacks {len(missing)} selected qids, e.g. {missing[:3]}")
        return {q: rows[q] for q in qids}, {"mode": "replay", "source": str(source), "api_calls": 0,
                                            "prompt": "original strict prompt v2 not preserved"}
    prompts = {q: DECOMPOSE_PROMPT.format(query=corpus.query(q)) for q in qids}
    parsed, stats = run_live(prompts, out / "decision_cache", args.workers)
    decisions = {}
    for q in qids:
        subqueries = [str(s).strip() for s in parsed[q].get("subqueries", []) if str(s).strip()] or [corpus.query(q)]
        decisions[q] = {"qid": q, "query": corpus.query(q), "decompose": bool(parsed[q].get("decompose")) and len(subqueries) > 1,
                        "reason": str(parsed[q].get("reason", "")), "subqueries": subqueries, "decision_model": LLM_MODEL}
    return decisions, {"mode": "live", "model": LLM_MODEL, "prompt_version": "reconstructed_decompose_v1", **stats}


def cmd_decompose(args: argparse.Namespace) -> Path:
    out = args.output_dir or DEFAULT_OUT / "decompose"
    corpus = Corpus()
    started = time.perf_counter()
    baseline = {q: ranked_metrics(corpus.saved_hybrid[q], corpus.gold[q])["metrics"]["recall@100"] for q in corpus.qids}
    qids = [q for q in corpus.qids if baseline[q] < args.threshold] if args.subset == "low_recall" else list(corpus.qids)
    keywords = {row["qid"]: row["keyword_query"] for row in read_jsonl(args.keywords_from or ORIGINAL_KEYWORDS)}
    decisions, llm_info = get_decompositions(corpus, qids, args, out)

    rows = []
    for qid in qids:
        decision = decisions[qid]
        subqueries = decision["subqueries"]
        sub_rankings = [corpus.top(corpus.bm25_scores(s), max(SUBQUERY_KS)) for s in subqueries]
        hybrid = corpus.saved_hybrid[qid]
        keyword_top = corpus.top(corpus.bm25_scores(keywords[qid]))
        pools = {
            "saved_original_hybrid_top100": hybrid[:100],
            "bm25_full_top100": corpus.top(corpus.bm25_scores(corpus.query(qid))),
            "bm25_keyword_top100": keyword_top,
            "bm25_keyword_top20": keyword_top[:20],
            **{f"decomp_union_top{k}_each": dedupe(p for r in sub_rankings for p in r[:k]) for k in SUBQUERY_KS},
            f"saved_original_hybrid_top{HYBRID_PREFIX}": hybrid[:HYBRID_PREFIX],
            **{f"hybrid{HYBRID_PREFIX}_plus_subquery_top{k}_union": dedupe([*hybrid[:HYBRID_PREFIX], *(p for r in sub_rankings for p in r[:k])])
               for k in SUBQUERY_KS},
        }
        rows.append({
            "qid": qid, "query": corpus.query(qid), "baseline_recall@100": baseline[qid],
            "decompose": decision["decompose"], "decision_reason": decision.get("reason", ""), "subqueries": subqueries,
            "gold_pages": len(corpus.gold[qid]), "gold_files": len({doc_of(p) for p in corpus.gold[qid]}),
            "subquery_top20": [{"subquery": s, "top20_pages": r[:20]} for s, r in zip(subqueries, sub_rankings)],
            "arms": {arm: pool_metrics(corpus, qid, pool) for arm, pool in pools.items()},
        })

    metrics = {
        "subset_rule": f"saved original hierarchical hybrid Discovery Recall@100 < {args.threshold:.2f}" if args.subset == "low_recall" else "all Industrial queries",
        "queries": len(rows), "decision": {**llm_info, "qrels_or_answers_provided": False},
        "keyword_query_source": str(args.keywords_from or ORIGINAL_KEYWORDS),
        "decompose_counts": {"decompose": sum(r["decompose"] for r in rows), "single_query": sum(not r["decompose"] for r in rows)},
        "subquery_count_distribution": dict(Counter(str(len(r["subqueries"])) for r in rows)),
        "retrieval": "BM25 per original or subquery using frozen enfr analyzer; union deduplicated by page ID",
        "reranker": "cached EVIE-4.5B page-query scores for the original query, no new inference",
        "retrieval_k_per_subquery": list(SUBQUERY_KS),
        "metrics_macro": macro(rows, metric_key=None),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "driver": "research/query_preparation/query_preparation_screen.py decompose",
    }
    write_jsonl(out / "decompositions.jsonl", [{k: decisions[q][k] for k in ("qid", "query", "decompose", "reason", "subqueries", "decision_model") if k in decisions[q]} for q in qids])
    write_jsonl(out / "per_query.jsonl", rows)
    write_json(out / "metrics.json", metrics)
    write_decompose_report(out, metrics)
    print(f"wrote {out}")
    return out


def dedupe(pages) -> list[str]:
    return list(dict.fromkeys(pages))


def write_decompose_report(out: Path, metrics: dict[str, Any]) -> None:
    m = metrics["metrics_macro"]

    def table(arms: list[tuple[str, str]]) -> list[str]:
        return ["| Arm | Mean candidate pages | Mean candidate files | Candidate page recall | Gold-file recall | EVIE Recall@10 |",
                "|---|---:|---:|---:|---:|---:|",
                *[f"| {label} | {m[a]['candidate_pages']:.1f} | {m[a]['candidate_files']:.1f} | {pct(m[a]['candidate_page_recall'])} | "
                  f"{pct(m[a]['candidate_gold_file_recall'])} | {pct(m[a]['evie_recall@10'])} |" for a, label in arms]]

    d = metrics["decision"]
    lines = [
        "# Industrial query decomposition screen", "",
        f"- Selection: {metrics['queries']} queries ({metrics['subset_rule']}). Benchmark-selected subsets are exploratory.",
        f"- Decision: `{d['mode']}` mode ({d.get('source') or d.get('prompt_version')}); only query text was sent to the LLM.",
        "- Retrieval: frozen `enfr` BM25 over 5,091 PP-OCRv5 pages; subquery results unioned and deduplicated by page ID.",
        "- Reranking: cached EVIE-4.5B scores for the original query; candidate metrics are measured before EVIE.", "",
        *table([("saved_original_hybrid_top100", "Saved hybrid top-100"), ("bm25_full_top100", "BM25 full query top-100"),
                ("bm25_keyword_top100", "BM25 pain-point keywords top-100"), ("bm25_keyword_top20", "BM25 pain-point keywords top-20"),
                *[(f"decomp_union_top{k}_each", f"Subquery BM25 union top-{k} each") for k in SUBQUERY_KS]]), "",
        f"Decompose: {metrics['decompose_counts']['decompose']}; keep one concise query: {metrics['decompose_counts']['single_query']}. "
        f"Subquery counts: {json.dumps(metrics['subquery_count_distribution'])}.", "",
        "## Baseline plus subquery candidate unions", "",
        *table([(f"saved_original_hybrid_top{HYBRID_PREFIX}", f"Saved hybrid top-{HYBRID_PREFIX}"),
                *[(f"hybrid{HYBRID_PREFIX}_plus_subquery_top{k}_union", f"Hybrid top-{HYBRID_PREFIX} + subquery top-{k} each") for k in SUBQUERY_KS]]),
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------- focus

def cmd_focus(args: argparse.Namespace) -> Path:
    out = args.output_dir or DEFAULT_OUT / f"focus_{args.qid.split('::')[-1]}"
    corpus = Corpus()
    qid = args.qid
    query, gold = corpus.query(qid), corpus.gold[qid]
    keyword_query = args.keyword_query
    compact = args.compact_query
    if keyword_query is None or compact is None:
        original = json.loads(ORIGINAL_FOCUS.read_text(encoding="utf-8")) if ORIGINAL_FOCUS.exists() else {}
        if original.get("qid") == qid:
            keyword_query = keyword_query or original["bm25_keyword_query"]
            compact = compact or original["compact_embedding_query"]
    if keyword_query is None:
        keyword_query = next((r["keyword_query"] for r in read_jsonl(ORIGINAL_KEYWORDS) if r["qid"] == qid), None)
    if keyword_query is None:
        raise ValueError("no keyword query; pass --keyword-query")

    bm_full, bm_kw = corpus.bm25_scores(query), corpus.bm25_scores(keyword_query)
    dense_orig = corpus.dense_scores(corpus.query_vector(query, args.allow_embedding_api))
    arms = {
        "bm25_full_query": ranked_metrics(corpus.top(bm_full), gold),
        "bm25_pain_only": ranked_metrics(corpus.top(bm_kw), gold),
        "original_hybrid_context": ranked_metrics(corpus.saved_hybrid[qid], gold),
        "pain_bm25_plus_original_embedding": ranked_metrics(corpus.top(corpus.fused(bm_kw, dense_orig)), gold),
    }
    if compact:
        vector = None
        if qid == "vidore_industrial::192" and ORIGINAL_FOCUS_EMBEDDING.exists() and compact == json.loads(ORIGINAL_FOCUS.read_text(encoding="utf-8"))["compact_embedding_query"]:
            vector = np.asarray(json.loads(ORIGINAL_FOCUS_EMBEDDING.read_text()), dtype=np.float32)
        if vector is None:
            vector = corpus.query_vector(compact, args.allow_embedding_api)
        arms["pain_bm25_plus_compact_embedding"] = ranked_metrics(corpus.top(corpus.fused(bm_kw, corpus.dense_scores(vector))), gold)

    payload = {"qid": qid, "original_query": query, "bm25_keyword_query": keyword_query, "compact_embedding_query": compact,
               "embedding_model": EMBED_MODEL, "weighting": f"{W_BM25:.2f} normalized BM25 + {W_DENSE:.2f} normalized dense; "
               f"min-max among each signal top {NORM_DEPTH}; max pool chunk cosine per page",
               "gold_pages": gold,
               "results": [{"arm": arm, "metrics": r["metrics"], "gold_ranks": [r["gold_ranks"][p] for p in gold], "top20": r["top20"]}
                           for arm, r in arms.items()],
               "driver": "research/query_preparation/query_preparation_screen.py focus"}
    write_json(out / "experiment.json", payload)
    lines = [f"# Pain-point fusion: {qid}", "", f"- Query: {query}", f"- BM25 keyword query: `{keyword_query}`",
             f"- Compact semantic query: `{compact}`" if compact else "- Compact semantic query: none", "",
             "| Arm | Gold ranks | Recall@10 | Recall@50 | Recall@100 |", "|---|---|---:|---:|---:|"]
    for r in payload["results"]:
        ranks = ", ".join(str(x) if x else ">100" for x in r["gold_ranks"])
        lines.append(f"| {r['arm']} | {ranks} | {pct(r['metrics']['recall@10'])} | {pct(r['metrics']['recall@50'])} | {pct(r['metrics']['recall@100'])} |")
    lines += ["", "Single-query diagnostic; fusion rows are page-level and omit the hierarchical file-parent boost."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


# ---------------------------------------------------------------------- verify

def cmd_verify(args: argparse.Namespace) -> int:
    original, rebuilt = args.original, args.rebuilt
    if (original / "experiment.json").exists() and not (original / "per_query.jsonl").exists():
        a = {r["arm"]: r for r in json.loads((original / "experiment.json").read_text(encoding="utf-8"))["results"]}
        b = {r["arm"]: r for r in json.loads((rebuilt / "experiment.json").read_text(encoding="utf-8"))["results"]}
        bad = [arm for arm in a if arm in b and (a[arm]["gold_ranks"] != b[arm]["gold_ranks"]
                                                 or [x["page_id"] if isinstance(x, dict) else x for x in a[arm]["top20"]] != b[arm]["top20"])]
        for arm in a:
            print(f"{arm:40s} original={a[arm]['gold_ranks']} rebuilt={b.get(arm, {}).get('gold_ranks')}")
        print("MISMATCH arms:" if bad else "all shared arms match", bad or "")
        return 1 if bad else 0

    a = {r["qid"]: r for r in read_jsonl(original / "per_query.jsonl")}
    b = {r["qid"]: r for r in read_jsonl(rebuilt / "per_query.jsonl")}
    if set(a) != set(b):
        print(f"qid sets differ: original {len(a)}, rebuilt {len(b)}, only-original {sorted(set(a) - set(b))[:5]}, only-rebuilt {sorted(set(b) - set(a))[:5]}")
        return 1
    mismatches: Counter = Counter()
    for qid in a:
        for arm, ra in a[qid]["arms"].items():
            rb = b[qid]["arms"].get(arm)
            if rb is None:
                mismatches[(arm, "missing")] += 1
                continue
            for field in ("top20", "evie_top10", "gold_ranks"):
                if field in ra and ra[field] != rb.get(field):
                    mismatches[(arm, field)] += 1
            for name, value in ra.get("metrics", ra).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and abs(value - rb.get("metrics", rb).get(name, float("nan"))) > 1e-9:
                    mismatches[(arm, name)] += 1
    def load_lenient(path: Path) -> dict[str, Any]:  # some originals carry trailing junk after the object
        return json.JSONDecoder().raw_decode(path.read_text(encoding="utf-8-sig").lstrip())[0]

    ma = load_lenient(original / "metrics.json")["metrics_macro"]
    mb = load_lenient(rebuilt / "metrics.json")["metrics_macro"]
    print(f"{len(a)} queries compared")
    for arm in ma:
        diffs = {k: (v, mb.get(arm, {}).get(k)) for k, v in ma[arm].items() if mb.get(arm, {}).get(k) is None or abs(v - mb[arm][k]) > 1e-9}
        print(f"  {arm:40s} {'OK' if not diffs else 'DIFF ' + json.dumps(diffs)}")
    if mismatches:
        print("per-query mismatches (arm, field): count")
        for key, count in sorted(mismatches.items()):
            print(f"  {key}: {count}")
        return 1
    print("all per-query rankings and metrics match")
    return 0


# ------------------------------------------------------------------------ main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def llm_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--llm-mode", choices=("replay", "live"), default="replay")
        p.add_argument("--replay-from", type=Path, help="decision/keyword JSONL to replay (default: original strict artifacts)")
        p.add_argument("--workers", type=int, default=18, help="concurrent LLM requests in live mode")
        p.add_argument("--output-dir", type=Path)
        p.add_argument("--allow-embedding-api", action="store_true", help="embed uncached queries through OpenRouter")

    p = sub.add_parser("keywords", help="pain-point keyword screen over all 75 Industrial queries")
    llm_options(p)
    p = sub.add_parser("decompose", help="keep/decompose screen with subquery BM25 unions and EVIE")
    llm_options(p)
    p.add_argument("--subset", choices=("low_recall", "all"), default="low_recall")
    p.add_argument("--threshold", type=float, default=LOW_RECALL_THRESHOLD)
    p.add_argument("--keywords-from", type=Path, help="keyword_queries.jsonl for the keyword arms")
    p = sub.add_parser("focus", help="single-query pain-point diagnostic")
    p.add_argument("--qid", default="vidore_industrial::192")
    p.add_argument("--keyword-query")
    p.add_argument("--compact-query")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--allow-embedding-api", action="store_true")
    p = sub.add_parser("verify", help="compare rebuilt outputs with an original artifact directory")
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--rebuilt", type=Path, required=True)

    args = parser.parse_args()
    for name in ("output_dir", "replay_from", "keywords_from", "original", "rebuilt"):
        value = getattr(args, name, None)
        if value is not None and not value.is_absolute():
            setattr(args, name, ROOT / value)
    if args.command == "verify":
        return cmd_verify(args)
    {"keywords": cmd_keywords, "decompose": cmd_decompose, "focus": cmd_focus}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
