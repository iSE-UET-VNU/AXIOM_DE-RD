# src/retrieval

Sparse, dense and fused retrieval over pipeline output.

**Use it as a library today.** The HTTP service is scaffolding — `app.py`,
`search.py`, `artifacts.py`, `upstream.py` and `rerank.py` raise
`NotImplementedError`. Nothing listens on a port yet. See
[Service](#service-not-yet-serving) for the shape it will take.

---

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.11 and numpy ≥ 1.26.

## Build an index

Artifacts are built offline from a pipeline run into
`src/retrieval/artifacts/<index_id>/`:

```bash
python -m src.retrieval.build_artifacts --run-id <RUN_ID> --analyzer auto
```

| Flag | Default | Notes |
|---|---|---|
| `--run-id` | required | pipeline run whose output to index |
| `--analyzer` | `auto` | `auto` scans the corpus once and picks `cjk_bigram` if **any** CJK is present |
| `--metric` | `cosine` | or `ip` |
| `--output` | `src/retrieval/artifacts` | |

Writes `manifest.json`, `bm25.json`, `chunks.jsonl`, `vectors.npy`,
`chunk_ids.json`. The manifest records `analyzer_id`, `embedder_id`, `dim`,
`normalized`, `metric` and `corpus_hash` — load-time assertions compare them, so
a mismatched artifact set fails loudly instead of scoring garbage.

## Retrieve

```python
from src.retrieval.index import load_artifacts
from src.retrieval import retrievers

# `embedder` is only needed by arms with a dense leg; bm25 works without one.
index = load_artifacts("src/retrieval/artifacts/<index_id>", embedder=embedder)
arm = retrievers.build("rrf", index)          # bm25 | dense | rrf | alpha0.7

for hit in arm.retrieve("thông tin tuyển sinh", k=10):
    print(hit.rank, hit.score, hit.doc_id, hit.text[:80])
```

`scope` restricts retrieval to a document subset. It is applied **inside**
scoring, not as a post-filter, so you always get `k` in-scope results:

```python
arm.retrieve(query, k=10, scope=["a.pdf", "b.pdf"])
```

## Persist a run

```python
from src.retrieval import runs

record = runs.RunRecord.build(qid, query, arm.retriever_id,
                              index.index_id, params_hash, hits)
runs.write(path, [record])
```

One JSON object per question, cached on
`(index_id, retriever_id, params_hash, query_set_hash)`.

## Two invariants worth knowing before you extend this

**The index owns query encoding.** Retrievers call `index.encode_query(text)`
and never hold an embedder. Models with required prefixes (E5, GTE) would
otherwise silently lose them in one arm and not another. A retriever asking for
an encoding the index cannot produce raises `UnsupportedEncoding` rather than
falling back.

**The analyzer is resolved once per corpus, not per document.** Choosing it per
text lets a document index as `plain` while a query analyzes as `cjk_bigram` —
the intersection is empty, so the arm scores zero and reports success. Resolution
is by presence, not proportion: a corpus that is 7% CJK still needs bigrams.

---

## Service (not yet serving)

Planned insertion, one environment variable on Methods-Hub:

```
agent ──MCP──> Methods-Hub :8000 ──> [ retrieval :8081 ] ──> corpus-service :8002
                                              └──> model-service :8006 (rerank)

CORPUS_SERVICE_URL=http://retrieval-service:8081
```

Methods-Hub reaches every corpus path through one cached HTTP client, so routing
is all-or-nothing across five paths: `vector-search`, `keyword-search`,
`context`, `neighbor-chunks`, `ingested-data`. `RETRIEVAL_ENHANCE` is empty by
default, making adoption a byte-preserving reverse proxy — enhancement is turned
on per path afterwards.

Defaults in `settings.py` carry their evidence. Note that the headline
configuration (R@1 0.425 → 0.744) is **not** statistically significant at n=49
(p=0.060) — a strong candidate, not a proven improvement. `reranker` is `none`
for that reason.
