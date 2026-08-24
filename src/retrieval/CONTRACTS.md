
# Retrieval service — contracts

**Status: design agreed, stubs written, not implemented.** Every module in this
package except `sparse.py`, `fusion.py`, `ids.py` and `build_artifacts.py` is a
stub whose bodies raise `NotImplementedError`. Nothing here serves traffic yet.

---

## 1. Where this service sits

```
agent ──MCP──> Methods-Hub :8000 ──> [ this service :8081 ] ──> corpus-service :8002
                     │                        │
                     │                        └──> model-service :8006  (rerank)
                     └──> OpenRouter  (query embedding, BEFORE we are called)
```

Insertion is one environment variable on Methods-Hub:

```
CORPUS_SERVICE_URL=http://retrieval-service:8081
```

Methods-Hub reaches every corpus path through a single `lru_cache(maxsize=1)`
HTTP client, so there is no way to route some tools to us and others to
corpus-service. We serve all five paths or none.

## 2. Ownership

| Concern | Owner |
|---|---|
| Chunking, embedding, lexical payloads | AXIOM_DE-RD pipeline (`src/chunking_embedding/`) |
| Vector storage, ANN search | Platform (corpus-service + pgvector) |
| Sparse index, fusion, reranking | **This service** |
| Query embedding | Methods-Hub (OpenRouter direct) |

We do **not** call corpus-service's `keyword-search` or `hybrid-search`. That is
a deliberate divergence: their tokenizer is `\b\w+\b`, which leaves most
Vietnamese and Chinese query terms unmatchable, and their IDF is computed over
the filtered candidate set of the current query rather than the corpus.

## 3. Inbound — what Methods-Hub sends

| Path | Carries | What we can do |
|---|---|---|
| `/api/v1/retrieval/vector-search` | `query_embedding`, no `query` | dense only — no text means no sparse leg and no rerank |
| `/api/v1/retrieval/keyword-search` | `query`, no embedding | our BM25 |
| `/api/v1/retrieval/context` | **both** | full stack: dense + sparse + fuse + rerank |
| `/api/v1/retrieval/neighbor-chunks` | positional lookup | passthrough |
| `/api/v1/documents/ingested-data` | not a retrieval path | passthrough |

Two behaviours to design around: `None` values are dropped before send, so
optional filters arrive **absent, never null**; and non-`None` defaults
(`top_k`, `vector_weight`, `keyword_weight`, `case_sensitive`) **always** arrive,
even when the agent said nothing about them.

## 4. Outbound — what Methods-Hub requires

```python
decoded = json.loads(raw_body)           # must be valid JSON
if not isinstance(decoded, dict): raise  # must be a JSON object
```

That is the entire enforced contract. No field is read, validated, or
schema-declared; FastMCP generates no `outputSchema`. Our body is wrapped as
`{"method", "result": <our body>, "metadata"}` and handed to the LLM verbatim.

**The real constraint is agent legibility, not code.** We mirror corpus-service's
response shape because the model has been reading those field names. We add only
inside `scores` (`dense`, `bm25`, `fused`, `rerank`) plus top-level `degraded` /
`degraded_reason`. `retrieval_source` stays inside their
`Literal["keyword","vector","hybrid"]`.

## 5. Adoption is a no-op by default

`RETRIEVAL_ENHANCE` is empty by default, which makes all five paths a
byte-preserving reverse proxy. Pointing `CORPUS_SERVICE_URL` at this service is
then a verifiable behavioural no-op; enhancement is enabled per path afterwards.

## 6. What we can detect that nothing else can

Requests carry `embeddings_model`, stamped by Methods-Hub from its own
`EMBEDDING_MODEL` env var. Our artifacts record which model actually produced the
stored vectors. Nothing else in the system holds both facts.

A mismatch today returns HTTP 200 with an empty list from corpus-service and
produces no signal anywhere — the dense leg goes dark while every component
reports healthy. Here it becomes an ERROR log and `degraded=true`.

## 7. Known gap: artifact freshness

The BM25 index is built offline from pipeline output into
`src/retrieval/artifacts/<config_hash>/` (`manifest.json`, `bm25.json`,
`chunks.jsonl`). It goes stale the moment a document is indexed that it does not
contain.

corpus-service already stores what we would need — `document_embeddings.lexical`
holds `{tf, dl}` and `document_lexical_stats` holds corpus statistics, both
written by our own pipeline. Only a bulk-read endpoint is missing. Until that
exists, `check_staleness` returns `UNKNOWN` rather than `FRESH`, and `UNKNOWN`
must never be rendered as healthy.

This is also the residual risk we cannot close: the same embedding model with
different chunking produces rows indistinguishable from correct ones. See
`docs/platform_config_hash_proposal.md`.

## 8. Measured defaults

Every retrieval default in `settings.py` carries its source. Evidence lives in
`docs/incorpus_benchmark_findings.md` (555 docs, 49 scored questions).

| Setting | Value | Basis |
|---|---|---|
| `alpha` | 0.7 | measured optimum; 0.5 scores below dense alone |
| `k1_dense` / `k2_sparse` | 200 | matches the winning run's `--top-k`; clamped to 100 on the wire by corpus-service's `MAX_TOP_K` |
| `k3_rerank` | 20 | distinct documents, not chunks — measured +5 R@1 at identical cost |
| `reranker` | `none` | the gain is real but not statistically separable at n=49; enable deliberately |

The headline benchmark result (R@1 0.425 → 0.744) is **not** statistically
significant (p=0.060). Treat it as a strong candidate configuration, not a proven
improvement.
