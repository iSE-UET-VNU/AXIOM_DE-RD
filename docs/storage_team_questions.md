# Questions for the corpus-service (storage) team

Verified against `origin/main` @ `3327c30` on 2026-08-02, and against a running
corpus-service at `localhost:38002`.

## 0. What we actually depend on

Our retrieval service never touches the vector database. It speaks HTTP to
corpus-service and nothing else:

| What we call | Why |
|---|---|
| `POST /api/v1/retrieval/vector-search` | The dense leg. We send `query_embedding`, never text. |
| `POST /api/v1/documents/ingested-data` | Build time: recover each chunk's stored `embedding_id`. |
| `GET /api/v1/health` | Degradation check. |

We never call `keyword-search` or `hybrid-search` — we send `vector_weight: 1.0`
so their fusion is disabled and ours is the only one that runs. We never write.

`scripts/load_corpus_service.py` does write directly to Postgres, but it is a
development stand-in for the Spark job in `services/indexing-streaming` and is
not part of any production path.

**So: endpoints are sufficient. No database access is requested.** The questions
below are about guarantees and gaps in those endpoints.

---

## Q1. Does `ingested-data` scope chunks to a single run? (highest impact)

`FileIngestedDataResponseDto` returns `document.latest_run_id` alongside a
`chunks[]` array. We used `latest_run_id` to re-derive each chunk's
`embedding_id` with the same `short_hash(run_id, item_id, item_type)` the
indexing job uses. On a freshly indexed document this agrees exactly — 10 of 10
ids matched in our test.

But in `corpus_retrieval_repository.py` the embeddings are selected by document
only:

```python
embeddings = session.execute(
    select(DocumentEmbeddingModel)
    .where(DocumentEmbeddingModel.document_id == document.document_id)
).scalars().all()
```

There is no `run_id` predicate, while `processing_run` immediately above *is*
filtered to `latest_run_id`.

**Why it matters.** Re-index a document and `document_embeddings` holds rows from
both runs. `chunks[]` then mixes them, but `latest_run_id` names only the newer
one. Ids we derive for the older rows are wrong, and the ones we derive for the
newer rows collide with text that belongs to the older ones. Nothing raises.

**Example.** Document `D` indexed at run `r1` (10 chunks), edited, re-indexed at
run `r2` (12 chunks):

```
ingested-data returns : 22 chunks, latest_run_id = r2
we derive ids for     : 22 chunks, all using r2
correct for           : 12 of them
```

**What we would like to know**
1. Is `document_embeddings` pruned when a document is re-indexed? If yes, this is
   a non-issue and we would like to record that guarantee.
2. If not, could `ingested-data` accept an optional `run_id` filter, or scope to
   `latest_run_id` by default?

**Our mitigation either way:** stop recomputing and read `embedding_id` straight
from the `chunks[]` response. That is correct regardless of the answer, and it
deletes our copy of their hash function (`src/retrieval/ids.py`). We plan to do
this; the question is whether the endpoint's run semantics still need fixing for
other consumers.

---

## Q2. Can `top_k` exceed 100 for server-side callers?

```python
# corpus_service/dto/retrieval_dto.py
MAX_TOP_K = 100
top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)
```

Our measured-best dense depth is **200**. We clamp to 100 and log it
(`src/retrieval/dense.py`), so the dense candidate pool is bounded by the API
rather than by our tuning. Reranking then has fewer real candidates to work with.

**Ask:** is 100 a deliberate protection for browser/agent callers, or an
incidental default? If the former, could an internal caller opt into a higher
ceiling? We are not asking for it to be removed globally.

---

## Q3. Nothing records *how* a vector was produced

`document_embeddings` carries `embeddings_model` (indexed) and no other
provenance. Searching `origin/main` for `config_hash` in corpus-service returns
nothing.

Two vector sets built from the same model but different chunking — say
`fixed_overlap` 512/128 versus `recursive` 400 — are indistinguishable once
stored. A corpus indexed twice with different settings silently mixes chunk
granularities in a single result set.

This is the subject of `docs/platform_config_hash_proposal.md`; that document
still stands. The minimal version is one nullable `config_hash` column plus an
optional equality filter on `vector-search`.

---

## Q4. A wrong `embeddings_model` returns zero rows with no signal

`vector-search` accepts `embeddings_model` and applies it as an equality filter:

```python
if embeddings_model:
    ... DocumentEmbeddingModel.embeddings_model == embeddings_model
```

If the value does not match what was stored, the response is **HTTP 200 with an
empty result set** — identical to "this corpus genuinely has no similar chunks".

We hit this in testing. Our vectors were stored under the gateway alias
`openrouter-embedding`, while our service defaulted to the upstream id
`openai/text-embedding-3-small`:

```
stored in Postgres : openrouter-embedding
filter we sent     : openai/text-embedding-3-small
rows returned      : 0        (HTTP 200, no warning)
```

Retrieval still answered, on BM25 alone. Top fused score was capped at 0.300 —
the sparse-only ceiling at alpha 0.7 — where a working dense leg gives 1.000.
Nothing in the response said the dense half had gone dark. We only noticed
because we knew what the number should be.

**Ask:** could the response distinguish "filter matched no rows" from "no
neighbours found"? A `filtered_out: true` flag, or a `models_present[]` echo,
would turn a silent wrong answer into a fixable error. We would also take a
`GET` that lists distinct `embeddings_model` values present for a document.

---

## Summary

| # | Question | Blocking? | Our workaround |
|---|---|---|---|
| Q1 | Is `ingested-data` run-scoped? | No | Read `embedding_id` from the response instead of deriving it |
| Q2 | `top_k` above 100 for internal callers? | No | Clamp to 100, log it |
| Q3 | Record chunking provenance | No | One corpus per config, enforced by convention |
| Q4 | Signal an empty-because-filtered result | No | Set `RETRIEVAL_EMBED_UPSTREAM` to the stored alias, assert at startup |

None of these block us. Q1 and Q4 are the two where a silent wrong answer is
possible today, so they are the two we would most like a definite answer on.
