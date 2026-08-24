# Proposal: record the indexing config that produced each embedding

**To:** Platform team (corpus-service, indexing-streaming)
**From:** Data Engineering (chunking / embedding / retrieval)
**Status:** proposal, no code written

---

## The one-sentence problem

`document_embeddings` records *which model* made each vector, but not *how the text
was cut up before it was embedded* — so two incompatible indexes look identical in
the database, and a search across both silently returns wrong results.

---

## 1. What a vector actually depends on

An embedding is a function of two things:

```
vector = embed( model , text_chunk )
```

The table records the first and not the second.

`text_chunk` is not raw file content. It is the output of a chunking config:

| knob | example A | example B |
|---|---|---|
| chunk size | 512 words | 256 words |
| overlap | 128 words | 64 words |
| title prefix | on | off |

A and B use the **same embedding model** and produce **completely different
vectors** from the same document, because they embed different strings.

### A worked example

Take one file, `Album.csv`, containing rows like
`AlbumId: 1; Title: For Those About To Rock; ArtistId: 1`.

**Config A** — 512 words, title prefix on:

```
chunk 1 = "Album
           AlbumId: 1; Title: For Those About To Rock; ArtistId: 1
           AlbumId: 2; Title: Balls to the Wall; ArtistId: 2
           ... 40 more rows ..."
```

**Config B** — 256 words, no prefix:

```
chunk 1 = "AlbumId: 1; Title: For Those About To Rock; ArtistId: 1
           AlbumId: 2; Title: Balls to the Wall; ArtistId: 2
           ... 20 more rows ..."
```

Both get embedded by `openai/text-embedding-3-small`. Both land in
`document_embeddings` with:

```
embeddings_model = "openai/text-embedding-3-small"
```

**The rows are indistinguishable.** Nothing in the table says one is 512-word
prefixed chunks and the other is 256-word bare chunks.

---

## 2. Why this is a correctness bug, not an inconvenience

Vector search ranks by cosine similarity. Mixing two chunkings in one collection
means comparing distances that were never meant to be compared.

Concretely, on our corpus the title prefix alone changes retrieval quality by a
large margin — because a chunk of bare spreadsheet values like

```
24.0 | x | x | 23.0 | 21.0 | 19.0 | 17.0 | N | N | 22.0
```

has no retrievable signal at all, whereas the same chunk with its title

```
FIT 30 Nam CNTT | So do
24.0 | x | x | 23.0 | 21.0 | 19.0 | 17.0 | N | N | 22.0
```

does. Those two vectors point in meaningfully different directions.

**The failure is silent.** No error, no warning. A query returns ten results, they
look plausible, and some fraction of them are ranked against the wrong baseline.
Nobody finds out until someone manually audits an answer.

### The scenario we expect to hit

1. We index the corpus with config A.
2. We improve chunking and re-index with config B.
3. The re-index doesn't cover every document — a Spark job fails, a file is
   skipped, a run is partial.
4. The collection now holds A-rows and B-rows side by side.
5. Every subsequent search silently mixes them.

Step 3 is not hypothetical. It is the normal state of any incremental pipeline.

---

## 3. What exists today

`document_embeddings` (`services/corpus-service/.../document_embedding_model.py`):

| column | records |
|---|---|
| `embedding_id` | `short_hash(run_id, item_id, item_type)` |
| `document_id` | which file |
| `run_id` | which pipeline run |
| `type` | chunk type |
| `position` | where in the document |
| `content` | the chunk text and metadata (JSONB) |
| `embedding` | the vector |
| `embeddings_model` | **which model** |

`embeddings_model` is genuinely useful and we already filter on it. It rules out
*wrong model*. It cannot rule out *wrong chunking*, and chunking is the half that
changes most often, because it's the half we tune.

`run_id` looks like it might help, but it identifies a *run*, not a *config* — two
runs with the same config get different `run_id`s, and one run could in principle
use a different config than the last.

---

## 4. What we propose

We already compute a stable hash of the full indexing config on our side.
`ChunkEmbedConfig.config_hash()` hashes:

```
chunker, chunker_params, max_rows_per_chunk,
embedder, embedder_params, retrieval_profile, llm
```

Example value:

```
a3f32c6472456381f9a427336860f0d994431d551dcb430ef17389b22d6d7288
```

Change the chunk size from 512 to 256 and the hash changes. Turn the title prefix
off and the hash changes. Same config, different day, different machine → **same
hash**.

We would like that value stored next to the vector.

### Option 1 — new column *(preferred)*

Add one nullable column to `document_embeddings`:

```python
config_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
```

plus an index, since it becomes a filter:

```python
Index("ix_document_embeddings_config_hash", "config_hash")
```

and one optional request field on `vector-search`:

```python
class VectorSearchRequestDto(BaseModel):
    query_embedding: list[float]
    embeddings_model: str | None = None
    config_hash: str | None = None      # <-- new, filters like embeddings_model
```

**Nullable, and the filter is optional, so nothing breaks.** Existing rows keep
`config_hash = NULL`; existing callers that don't send the field see today's
behaviour exactly.

The value would come from us. Each retrieval item we return from `/v1/dataeng`
would carry it alongside the model:

```json
{
  "item_id": "...",
  "type": "text_chunk",
  "embeddings": [{
    "model": "openai/text-embedding-3-small",
    "dimension": 1536,
    "values": [...],
    "config_hash": "a3f32c64...2d6d7288"
  }]
}
```

and `persist_embeddings` in the Spark job would read it the same way it already
reads `model` and `values`:

```python
first  = embeddings[0] if embeddings else {}
values = first.get("values")
model  = first.get("model")
config_hash = first.get("config_hash")     # <-- new
```

**Total change: one column, one index, one optional request field, one line in the
Spark job.**

### Option 2 — no schema change *(fallback)*

We already control the `content` JSONB, which the Spark job writes verbatim from
our payload. We can put the hash there ourselves today:

```json
"content": {"text": "...", "config_hash": "a3f32c64..."}
```

That records provenance with **zero changes on your side** — you could ship
nothing and we'd still be able to audit after the fact.

But it doesn't solve the actual problem, because we can't *filter* on it: a JSONB
predicate on `content->>'config_hash'` would have to be exposed by
`vector-search`, and it can't use the HNSW index efficiently. Option 2 gives us
forensics; Option 1 gives us prevention.

We will implement Option 2 regardless, as a stopgap.

---

## 5. What we are not asking for

- No change to the embedding pipeline or the Spark job's structure
- No backfill of existing rows — nullable is fine, and `NULL` correctly means
  "unknown provenance"
- No new endpoint
- No change to `hybrid-search` or `keyword-search`

---

## 6. Why we're asking rather than working around it

We considered three workarounds and none of them close the gap:

**Filter by `run_id`.** Requires us to track which runs used which config, in a
place the database doesn't know about. Any consumer that doesn't know our
bookkeeping gets wrong results.

**Filter by `embeddings_model`.** What we do today. Catches a wrong model, misses
a wrong chunking entirely.

**Maintain our own parallel index.** Duplicates your storage and guarantees the
two drift.

The database is the only place that can enforce this, because it's the only place
that sees every row regardless of who wrote it.

---

## 7. Impact if we do nothing

We can ship without it. The risk we accept is that a partial re-index produces a
mixed collection, retrieval quality degrades by an amount nobody can measure, and
the failure is invisible until someone audits answers by hand.

Given that we expect to tune chunking repeatedly — it's the parameter we
experiment with most — we think that's a bad risk to carry, and a cheap one to
remove.

---

## Summary

| | |
|---|---|
| **Problem** | A vector depends on the model *and* the chunking; only the model is recorded |
| **Consequence** | Two incompatible indexes are indistinguishable; mixing them silently degrades results |
| **Ask** | One nullable `config_hash` column + one optional filter on `vector-search` |
| **Cost** | ~1 column, 1 index, 1 request field, 1 line in the Spark job |
| **Breaking** | Nothing — nullable column, optional filter |
| **We provide** | The hash, in the retrieval payload we already send |
