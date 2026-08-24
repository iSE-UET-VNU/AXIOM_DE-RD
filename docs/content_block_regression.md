# `content` block removed from output documents — regression and fix

**Audience:** AXIOM_DE-RD team (internal). No action required from corpus-service,
indexing-streaming, or Methods-Hub — this restores prior behaviour rather than
changing it.

**Introduced:** `9433190 refactor code` (2026-08-03), on `develop`.
**Fixed:** `src/artifacts/pipeline_output.py`, this change.
**Impact while present:** every document indexed by an affected build has zero
rows in `document_contents`.

---

## What changed

The refactor replaced the output document's `content` block with three
stage-oriented blocks:

```diff
  {
    "document":  {...},
-   "content":   {main_text, tables, figures, formulas, blocks, reading_order, ...},
+   "ingest":    {"data": ParsedData, "source": DataObject},
+   "clean":     {"data": CleanedData},
+   "enrich":    {"data": EnrichedData},
    "retrieval": {items, lexical_stats}
  }
```

The stage blocks are a genuine improvement — each stage keeps its native
contract and the boundaries are explicit. The problem is only that `content`
went away with them.

## Why that breaks storage silently

`services/indexing-streaming/jobs/process_s3_events_lib/database.py`:

```python
def persist_contents(statement, document_id, run_id, content_payload: dict) -> int:
    count = 0
    for content_type, content_value in content_payload.items():
        statement.setString(1, build_content_id(run_id, content_type))
        ...
        count += 1
    return count
```

called as:

```python
persist_contents(..., result.get("content") or {}, ...)
```

With `content` absent this iterates an empty dict. No exception, no warning —
`count` is 0 and the transaction commits successfully. **The failure is
invisible on both sides.**

The Spark job does not read `ingest`, `clean`, or `enrich` at all, so the
replacement blocks add response size without reaching storage.

## Downstream consequences

1. **`document_contents` is empty** for affected documents.
2. **`corpus_get_file_ingested_data` returns empty `contents`** — the MCP tool an
   agent uses to read a document.
3. **Keyword search loses its fallback.** corpus-service's
   `_keyword_search_contents` scans `document_contents` as plain text whenever no
   row in the filtered set has a usable `lexical` payload. With the table empty
   there is nothing to fall back to, so a document whose `lexical` is missing or
   malformed now returns nothing at all instead of degraded results.

Point 3 is the one worth internalising: `document_contents` is not decorative.
It is the safety net under BM25.

## The fix

Restore `content` **alongside** the new stage blocks rather than reverting them:

```python
"document": _output_document(data_object, document, enriched),
"content":  _output_content(document, enriched, parsed),   # ← restored
"ingest":   _stage_data(parsed, source=data_object),
"clean":    _stage_data(cleaned),
"enrich":   _stage_data(enriched),
"retrieval": _compact_retrieval(...),
```

`_output_content` is restored verbatim from the pre-refactor implementation,
along with the `reading_order_from_rows` import the refactor dropped. Every
helper it calls (`_strip_parser_audit`, `_extraction_source_refs`) survived the
refactor untouched.

## Verification

A full local pipeline run (`provider: deferred`, `embedder: local_hash`),
then replaying the Spark persistence logic against the produced document:

```
top-level keys: ['document', 'content', 'ingest', 'clean', 'enrich', 'retrieval']

persist_contents      → 8 rows   (was 0)
    main_text, tables, figures, formulas,
    blocks, reading_order, reading_order_meta, source_refs
persist_embeddings    → 1 row
persist_lexical_stats → 1 row
```

## Open questions for the team

1. **Are the stage blocks meant to reach storage?** If `ingest`/`clean`/`enrich`
   are for debugging, they cost response size on every document and could be
   gated. If they are meant to be persisted, indexing-streaming needs to learn
   about them — that is a cross-team change.

2. **Should `contract_version` come back?** The refactor also stopped emitting it
   (`src/api/output.py` now strips it and `lineage` recursively from the public
   response). Consumers currently have no way to detect a shape change — which is
   precisely how this regression stayed invisible.

3. **Do already-indexed documents need re-indexing?** Anything processed by a
   build between `9433190` and this fix has an empty `document_contents`. Whether
   that matters depends on whether those runs reached a real corpus or only local
   test data.

## How to not repeat this

The output document is a **write instruction**, not a view: every top-level key
names a destination table, and a missing key is silently zero rows rather than an
error. A contract test that asserts the key set — and replays `persist_*` against
a produced document, as the verification above does — would have caught this at
the point of change.

Note also that `tests/` is currently in `.gitignore`, so the suite that exists
locally provides no signal to anyone else on the team.
