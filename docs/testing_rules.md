# Two testing rules, and why they exist

Five bugs in this project reached a measured number before anyone noticed. None
raised an exception. None was caught by a unit test. Every one would have
survived into a paper.

| # | Bug | Caught by |
|---|---|---|
| 1 | CJK analyzer chosen per text, so index and query tokenized differently | index-then-query round trip |
| 2 | NFD queries lose 54.2% of terms against an NFC index | byte-level tokenization comparison |
| 3 | `doc_name` carries `.pdf` in annotations, not in the parquets | join rate against a published label count |
| 4 | `Region.doc_id` was a layout id, so page-level region recall read 0 | known-true page assertion |
| 5 | BM25 filtered scope *after* a global top-N, under-retrieving | old-path vs new-path equivalence |
| 6 | `embedding-default` resolves to a mock provider returning 8-dim vectors | alias resolved against the registry before any vector is written |

---

## Rule 1 — For any new seam, write the spanning assertion first

Every one of the five lived at a **boundary between two things that were each
internally consistent**: two analyzers, two annotation files, two document-name
conventions, two implementations of one search, two metric granularities.

Unit tests scope to one side of a boundary *by construction*. That is a property
of the technique, not a lapse in discipline, and it explains the whole pattern:
our tests passed because we normalized on both sides; corpus-service's passed
because they normalized on neither. The bug lived in the composition, which
neither suite could see.

A spanning assertion states a fact that is only true if both sides agree:

- index a document, query a term inside it, get that document back
- join two tables, assert the match rate against an independently published count
- write a record with one module, read it with the other, assert equality
- run both implementations of an operation, assert identical output

**Write it before the code, not after the bug.**

## Rule 2 — Any regression test must be shown to fail against the bug it guards

A regression test that has never been observed to fail is an assumption wearing
a test's clothing. Worse than no test: it converts an unchecked assumption into a
documented guarantee, and everything downstream then rests on it.

This is not hypothetical. The first version of `test_retrieval_scope_audit.py`
passed on **both** the buggy and the fixed path — a uniform fixture put the
scoped document inside the over-fetch window by tie-breaking, so the assertion
held either way. It was reported as discriminating. It was not.

The fix was to make the fixture adversarial: `doc0` mentions the query terms
once while 119 other documents repeat them five times, so all 595 competing units
outrank it and no global top-200 window can reach it. Then:

```
OLD (post-filter)        returned 0/5  -> FAIL
NEW (scope in scoring)   returned 5/5  -> PASS
```

If the buggy version is already gone, reintroduce it temporarily and watch the
test go red. Delete it once you have seen the failure.

## Rule 3 — A fixture must exhibit the shape the real data has

Red-verification proves a test *can* fail. It does not prove it fails for the
right reason, on the right shape. Four tests in this project have now passed
for a reason other than the one stated:

| Test | Passed because | Real shape |
|---|---|---|
| `test_retrieval_scope_audit.py` | uniform fixture put the scoped doc inside the over-fetch window | 595 competing units outrank it |
| ViDoRe namespacing | claimed unit ids could collide | `doc_id` is globally unique, so only `query_id` can |
| `test_pipeline_unit_reconstruction.py` | fixture `file_name` was `<name>.pdf` | real output is `pdfs/<name>.pdf` |
| `test_benchmark_registry.py` | asserted a `gold` method on the Benchmark protocol | it is `gold_docs` / `gold_pages` / `gold_regions` |

The fourth is the first where the assumed shape was in **our own code** rather
than external data, and it is the easier mistake to make: an interface you wrote
feels known, so it gets recalled instead of read. It failed on all three adapters
until `base.py` was opened. Read the interface, do not remember it.

The third is the clearest. `canonical_doc = file_name[:-4]` matched the fixture
and matched **0/42** real documents; basename-then-strip matches 42/42. The test
was green, red-verifiable, and wrong.

**Check the fixture against the real data before trusting the test.** Where the
data cannot be held as a fixture, verify the assertion against it once and record
the count -- 42/42 documents, 1,674/1,674 units, 2,178/2,178 judgements. A
verified count is evidence; a green test on an invented shape is not.

The same standard applies to config parity. `table_refinement.enabled` recorded
false on 42/42 documents with 0 attempted is verifiable; reading two YAML files
and concluding they agree is not.

---

## Seams in this repo

| Seam | Spanning assertion | Where |
|---|---|---|
| index-time vs query-time analyzer | round trip, three scripts | `test_retrieval_symmetry.py` |
| `chunk_ids.json` order vs `vectors.npy` rows | embed a known text, assert nearest neighbour is itself | `test_vector_alignment.py` |
| `runs.py` writer vs `run_answer.py` reader | write then read, adversarial ordering | `test_retrieval_protocol.py` |
| `Benchmark` protocol vs each adapter | conformance + gold semantics per adapter | `test_benchmark_adapter.py`, `test_mmdocir_adapter.py` |
| scope filtering vs scoring, per arm | full-k under a low-ranking scope | `test_retrieval_scope_audit.py` |
| MMDocIR annotations vs parquet tables | join rate vs the paper's label count | `MMDocIR.join_stats` |
| `reading_order` vs `main_text` order | blocks earlier in reading_order start earlier in main_text | `test_reading_order_seam.py` |
| ViDoRe qrels `corpus_id` vs corpus `corpus_id` | join rate against the measured 74,016 judgements, 0 unreachable | `ViDoreV3.join_stats`, `test_vidore_v3_adapter.py` |
| ViDoRe language filter vs qrel drop | English `hr` yields 318 queries / 1,731 qrels, not 1,908 / 10,386 | `test_vidore_v3_adapter.py` |
| ViDoRe per-subset id namespacing | two subsets loaded together keep their separate `query_id` 0 and `corpus_id` 10 | `test_vidore_v3_adapter.py` |
| judge three-way label vs the two aggregations | a fixture containing Partial produces two *different* scores | `test_vidore_v3_generation.py` |
| pinned prompts vs the paper's Figures 24/25 | constants byte-identical to the published text | `test_vidore_v3_generation.py` |
| `Benchmark` protocol vs `ViDoreV3` adapter | conformance + graded gold semantics | `test_vidore_v3_adapter.py` |
| benchmark arm vs the provider its alias reaches | no arm resolves to a mock adapter; resolution recorded in the manifest | `test_model_guard.py` |

When adding a seam, record it here **before** writing the code, and mark it
unspanned if the assertion does not exist yet -- an absent row and a passing row
must never look the same.

A note on the two ViDoRe id seams: they are listed separately because they fail
in opposite directions. A broken `corpus_id` join makes every score read zero,
which is loud. A missing language filter silently *inflates* the gold set 6x --
each qrel file carries all six language variants' judgements -- so recall rises
and nothing errors. The published `vidore_v3_*_mteb_format` configs make this
trap the default: they ship all 10,386 `hr` qrels alongside only 318 queries of
one language, so a naive join reports 32.7 gold pages per query where the correct
answer is 5.44.

A note on the reading-order seam specifically: a missing `reading_order` is
reported as `source: "unavailable"` and `complete: False` rather than skipped.
A seam left unspanned because the data is absent is otherwise indistinguishable
from one that passes, which defeats the point of keeping an inventory.


## Bug #7 and its mirror — omission versus duplication in the identity key

The same design question produces opposite errors, and both report a plausible
number rather than an error.

| | Error | What it reports |
|---|---|---|
| **Omission** (bug #7) | corpus missing from `index_id` | two parsers share a run cache, so a real difference reads as **exactly zero** |
| **Duplication** | chunker folded into `corpus_identity` when it is already its own key | every cached run invalidates, so a no-op refactor reads as a **full re-embed** |

Omission is the dangerous one because zero is believable. Duplication is merely
expensive, and it announces itself as a cache miss.

The rule that separates them: **a key belongs to exactly one component of the
identity.** `corpus_identity` names the corpus -- its subset and the parser
config that produced it. It does not name the chunker, the embedder, the
retriever or the depth, because `index_identity` already does.

Asserted by `test_corpus_source.py::test_chunking_does_not_change_corpus_identity`
and `test_identity_keys.py`, which enumerates the eight ladder arms and requires
eight distinct **cache paths**.

### The parser axis

`corpus_identity` on a pipeline run hashes the settings recorded in the run, not
the parser name. chandra2 and kdl carry differently-shaped config -- KDL has
`bbox_max_workers` and per-element token budgets -- so a name-keyed identity
would collide whenever configs differ and report the parsers as identical.

Two collisions were found by checking against the three real parsed runs after
the unit tests were green:

1. Parser settings live in the **ingested** stage; output-stage documents drop
   the `parsed` block. All three runs hashed to the SHA of `{}`.
2. Identity omitted the subset, so chandra2 over physics and over
   pharmaceuticals shared one token.

Both are Rule 3: the fixture had a shape the real data does not have.
