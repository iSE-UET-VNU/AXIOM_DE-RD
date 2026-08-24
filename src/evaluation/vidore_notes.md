# ViDoRe V3 adapter: the reasoning that used to live in code comments

Kept here so the modules stay terse. Full detail in `docs/vidore_v3_setup.md`.

- **Open-corpus per subset.** Unlike MMDocIR there is no within-document
  restriction; 9.5% of queries have gold in more than one document.
- **`query_id` and `corpus_id` are per-subset ints restarting near zero.** Across
  the 8 public subsets 14,514 query rows collapse to 2,184 distinct ints;
  `corpus_id` runs 0..1673 in physics and 0..1359 in computer_science.

- **We shadow the release's `doc_id`, and the name is a trap.** In
  `corpus.parquet` `doc_id` is *file*-level (42 files over 1,674 physics pages).
  `SourceDoc.doc_id` instead carries `unit_id()` = `subset::<file>#page=N`, a
  *page*. The file id survives only in `meta["doc_id"]`. Everything downstream
  inherits the shadowed sense, so `gold_doc_ids` and `context_recall` in the
  reports are **page**-level despite their names.

  The scored unit must be the page because qrels map `query_id -> corpus_id` and
  `corpus_id` is one row per page — upstream keys them the same way. The reason
  the key is `file#page` rather than `subset::corpus_id`, which would also be
  unique, is that **chandra2 and KDL cannot produce a `corpus_id`**: a parser
  reads a PDF and knows `(file_name, page)` only. `file#page` is the one key both
  the release's own text and our parses can generate, which is what makes any
  parser-vs-ViDoRe comparison possible.
- **Each language variant is its own query**, and one qrel table holds all six.
  Selecting a language means filtering queries then dropping orphaned qrels.
- **Relevance is graded 1/2**, passed to `pytrec_eval` untranslated.
- **Language has no default.** BM25S scores 53.3 on English and 19.1 across six
  languages; the gap is cross-lingual mismatch, not retrieval quality.
- **`image` is never projected** — ~12 GB of page renders no text arm reads.
- **Only 8 public subsets exist.** The paper's 3,099 queries / 26,000 pages
  counts two private hold-outs.
- **Regions are a localization target, not a retrievable unit.** Deferred: the
  paper's baselines are a VLM emitting boxes inline, a different task.
- **84 released boxes are degenerate** and are dropped rather than kept at zero
  area.
- **96 qrels carry `content_type = "N/A (If relevance score=0)"`** while scoring
  1 or 2; a leaked form option, not a modality.

## Judge

- Prompts are Figures 24 and 25, verbatim except Jinja `{{ x }}` → `{x}` and the
  paper's `{{ true_answer} }` typo.
- The rubric is three-way; Appendix H reports a binary metric without saying how
  Partial folds in. The raw label is stored and both aggregations come from one
  judge pass.
- The paper's prompt has no abstain option, so a fourth bucket would make our
  denominator differ from theirs.

## Loader mirror

`illuin-tech/vidore-benchmark` at `a70f23af`,
`src/vidore_benchmark/pipeline_evaluation/dataset_loader.py` lines 66-111.
Not called directly: line 88 materialises every page render, line 117 raises if
that list is empty, and importing the package pins `sentence-transformers<4.0.0`
against our 5.6.0. Mirrored exactly: the `str()` casts on both id columns, the
filter-then-drop order, and the `{query_id: {corpus_id: score}}` shape.
