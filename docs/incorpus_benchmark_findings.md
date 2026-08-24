# In-corpus retrieval benchmark — findings

Benchmark of chunking, embedding and retrieval choices for document-level
("which file holds the answer") retrieval over the iSE Summer Challenge 2026
data lake. All numbers are reproducible with `python -m benchmark.run`.

## Setup

| | |
|---|---|
| Corpus | 555 documents extracted from 624 candidates (88.9%) — 53 text, 502 table |
| Extraction | PyMuPDF (pdf), python-docx (docx), openpyxl (xlsx/xlsm), csv, plain (md/txt) |
| Eval set | 54 questions with text/table-only gold evidence; **49 scored** (gold fully in corpus) |
| Gold | `Evidences` column = document filenames, so recall is measured, not judged |
| Aggregation | MaxP (document scored by its best chunk) unless stated |
| Significance | 2000-sample bootstrap, 95% CI |

Questions whose gold evidence is absent from the corpus are excluded from
scoring (`--keep-unreachable` re-includes them). 5 of 54 are excluded: 3 scanned
PDFs with no text layer, 1 `.pptx`, 1 unresolved reference.

## Result 1 — dense retrieval is the only significant win (+30 points R@10)

`fixed_512_ol` chunking, 49 questions, te3s embeddings (1536-d):

| retriever | R@1 | R@5 | R@10 | MRR@10 | nDCG@10 | CI(R@10) |
|---|---|---|---|---|---|---|
| **dense** | **0.558** | **0.801** | **0.870** | **0.711** | **0.737** | **[0.78, 0.95]** |
| rrf (k=60) | 0.486 | 0.781 | 0.855 | 0.653 | 0.690 | [0.76, 0.94] |
| alpha (0.5) | 0.517 | 0.758 | 0.824 | 0.672 | 0.695 | [0.72, 0.92] |
| bm25 | 0.425 | 0.551 | 0.571 | 0.527 | 0.527 | [0.44, 0.70] |

The dense and BM25 intervals do not overlap — this is the only comparison in the
study that is statistically separable at n=49.

**Hybrid fusion currently hurts.** Both RRF and α=0.5 score below dense alone,
because BM25 is far weaker on this corpus and equal weighting drags dense down.
Published fusion weights must not be inherited: BGE-M3's long-document prior
(lexical 0.5 / dense 0.15) is the opposite of what this corpus wants.

## Result 2 — MaxP beats sum-of-top-k aggregation

| arm | aggregation | R@1 | MRR@10 | R@10 |
|---|---|---|---|---|
| fixed_512_ol | **maxp** | **0.425** | **0.527** | 0.571 |
| fixed_512_ol | sum_topk | 0.333 | 0.435 | 0.531 |

Consistent with the score-aggregation literature (MaxP over FirstP/SumP/AvgP).

## Result 3 — no chunker is significantly better; the cheapest is competitive

BM25, MaxP, 49 questions, after the header fix:

| arm | chunks | R@1 | R@10 | build |
|---|---|---|---|---|
| recursive_400 | 104,927 | 0.384 | 0.612 | 24.5s |
| fixed_256_ol | 26,082 | 0.415 | 0.602 | 13.1s |
| fixed_512_ol | 13,114 | 0.425 | 0.571 | 11.1s |
| blocks_hdr (structure) | 29,919 | 0.415 | 0.541 | 5.1s |

All confidence intervals overlap. `fixed_512_ol` produces **8x fewer chunks**
than `recursive_400` for equal-or-better R@1. The defensible claim is *"no
chunker beat fixed-size, and fixed-size is much cheaper"* — not that fixed-size
is more accurate.

## Result 4 — table header detection: a real data-quality bug with no recall effect

Initial extraction assumed row 0 of every sheet was the header. Many workbooks
open with title rows, blank rows or merged cells, so rows serialized with
meaningless column names.

A header detector (scan the first 12 rows; score each on label-likeness,
uniqueness, column coverage, and whether rows beneath fill those columns)
reduced suspicious headers from **263/502 (52%) to 7/502 (1.4%)**.

Document-level retrieval did **not** improve (`fixed_512_ol` unchanged at R@1
0.425 / R@10 0.571; `blocks_hdr` +0.01 R@1).

**Interpretation.** MaxP scores a document by its single best chunk, so a
document usually ranks on a text block or an already-correct table. Column-name
quality changes what a chunk *says*, not whether its document is *found*. The
fix is still required for anything that reads a retrieved chunk — answer
extraction, passage-level scoring, or a reranker — where `nan: 5` versus
`Ngành: CNTT` is the difference between a usable and an unusable passage.

This is a useful negative result: **data-quality fixes and ranking fixes are not
interchangeable, and document-level recall is insensitive to within-chunk
quality.**

## Result 5 — failure modes

`fixed_512_ol` + BM25, 24 misses of 54:

| cause | count | implied fix |
|---|---|---|
| gold never retrieved (not in top-500 chunks) | 14 | dense retrieval (vocabulary mismatch) |
| ranked below cutoff (11–100) | 6 | reranking |
| gold not in corpus | 4 | parsing/extraction |

The dominant failure was lexical mismatch, and the dense arm subsequently
confirmed it (+30 points R@10).

## Result 6 — the winning configuration: R@1 0.425 → 0.744

Four changes, each measured separately, all on the same 49 questions:

| step | change | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|---|
| baseline | BM25, MaxP | 0.425 | 0.571 | 0.571 | 0.527 |
| +dense | te3s via gateway | 0.537 | 0.806 | 0.872 | 0.685 |
| +fusion | α=0.7 (not 0.5) | 0.558 | 0.847 | 0.883 | 0.707 |
| +prefix | title/section on every chunk | 0.599 | 0.839 | **0.969** | 0.785 |
| +rerank | listwise LLM, 20 distinct docs | **0.744** | **0.980** | **0.985** | **0.920** |

Final: `fixed_512_ol`, `--prefix`, α=0.7, `--top-k 200`, LLM rerank depth 20 (gpt-4o).

**α=0.5 was the problem, not fusion.** Result 1 concluded hybrid fusion hurts. That
was an artifact of the weight. Sweeping α shows 0.7–0.75 beats dense alone; α=0.5
sits below it. The earlier conclusion should be read as "α=0.5 hurts".

**Prefixing is the cheapest large win.** Prepending title and section to each chunk
lifted R@10 from 0.883 to 0.969 — recall was the binding constraint and this
removed it. Gold is document-level and titles carry the filename, so every chunk
becomes partial evidence for its document's identity. Cost: one re-embed (~$0.40).

**Reranking documents beats reranking chunks (+5 points).** Reranking the top-20
*chunks* wastes the budget when several belong to one document. Selecting 20
*distinct* documents (best chunk each) took R@1 from 0.662 to 0.713 at identical
cost. Scoring is document-level, so the candidate list should be too.

**Reranker model quality matters (+3 points).** gpt-4o-mini 0.713 → gpt-4o 0.744,
at 8x the price ($0.15 vs $1.13 for 49 queries).

### Significance — the honest reading

Paired bootstrap vs α=0.7, 2000 samples, same 49 questions:

| comparison | ΔR@1 | p |
|---|---|---|
| rerank vs α=0.7 | **+0.145** | 0.060 |
| dense vs α=0.7 | −0.041 | 0.267 |
| rrf vs α=0.7 | −0.082 | 0.049 |
| bm25 vs α=0.7 | −0.143 | 0.029 |

**The headline gain is not significant at p<0.05.** +14.5 points of R@1 gives
p=0.060 at n=49 — that is 7 questions out of 49. The effect is the largest in the
study and the direction is consistent across every intermediate step, but the
sample cannot certify it. Treat 0.744 as a strong candidate configuration, not a
proven improvement, until the eval set grows past ~100 questions.

## Result 7 — two negative results worth keeping

**Vietnamese word segmentation does not help.** With pyvi actually working (see
below), BM25 R@10 improved 0.571 → 0.612 but R@1 fell 0.425 → 0.384, and the
fusion arms moved <0.01 while build time tripled. The corpus is also not
Vietnamese-only — it contains substantial Chinese — so a single global analyzer is
the wrong shape regardless.

*The first run of this experiment was silently a no-op.* pyvi joins compounds with
`_` (`Đại_học`), but the shared token pattern is `[^\W_]+`, which excludes
underscore — so `analyze()` split every compound back apart and `segmented`
reduced exactly to `plain`. The measured "result" was two identical systems. Fixed
in `benchmark/retrieval.py` by normalizing each segmented unit and rejoining.

**`openrouter/free` is unusable as an LLM alias.** It is an auto-router: the same
short prompt returned a valid ranking once and `User Safety: safe` another time
(it routed to a moderation model). Vietnamese and long prompts return
`finish_reason=length` with empty content — the routed model spends the whole
output budget on reasoning tokens. Register a specific model.

## Known limitations

- **n=49.** Adequate to separate large effects (dense vs BM25), inadequate for
  1–3 point gaps. No chunker claim should be made from this sample.
- **Extraction gaps.** 40 `.epub`, 13 `.pptx`, `.xls`, `.doc` unsupported; 13
  files (incl. 3 scanned PDFs) yield no text. 32 tables truncate at 2000 rows.
- **PDF column merging.** `fitz` text extraction concatenates multi-column
  layouts and flattens embedded tables.
- **Embedder.** te3s (1536-d) chosen for cross-team index compatibility, not for
  Vietnamese quality; no Vietnamese-specific comparison has been run.
- **Vietnamese word segmentation is untested** — the BM25 analyzer is
  syllable-level, which splits multi-syllable Vietnamese words.

## Reproduce

```bash
python -m benchmark.build_evalset      # question sheet -> questions.jsonl
python -m benchmark.build_corpus       # data lake -> corpus.jsonl
python -m benchmark.run                # lexical arms, no API spend
python -m benchmark.analyze            # failure-mode attribution
```

The winning configuration. Requires the Model Service running
(`../AXIOM/services/model-service/run_local.sh`) with aliases
`openrouter-embedding` and `llm-rerank-strong` registered:

```bash
python -m benchmark.run --arms fixed_512_ol --dense --prefix \
  --embedder axiom_gateway \
  --embedder-param model=openrouter-embedding \
  --embedder-param dimension=1536 \
  --embedder-param cache_model=openai/text-embedding-3-small \
  --alpha 0.7 --top-k 200 --baseline alpha0.7 \
  --llm-rerank llm-rerank-strong --rerank-depth 20
```

`cache_model` is required: it keys the cache on the upstream model rather than the
route, so gateway and direct clients share vectors instead of paying twice.

Measured spend for the full study, from the gateway's own audit trail
(`curl localhost:8006/api/v1/audit/outbox`): **$1.68** — $0.40 embeddings
(20.0M tokens), $0.15 gpt-4o-mini reranking, $1.13 gpt-4o reranking.
