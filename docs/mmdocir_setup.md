# MMDocIR: setup and acceptance run

MMDocIR (EMNLP 2025, [arXiv 2501.08828](https://arxiv.org/abs/2501.08828)) is the
external benchmark that gives us **layout-level gold labels with bounding boxes
and modality tags** — the annotation our lake does not have, and the one that
makes chunker evaluation possible without hand-annotating anything.

## Download

```bash
mkdir -p data/benchmark/mmdocir
huggingface-cli download MMDocIR/MMDocIR_Evaluation_Dataset \
    --repo-type dataset --local-dir data/benchmark/mmdocir
```

| File | Size | Needed for |
|---|---|---|
| `MMDocIR_annotations.jsonl` | 770 kB | questions + gold. **Enough on its own** for questions, page gold, and modality-tagged regions |
| `MMDocIR_pages.parquet` | 1.6 GB | page-level corpus (`ocr_text`, `vlm_text`) |
| `MMDocIR_layouts.parquet` | 2.5 GB | layout-level corpus; also upgrades region ids from geometric to real `layout_id` |

Total 11.2 GB. The adapter degrades gracefully: without the layouts table,
regions keep a geometric id and the question's own modality tag, so the
per-modality breakdown still works.

## Scope discipline

**Only the expert-annotated questions are exposed.** The ~173,843
bootstrapped-label questions are training-scale supervision, not an evaluation
set, and are unreachable through this adapter. Do not report on them.

## What the real annotations contain

Measured after download, not taken from the paper:

| | |
|---|---|
| Documents | 313 |
| Questions | **1,658** |
| Domains | 10 |
| Pages per document | mean 64.2, median 26, range 2–843 |
| Layouts per document | mean 543, max 5,886 |
| Gold pages per question | mean 1.27, max 6 |
| Gold layout labels per question | mean 1.59, max 12, **none missing** |
| Layout labels total | 2,638, **0 malformed** |

**The paper contradicts itself on the question count; 1,658 is correct.** The
abstract says 1,685, but the body, Table 1 and Table 3 all say 1,658 questions
with 2,107 page-level and 2,638 layout-level labels. Our parse matches the body
on all three: 1,658 questions, 2,638 layout labels, and 2,107/1,658 = 1.27 gold
pages per question, exactly the measured mean. **Nothing is missing** -- the
abstract is the outlier. 64.2 vs 65.1 pages/document is rounding, not a gap.
Cite 1,658 and note the inconsistency so nobody else chases it.

**`type` carries two disjoint labelling schemes**, because MMDocIR is assembled
from two source datasets with different annotation procedures. The paper: page
labels were annotated from scratch for all 864 DocBench questions, while the 794
MMLongBench-Doc questions were reviewed and validated (10 answers and 169 page
labels corrected). 794 + 864 = 1,658.

| Taxonomy | Source | Documents | Questions | Labels |
|---|---|---|---|---|
| `mmlongbench` | MMLongBench-Doc | 135 | 794 | stringified Python lists — `"['Figure', 'Table']"` |
| `docbench` | DocBench | 178 | 864 | bare strings — `text-only`, `multimodal-t`, `multimodal-f`, `meta-data`, `una-web` |

They never co-occur within a document. The list form is a **Python repr, not
JSON** — single-quoted, so `json.loads` rejects it and `ast.literal_eval` is the
correct reader. Normalized evidence labels come out as
`text 270 · figure 270 · table 211 · chart 166 · layout_text 102 · none 17`.

`Question.taxonomy` travels with the labels so the metric layer can split them.
**Do not average across the two** — a breakdown that mixes `text-only` with
`Pure-text (Plain-text)` looks fine and means nothing.

## The bbox join

Layout-level gold linkage is entirely this join, so it is instrumented rather
than assumed: `MMDocIR.join_stats` reports `attempted / matched / no_table`.

Annotation boxes are **absolute pixel coordinates** (verified across all 2,638 —
none normalized), matching the layouts table's space. Page sizes vary widely
(612×792, 595×842, 768×576, 880×1583), which is why matching is by
**IoU ≥ 0.5** rather than a pixel tolerance: a fixed tolerance is simultaneously
too tight for a re-detected box and too loose for the 18px-tall boxes in this
set, and IoU is scale-invariant across those page sizes. Best overlap on the page
wins; below the floor is reported as a miss rather than silently taking the
nearest box, because a wrong layout id is worse than an honest gap.

**Measured: 2,527 / 2,638 = 95.8%**, and the residual is explained rather than a
defect. Layouts come from MinerU, and the paper states that where MinerU failed
to detect an evidentiary element the boxes were drawn manually -- about 7% of
layout labels, ~185 of 2,638. Those have no MinerU row to match by construction.
Our 111 misses (58 with *no* overlapping row at all) sit inside that subset.

The best-IoU distribution confirms it is not a coordinate problem:

```
>=0.9  2,524      0.5-0.9  3      0.1-0.5  53      <0.1  58      median 1.000
```

A median of exactly 1.000 means the two sides share an origin and a render
scale. A systematic offset or DPI mismatch would shift everything together into
a band just below the floor; this is bimodal instead. Only 3 labels sit between
0.5 and 0.9, so lowering the floor would gain nothing. **Do not loosen it.**

### Binary vs graded: our layout metric is not their layout metric

The paper scores layout retrieval by *graded overlap* between retrieved and gold
boxes, because layout detectors produce differing boxes for the same content and
a binary matched/not-matched verdict discards that. Our `region_recall` is binary
(IoU >= 0.5). Both are now reported -- `region_recall@k` (ours, binary) and
`region_recall_graded@k` (mean best-IoU, comparable to theirs). **Only the graded
figure may be compared to published layout numbers**, and every table must say
which it is.

## Running

```bash
# page-level, VLM text, sparse + dense + fusion
python -m src.evaluation.run_retrieval --benchmark mmdocir \
    --arms bm25,dense,rrf --embedder openrouter_te3s --k 10

# the published OCR-vs-VLM comparison
python -m src.evaluation.run_retrieval --benchmark mmdocir --text-source ocr_text --arms bm25
python -m src.evaluation.run_retrieval --benchmark mmdocir --text-source vlm_text --arms bm25

# layout-level retrieval
python -m src.evaluation.run_retrieval --benchmark mmdocir --level layout --arms bm25,dense,rrf
```

Runs are cached on `(index_id, retriever_id, params_hash, query_set_hash)`, and
`index_id` includes the analyzer and the embedder, so a run produced under a
different tokenizer cannot be silently reused.

## Three adapter decisions worth knowing

**Retrieval is within a document — the trap anyone reproducing this will hit.**
MMDocIR's page task is "find the relevant pages inside this document", so each
question's search space is its own document's units, passed as `scope`. Run it
open-corpus and you are searching ~20,000 pages instead of a median of 26. The
numbers come out low, look like a broken retriever, and are actually a different
and much harder task. Nothing errors. Check `scope_doc_ids` in the run records
before believing a bad score.

**Page identity lives in the `doc_id`, not in our `chunk_id`.** Units are
`doc#page=7` and `doc#page=7#layout=11`. Our `chunk_id` stays offset-based;
encoding pages into chunk identity would change a pipeline contract for the
benefit of one benchmark.

**`layout_mapping` gives a bbox, not a layout id.** Linking a question to its
gold layouts is a geometric join against the layouts table (IoU ≥ 0.5, see
above), not a key lookup.

**`doc_name` does not join across the three files.** The annotations carry a
`.pdf` suffix; both parquet tables do not — 0/313 documents match raw, 313/313
match stripped. Nothing errors on the mismatch: the layout join finds no rows and
falls back to geometric ids, and page units get a `doc_id` the gold labels never
contain, so **every recall number comes out zero and reads as a broken
retriever**. `canonical_doc` normalizes on the way in.

## Acceptance test

The bar is **plausible relative to the published leaderboard**, not good. If
`bm25` / `dense` / `rrf` land wildly off in either direction, the pipeline is
miswired and we find out before building conclusions on it.

Harness validation is already done on a controlled fixture where the mechanism is
known — OCR mangles the single disambiguating token in each page:

```
text_source=vlm_text   recall@1 = 0.50
text_source=ocr_text   recall@1 = 0.25
```

That reproduces the **direction** of the paper's finding — *"text retrievers
leveraging VLM-text significantly outperforms retrievers relying on OCR-text"* —
on data where we control the cause. The 0.50 ceiling is by construction: each
`(topic, detail)` pair occurs twice per document, so `k=1` can reach at most half.

This validates that the metrics respond correctly. It is **not** evidence about
MMDocIR itself; that requires the real download.

## Coverage gap to state in the writeup

MMDocIR is document-centric — text, tables, charts, figures. It does not cover
audio, video, or 3D, which are 18 of our resolvable iSE questions. No external
benchmark in our set does. The iSE evalset remains the only evidence there, and
the writeup should say so rather than let a reader infer coverage we do not have.
