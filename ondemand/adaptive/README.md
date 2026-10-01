# On-demand with adaptive refinement — CLI guide

```
Raw data ──► Light prep ──► Light retrieval ──► SubData ──► Router ──► branch ──► Chunk&Rank ──► QA
            (pdf-inspector    (BM25 + te3-small,   (top-k,      (score/     1 light
             -> PP-OCRv5)      file_k=3)            adaptive)    cost)      2 enrich
                                                                            3 visual + KDL
```

Everything is a flag. Nothing about the pipeline is hardcoded except the defaults listed here.

## Setup

```bash
conda activate axiom-de-rd        # or use the env's python directly
git switch adaptive
python -m ondemand.adaptive artifacts
```

`artifacts` prints which caches this bundle has and which gates exist. Pull anything listed under
`missing` from the team Drive folder into `data/work/ondemand_v2/<fingerprint>/` before running.

| artifact | path under `data/work/ondemand_v2/<fp>/` | produced by |
|---|---|---|
| light prep pages | `light_prep/pages_ocr_ppocr.jsonl` | `python -m ondemand light-prep` + `merge-ocr` |
| gate | `light_prep/gate_locked_all.json` | `python -m ondemand light-retrieval` |
| ColVec scores | `colvec/colvec_scores.npy` | `colab/01_colvec_full_corpus.ipynb` |
| KDL pages | `kdl/kdl_pages.jsonl` | `colab/02_kdl_full_corpus.ipynb` |
| KDL chunk vectors | `embeddings/kdl/chunk_vectors.npy` | `python -m ondemand chunks` |
| te3-small cache | `embedding_cache/te3s_ppocr/` | written as a side effect of light retrieval |

## Build the gate

```bash
python -m ondemand light-retrieval --pages pages_ocr_ppocr.jsonl --name gate_locked_all.json
```

| flag | default | meaning |
|---|---|---|
| `--pages` | `pages_ocr_ppocr.jsonl` | light-prep output to retrieve over |
| `--signals` | `all` | `all` uses BM25 everywhere; `lang` drops BM25 where the query language differs from the page language |
| `--analyzer` | `enfr` | BM25 tokens: `enfr` (stopwords + EN/FR stemming) or `plain` |
| `--file-mode` | `dense_pool` | file selection by pooled dense score; `bm25_blend` adds a file-level BM25 index |
| `--file-k` | `3` | files kept before page ranking |
| `--k` | `20` | pages written to the gate |
| `--w-dense` | `0.70` | dense weight; BM25 gets the remainder |
| `--parent` | `0.15` | weight of the parent file score in the page score |
| `--allow-api` | off | permit embedding calls for texts not already cached |

The defaults reproduce the locked row: gate recall@20 71.34, file recall@20 91.82.

## Run the pipeline

```bash
python -m ondemand.adaptive run --tag my_experiment \
  --gate gate_locked_all.json --router fixed:visual --ranker hybrid --store kdl
```

| flag | default | meaning |
|---|---|---|
| `--tag` | required | output folder under `results/` |
| `--gate` | `gate_locked_all.json` | gate file from the step above |
| `--pages` | `pages_ocr_ppocr.jsonl` | light-prep text, used by the light branch |
| `--k` | `fixed:20` | how many SubData pages reach the router |
| `--router` | `fixed:visual` | `fixed:light`, `fixed:enrich`, `fixed:visual`, or `rule` (not implemented) |
| `--refine-k` | `fixed:10` | pages kept after the ColVec re-ranking in the visual branch |
| `--ranker` | `hybrid` | `hybrid` (chunk BM25 + te3-small, α=0.7 dense) or `page_order` (fill by page rank) |
| `--chunk` | `fixed:512:128` | chunker, `fixed:<words>:<overlap>` |
| `--top-chunks` | `10` | chunks passed to QA |
| `--store` | none | precomputed chunk vectors, e.g. `kdl`; avoids embedding calls |
| `--cache` | `te3s_ppocr` | embedding cache used when the store misses |
| `--branch-arg` | none | per-branch override, e.g. `--branch-arg enrich=cached` |
| `--limit` | none | first N queries, for smoke tests |
| `--qa` | off | run generation and judging |
| `--allow-api` | off | permit uncached embedding calls |

Both `--k` and `--refine-k` take `fixed:N` or `adaptive` (not implemented). Nothing is hardcoded to 20 or 10.

Each run writes to `data/work/ondemand_v2/<fp>/results/<tag>/`:

- `metrics.json` — every setting, per-group retrieval metrics, per-branch cost, embedding call counts
- `per_query.jsonl` — SubData, chosen branch, branch pages, the 10 chunks, costs
- `qa_summary.json`, `qa.jsonl`, `qa.csv` — only with `--qa`

## Examples

```bash
# Branch 1, cheapest: light-prep text straight to Chunk&Rank
python -m ondemand.adaptive run --tag light_only --router fixed:light --ranker page_order

# Branch 3, the locked configuration (reproduces 45.45 / 80.45 from the QA cache)
python -m ondemand.adaptive run --tag locked --router fixed:visual \
  --refine-k fixed:20 --ranker page_order --qa

# Branch 3 with the hybrid chunk ranker and a smaller refined set
python -m ondemand.adaptive run --tag hybrid_k10 --router fixed:visual \
  --refine-k fixed:10 --ranker hybrid --store kdl

# Sweep the refined page budget
for k in 5 10 15 20; do
  python -m ondemand.adaptive run --tag refine_$k --refine-k fixed:$k --ranker hybrid --store kdl
done

# What is registered
python -m ondemand.adaptive list
```

## Costs

Runs are free when every embedding resolves from `--store` or `--cache`; `metrics.json` reports
`embedding_api_calls`, `embedding_store_hits` and `embedding_cache_hits` so you can confirm.
Without `--allow-api` a cache miss stops the run instead of spending. `--qa` calls the generator and
judge; answers and verdicts are cached under `qa_cache/`, so repeating an identical configuration is free,
but any change to the chunks produces new prompts and real spend.

## Not implemented on purpose

| component | state |
|---|---|
| `--router rule` | the score/cost function is not decided; use `fixed:<branch>` |
| `--k adaptive`, `--refine-k adaptive` | no adaptive rule agreed |
| `--branch-arg enrich=chandra` | needs a Chandra endpoint and a cache under `enrich/` |

Each raises a `NotImplementedError` naming what is missing. To add one, register it in the matching
module — `router.py`, `kpolicy.py`, `branches/enrich.py` — and it appears in `list` and on the CLI
with no other changes.
