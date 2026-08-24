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

## Configs

Configs live in `configs/` as `.yaml`. Their **contents are JSON**, and the
notebooks read them with `json.loads`, so a `#` comment breaks a run even though
the extension suggests otherwise. `load_config` accepts either.

| Config | Parser | Chunker | Embedder | Purpose |
|---|---|---|---|---|
| `pipeline.yaml` | chandra2, queue | `fixed_overlap` 512/128 | `openrouter_te3s` | the default local run |
| `pipeline.chandra2.yaml` | chandra2, no queue | `recursive` 400 | `openrouter_te3s` | older single-shot parse |
| `pipeline.chandra2.free.yaml` | chandra2, no queue | `recursive` 400 | `local_hash` | no paid API; runs `indexing_cataloging`, not `chunking_embedding`; **`local_hash` vectors are not real embeddings** |
| `pipeline.vidore-v3-chandra2.yaml` | chandra2, queue | `fixed_overlap` 512/128 | `openrouter_te3s` | ViDoRe benchmark parse, Colab |
| `pipeline.vidore-v3-kdl.yaml` | kdl, queue | `fixed_overlap` 512/128 | `openrouter_te3s` | ViDoRe benchmark parse, Colab |
| `pipeline.vidore-v3-ingest.yaml` | chandra2, queue | -- | -- | ingestion only, `enabled_modules` stops at artifacts |
| `pipeline.vidore-v3-kdl-ingest.yaml` | kdl, queue | **none** | **none** | enables `chunking_embedding` but configures neither -- see below |

### The blocks that matter

`parsing.provider` selects `chandra2` or `kdl`, and the sibling block named for
it carries that parser's settings. The two blocks are **differently shaped** --
KDL adds `bbox_max_workers` and per-element token budgets, chandra2 has
`request_batch_size` and `table_refinement`. Corpus identity hashes the whole
block, so a settings change is a different corpus.

    "parsing": {"provider": "chandra2", "chandra2": {
      "method": "vllm", "continuous_page_queue": true,
      "max_workers": 48, "render_processes": 32, "request_batch_size": 1,
      "max_output_tokens": 4096, "table_refinement": {"enabled": false}}}

`max_workers` and `max_output_tokens` in `pipeline.yaml` are **machine-tuned
overrides** -- 48 / 4096 against upstream's 256 / 12384, which are sized for the
notebook's H100/A100 branch. Do not let a merge restore upstream's numbers.

`local_input.include_extensions` on every `vidore-v3-*` config is `[".pdf"]`.
Pointing one at the lake's pptx or audio silently indexes nothing.

`enabled_modules` decides how far the run goes. `["ingestion", "artifacts"]`
parses only; the full list through `chunking_embedding`, `integration`,
`artifacts` also chunks and embeds.

Two mismatches worth knowing before you copy a config:
`pipeline.vidore-v3-kdl-ingest.yaml` **enables `chunking_embedding` but ships no
`chunking_embedding` block**, unlike its chandra2 counterpart which stops at
`["ingestion", "artifacts"]`. And `pipeline.chandra2.free.yaml` enables
`indexing_cataloging` where every other config enables `chunking_embedding`.
Neither is something this repo's evaluation code depends on, but both mean the
`-ingest` pair are not symmetric.

---

## Running

    PY=/usr/local/Caskroom/miniconda/base/envs/axiom-de-rd/bin/python

### 1. Parse documents into a run directory

    $PY scripts/run_pipeline.py --config configs/pipeline.vidore-v3-chandra2.yaml \
        --local-raw <dir-of-pdfs>

Writes `data/{ingested,cleaned,enriched,embedded,output}/benchmarks/<run>/<run_id>/`.
The GPU parsers need vLLM on localhost, which is why the Colab notebooks exist --
`Chandra_serving_de.ipynb` and `KDL_serving_de.ipynb` patch the config at runtime
and shell out to this same script. Only artifacts travel back.

**Parser settings live in `ingested/`.** Later stages drop the `parsed` block, so
that is where `parser`, `status` and `table_refinement` are readable.

### 2. Retrieval over a benchmark

    $PY -m src.evaluation.run_retrieval \
        --benchmark vidore_v3 --subset physics --language french \
        --arms bm25,dense,rrf,alpha0.7 --embedder openrouter_te3s --k 10

`--language` is required for ViDoRe and has no default: a multilingual average
is not a number anyone wants to report by accident.

| Flag | Effect |
|---|---|
| `--arms` | `bm25`, `dense`, `rrf`, `alpha<w>` -- dense and fusion need `--embedder` |
| `--chunker` + `--chunk-param k=v` | re-chunk units before indexing |
| `--prefix` | prepend the document title to each chunk |
| `--rerank llm --rerank-model <alias> --rerank-depth N` | LLM rerank |
| `--corpus PATH` | corpus file for adapters that read one (iSE) |
| `--depth` | chunks written per question, default 100 |

Runs cache on `(index_id, retriever_id, params_hash, query_set_hash)`. Two
configs never share a cache -- the corpus, chunker, embedder, prefix, rerank
model and depth are all in the identity.

### 3. Retrieval over a parsed run

`PipelineRunCorpus` reads a run directory as a corpus. **No CLI flag yet** -- use
it from a script:

    from src.evaluation.corpus_source import PipelineRunCorpus

    source = PipelineRunCorpus(
        "data/output/benchmarks/vidore-v3-pharmaceuticals-chandra2/<run_id>",
        subset="pharmaceuticals",   # namespaces unit ids and enters the identity
        granularity="page",         # or "chunk"
    )

`page` groups blocks by `blocks[].page` in `reading_order` -- the only unit that
compares fairly against ViDoRe's page corpus. `chunk` uses the pipeline's own
`retrieval.items` and **rejects `--chunker`**, because those chunks were fixed at
parse time and silently ignoring the flag would report a lie.

`research/experiments/physics_ladder.py` is the working template.

### 4. End-to-end generation and judging

    $PY research/experiments/physics_e2e.py

Four arms: gold-context oracle per parser, plus retrieved context per parser.
Generator and judge are required and must differ. Checkpointed per arm -- delete
an arm's JSON to redo just that one.

### 5. Compare arms

    $PY -m src.evaluation.compare_arms data/benchmark/runs/*.report.json

Refuses ragged arm sets, and reports coverage separately from accuracy. A parser
that reaches more documents is scored on a different question set, so only
`acc|common` may carry a cross-arm claim.

### 6. Check a result did not move

    $PY research/experiments/gate_check.py <frozen baseline.json>

SHA-256 over the sorted per-question NDCG map. Exact, not tolerance-based -- an
aggregate can match while individual questions move.

---

## Before trusting a new parser output

Run `research/experiments/physics_format_report.py` first. It checks the
document-name join, page coverage, block-type inventory, unmapped types, parse
status and config parity. A coverage gap otherwise reads as a quality
difference: the KDL pharmaceuticals run is missing 452 pages from one quarantined
document, and 23 of 364 queries have no gold evidence available in it at all.

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
