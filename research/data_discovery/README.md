# On-demand PDF data discovery

This folder contains the experimental on-demand ingestion pipeline. It is
separate from the main benchmark runners so that the discovery workflow can be
developed and evaluated independently.

## Pipeline overview

The light retrieval unit is one PDF page. A lightweight `pdf-inspector` parser
extracts page text once and a local BM25 index retrieves candidate pages for
each query.

There are currently three answer-generation settings:

```text
1. data_discovery
   -> BM25 retrieved pages
   -> accurate KDL + pdf-inspector ingestion
   -> chunking + embeddings + hybrid retrieval
   -> retrieved chunks
   -> generator

2. data_discovery
   -> BM25 retrieved pages
   -> accurate KDL + pdf-inspector ingestion
   -> page text / OCR evidence
   -> generator

3. data_discovery (on-demand-per-query)
   -> BM25 retrieved pages for one query
   -> KDL + pdf-inspector only for missing pages
   -> fixed 512-word chunks + text embeddings for newly parsed pages
   -> hybrid keyword/dense retrieval (dense alpha=0.7)
   -> generator
```

The `chunks` arm evaluates the first flow. The `pages` arm evaluates the
second flow and intentionally skips the second chunking, embedding and
retrieval stage.

The `on-demand-per-query` mode evaluates the third flow. It keeps a persistent
page-parse cache and prepared-chunk cache, so a page parsed for an earlier
query is not parsed or embedded again. Independent query workers share a short
KDL micro-batch queue; this preserves query-local BM25 and hybrid scopes while
allowing the hosted KDL model to receive batches.

## Current settings

| Component | Current setting |
| --- | --- |
| Light retrieval unit | PDF page |
| Light retriever | BM25 over fast pdf-inspector page text |
| Candidate pages | `--top-k-pages 10` by default; can be increased to 50 or 100 |
| Accurate parser | KDL + pdf-inspector |
| Parser scheduler | `global_two_phase` |
| Parser workers | `max_workers=8`, `render_processes=8` |
| vLLM request concurrency | `request_workers=24`, `request_batch_size=8` |
| vLLM sequence limit | `max_model_sequences=128` |
| Chunker | fixed overlap, 512 words with 128-word overlap |
| Embedder | OpenRouter `openai/text-embedding-3-small` |
| Embedding batch size | 64 |
| Accurate retrieval | hybrid dense/sparse retrieval, top-10 chunks |
| Generator | OpenRouter `deepseek/deepseek-v4-flash` |
| Judge | OpenRouter-compatible `openai/gpt-4o` |

The parser settings above are defined in
`configs/pipeline.data-discovery.yaml`. The chunking and
embedding settings used by the E2E runner are read from its
`--chunking-config` argument.

## Commands

Run a single-query discovery experiment:

```powershell
python -m research.data_discovery.cli `
  --input data/raw/my-lake `
  --index-dir data/work/page-discovery `
  --query "revenue recognition" `
  --top-k-pages 10
```

Add accurate ingestion, and optionally chunking and embeddings:

```powershell
python -m research.data_discovery.cli `
  --input data/raw/my-lake `
  --index-dir data/work/page-discovery `
  --query "revenue recognition" `
  --top-k-pages 10 `
  --ingest `
  --chunk `
  --pipeline-config configs/pipeline.data-discovery.yaml
```

Run light BM25 retrieval for the complete Physics subset:

```powershell
python -m research.data_discovery.run_vidore_physics `
  --subset physics `
  --language french `
  --top-k 10 `
  --output data/benchmark/vidore_v3/results/physics_discovery_bm25_french.jsonl
```

Run the end-to-end chunks arm:

```powershell
python -m research.data_discovery.run_vidore_e2e `
  --subset physics `
  --language french `
  --top-k-pages 10 `
  --top-k-chunks 10 `
  --arms chunks `
  --workers 24 `
  --parser-config configs/pipeline.data-discovery.yaml `
  --chunking-config configs/pipeline.data-discovery.yaml
```

Run the pages arm:

```powershell
python -m research.data_discovery.run_vidore_e2e `
  --subset physics `
  --language french `
  --top-k-pages 10 `
  --arms pages `
  --workers 24 `
  --parser-config configs/pipeline.data-discovery.yaml
```

Run the stateful online pipeline (one query at a time, with concurrent query
workers and shared KDL micro-batching):

```powershell
python -m research.data_discovery.run_vidore_e2e `
  --subset physics `
  --language french `
  --on-demand-per-query `
  --query-workers 4 `
  --arms chunks `
  --parser-config configs/pipeline.data-discovery.yaml `
  --chunking-config configs/pipeline.data-discovery.yaml
```

The online cache defaults to `<work-dir>/on-demand-cache`. Override it with
`--on-demand-cache-dir`. The micro-batch defaults are a `0.30` second window
and at most `32` unique pages. KDL concurrency defaults in this mode are
`max_workers=8`, `render_processes=8`, `bbox_max_workers=8`,
`request_workers=24`, `request_batch_size=8`, and
`max_model_sequences=128`; each can be overridden with the corresponding
`--kdl-*` option.

To test larger light-retrieval coverage, change only
`--top-k-pages`, for example to `50` or `100`. The selected pages are deduplicated
across queries before accurate ingestion.

If parser artifacts already exist, reuse them with:

```powershell
  --reuse-parser-artifacts data/work/vidore_v3/physics/discovery_e2e/parser-assets
```

Here `--workers` controls concurrent generator/judge requests. It is separate
from the parser's `request_workers` and `request_batch_size` settings in the
parser config. In `--on-demand-per-query` mode, `--query-workers` controls
concurrent independent pipeline queries; `--workers` still controls only
generator/judge requests.

## DocBench: on-demand basic (native Python runner)

`run_docbench_e2e.py` is the repository runner for the requested DocBench
baseline. It does not use a notebook and keeps the expensive stages online and
page-selective:

```text
light preparation: pdf-inspector page text
light retrieval:   BM25 page retrieval
accurate parse:    KDL + pdf-inspector for selected pages only
baseline_legacy:   fixed_overlap 512/128 -> text-embedding-3-small -> hybrid
answer/judge:      generator + binary DocBench judge
```

The original DocBench checkout is expected to have this shape. The runner
accepts either `DocBench` or `DocBench/data` as `--docbench-root`:

```text
DocBench/
  data/
    0/0.pdf
    0/0_qa.jsonl
    1/1.pdf
    1/1_qa.jsonl
    ...
```

Set the remote KDL/vLLM endpoint from Colab and the local OpenRouter key in
`AXIOM_DE-RD/.env` (the endpoint must be OpenAI-compatible):

```dotenv
VLLM_API_BASE=https://<your-colab-tunnel>/v1
VLLM_MODEL_NAME=<served-kdl-model-name>
VLLM_API_KEY=<optional>
OPENROUTER_API_KEY=<your-openrouter-key>
```

Run the complete dataset with retrieval over the selected/evaluated document
set (`file` mode):

```powershell
python -m research.data_discovery.run_docbench_e2e `
  --docbench-root ..\DocBench `
  --retrieval-scope file
```

Run the lake-retrieval variant, where each question searches all indexed
DocBench PDFs, including documents outside the evaluated subset:

```powershell
python -m research.data_discovery.run_docbench_e2e `
  --docbench-root ..\DocBench `
  --retrieval-scope lake
```

Before spending a full run, use a small smoke run. `--max-documents` selects
the first numeric document folders and `--limit` limits questions after that:

```powershell
python -m research.data_discovery.run_docbench_e2e `
  --docbench-root ..\DocBench `
  --retrieval-scope file `
  --max-documents 2 `
  --limit 10
```

The default config is
`configs/pipeline.docbench-on-demand-basic.yaml`. It pins the requested
`fixed_overlap` 512-word chunks with 128-word overlap, OpenRouter
`openai/text-embedding-3-small`, hybrid alpha `0.7`, KDL's
`global_two_phase` scheduler, and the existing generator/judge defaults.
Override a parser concurrency setting with a CLI option such as
`--kdl-request-batch-size 1`; this is useful when a tunnel or server does not
implement the optional `/chat/completions/batch` endpoint. The parser already
falls back to individual requests when that endpoint returns a 4xx.

`--skip-qa` runs only retrieval. `--skip-judge` runs generation but leaves the
score unset. Results are checkpointed after each question under
`data/benchmark/docbench_on_demand_basic/` (or `--output-dir`):

```text
indexes/                    cached pdf-inspector page BM25 indexes
cache/<scope>/               parsed pages and prepared chunks
retrieval/<scope>_*.jsonl    page/chunk retrieval rows and timings
qa/<scope>_*.jsonl           generated answers and judge results
reports/<scope>_*.json       aggregate DocBench report
manifest_<scope>.json        resolved run contract and paths
```

## Output and caching

- Discovery pipeline artifacts are isolated below
  `data/{ingested,cleaned,enriched,embedded,output}/discovery/` and
  `data/work/discovery/`.
- BM25 indexes are written below `data/work/vidore_v3/`.
- E2E reports are written below `data/benchmark/vidore_v3/results/`.
- On-demand-per-query parser/chunk caches, per-query retrieval JSONL and
  timing JSONL are written below the selected `--work-dir`/`--output-dir`.
- Generated data, embeddings, parser outputs and `.env` are intentionally not
  committed to Git.
- Reusing parser artifacts avoids re-running accurate ingestion when comparing
  the `pages` and `chunks` arms or changing only the downstream evaluation.
