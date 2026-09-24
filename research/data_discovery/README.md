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
answer/judge:      generator + 0/0.5/1 DocBench judge
```

### Flat `0. BENCHMARK` bundle

The exported bundle at
`data/raw/0. BENCHMARK-*/0. BENCHMARK/` is also accepted by the same runner.
It is detected by the presence of `documents.jsonl` and `queries.jsonl`; PDF
paths are resolved relative to that directory, and `qrels.jsonl` is used for
the retrieval report. The adapter handles the three sources in one lake:
`vidore_physics` (22 PDFs/80 questions), `vidore_industrial` (16/75), and
`mpdocvqa` (63/65), for 101 PDFs and 220 questions total.

Use the dedicated config and run a small smoke test first:

```bash
.venv/bin/python -m research.data_discovery.run_docbench_e2e \
  --config configs/pipeline.0-benchmark-on-demand-basic.yaml \
  --docbench-root "data/raw/0. BENCHMARK-20260915T030813Z-1-001/0. BENCHMARK" \
  --retrieval-scope lake \
  --max-documents 3 \
  --limit 10 \
  --skip-qa
```

For the complete end-to-end run (KDL endpoint and `OPENROUTER_API_KEY` must be
available), remove `--skip-qa` and the smoke limits:

```bash
.venv/bin/python -m research.data_discovery.run_docbench_e2e \
  --config configs/pipeline.0-benchmark-on-demand-basic.yaml \
  --docbench-root "data/raw/0. BENCHMARK-20260915T030813Z-1-001/0. BENCHMARK" \
  --retrieval-scope lake
```

The output is under `data/benchmark/0_benchmark_on_demand_basic/`. In the
report, `correct_only` is the strict QA accuracy, while
`correct_plus_partial` credits both 1 and 0.5 judgments. The bundle qrels add
`file_recall` over unique PDF documents in the light top-20 page ranking. All
other retrieval `recall@k` and `NDCG@k` fields are page-level metrics; the
accurate ranking maps chunks back to their source pages before scoring. Timing
fields map directly to the requested table:
`light_preparation_seconds_all_data`,
`online_latency_seconds_per_query` (`Light Retrieval`, `Parsing`,
`Chunk&Embed&Index`, `Retrieval`, `Overall`), and
`infer_time_seconds_per_query`.

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

Run the complete dataset with per-question retrieval restricted to the
DocBench document that owns the question (`file` mode):

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

The three DocBench runners write timestamped text and JSONL events below
`<output-dir>/logs/`. After `host_failure_threshold` consecutive KDL 5xx,
408/429, timeout, or connection errors (default `3`), the KDL circuit opens
and the current run exits. Retrieval/QA JSONL rows and parser caches are
checkpointed as work completes; rerunning the same command skips successful
questions/pages and resumes the rest. The threshold can be changed with
`--kdl-host-failure-threshold N` or the corresponding `kdl` config field.

All three DocBench runners handle the generator abstention sentinel
`KHONG_DU_THONG_TIN` with one retry. If the first generation returns the
sentinel, the runner generates and judges that question once more immediately.
The QA JSONL stores `unanswerable_retry_count` (and `initial_sys_ans` when a
retry occurred), so a later invocation automatically processes only completed
questions that returned the sentinel and have not used their retry yet. A
question whose retry also returns the sentinel is not regenerated repeatedly.

## DocBench: pages-only generation

For the page-context experiment, use the separate
`run_docbench_pages.py` runner. It keeps BM25 as the light page-discovery
step, parses the selected pages with KDL + pdf-inspector, and sends each
question's parsed page text directly to the generator. Selected pages are
submitted in bounded batches (configurable with `--parse-batch-size`) so the
KDL global scheduler can batch requests across pages. Missing or quarantined
pages are rebatched and retried within the same run; `--parse-retry-attempts`
controls the number of retries after the initial attempt (default `2`). Each
batch is checkpointed for resume. It does not
run fixed chunking, embeddings, or a second retrieval stage:

```text
pdf-inspector page index -> BM25 pages -> KDL + pdf-inspector parse
-> parsed page text -> generator -> DocBench judge
```

Example smoke run:

```powershell
python -m research.data_discovery.run_docbench_pages `
  --config configs/pipeline.docbench-pages.yaml `
  --docbench-root ..\DocBench `
  --retrieval-scope file `
  --max-documents 2 `
  --limit 10 `
  --top-k-pages 10 `
  --parse-batch-size 32 `
  --parse-retry-attempts 2 `
  --workers 4
```

Use `--retrieval-scope lake` to let each question search every PDF in the
lake while evaluating only the selected documents. In `file` mode, each
question's page retrieval is filtered to its own document. `--workers` controls
parallel generator/judge requests. Parsed page text is passed as one context
unit per page; `--parse-batch-size` controls how many one-page inputs share
each KDL parse call. `--max-page-chars` and `--max-context-chars` only cap the LLM
context and do not create chunks. The runner writes discovery rows to
`discovery/<scope>_pages.jsonl`, answers to `qa/<scope>_pages.jsonl`, and a
pages-specific report and manifest.

Page-level retries are separate from KDL's `parsing.kdl.max_retries`: the KDL
setting retries individual HTTP requests, while `parse_retry_attempts` retries
whole pages that still have no usable enriched record after a batch finishes.

If a run is interrupted, rerun the same command. Successful QA rows and
successfully parsed pages are reused automatically; only missing or non-`ok`
rows/pages are sent again. A KDL host circuit opens after three consecutive
5xx, 408/429, timeout, or connection failures and stops the current run. The
checkpoint is left intact for the next invocation.

Each run also writes timestamped logs under `logs/`:

```text
logs/<scope>_on_demand_*.log       human-readable log with timestamps
logs/<scope>_kdl_events.jsonl      request health/circuit events
logs/<scope>_events.jsonl          pipeline/query/page events
```

To avoid calling KDL for already persisted parser artifacts, pass the
parser-artifact directory:

```bash
.venv/bin/python -m research.data_discovery.run_docbench_pages \
  --config configs/pipeline.docbench-pages.yaml \
  --docbench-root /path/to/DocBench \
  --output-dir data/benchmark/docbench_pages \
  --retrieval-scope lake \
  --reuse-parser-artifacts data/benchmark/docbench_pages/parser-assets/pages_lake \
  --workers 4
```

With `--reuse-parser-artifacts`, the KDL endpoint is not contacted. The
OpenRouter key is still required because failed questions are regenerated.

`--skip-qa` runs only retrieval. `--skip-judge` runs generation but leaves the
score unset. Results are checkpointed after each question under
`data/benchmark/docbench_on_demand_basic/` (or `--output-dir`):

```text
indexes/                    cached pdf-inspector page BM25 indexes
cache/<scope>/               parsed pages and prepared chunks
retrieval/<scope>_*.jsonl    page/chunk retrieval rows and timings
qa/<scope>_*.jsonl           generated answers and judge results
reports/<scope>_*_verN.json  immutable aggregate/timing reports for each run
manifest_<scope>_verN.json   resolved run contract and report paths for each run
logs/                        timestamped text and JSONL runtime events
```

### Review retrieval and QA in one HTML file

Generate a self-contained qualitative review page for a completed run. The
left rail supports search, score/retrieval filters, and sorting; the detail
view combines the gold answer, generated answer, judge output, qrels, BM25
page ranking, accurate chunks, QA context markers, and timing metadata:

```bash
.venv/bin/python scripts/visualize_docbench_run.py \
  --run-dir data/benchmark/on_demand_basic \
  --output data/benchmark/on_demand_basic/retrieval_qa_review.html
```

The default output is `<run-dir>/retrieval_qa_review.html`. The page is fully
offline and does not require a web server; open it directly in a browser.
Use `j`/`k` to move between questions, or use the previous/next buttons.
For each question, the **Gold page coverage** table is the main retrieval
check: it shows every gold page, its relevance grade, the Light BM25 rank and
score, the Accurate chunk rank and score, and whether the page entered the QA
context. `HIT` means the page is within the configured top-k; clicking a rank
jumps to the corresponding page/chunk evidence card.

### Compare the canonical baseline with on-demand

The canonical `baseline_pdf_inspector_tesseract` package and the on-demand run
use different page-ID conventions: baseline IDs are zero-based, while
on-demand IDs are one-based PDF page numbers. The comparison tool normalizes
both before computing page overlap, gold-page hits, rank movement, and parse
similarity:

```bash
.venv/bin/python scripts/compare_docbench_runs.py \
  --baseline-dir data/benchmark/baseline_pdf_inspector_tesseract \
  --ondemand-dir data/benchmark/on_demand_basic_1 \
  --output data/benchmark/on_demand_basic_1/baseline_comparison.html
```

The generated page has separate **Parse compare** and **Query compare** modes.
Parse compare is document/page based and shows baseline text beside on-demand
KDL text, availability, OCR provenance, text similarity, and parser component
counts. Query compare shows gold-page rank coverage, top-20 overlap, baseline
only/on-demand only pages, scores, page text, and on-demand QA. The canonical
baseline README states that it does not include QA JSONL, so that side is shown
as unavailable rather than being treated as an empty answer.

Each invocation uses the next `verN` number, so it never overwrites an earlier
aggregate report or timing summary in the same output directory.

The on-demand timing summary separates elapsed latency from shared-resource
service work. `total_runtime_seconds_all_data.Overall` is the wall-clock
runtime for the run; `online_latency_seconds_per_query.Overall` is the
end-to-end query latency and includes queueing. KDL queue wait and chunk-lock
wait are reported in `queue_wait_seconds_per_query`, while
`amortized_service_work_seconds_per_query` divides unique phase work by the
number of completed queries. `query_latency_percentiles_seconds` contains
p50/p95/p99 query latency, and `online_throughput_queries_per_second` reports
run throughput. Phase service timings exclude KDL queue wait and chunk-lock
wait, so they should not be added together to reproduce wall-clock runtime.

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
