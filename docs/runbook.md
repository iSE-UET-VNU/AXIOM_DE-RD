# Runbook — from an empty machine to an answer-level number

Two independent things live in this repo and they are easy to confuse:

- **The pipeline** (`src/`, `scripts/run_pipeline.py`) — ingestion → cleaning →
  enrichment → chunking/embedding → integration → artifacts. This is the
  product.
- **The harness** (`research/harness/`) — builds a benchmark corpus and an
  evalset from the challenge data lake, runs retrieval, generates answers and
  judges them. This is how every number we report is produced.

They share `src/chunking_embedding/` and `src/retrieval/`. The harness imports
`src/`; `src/` never imports the harness.

Run **Part A** to see the pipeline work. Run **Parts B–D** to get a measured
number. They do not depend on each other.

---

## 0. Prerequisites

```bash
conda env create -f environment.yml      # once
conda activate axiom-de-rd
pip install -e .
```

`.env` at the repo root needs `OPENROUTER_API_KEY` — used for embeddings in
Part C and, indirectly, for generation in Part D.

**model-service must be running for Part D.** It lives in the platform repo:

```bash
cd ../AXIOM/services/model-service
OPENROUTER_API_KEY=$(grep -m1 '^OPENROUTER_API_KEY' ../../AXIOM_DE-RD/.env | cut -d= -f2-) \
  docker compose up -d --build
curl -s localhost:8006/api/v1/health/ready
```

Two inputs are **not in git** and must be obtained separately — the data lake
(2.0 GB, 923 files) and the questions CSV. Place them at the repo root as
`[iSE Summer Challenge 2026] Data Lake/` and
`[iSE Summer Challenge 2026] Questions - Q&A(1).csv`.

---

## Part A — the pipeline, one document

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.mock.yaml \
  --local-raw <a-directory-of-files>
```

Writes to `data/{ingested,cleaned,enriched,embedded,output}/<run_id>/`. Success
looks like `Built N retrieval record(s) and N vector record(s)`.

**Why `pipeline.mock.yaml`.** The default config parses with `lift_api`, which
currently returns `402 Payment Required` — Datalab is out of credit — and every
PDF is quarantined. `chandra2` is the other real parser and needs a hosted vLLM
endpoint (see Part E). `mock_vlm` reads a PDF's existing text layer and emits a
labelled placeholder for pages that have none, so stages 2–6 can run.

**Nothing produced under `mock_vlm` is a parsing result.** Every synthetic block
carries `parser_source: "mock_vlm"` and `mock: true`, and the run metadata
counts `synthetic_pages`. Check those before quoting anything.

---

## Part B — lake → benchmark inputs

Two scripts, both read the lake, and neither is needed again afterwards.

```bash
# 923 files -> data/benchmark/corpus.jsonl  (~92 MB) + corpus_report.json
python -m research.harness.build_corpus \
  --lake "[iSE Summer Challenge 2026] Data Lake"

# CSV + lake -> data/benchmark/questions.jsonl + evalset_report.json
python -m research.harness.build_evalset \
  --questions "[iSE Summer Challenge 2026] Questions - Q&A(1).csv" \
  --lake "[iSE Summer Challenge 2026] Data Lake"
```

`build_evalset` needs the lake as well as the CSV: it resolves the gold
filenames in each question to document ids, and records how each one resolved.

Add `--limit 20` to `build_corpus` for a fast subset. Read
`corpus_report.json` before going further — `coverage` is the share of the lake
that parsed, and `failure_reasons` splits into `unsupported` (no handler) and
`empty` (parsed, produced nothing — image-only PDFs).

Everything under `data/benchmark/` is gitignored: it is derived, large, and
`questions.jsonl` contains the challenge's gold answers.

---

## Part C — retrieval

```bash
python -m research.harness.run_retrieval \
  --benchmark ise \
  --arms bm25,dense,rrf \
  --chunker fixed_overlap \
  --embedder openrouter_te3s \
  --k 10
```

Costs roughly $0.37 in embeddings for the full corpus at
`text-embedding-3-small`. Add `--limit N` to cap the question count while
testing.

Writes `data/benchmark/runs/<index_id>/<arm>__<params>__<queryset>.jsonl` and a
`<index_id>.report.json` beside it. `index_id` is
`benchmark.level.text_source.chunker.embedder.analyzer`.

> **Known label defect.** `--text-source` defaults to `vlm_text` and is baked
> into `index_id` even for iSE, which has no VLM text at all — that corpus is
> `extract.py` output. The numbers are unaffected; the name is wrong. Do not
> read `ise.page.vlm_text.…` as a VLM arm.

Read `coverage` in the report first. It is the share of questions whose gold
evidence is present in the corpus at all — the ceiling on everything below it,
and the parsing result rather than a retrieval one.

---

## Part D — answers

```bash
python -m research.harness.run_answer \
  --run data/benchmark/runs/<index_id>/rrf__*.jsonl \
  --arm rrf \
  --generator llm-rerank \
  --judge llm-rerank-strong
```

**The model aliases matter.** The harness defaults to `llm-default` and
`llm-judge`, which are not registered in model-service. Of what is registered,
`agent-llm-default` and `retry-llm` are backed by the **fake** provider: they
return HTTP 200 with `"Fake response: <your prompt>"`, never error, and every
answer scores wrong while looking like a retrieval failure. The real LLM
aliases are `llm-rerank` (gpt-4o-mini), `llm-rerank-strong` (gpt-4o) and
`openrouter-llm-free`. Check before trusting a run:

```bash
curl -s localhost:8006/api/v1/model-registry/models
```

Generation packs context to a **character** budget (12,000 by default), not a
chunk count, so chunk size does not silently become the treatment. Answers that
cannot be supported return the abstention sentinel `KHONG_DU_THONG_TIN`, which
is scored as wrong but reported separately from errors.

Comparing arms:

```bash
python -m research.harness.compare_arms data/benchmark/runs/*.report.json
```

Only `acc|common` — accuracy on the intersection of every arm's reachable set —
may carry a cross-arm claim. `coverage` belongs beside it, never averaged in.

---

## Part E — the VLM parsing arm (blocked)

`configs/pipeline.chandra2.yaml` exists but needs a hosted endpoint. Run
`ModelHosting.ipynb` on Colab with an **L4 or A100** — cell 3 refuses anything
under 20 GiB VRAM and requires bf16, so a free-tier T4 fails. The last cell
prints the tunnel URL. Then:

```bash
export VLLM_API_BASE="https://<ngrok>.ngrok-free.app/v1"
export VLLM_API_KEY="axiom-dev-9f2a"
export VLLM_MODEL_NAME="chandra"
curl -s -H "Authorization: Bearer $VLLM_API_KEY" "$VLLM_API_BASE/models"

python scripts/run_pipeline.py --config configs/pipeline.chandra2.yaml --local-raw <dir>
```

**Gap:** `build_corpus.py` has no `--parser` flag — it always uses
`extract.py`. There is no wired path from `run_pipeline.py` output into
`corpus.jsonl`, so Chandra output cannot yet reach Parts C–D. Hosting Chandra
gets parsed documents, not an answer-level number.

---

## What is measured where

| Level | Where | Reported as |
|---|---|---|
| coverage | Part B report, Part C report | share of questions whose evidence exists in the corpus |
| retrieval | Part C | recall@k, split single- vs multi-evidence, by modality |
| answer | Part D | accuracy, abstention rate, error rate, context recall |

Coverage and accuracy are kept apart on purpose. A parser answering 64/111 at
70% beats one answering 49/111 at 78%, and averaging the two hides it.
