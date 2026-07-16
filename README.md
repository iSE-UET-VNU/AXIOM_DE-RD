# AXIOM_DE-RD

AXIOM_DE-RD is an early-stage data pipeline for document ingestion and indexing.
The current workflow focuses on stable artifacts for downstream retrieval and
analytics systems.

## Current Pipeline

```text
raw document
-> ingestion: route by file type
   - TextParserBackend: local text parsing for TXT, Markdown, CSV, JSON,
     JSONL, YAML, and YML; CSV is emitted as a canonical table
   - TableParser: local XLS/XLSX parsing, one table per non-empty sheet
   - DocumentParser: PDF and supported image formats use Lift API by default;
     deferred mode and self-hosted Chandra2 remain configurable options
-> cleaning: pass-through
-> enrichment: pass-through
-> indexing_cataloging:
   - metadata catalog
   - document index records
   - text chunk index records
   - table index records
   - figure index records
   - catalog index records
   - index quality report
   - OpenRouter text chunk embeddings
-> integration: pass-through
-> storage:
   - local JSON artifacts
   - Milvus vector collection for text chunks
```

The main config is:

```text
configs/pipeline.yaml
```

Parser routing is the default. Text and table inputs are parsed locally without
network calls. Document inputs (`.pdf`, `.pptx`, and supported image formats)
are routed to Lift API by default. Use `document.provider="deferred"` to disable
document extraction, or select Chandra2 explicitly to use a self-hosted vLLM
server that emits page-ordered Markdown. If
`document.fallback_to_deferred` is enabled, a provider failure becomes
`deferred`; otherwise it is recorded as a file-scoped parsing failure.

It is configured for a small final-test input:

```text
data/raw/final_test
```

## Environment

Python 3.11 or newer is required. Choose one of the following setup options.

### Conda

```bash
conda env create -f environment.yml
conda activate axiom-de-rd
```

### Python venv and pip

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install the project and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install only the optional document provider you use:

```bash
python -m pip install -e ".[lift]"
python -m pip install -e ".[chandra2]"
```

The pipeline loads `.env` from the project root automatically. Text/table-only
runs do not require `DATALAB_API_KEY`, but the default Lift document route does.
Chandra2 reads the vLLM connection settings used by its official package. The
indexing and vector storage stages may still require their provider credentials:

```dotenv
DATALAB_API_KEY=
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL_NAME=chandra
VLLM_API_KEY=EMPTY
OPENROUTER_API_KEY=
MILVUS_URI=
MILVUS_TOKEN=
```

The Chandra code package and model weights use different licenses. Review the
[model's commercial-use terms](https://github.com/datalab-to/chandra#commercial-usage)
before deploying Chandra2 in production.

## Run

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Outputs are written to:

```text
data/work/final_test/<run_id>
data/processed/final_test
data/cleaned/final_test
data/enriched/final_test
data/output/final_test
```

Important artifact directories:

```text
data/work/final_test/<run_id>/
data/work/final_test/<run_id>/datalab/ (Lift mode only)
data/work/final_test/<run_id>/chandra2/ (Chandra2 mode only)
data/processed/final_test/
data/processed/final_test/normalization/
data/output/final_test/data/
data/output/final_test/reports/
```

Provider debug outputs are retained in run-scoped document bundles under
`data/work`. Lift stores its response payloads and decoded assets; Chandra2
stores merged `result.md` and page-level `metadata.json`. Canonical ingestion
records are JSONL under `data/processed`; large in-memory parsing intermediates
are not persisted.

Routing outcomes for every inventoried input are stored in
`data/processed/final_test/parsing_results.jsonl`. Each record includes the
selected route, concrete backend/provider, parse status, and reason where
applicable. The processed manifest uses `processed-manifest-v2`, summarizes
counts by backend and status, and lists deferred, unsupported, and failed source
IDs. A dataset is `complete` only when every input parses successfully and
contains content; otherwise it is `partial`.

The provider extraction contract and stage-specific schemas remain at their
source or stage paths:

```text
src/ingestion/parsing/lift/schemas/document_components.json (Lift mode only)
data/processed/final_test/parsing_results.jsonl
data/processed/final_test/manifest.json
data/cleaned/final_test/cleaned_schemas.json
data/enriched/final_test/enriched_schemas.json
```

`data/output/final_test/data/schemas.json` is a self-contained JSON Schema for
one logical document. Its document metadata maps to `documents.jsonl`; its
`texts`, `tables`, `images`, and `formulas` arrays map to the normalized JSONL
files through `document_id`. The schema includes storage mappings, checksums,
record counts, observed types, and dataset-level values without duplicating
document content. `document_components.json` remains only the optional
extraction schema sent to Datalab when Lift is selected explicitly.

`pipeline_state.json` is a lightweight manifest with run metadata, artifact
paths, counts, and report statuses; it does not duplicate records or embeddings.

`documents.jsonl` is a document registry containing source/parser metadata and
component counts. Component content lives only in the three normalization JSONL
files. `index_records.json` remains the downstream indexing contract.

The configured Milvus collection is:

```text
axiom_text_chunks_openrouter_text_embedding_3_small_1536
```

## Contracts

`IndexRecord.index_type` currently supports:

```text
document
text_chunk
table
image
figure
catalog
```

The sample config targets `text_chunk`, `table`, and `image` records for
embedding. Embeddings use OpenRouter with `openai/text-embedding-3-small` and
dimension `1536`.
