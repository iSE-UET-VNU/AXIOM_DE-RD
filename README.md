# AXIOM_DE-RD

AXIOM_DE-RD is a document-processing pipeline that converts documents from S3
or a local raw path into parsed records, index records, and vector embeddings.
Spreadsheet workbooks can be delegated to a running TableAgent API and
normalized into the same final output contract.


## Project Structure

```text
AXIOM_DE-RD/
├── configs/
│   └── pipeline.yaml                 # Pipeline and provider configuration
├── data/
│   ├── raw/                          # Local inputs, S3 inventories, downloaded S3 files
│   ├── ingested/                     # Provider output and parsed records
│   ├── cleaned/                      # Cleaning-stage snapshot
│   ├── enriched/                     # Enrichment-stage snapshot
│   ├── embedded/                     # Per-document indexes and vectors
│   └── output/                       # Consolidated end-to-end documents
├── scripts/
│   └── run_pipeline.py               # Main pipeline entrypoint
├── src/
│   ├── s3_reader.py                  # S3 URI and presigned inventory reader
│   ├── local_reader.py               # Local file and directory discovery
│   ├── ingestion/                    # Local-file parsing and schema inference
│   ├── cleaning/                     # Cleaning stage
│   ├── enrichment/                   # Enrichment stage
│   ├── indexing_cataloging/          # Chunking, indexing, quality, embeddings
│   ├── integration/                  # Integration stage
│   ├── artifacts/                    # Per-document artifact writers
│   ├── models.py                     # Shared pipeline data contracts
│   └── pipeline.py                   # Pipeline orchestration
├── .env.example
├── environment.yml
└── pyproject.toml
```

The cleaning, enrichment, and integration stages currently provide scaffolded
pass-through behavior for future domain-specific processing.

## Requirements

Python 3.11 or newer is required.

## Installation

### Option1: Conda

```bash
conda env create -f environment.yml
conda activate axiom-de-rd
```

### Option2: Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Environment Variables

Copy the example environment file and provide the required credentials:

```bash
cp .env.example .env
```

## Configuration

The main configuration file is `configs/pipeline.yaml`. It is intentionally
JSON-compatible so it can be loaded even when PyYAML is unavailable.

Important settings include:

| Setting | Purpose |
|---|---|
| `input.mode` | Selects `s3_uri`, `presigned_info`, or `local_raw` |
| `s3_input` | S3 URI and presigned-inventory settings |
| `local_input` | Local path, recursion, extension filters, and file limit |
| `raw_dir` | Root for downloaded S3 objects; local raw inputs remain in place |
| `ingested_dir` | Provider output plus parsed ingestion artifacts |
| `cleaned_dir` | Cleaning-stage artifacts |
| `enriched_dir` | Enrichment-stage artifacts |
| `embedded_dir` | Index, embedding, schema, and report artifacts |
| `output_dir` | Final consolidated document artifacts |
| `enabled_modules` | Pipeline stages to enable; execution follows the order defined by the orchestrator |
| `parsing` | Parser provider and Lift options |
| `table_agent` | TableAgent API routing, authentication, timeout, and workbook extensions |
| `indexing.embeddings` | Embedding provider, model, dimensions, and targets |

### TableAgent integration

Start TableAgent on a separate port, for example `8001`, then set:

```bash
TABLE_AGENT_BASE_URL=http://127.0.0.1:8001
TABLE_AGENT_SERVICE_API_KEY=
```

When `table_agent.enabled` is true, every supported AXIOM entrypoint first uses
the shared `src.dispatcher.dispatch_dataeng_inputs` boundary. REST presigned
inventories, Streamlit uploads, local paths, direct S3 URIs, and file-backed
presigned inventories all use the same extension-based routing:

- supported Excel workbooks are uploaded to TableAgent's existing
  `/v1/jobs/upload` endpoint with `stage=structure`;
- non-workbook files continue through the existing document pipeline;
- both branches are normalized and merged into the existing `metadata` plus
  `documents` response with `output-document-v4` documents.

`src.pipeline.run_pipeline` remains the internal non-table document engine and
is called by the dispatcher only after workbook inputs have been removed from
the document branch. This keeps TableAgent state out of document models,
ingestion, indexing, and persisted intermediate artifacts.

The Streamlit uploader uses the same routing behavior for local uploads.
Workbook-only uploads do not require `DATALAB_API_KEY`; mixed batches send
Excel files to TableAgent and keep PDF/image files on the existing document
pipeline. For the demo explorer only, Streamlit persists the final normalized
metadata and document JSON under `data/output/<run_id>/`; it does not create
workbook ingestion, cleaning, enrichment, or indexing artifacts. Workbook
results therefore open in **Explore results** with the same tabs and downloads
as other final output documents.

The current TableAgent source response does not include temporary retrieval
artifact files. AXIOM will preserve `retrieval_items` or `retrieval.items` when
the deployed API returns them; otherwise it builds retrieval items from the
returned structures with empty `embeddings` arrays.

Normalization consumes known TableAgent fields into their canonical
`output-document-v4` locations instead of copying the complete source response.
Unmapped response, table, retrieval-item, and embedding fields are retained
once under an `extensions` mapping at the corresponding level. This avoids a
duplicated `raw_response` while allowing future or deployment-specific
TableAgent fields to pass through without being discarded.

## Running the Pipeline

The CLI uses the shared dispatcher, including when input mode comes from
`configs/pipeline.yaml`:

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Process a local workbook through TableAgent:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --local-raw path/to/workbook.xlsx
```

Programmatic callers that need source routing should use:

```python
from src.dispatcher import dispatch_dataeng_inputs

result = dispatch_dataeng_inputs(
    "configs/pipeline.yaml",
    local_raw="path/to/input",
)
response = result.response
```

#### Use the existing `s3://bucket/key` CLI option:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --s3-uri 's3://bucket/path/to/document.pdf'
```

#### Or select a presigned URL from a local S3 inventory:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --s3-info-file data/raw/s3.info.txt \
  --s3-object-key 'path/to/document.pdf'
```

#### Process every presigned object from the inventory in one pipeline run:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --s3-info-file data/raw/s3.info.txt \
  --s3-all-objects
```

The equivalent configuration is:

```json
{
  "input": {
    "mode": "presigned_info"
  },
  "s3_input": {
    "info_file": "data/raw/s3.info.txt",
    "all_objects": true,
    "object_key": null
  }
}
```

#### Or override the config with a local raw file or directory:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --local-raw data/raw/omnidocbench_subset
```

## REST API

Start the synchronous REST service with:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Submit a batch of presigned S3 objects:

```bash
curl -X POST http://localhost:8000/v1/dataeng \
  -H 'Content-Type: application/json' \
  -d '{
    "bucket": "example-bucket",
    "files": [
      {
        "key": "document.pdf",
        "presigned_url": "https://example.invalid/presigned-document-url"
      }
    ]
  }'
```

The response groups the existing final artifacts without changing their
contracts: `metadata` contains `data/output/<run_id>/metadata.json`, and
`documents` contains the corresponding `output-document-v4` JSON objects.
The existing CLI and all other input modes remain available.

Health checks are exposed at `/health/live` and `/health/ready`. Interactive
OpenAPI documentation is available at `/docs`.

## Streamlit Demo

Start the interactive demo from the repository root:

```bash
streamlit run streamlit_app.py
```


## Output Artifacts

Each parsed document is persisted immediately before ingestion continues with
the next input. A document is quarantined when the parser returns
`page_count=0`, a null Lift extraction, or no usable text/structured component.
Quarantined documents remain under `data/ingested` with failure reasons but do
not continue into cleaning, enrichment, indexing, embedding, or final output.
`metadata.json` reports `in_progress` during a batch and changes to `completed`
or `completed_with_errors` after all expected documents are parsed. With the
default configuration, one run is organized as follows:

```text
data/raw/<run_id>/objects/              # S3 and presigned input only

data/ingested/<run_id>/
├── assets/
│   └── <document_id>/
│       ├── images/                     # Images extracted during parsing
│       └── debug/                      # Optional raw Lift outputs
├── documents/
│   └── <document_id>.json              # Parsed result or quarantine record
└── metadata.json                       # Progress, common info, and initial schemas

data/cleaned/<run_id>/
├── documents/
│   └── <document_id>.json              # Cleaned data for one document
└── metadata.json                       # Common info and cleaned schemas

data/enriched/<run_id>/
├── documents/
│   └── <document_id>.json              # Enriched data for one document
└── metadata.json                       # Common info and enriched schemas

data/embedded/<run_id>/
├── documents/
│   └── <document_id>.json              # Compact retrieval items with nested vectors
└── metadata.json                       # Retrieval schema and compact run summary

data/output/<run_id>/
├── documents/
│   └── <document_id>.json              # Semantic content, source blocks, reading order, retrieval
└── metadata.json                       # Common document schema and compact run summary
```

`output-document-v4` stores parser-native source components in
`content.blocks`, their ordered component IDs in `content.reading_order`, and
provenance/completeness in `content.reading_order_meta`. New Lift runs use JSON
output so the order includes text, headings, equations, figures, captions, and
tables. Older artifacts can be reconstructed from structured-extraction
citations, with `reading_order_meta.complete` set to `false`.

An ingested quarantine record uses `status: "quarantined"`, has no `schema_id`,
and includes machine-readable failure reasons such as `zero_page_count`,
`null_extraction`, and `empty_parsed_content`. The CLI reports
`completed_with_errors` and the number of quarantined documents while allowing
the valid documents in the same batch to finish.

For `--local-raw`, source files are read in place and are not copied into a new
`raw/<run_id>` directory.

By default, `parsing.lift_api.save_raw_outputs` is `false`. Set it to `true`
only when full Lift responses and rendered JSON/Markdown are needed under each document's
`assets/<document_id>/debug/` directory.
