# AXIOM_DE-RD

AXIOM_DE-RD is a document-processing pipeline that converts documents from S3
or a local raw path into parsed records, index records, and vector embeddings.


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

### Conda

```bash
conda env create -f environment.yml
conda activate axiom-de-rd
```

### Python virtual environment

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

```dotenv
S3_URI=s3://bucket/path/to/document.pdf
# AWS_PROFILE=default
# AWS_REGION=ap-southeast-1
# S3_ENDPOINT_URL=http://localhost:9000
# S3_OBJECT_KEY=path/to/document.pdf
DATALAB_API_KEY=
OPENROUTER_API_KEY=
# Optional: only used when configured as the OpenRouter HTTP referer.
OPENROUTER_HTTP_REFERER=
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
| `indexing.embeddings` | Embedding provider, model, dimensions, and targets |

## Running the Pipeline

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Use the existing `s3://bucket/key` CLI option:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --s3-uri 's3://bucket/path/to/document.pdf'
```

Or select a presigned URL from a local S3 inventory:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --s3-info-file data/raw/s3.info.txt \
  --s3-object-key 'path/to/document.pdf'
```

Process every presigned object from the inventory in one pipeline run:

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

`--s3-all-objects` and `--s3-object-key` cannot be combined. Batch downloads
are persisted under `data/raw/<run_id>/objects/`, while each document retains
its own `s3://bucket/key` lineage. S3 folder markers such as `test/` are skipped,
and presigned URLs are never written to artifacts.

Or override the config with a local raw file or directory:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline.yaml \
  --local-raw data/raw/omnidocbench_subset
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
│   └── <document_id>.json              # Unified retrieval items with nested vectors
└── metadata.json                       # Common schemas and run reports

data/output/<run_id>/
├── documents/
│   └── <document_id>.json              # Consolidated ingest-to-embedding result
└── metadata.json                       # Common document info, schemas, and reports
```

An ingested quarantine record uses `status: "quarantined"`, has no `schema_id`,
and includes machine-readable failure reasons such as `zero_page_count`,
`null_extraction`, and `empty_parsed_content`. The CLI reports
`completed_with_errors` and the number of quarantined documents while allowing
the valid documents in the same batch to finish.

For `--local-raw`, source files are read in place and are not copied into a new
`raw/<run_id>` directory.

By default, `parsing.lift_api.save_raw_outputs` is `false`. Set it to `true`
only when full Lift responses and Markdown are needed under each document's
`assets/<document_id>/debug/` directory.

### Vector embeddings

The parsed document may keep text, tables, and figures in separate fields to
preserve the source structure. From the embedded stage onward, searchable text,
tables, and images are normalized into one `retrieval.items` array. Every item
uses the same identity and linkage fields and contains its own `embeddings`
array, so retrieval and frontend consumers do not need to join separate index
and vector collections or iterate over multiple component arrays.

```json
{
  "retrieval": {
    "document": {
      "record_id": "...",
      "index_type": "document",
      "document_id": "..."
    },
    "catalog": {
      "metadata": {},
      "index": {}
    },
    "items": [
      {
        "item_id": "...",
        "type": "text",
        "record_id": "...",
        "document_id": "...",
        "source_object_id": "...",
        "position": {
          "index": 0,
          "start_char": 0,
          "end_char": 1200
        },
        "content": {
          "text": "..."
        },
        "embedding_text": "...",
        "embeddings": [
          {
            "vector_id": "...",
            "model": "openai/text-embedding-3-small",
            "dimension": 1536,
            "status": "embedded",
            "values": [0.0]
          }
        ],
        "metadata": {}
      }
    ]
  }
}
```

`type` is `text`, `table`, or `image`. `content` keeps the type-specific parsed
payload, while `item_id`, `record_id`, `document_id`, `source_object_id`,
`position`, and `embeddings` have the same meaning for every item. The same
contract is written to both `data/embedded/<run_id>/documents/` and the final
`data/output/<run_id>/documents/` files.
