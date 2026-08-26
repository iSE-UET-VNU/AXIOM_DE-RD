# AXIOM_DE-RD

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

## Running the Pipeline

Every entrypoint defaults to `configs/pipeline.kdl-pdf-inspector.yaml`. Set
`AXIOM_PIPELINE_CONFIG` or pass `--config` to select another profile.

```bash
python scripts/run_pipeline.py
```

Process a local PDF or image through the hybrid parser:

```bash
python scripts/run_pipeline.py \
  --local-raw path/to/document.pdf
```

The legacy `configs/pipeline.yaml` profile remains available for Lift API
workflows that need its broader input configuration.

#### Use the existing `s3://bucket/key` CLI option:

```bash
python scripts/run_pipeline.py \
  --s3-uri 's3://bucket/path/to/document.pdf'
```

#### Or select a presigned URL from a local S3 inventory:

```bash
python scripts/run_pipeline.py \
  --s3-info-file data/raw/s3.info.txt \
  --s3-object-key 'path/to/document.pdf'
```

#### Process every presigned object from the inventory in one pipeline run:

```bash
python scripts/run_pipeline.py \
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
  --local-raw data/raw/omnidocbench_subset
```

## REST API

The REST API uses the same hybrid default. Install the
hybrid extra and point AXIOM at an externally hosted OpenAI-compatible KDL/vLLM
service:

```bash
python -m pip install -e ".[pdf-inspector]"
export AXIOM_PIPELINE_CONFIG="configs/pipeline.kdl-pdf-inspector.yaml"
export KDL_NANO_ENDPOINT_URL="http://<kdl-host>:8000/v1"
export KDL_NANO_MODEL="kdl-frontier-parser-nano"
```

Start the synchronous REST service with:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The ready-to-run hybrid config uses `request_batch_size: 8`, so the model
service must expose the synchronous `/v1/chat/completions/batch` route. Set
`request_batch_size: 1` when the service only exposes the standard
`/v1/chat/completions` route. Set
`AXIOM_PIPELINE_CONFIG=configs/pipeline.yaml` to explicitly select the legacy
Lift API profile.

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

The response contains one common `metadata` object and a `documents` array.
Each document element separates `document`, `ingest`, `clean`, `enrich`, and
`retrieval`; native stage data is kept under each stage's `data` field. Schema
objects stay in internal stage artifacts and are not returned by the public
API. The existing CLI and all other input modes remain available.


## Streamlit Demo

Start the interactive demo from the repository root:

```bash
streamlit run streamlit_app.py
```
