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