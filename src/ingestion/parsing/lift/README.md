# Lift API Parsing

This adapter calls the Datalab hosted Lift extraction API directly and maps the
result into AXIOM `ParsedData`.

## Setup

Install dependencies in your AXIOM environment:

```bash
pip install -e .
pip install datalab-python-sdk
```

Set the API key in `.env` at the project root:

```dotenv
DATALAB_API_KEY="your_api_key_here"
```

Local scripts load `.env` automatically. A shell-level `DATALAB_API_KEY` still
takes precedence when present.

## Config

`configs/pipeline.yaml` defaults to Lift API:

```json
"parsing": {
  "provider": "lift_api",
  "lift_api": {
    "api_key_env": "DATALAB_API_KEY",
    "mode": "balanced",
    "schema_path": "src/ingestion/parsing/lift/schemas/document_components.json",
    "output_dir": "data/processed/lift_outputs",
    "fallback_to_local": true
  }
}
```

`fallback_to_local` lets unsupported files or API failures continue through the
local scaffold parser. Set it to `false` when you want parsing to fail fast.

## Default Schema

By default, AXIOM uses a general document-components schema:

- `src/ingestion/parsing/lift/schemas/document_components.json`
- output fields: `document_type`, `language`, `title`, `main_text`, `tables`, `figures`, `formulas`

The `omnidoc_markdown.json` schema is still available for OmniDocBench-style
benchmark runs that require one markdown prediction field.

The raw API response is saved under `data/processed/lift_outputs/`. The parsed
AXIOM row stores:

```json
{
  "extraction": {
    "document_type": "...",
    "language": "...",
    "title": "...",
    "main_text": "...",
    "tables": [],
    "figures": [],
    "formulas": []
  },
  "text": "main_text or markdown when present"
}
```

To use your own schema:

```json
"lift_api": {
  "schema_path": "configs/my_lift_schema.json"
}
```

## Relation To The `lift` Repo

This adapter does not import or execute the sibling `lift` repo. It follows the
same API pattern as `lift/scripts/run_omnidocbench_datalab.py`:

```python
from datalab_sdk import DatalabClient, ExtractOptions
```

## Run Parsing Only

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD
python -m src.ingestion.parsing.run_parsing --config configs/pipeline.yaml
```

Stage artifacts:

```text
data/processed/
├── data_objects.json
├── parsed_data.json
└── initial_schemas.json
```

Raw Lift API responses:

```text
data/processed/lift_outputs/<filename>.json
```

## Run OmniDocBench Subset Like `lift`

This mirrors the sibling repo commands that run Datalab on the
OmniDocBench subset with the markdown schema.

Dry run first, without calling the API:

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD
python -m src.ingestion.parsing.lift.run_omnidocbench_subset --dry-run
```

Run the API:

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD
python -m src.ingestion.parsing.lift.run_omnidocbench_subset \
  --dataset-root data/raw/omnidocbench_subset \
  --schema src/ingestion/parsing/lift/schemas/omnidoc_markdown.json \
  --manifest data/raw/omnidocbench_subset/manifest.csv \
  --output-dir data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100
```

Convert API result JSON files to OmniDocBench markdown predictions:

```bash
python -m src.ingestion.parsing.lift.convert_results_to_markdown \
  data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100 \
  data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100_mds
```

Subset run output:

```text
data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100/
├── manifest.json
├── schema.json
├── 001_<image_stem>.json
├── 002_<image_stem>.json
└── ...
```

Each result JSON has the same basic shape as the `lift` repo run:

```json
{
  "input": {
    "index": 1,
    "file_name": "page-example.png",
    "path": "data/raw/omnidocbench_subset/images/page-example.png"
  },
  "status": "...",
  "page_count": 1,
  "latency_seconds": 1.23,
  "extraction": {
    "markdown": "..."
  }
}
```
