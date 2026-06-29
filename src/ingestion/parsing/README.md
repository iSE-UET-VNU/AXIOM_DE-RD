# Parsing Module

This module converts raw files into AXIOM `ParsedData` and `InitialSchema`
artifacts.

Current behavior:

- CSV and JSON files are parsed into row dictionaries.
- Text-like files are read into a `text` field.
- Images/PDFs are sent to Datalab hosted Lift when `provider` is `lift_api`.
- Unsupported files and Lift failures can fall back to the local scaffold parser.
- The module exposes a stable interface for adding OCR/model adapters later.

## Config

The default parsing config uses Lift API with local fallback:

```json
{
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
}
```

Install the SDK:

```bash
pip install -e .
pip install datalab-python-sdk
```

Set the API key in `.env` at the project root:

```dotenv
DATALAB_API_KEY="your_api_key_here"
```

Parsing scripts load `.env` automatically. A shell-level `DATALAB_API_KEY`
still takes precedence when present.

If `DATALAB_API_KEY` or `datalab-python-sdk` is missing, the default config keeps
running through local fallback because `fallback_to_local` is `true`.

Lift raw API responses are written to `data/processed/lift_outputs/`.

## Schema

The project default points Lift to a general document-components schema:

- `src/ingestion/parsing/lift/schemas/document_components.json`
- output fields: `document_type`, `language`, `title`, `main_text`, `tables`, `figures`, `formulas`

The adapter uses the same schema as its default when `schema_path` is omitted.
The older `omnidoc_markdown.json` schema is still available for benchmark runs
that need a single `markdown` field.

To use a custom JSON schema, set:

```json
"schema_path": "configs/my_lift_schema.json"
```

inside `parsing.lift_api`.

## Run OmniDocBench Subset Like `lift`

For the dedicated OmniDocBench subset flow:

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD
python -m src.ingestion.parsing.lift.run_omnidocbench_subset --dry-run
```

Run the API and then convert results to markdown:

```bash
python -m src.ingestion.parsing.lift.run_omnidocbench_subset \
  --dataset-root data/raw/omnidocbench_subset \
  --schema src/ingestion/parsing/lift/schemas/omnidoc_markdown.json \
  --manifest data/raw/omnidocbench_subset/manifest.csv \
  --output-dir data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100

python -m src.ingestion.parsing.lift.convert_results_to_markdown \
  data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100 \
  data/processed/lift_outputs/omnidocbench_subset_markdown_schema_100_mds
```

## Does It Call The `lift` Repo?

No. AXIOM calls the hosted API directly with:

```python
from datalab_sdk import DatalabClient, ExtractOptions
```

The sibling `lift` repo was used as a reference for API usage. It is not imported
or executed by AXIOM.

## Run Parsing Only

Run this from the project root:

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD
python -m src.ingestion.parsing.run_parsing --config configs/pipeline.yaml
```

Parsing-only artifacts are written to:

```text
data/processed/
├── data_objects.json
├── parsed_data.json
└── initial_schemas.json
```

## Run A Smaller Input Folder

For safer tests, create a tiny folder under `data/raw` and point parsing at it:

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD

python -m src.ingestion.parsing.run_parsing \
  --config configs/pipeline.yaml \
  --input-dir data/raw/smoke \
  --output-dir data/processed/smoke
```

## Run Full Pipeline

After parsing works, run the whole AXIOM pipeline:

```bash
cd /home/halinh/DE_R&D/AXIOM_DE-RD
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Full pipeline outputs are written to:

```text
data/processed/   # parsing stage
data/cleaned/     # cleaning stage
data/enriched/    # enrichment stage
data/output/      # catalog/index/integration/system outputs
```
