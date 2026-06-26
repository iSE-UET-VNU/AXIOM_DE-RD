# Parsing Module

This module converts raw files into AXIOM `ParsedData` and `InitialSchema`
artifacts.

Current behavior:

- CSV and JSON files are parsed into row dictionaries.
- Text-like files are read into a `text` field.
- Images/PDFs and other binary files are represented as metadata-only records.
- The module exposes a stable interface for adding OCR/model adapters later.

## Config

The current parsing config is local-only:

```json
{
  "parsing": {
    "provider": "local"
  }
}
```

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
