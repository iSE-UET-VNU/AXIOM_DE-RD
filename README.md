# AXIOM_DE-RD

AXIOM_DE-RD is an early-stage data pipeline for document ingestion, chunking,
and embedding. The current workflow produces local artifacts for downstream
retrieval and analytics systems; it does not write to a database.

## Current Pipeline

```text
raw document
-> ingestion: parse with Lift API and the document-components schema
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
   - local JSON artifacts, including text chunk embeddings
```

The main config is:

```text
configs/pipeline.yaml
```

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

The pipeline loads `.env` from the project root automatically. Required values
for the current full workflow:

```dotenv
DATALAB_API_KEY=
OPENROUTER_API_KEY=
```

## Run

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Outputs are written to:

```text
data/work/final_test/<run_id>/datalab
data/processed/final_test
data/cleaned/final_test
data/enriched/final_test
data/output/final_test
```

Important artifact directories:

```text
data/work/final_test/<run_id>/datalab/
data/processed/final_test/
data/processed/final_test/normalization/
data/output/final_test/data/
data/output/final_test/reports/
```

Provider responses and decoded assets are retained in run-scoped document
bundles under `data/work`. Canonical ingestion records are JSONL under
`data/processed`; the large in-memory parsing intermediates are not persisted.

The provider extraction contract and stage-specific schemas remain at their
source or stage paths:

```text
src/ingestion/parsing/lift/schemas/document_components.json
data/processed/final_test/manifest.json
data/cleaned/final_test/cleaned_schemas.json
data/enriched/final_test/enriched_schemas.json
```

`data/output/final_test/data/schemas.json` is a self-contained JSON Schema for
one logical document. Its document metadata maps to `documents.jsonl`; its
`texts`, `tables`, `images`, and `formulas` arrays map to the normalized JSONL
files through `document_id`. The schema includes storage mappings, checksums,
record counts, observed types, and dataset-level values without duplicating
document content. `document_components.json` remains only the extraction schema
sent to Datalab.

`pipeline_state.json` is a lightweight manifest with run metadata, artifact
paths, counts, and report statuses; it does not duplicate records or embeddings.

`documents.jsonl` is a document registry containing source/parser metadata and
component counts. Component content lives only in the three normalization JSONL
files. `index_records.json` remains the downstream indexing contract.
`data/output/final_test/data/vector_records.json` is the final embedding output
and includes each vector together with its chunk text and lineage metadata. No
VectorDB or relational/graph database write is performed.

## Contracts

`IndexRecord.index_type` currently supports:

```text
document
text_chunk
table
figure
catalog
```

Only `text_chunk` records are embedded in this phase. Embeddings use OpenRouter
with `openai/text-embedding-3-small` and dimension `1536`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests use deterministic local embeddings. The production pipeline config
uses OpenRouter and writes the generated vectors only to local JSON artifacts.
