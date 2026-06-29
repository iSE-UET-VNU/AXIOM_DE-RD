# AXIOM_DE-RD

AXIOM_DE-RD is an early-stage data pipeline for document ingestion and indexing.
The current workflow focuses on stable artifacts for downstream retrieval and
analytics systems.

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
   - local JSON artifacts
   - Milvus vector collection for text chunks
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

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate axiom-de-rd
```

The pipeline loads `.env` from the project root automatically. Required values
for the current full workflow:

```dotenv
DATALAB_API_KEY=
OPENROUTER_API_KEY=
MILVUS_URI=
MILVUS_TOKEN=
```

## Run

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

Outputs are written to:

```text
data/processed/final_test
data/cleaned/final_test
data/enriched/final_test
data/output/final_test
```

Important output artifacts:

```text
data/documents.json
data/metadata_catalog.json
data/schemas.json
data/index_records.json
data/vector_records.json
reports/embedding_report.json
reports/index_quality_report.json
reports/integration_updates.json
reports/pipeline_state.json
reports/vector_db_report.json
```

Schema payloads are stored with their stage outputs:

```text
data/processed/final_test/initial_schemas.json
data/cleaned/final_test/cleaned_schemas.json
data/enriched/final_test/enriched_schemas.json
```

`data/output/final_test/data/schemas.json` is only a lightweight registry that
points to those files. `pipeline_state.json` is also a lightweight manifest with
run metadata, artifact paths, counts, and report statuses; it does not duplicate
records or embeddings.

`data/output/final_test/data/documents.json` is the document-centric artifact. It
stores normalized document data and elements such as main text, tables, figures,
formulas, and text chunks. `index_records.json` is the indexing contract built
from those document components.

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
figure
catalog
```

Only `text_chunk` records are embedded in this phase. Embeddings use OpenRouter
with `openai/text-embedding-3-small` and dimension `1536`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Unit tests use deterministic local embeddings and mock vector storage where
needed. The production pipeline config uses OpenRouter and Milvus.
