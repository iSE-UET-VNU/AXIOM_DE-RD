# AXIOM - Data Engineering R&D Project

AXIOM_DE-RD is an early-stage research scaffold for a modular data-engineering
pipeline. The current implementation focuses on architecture, module
boundaries, data contracts, and local artifact flow rather than production
algorithms or infrastructure.

## Folder Structure

```text
AXIOM_DE-RD/
├── configs/                  # Pipeline and storage configuration
├── data/
│   ├── raw/                  # Input data lake objects
│   ├── processed/            # Parsed data and initial schemas
│   ├── cleaned/              # Cleaned data and schemas
│   ├── enriched/             # Enriched data and schemas
│   └── output/               # System-level catalog/index/integration outputs
├── scripts/
│   └── run_pipeline.py       # CLI entrypoint
└── src/
    ├── cleaning/             # Standardization, error processing, imputation
    ├── enrichment/           # Annotation and profiling
    ├── indexing_cataloging/  # Metadata catalog and index record builders
    ├── ingestion/            # Parsing, format detection, initial schemas
    ├── integration/          # Schema/entity matching and relationships
    ├── storage/              # Centralized local/DB/vector/graph persistence
    ├── utils/                # Shared helpers
    ├── models.py             # Shared dataclass contracts
    ├── pipeline.py           # Orchestrator
    └── main.py               # Thin entrypoint wrapper
```

## Pipeline Workflow

The first workflow is intentionally simple:

```text
raw data
-> ingestion: parse local files or optional Lift API outputs, then infer initial schemas
-> cleaning: pass parsed data through until cleaning logic is implemented
-> enrichment: pass cleaned/parsed data through until enrichment logic is implemented
-> indexing_cataloging: create metadata, DB, vector, and graph index records
-> integration: pass indexing records through until integration logic is implemented
-> storage: write local JSON artifacts
```

Shared intermediate artifacts live in `src/models.py`:

- `DataObject`
- `ParsedData`
- `InitialSchema`
- `CleanedData`
- `CleanedSchema`
- `EnrichedData`
- `EnrichedSchema`
- `MetadataRecord`
- `IndexRecord`
- `SchemaMatch`
- `EntityMatch`
- `RelationshipRecord`
- `PipelineState`

## How To Run

From the project root:

```bash
python scripts/run_pipeline.py --config configs/pipeline.yaml
```

The sample config is in `configs/pipeline.yaml`. It is JSON-compatible YAML so
the scaffold can run without adding dependencies. If `PyYAML` is installed,
regular YAML syntax will also work.

The default parser uses the Lift hosted API for images/PDFs, with local fallback
enabled for unsupported files or missing API setup:

```bash
pip install datalab-python-sdk
export DATALAB_API_KEY="your_api_key_here"
```

Then set this in `configs/pipeline.yaml`:

```json
"parsing": {
  "provider": "lift_api"
}
```

AXIOM calls the hosted API directly through `datalab_sdk`; it does not import or
execute the sibling `lift` repo.

Stage outputs are written to:

```text
data/processed/
data/cleaned/
data/enriched/
```

System-level outputs are written to:

```text
data/output/
```

Current system output files include the combined schema registry, metadata
catalog records, index records, integration updates, and the final pipeline
state.

## Current Limitations

- Parsing defaults to Lift API with local fallback. Real API calls require
  `datalab-python-sdk` and `DATALAB_API_KEY`.
- Binary files are represented as metadata-only placeholders.
- Schema extraction uses simple Python type names.
- Cleaning is currently pass-through.
- Enrichment is currently pass-through.
- Vector and graph indexing are demo records only.
- Integration is currently pass-through from indexing records.
- Storage writes local JSON artifacts; DB, VectorDB, and GraphDB adapters are
  mocks.

## Next Development Plan

1. Replace placeholder parsers with domain-specific parsers.
2. Add validation contracts and error quarantine behavior.
3. Improve schema inference for nested and semi-structured data.
4. Add semantic annotations, profiling metrics, and data-quality scoring.
5. Implement real database, vector database, and graph database adapters under
   `src/storage`.
6. Add tests around module interfaces and end-to-end pipeline state.
