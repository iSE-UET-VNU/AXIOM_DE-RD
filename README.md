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
│   ├── processed/            # Local pipeline artifacts
│   ├── cleaned/              # Future cleaned outputs
│   └── enriched/             # Future enriched outputs
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
-> ingestion: parse files and infer initial schemas
-> cleaning: standardize fields, process basic errors, impute missing values
-> enrichment: annotate and profile cleaned data
-> indexing_cataloging: create metadata, DB, vector, and graph index records
-> integration: match schemas/entities and extract relationships
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

Artifacts are written to:

```text
data/processed/artifacts/
```

Current artifact files include parsed data, cleaned data, enriched data,
schemas, metadata catalog records, index records, integration updates, and the
final pipeline state.

## Current Limitations

- Parsing is extension-based and only lightly handles CSV, JSON, and text.
- Binary files are represented as metadata-only placeholders.
- Schema extraction uses simple Python type names.
- Cleaning uses placeholder normalization and constant missing-value handling.
- Enrichment only computes basic annotations and profiles.
- Vector, graph, schema matching, entity matching, and relationship extraction
  are interface placeholders.
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
