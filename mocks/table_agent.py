"""Mock TableAgent — stands in for the spreadsheet team's service.

Implements just enough of ``POST /v1/jobs/upload`` for ``src/table_agent`` to
normalize the response: a ``structures`` list whose entries are ``status:
"good"``, plus optional retrieval items. Embeddings are emitted with the same
model and dimension the pipeline uses, because a mismatch there is the failure
this stub exists to catch early.

    uvicorn mocks.table_agent:app --port 8001
"""

from __future__ import annotations

from typing import Any
import hashlib
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

MODEL = os.getenv("MOCK_EMBEDDING_MODEL", "openai/text-embedding-3-small")
DIMENSION = int(os.getenv("MOCK_EMBEDDING_DIMENSION", "1536"))
EMBED_DIMENSION_SHORT = 8

app = FastAPI(title="Mock TableAgent", version="1.0.0")


def _values(text: str, dimension: int) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [((digest[i % len(digest)] / 127.5) - 1.0) for i in range(dimension)]


@app.post("/v1/jobs/upload")
async def upload(
    files: UploadFile = File(...),
    payload: str = Form(default="{}"),
) -> dict[str, Any]:
    try:
        options = json.loads(payload or "{}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"payload is not JSON: {exc}") from None

    stage = options.get("stage", "structure")
    if stage != "structure":
        raise HTTPException(status_code=400, detail=f"mock supports stage=structure, got {stage!r}")

    name = files.filename or "workbook.xlsx"
    body = await files.read()
    job_id = hashlib.sha1(body or name.encode()).hexdigest()[:16]
    sheet = "Sheet1"

    structure = {
        "workbook": name,
        "sheet": sheet,
        "status": "good",
        "structure": {
            "tables": [
                {
                    "range": "A1:C4",
                    "header_rows": 1,
                    "columns": ["Year", "Imports", "Exports"],
                    "row_count": 3,
                }
            ]
        },
    }

    items: list[dict[str, Any]] = []
    if options.get("embed"):
        for index, text in enumerate(
            [f"{name} {sheet} columns Year Imports Exports", f"{name} {sheet} rows 2016 2017 2018"]
        ):
            items.append(
                {
                    "item_id": f"{job_id}-{index}",
                    "type": "table",
                    "retrieval_level": "table" if index else "sheet",
                    "position": {"index": index},
                    "content": {"text": text},
                    "embeddings": [
                        {"model": MODEL, "dimension": DIMENSION, "values": _values(text, DIMENSION)}
                    ],
                }
            )

    return {
        "job_id": job_id,
        "structures": [structure],
        "schema_artifacts": {"schema.yaml": f"workbook: {name}\nsheets:\n  - {sheet}\n"},
        "metadata_artifacts": {"metadata.json": {"workbook": name, "sheets": [sheet]}},
        "retrieval_items": items,
        "answers": [],
    }


@app.get("/health/ready")
def ready() -> dict[str, str]:
    return {"status": "ok", "model": MODEL, "dimension": str(DIMENSION)}
