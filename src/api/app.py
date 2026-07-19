"""FastAPI application exposing the existing pipeline as a REST service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from ..pipeline import run_pipeline
from .output import build_dataeng_output
from .schemas import DataEngRequest, DataEngResponse, HealthResponse


app = FastAPI(
    title="AXIOM Data Engineering API",
    version="1.0.0",
)


@app.post("/v1/dataeng", response_model=DataEngResponse)
def dataeng(request: DataEngRequest) -> dict:
    """Run the current pipeline for every presigned object in the request."""
    try:
        state = run_pipeline(presigned_inventory=request.to_inventory())
        return build_dataeng_output(state)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health/live", response_model=HealthResponse)
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/ready", response_model=HealthResponse)
def ready() -> dict[str, str]:
    return {"status": "ok"}

