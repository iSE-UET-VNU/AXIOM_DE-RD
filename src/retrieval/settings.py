
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = PROJECT_ROOT / "src" / "retrieval" / "artifacts"

VECTOR_SEARCH_PATH = "/api/v1/retrieval/vector-search"
KEYWORD_SEARCH_PATH = "/api/v1/retrieval/keyword-search"
CONTEXT_PATH = "/api/v1/retrieval/context"
NEIGHBOR_CHUNKS_PATH = "/api/v1/retrieval/neighbor-chunks"
INGESTED_DATA_PATH = "/api/v1/documents/ingested-data"


ENHANCEABLE = (VECTOR_SEARCH_PATH, KEYWORD_SEARCH_PATH, CONTEXT_PATH)

CORPUS_MAX_TOP_K = 100


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _paths(raw: str) -> frozenset[str]:
   
    raise NotImplementedError


@dataclass(frozen=True)
class Settings:
    # --- Serving -----------------------------------------------------------
    port: int = int(_env("RETRIEVAL_PORT", "8081"))

    # --- Upstream ----------------------------------------------------------
    # corpus-service owns vector storage and ANN search. We are read-only.
    corpus_service_url: str = _env("CORPUS_SERVICE_URL", "http://corpus-service:8002")
    corpus_timeout_s: float = float(_env("CORPUS_TIMEOUT_S", "30"))

   
    enhance: frozenset[str] = field(default_factory=lambda: _paths(_env("RETRIEVAL_ENHANCE", "")))

    # --- Auth --------------------------------------------------------------
   
    require_token: bool = _flag("RETRIEVAL_REQUIRE_TOKEN")
    service_token: str = _env("RETRIEVAL_SERVICE_TOKEN", "")

    # --- Retrieval depths --------------------------------------------------
   
    k1_dense: int = int(_env("RETRIEVAL_K1", "200"))
    k2_sparse: int = int(_env("RETRIEVAL_K2", "200"))
    k3_rerank: int = int(_env("RETRIEVAL_K3", "20"))

   
    alpha: float = float(_env("RETRIEVAL_ALPHA", "0.7"))

    # --- Reranking ---------------------------------------------------------
    
    reranker: str = _env("RETRIEVAL_RERANKER", "none")
    rerank_timeout_s: float = float(_env("RETRIEVAL_RERANK_TIMEOUT_S", "15"))
    model_service_url: str = _env("AXIOM_MODEL_SERVICE_URL", "http://model-service:8006/api/v1")
    rerank_model: str = _env("RETRIEVAL_RERANK_MODEL", "cohere-rerank")
    rerank_llm_model: str = _env("RETRIEVAL_RERANK_LLM_MODEL", "openrouter-llm")

    # --- Artifacts ---------------------------------------------------------
    artifact_dir: Path = Path(_env("RETRIEVAL_ARTIFACT_DIR", str(ARTIFACT_ROOT)))
   
    config_hash: str = _env("RETRIEVAL_CONFIG_HASH", "")

    fail_on_stale: bool = _flag("RETRIEVAL_FAIL_ON_STALE")

    # --- Response ----------------------------------------------------------
    
    expose_scores: bool = _flag("RETRIEVAL_EXPOSE_SCORES", default=True)

    def enhances(self, path: str) -> bool:
       
        raise NotImplementedError


def load_settings() -> Settings:
    raise NotImplementedError
