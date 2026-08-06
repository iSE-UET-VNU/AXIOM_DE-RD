
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
    """Parse RETRIEVAL_ENHANCE into a set of enhanced paths.

    Accepts full paths or their last segment, so both of these are the same:

        RETRIEVAL_ENHANCE=/api/v1/retrieval/context,keyword-search
        RETRIEVAL_ENHANCE=context,keyword-search
    """
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    if not entries:
        return frozenset()
    if len(entries) == 1 and entries[0].lower() == "all":
        return frozenset(ENHANCEABLE)

    by_suffix = {path.rsplit("/", 1)[-1]: path for path in ENHANCEABLE}
    resolved: set[str] = set()
    for entry in entries:
        if entry in ENHANCEABLE:
            resolved.add(entry)
        elif entry in by_suffix:
            resolved.add(by_suffix[entry])
        else:
            raise ValueError(
                f"RETRIEVAL_ENHANCE names an unknown path {entry!r}. "
                f"Valid: {'all'!r}, {sorted(by_suffix)}, or a full path."
            )
    return frozenset(resolved)


@dataclass(frozen=True)
class Settings:

    # --- Serving -----------------------------------------------------------
    port: int = field(default_factory=lambda: int(_env("RETRIEVAL_PORT", "8081")))

    # --- Upstream ----------------------------------------------------------
    # corpus-service owns vector storage and ANN search. We are read-only.
    corpus_service_url: str = field(
        default_factory=lambda: _env("CORPUS_SERVICE_URL", "http://corpus-service:8002")
    )
    corpus_timeout_s: float = field(default_factory=lambda: float(_env("CORPUS_TIMEOUT_S", "30")))

    enhance: frozenset[str] = field(default_factory=lambda: _paths(_env("RETRIEVAL_ENHANCE", "")))

    # --- Auth --------------------------------------------------------------
    require_token: bool = field(default_factory=lambda: _flag("RETRIEVAL_REQUIRE_TOKEN"))
    service_token: str = field(default_factory=lambda: _env("RETRIEVAL_SERVICE_TOKEN", ""))

    # --- Retrieval depths --------------------------------------------------
    k1_dense: int = field(default_factory=lambda: int(_env("RETRIEVAL_K1", "200")))
    k2_sparse: int = field(default_factory=lambda: int(_env("RETRIEVAL_K2", "200")))
    k3_rerank: int = field(default_factory=lambda: int(_env("RETRIEVAL_K3", "20")))

    alpha: float = field(default_factory=lambda: float(_env("RETRIEVAL_ALPHA", "0.7")))

    # --- Reranking ---------------------------------------------------------
    reranker: str = field(default_factory=lambda: _env("RETRIEVAL_RERANKER", "none"))
    rerank_timeout_s: float = field(
        default_factory=lambda: float(_env("RETRIEVAL_RERANK_TIMEOUT_S", "15"))
    )
    model_service_url: str = field(
        default_factory=lambda: _env(
            "AXIOM_MODEL_SERVICE_URL", "http://model-service:8006/api/v1"
        )
    )
    rerank_model: str = field(
        default_factory=lambda: _env("RETRIEVAL_RERANK_MODEL", "cohere-rerank")
    )
    rerank_llm_model: str = field(
        default_factory=lambda: _env("RETRIEVAL_RERANK_LLM_MODEL", "openrouter-llm")
    )

    # --- Artifacts ---------------------------------------------------------
    artifact_dir: Path = field(
        default_factory=lambda: Path(_env("RETRIEVAL_ARTIFACT_DIR", str(ARTIFACT_ROOT)))
    )
    config_hash: str = field(default_factory=lambda: _env("RETRIEVAL_CONFIG_HASH", ""))

    fail_on_stale: bool = field(default_factory=lambda: _flag("RETRIEVAL_FAIL_ON_STALE"))

    # --- Response ----------------------------------------------------------
    expose_scores: bool = field(
        default_factory=lambda: _flag("RETRIEVAL_EXPOSE_SCORES", default=True)
    )

    def enhances(self, path: str) -> bool:

        return path in ENHANCEABLE and path in self.enhance

    @property
    def passthrough_only(self) -> bool:
        return not self.enhance


def load_settings() -> Settings:
    settings = Settings()

    if settings.require_token and not settings.service_token:
        raise ValueError(
            "RETRIEVAL_REQUIRE_TOKEN is set but RETRIEVAL_SERVICE_TOKEN is empty. "
            "Every request would 401, which reads as an outage, not a misconfiguration."
        )
    if not 0.0 <= settings.alpha <= 1.0:
        raise ValueError(f"RETRIEVAL_ALPHA must be in [0,1]; got {settings.alpha}.")
    for name, value in (("RETRIEVAL_K1", settings.k1_dense),
                        ("RETRIEVAL_K2", settings.k2_sparse),
                        ("RETRIEVAL_K3", settings.k3_rerank)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1; got {value}.")
    if settings.k3_rerank > min(settings.k1_dense, settings.k2_sparse):
        raise ValueError(
            f"RETRIEVAL_K3 ({settings.k3_rerank}) exceeds the retrieval depth it "
            f"reranks (k1={settings.k1_dense}, k2={settings.k2_sparse}). A reranker "
            "may only reorder what retrieval surfaced."
        )
    if settings.reranker not in {"none", "gateway", "llm"}:
        raise ValueError(
            f"RETRIEVAL_RERANKER must be none|gateway|llm; got {settings.reranker!r}."
        )
    return settings
