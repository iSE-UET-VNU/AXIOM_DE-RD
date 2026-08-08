"""Benchmark adapters. One module per dataset, all satisfying ``base.Benchmark``."""

from __future__ import annotations

from typing import Any

from .base import Benchmark, GoldSpec, Question, Region, SourceDoc, normalize_answer_type

_ADAPTERS: dict[str, str] = {
    "ise": "IseChallenge",
    "mmdocir": "MMDocIR",
    "vidore_v3": "ViDoreV3",
}


def load(name: str, **kwargs: Any) -> Benchmark:
    if name not in _ADAPTERS:
        raise ValueError(f"Unknown benchmark {name!r}. Known: {sorted(_ADAPTERS)}")
    module = __import__(f"src.evaluation.benchmarks.{name}", fromlist=[_ADAPTERS[name]])
    return getattr(module, _ADAPTERS[name])(**kwargs)


__all__ = [
    "Benchmark", "GoldSpec", "Question", "Region", "SourceDoc",
    "normalize_answer_type", "load",
]
