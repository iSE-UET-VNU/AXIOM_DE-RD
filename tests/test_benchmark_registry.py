"""Every registered adapter must actually import.

``benchmarks.load`` resolves adapters by string, so a package rename leaves the
registry syntactically fine and failing only when a run reaches it. Nothing else
in the suite imports all three by name.
"""

from __future__ import annotations

import importlib

import pytest

from src.evaluation.benchmarks import _ADAPTERS, Benchmark, load


def test_the_registry_lists_the_three_adapters():
    assert set(_ADAPTERS) == {"ise", "mmdocir", "vidore_v3"}


@pytest.mark.parametrize("name", sorted(_ADAPTERS))
def test_each_adapter_module_imports(name):
    module = importlib.import_module(f"src.evaluation.benchmarks.{name}")
    assert hasattr(module, _ADAPTERS[name]), f"{name} lacks {_ADAPTERS[name]}"


@pytest.mark.parametrize("name", sorted(_ADAPTERS))
def test_each_adapter_class_satisfies_the_protocol(name):
    module = importlib.import_module(f"src.evaluation.benchmarks.{name}")
    adapter = getattr(module, _ADAPTERS[name])
    for method in ("corpus", "questions", "gold_docs", "gold_pages", "gold_regions"):
        assert callable(getattr(adapter, method, None)), f"{name}.{method} missing"


def test_an_unknown_benchmark_is_refused_by_name():
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load("not_a_benchmark")


def test_the_refusal_lists_what_is_available():
    with pytest.raises(ValueError) as error:
        load("nope")
    assert "vidore_v3" in str(error.value)


def test_load_reaches_the_module_not_just_the_registry():
    """A stale import path passes the registry check and dies here."""
    with pytest.raises((TypeError, ValueError, FileNotFoundError, KeyError)):
        load("vidore_v3")  # missing required subset/language, not ModuleNotFoundError


def test_the_protocol_is_exported():
    assert Benchmark is not None
