"""A partial arm set must not read as a completed sweep."""

from __future__ import annotations

import pytest

from src.evaluation.compare_arms import check_arm_sets


def arms(spec: dict[str, list[str]]) -> dict[str, dict]:
    return {f"{index}::{arm}": {} for index, names in spec.items() for arm in names}


def test_identical_arm_sets_pass():
    check_arm_sets(arms({"hr": ["bm25", "dense", "rrf"],
                         "finance_en": ["bm25", "dense", "rrf"]}))


def test_a_missing_arm_is_refused():
    """The exact shape the embedder timeout produced: bm25 everywhere, dense nowhere."""
    with pytest.raises(SystemExit, match="dense"):
        check_arm_sets(arms({"hr": ["bm25", "dense", "rrf"],
                             "finance_en": ["bm25"]}))


def test_the_refusal_names_the_index_and_the_arms():
    with pytest.raises(SystemExit) as error:
        check_arm_sets(arms({"hr": ["bm25", "dense"], "industrial": ["bm25"]}))
    assert "industrial" in str(error.value)


def test_a_single_index_is_not_constrained():
    """One index with one arm is a valid, if uninteresting, report."""
    check_arm_sets(arms({"hr": ["bm25"]}))


def test_every_index_missing_the_same_arm_is_fine():
    """A bm25-only sweep is legitimate as long as it is uniform."""
    check_arm_sets(arms({"hr": ["bm25"], "finance_en": ["bm25"]}))
