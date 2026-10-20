"""Acceptance tests for generated Adaptive-K optimization artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "research" / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from adaptive_k_optimization_core import load_frozen_data, validate_data  # noqa: E402

OUTPUT = ROOT / "data" / "work" / "xgboost_adaptive_k_optimization_20261016"


def test_frozen_data_contract() -> None:
    checks = validate_data(load_frozen_data())
    assert checks["queries_220"]
    assert checks["pages_5091"]
    assert checks["candidates_200_each"]
    assert checks["score_monotone"]
    assert checks["finite_scores"]


def test_frozen_outer_folds_are_unique_and_stratified() -> None:
    with (OUTPUT / "folds.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 220
    assert len({row["qid"] for row in rows}) == 220
    assert {int(row["outer_fold"]) for row in rows} == set(range(5))
    for fold in range(5):
        fold_rows = [row for row in rows if int(row["outer_fold"]) == fold]
        assert len(fold_rows) == 44
        assert {row["source"] for row in fold_rows} == {"ohrbench", "vidore_industrial", "vidore_physics"}


def test_every_oof_file_has_one_prediction_per_query() -> None:
    paths = list(OUTPUT.glob("*/*_oof.csv"))
    assert paths
    for path in paths:
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 220, path
        assert len({row["qid"] for row in rows}) == 220, path
        choices = np.asarray([int(row["chosen_k"]) for row in rows])
        assert np.all((choices >= 1) & (choices <= 200)), path


def test_confusion_matrices_cover_all_queries() -> None:
    for path in (OUTPUT / "classification").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert int(np.asarray(payload["confusion_matrix"]).sum()) == 220, path


def test_primary_neighbor_declares_leakage_control() -> None:
    path = OUTPUT / "neighbor" / "neighbor_score_modality.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "training neighbours only" in payload["leakage_control"]
    assert "diagnostic only" in payload["leakage_control"]


def test_final_claim_obeys_acceptance_rule() -> None:
    payload = json.loads((OUTPUT / "final" / "report.json").read_text(encoding="utf-8"))
    champion = payload["champion"]
    expected = bool(champion and champion["mean_k"] < 60 and champion["macro_recall"] >= .87)
    assert payload["success"] is expected
