from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("run_xgb_hard_rescue_batch_comparison.py")
SPEC = importlib.util.spec_from_file_location("xgb_hard_rescue", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_xgb_mask_has_expected_budget() -> None:
    data = MODULE.load_frozen_data()
    hard, probability = MODULE.xgb_hard_mask(data)
    assert len(hard) == 220
    assert int(hard.sum()) == 134
    assert np.array_equal(hard, probability >= 0.2)


def test_balanced_pilot_has_four_per_source() -> None:
    data = MODULE.load_frozen_data()
    hard, probability = MODULE.xgb_hard_mask(data)
    selected = MODULE.select_pilot(data, hard, probability)
    assert len(selected) == 12
    assert all(hard[index] for index in selected)
    for source in sorted({str(row["source"]) for row in data.rows}):
        local = [index for index in selected if str(data.rows[index]["source"]) == source]
        safe = [
            MODULE.recall_at(data.rows[index], 100) > 0
            and MODULE.recall_at(data.rows[index], 50) == MODULE.recall_at(data.rows[index], 100)
            for index in local
        ]
        assert len(local) == 4
        assert sum(safe) == (4 if source == "ohrbench" else 2)
