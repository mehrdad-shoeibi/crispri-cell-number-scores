"""Tests of the value-by-value comparison used to check regenerated tables."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_regeneration.py"


def _load():
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("check_regeneration", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


check = _load()


def test_identical_tables_pass_with_renamed_and_added_columns():
    old = pd.DataFrame({"n": [1, 2], "rho": [0.5, np.nan]})
    new = pd.DataFrame({"n_rows": [1, 2], "rho": [0.5, np.nan], "extra": ["a", "b"]})
    report = check.compare_tables(old, new, {"n": "n_rows"})
    assert report["all_values_identical"] and report["all_values_within_tolerance"]
    assert report["renamed"] == {"n": "n_rows"}
    assert report["added"] == ["extra"]


def test_a_small_difference_fails_exactly_and_passes_within_its_tolerance():
    old = pd.DataFrame({"y": [0.1, 0.2, 0.3]})
    new = pd.DataFrame({"y": [0.1, np.nextafter(0.2, 1.0), 0.3]})
    exact = check.compare_tables(old, new, {})
    assert not exact["all_values_identical"]
    assert not exact["all_values_within_tolerance"]
    assert exact["columns"]["y"]["n_values_differing"] == 1
    tolerant = check.compare_tables(old, new, {}, tolerance=1e-12)
    assert not tolerant["all_values_identical"]
    assert tolerant["all_values_within_tolerance"]


def test_a_changed_missing_value_is_never_within_tolerance():
    old = pd.DataFrame({"y": [0.1, np.nan]})
    new = pd.DataFrame({"y": [0.1, 0.0]})
    report = check.compare_tables(old, new, {}, tolerance=1.0)
    assert not report["columns"]["y"]["missing_pattern_identical"]
    assert not report["all_values_within_tolerance"]


def test_a_change_in_row_count_fails():
    report = check.compare_tables(pd.DataFrame({"y": [1.0]}), pd.DataFrame({"y": [1.0, 2.0]}), {}, tolerance=1.0)
    assert not report["all_values_within_tolerance"]
