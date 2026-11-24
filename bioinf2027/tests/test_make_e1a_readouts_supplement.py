"""Tests of the E1a read-out supplement on synthetic tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e1a_readouts as readouts  # noqa: E402
import make_e1a_readouts_supplement as supplement  # noqa: E402

from bioinf2027 import endpoint  # noqa: E402


def _r2_tables(seed: int = 3):
    rng = np.random.default_rng(seed)
    labels, simulation = [], []
    for g in range(30):
        for b in range(6):
            n_pool = int(rng.integers(30, 90))
            y = 0.01 * n_pool + rng.normal(0, 0.2)
            key = {"split": ("train", "val", "test")[g % 3], "batch": f"B{b}", "target_gene": f"G{g}", "n_pool": n_pool}
            labels.append({**key, "existing_label": y + rng.normal(0, 0.1), "y": y, "y_mean": y + rng.normal(0, 0.01),
                           "n_guides": 1, "guides": "a"})
            for j, size in enumerate(rng.integers(27, n_pool + 1, size=5)):
                simulation.append({**key, "simulation": j, "size": int(size), "y": y + rng.normal(0, 0.05), "n_valid_draws": 25,
                                   "y_mean": y + rng.normal(0, 0.05)})
    return {"vcc_labels.csv": pd.DataFrame(labels), "pool_size_simulation.csv": pd.DataFrame(simulation)}


def test_r2_within_reproduces_the_first_pass_for_the_median_label(monkeypatch):
    monkeypatch.setattr(readouts, "B", 20)
    monkeypatch.setitem(readouts.TABLES, "T", _r2_tables())
    first = readouts.task_r2("T")["within_and_mechanical"]
    again = supplement.r2_within("T", "y")
    for split in first:
        for key, value in first[split].items():
            assert again[split][key] == pytest.approx(value, nan_ok=True), (split, key)
    other = supplement.r2_within("T", "y_mean")
    assert np.isfinite(other["all"]["excess"])


def test_residuals_with_the_deciding_seed_match_the_readout_residuals():
    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({name: rng.normal(size=n) for name in readouts.TRIVIAL})
    frame["gene"] = np.repeat([f"g{i}" for i in range(80)], 5)
    frame["y"] = frame["B1"] + 0.5 * frame["B2"] ** 2 + rng.normal(0, 0.3, n)
    expected = readouts.residuals(frame, readouts.TRIVIAL, "boosting")
    got, r2 = supplement.residuals_with_seed(frame, readouts.TRIVIAL, readouts.FOLD_SEED)
    assert np.array_equal(expected, got)
    assert 0.3 < r2 < 1.0
    other, _ = supplement.residuals_with_seed(frame, readouts.TRIVIAL, 20270410)
    assert not np.array_equal(other, got)


def test_ceiling_uses_the_same_control_draws_as_the_label():
    rng = np.random.default_rng(2)
    controls = rng.gamma(2.0, 0.5, size=900).astype(np.float32)
    controls[:200] = 0.0
    zeros = np.zeros(60, dtype=np.float32)
    label, _ = endpoint.subsampled_label(zeros, controls, 27, 788)
    assert supplement.ceiling_of(60, controls, 27, 788) == pytest.approx(label["y"])
    knocked_down = (controls[:60] * 0.2).astype(np.float32)
    partial, _ = endpoint.subsampled_label(knocked_down, controls, 27, 788)
    assert supplement.ceiling_of(60, controls, 27, 788) >= partial["y"]


def test_rank_partial_removes_a_shared_covariate():
    rng = np.random.default_rng(4)
    n = 500
    b1 = rng.normal(size=n)
    frame = pd.DataFrame({"B1": b1, "B0": rng.normal(size=n), "y": b1 + rng.normal(0, 0.2, n), "B2": b1 + rng.normal(0, 0.2, n)})
    raw = np.corrcoef(frame["y"], frame["B2"])[0, 1]
    rx, ry = supplement.rank_partial(frame, "y", "B2", ("B1",))
    assert raw > 0.8
    assert abs(np.corrcoef(rx, ry)[0, 1]) < 0.2


def test_replicable_share_and_guide_union():
    assert supplement.replicable_share(0.2, 0.95, 0.8) == pytest.approx(1 - 0.2 * 0.2 / 0.95)
    assert np.isnan(supplement.replicable_share(0.2, 0.0, 0.8))
    rows = pd.DataFrame({"split": ["train"] * 4, "target_gene": ["A", "A", "B", "B"], "guides": ["g1", "g2", "g3", "g3"]})
    assert supplement.guide_union_counts(rows, {"A", "B"}) == 1


def test_ceiling_rows_counts_rows_near_the_ceiling(monkeypatch):
    monkeypatch.setattr(readouts, "B", 20)
    rng = np.random.default_rng(5)
    n = 120
    y = rng.uniform(1, 4, n)
    frame = pd.DataFrame({"y": y, "B2": y + rng.normal(0, 0.1, n), "existing_label": y + rng.normal(0, 0.2, n)})
    ceiling = y + np.where(np.arange(n) < 30, 0.05, 0.5)
    rows = supplement.ceiling_rows("train", frame, ceiling, "T", np.repeat([f"g{i}" for i in range(60)], 2), True)
    assert rows[0]["share_within_0.1_of_ceiling"] == pytest.approx(0.25)
    assert rows[0]["rows_further_than_0.1"] == 90
    assert {r.get("label") for r in rows[1:]} == {"R3 on rows further than 0.1 below the ceiling", "R1 on rows further than 0.1 below the ceiling"}


def test_derived_cells_are_matched_on_their_row_keys(monkeypatch):
    monkeypatch.setattr(readouts, "B", 20)
    rng = np.random.default_rng(6)
    keys = pd.DataFrame({"split": np.repeat(["train", "val", "test"], 40), "batch": np.tile([f"B{i}" for i in range(4)], 30),
                         "target_gene": [f"G{i // 4}" for i in range(120)]})
    base = keys.assign(y=rng.normal(size=120))
    derived = keys.assign(y=base["y"] * 0.5 + 1.0).sample(frac=1.0, random_state=1)
    monkeypatch.setitem(supplement.DERIVED, "G1-DT-X", "G1")
    monkeypatch.setitem(readouts.TABLES, "G1", {"vcc_labels.csv": base})
    monkeypatch.setitem(readouts.TABLES, "G1-DT-X", {"vcc_labels.csv": derived})
    rows = supplement.task_derived("G1-DT-X")
    assert [r["split"] for r in rows] == ["train", "val", "test"]
    assert all(r["spearman"] == pytest.approx(1.0) for r in rows)
