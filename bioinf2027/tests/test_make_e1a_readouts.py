"""Tests of the E1a read-out internals on synthetic tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e1a_readouts as readouts  # noqa: E402


def test_centred_subtracts_the_group_mean():
    out = readouts.centred(np.array([1.0, 3.0, 10.0, 14.0]), np.array(["a", "a", "b", "b"]))
    assert out.tolist() == [-1.0, 1.0, -2.0, 2.0]


def test_icc1_is_high_for_grouped_values_and_low_for_noise():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(40), 4)
    grouped = np.repeat(rng.normal(0, 3, 40), 4) + rng.normal(0, 0.3, groups.size)
    noise = rng.normal(0, 1, groups.size)
    assert readouts._icc1(grouped, groups) > 0.9
    assert abs(readouts._icc1(noise, groups)) < 0.2


def test_residuals_are_out_of_fold_and_grouped_by_gene():
    rng = np.random.default_rng(1)
    n = 300
    frame = pd.DataFrame({name: rng.normal(size=n) for name in readouts.TRIVIAL})
    frame["gene"] = np.repeat([f"g{i}" for i in range(60)], 5)
    frame["y"] = frame["B1"] + rng.normal(0, 0.1, n)
    res = readouts.residuals(frame, ("B0", "B1", "B2"), "linear")
    assert np.isfinite(res).all()
    # The label is nearly a function of B1, so its rank-scaled residuals are small (a straight line
    # fits the rank-scaled label only roughly, since ranks of a normal variable are sigmoid in it).
    assert np.std(res) < 0.3 * np.std(readouts._rank01(frame["y"].to_numpy()))


def test_r5_reads_both_halves_and_the_gate(monkeypatch):
    rng = np.random.default_rng(2)
    rows = []
    for k in range(120):
        signal = rng.normal()
        for half in ("A", "B"):
            rows.append({"construct": f"c{k}", "target_ensg": f"E{k // 2}", "n_pool": 60, "level": 10,
                         "version": "pooled control halves", "half": half, "y": signal + rng.normal(0, 0.3),
                         **{name: rng.normal() for name in readouts.TRIVIAL}})
    monkeypatch.setitem(readouts.TABLES, "T", {"external_S_halves.csv": pd.DataFrame(rows)})
    out = readouts.task_r5("T", "S")
    assert len(out) == 1 and out[0]["rows"] == 120
    assert out[0]["residual_reliability"] > 0.5 and out[0]["gate_passed"]


def test_r2_returns_the_excess_and_the_stop_rule(monkeypatch):
    rng = np.random.default_rng(3)
    labels, simulation = [], []
    for g in range(30):
        for b in range(6):
            n_pool = int(rng.integers(30, 90))
            y = 0.01 * n_pool + rng.normal(0, 0.2)
            key = {"split": ("train", "val", "test")[g % 3], "batch": f"B{b}", "target_gene": f"G{g}", "n_pool": n_pool}
            labels.append({**key, "existing_label": y + rng.normal(0, 0.1), "y": y, "n_guides": 1, "guides": "['a']"})
            for j, size in enumerate(rng.integers(27, n_pool + 1, size=5)):
                simulation.append({**key, "simulation": j, "size": int(size), "y": y + rng.normal(0, 0.05), "n_valid_draws": 25, "y_mean": y})
    monkeypatch.setattr(readouts, "B", 20)
    monkeypatch.setitem(readouts.TABLES, "T", {"vcc_labels.csv": pd.DataFrame(labels), "pool_size_simulation.csv": pd.DataFrame(simulation)})
    out = readouts.task_r2("T")
    assert set(out["within_and_mechanical"]) == {"train", "val", "test", "all"}
    every = out["within_and_mechanical"]["all"]
    assert np.isfinite(every["excess"]) and every["excess_lo"] <= every["excess_hi"]
    assert out["stop_rule_fires"] == bool(every["excess_lo"] > readouts.R2_LIMIT)
    assert "95 % interval" in out["stop_rule"]


def test_corr_row_drops_missing_values_and_marks_few_clusters():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    row = readouts.corr_row(x, x * 2, np.array(list("aabbcc")), cell="T")
    assert row["n_rows"] == 5 and row["descriptive"] is True
    assert row["spearman"] == pytest.approx(1.0)
