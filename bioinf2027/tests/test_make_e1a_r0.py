"""Tests of R0's statistics and reading and of the builder-check helpers, on synthetic inputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e1a_r0 as r0  # noqa: E402
from bioinf2027 import endpoint  # noqa: E402

B = 200


def _unit(batch_sd: float, seed: int = 0, genes: int = 30, batches: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(genes):
        level = rng.normal(1.0, 0.5)
        pooled = level + rng.normal(0, 0.05)
        for b in range(batches):
            rows.append(
                {
                    "target_gene": f"g{g}",
                    "batch_own_null": level + rng.normal(0, batch_sd) + rng.normal(0, 0.05),
                    "pooled_draw_null": pooled,
                    "B1_pooled_control_mean": level + rng.normal(0, 0.1),
                    "existing_label": rng.normal(2.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def test_component_is_near_zero_without_a_batch_shift_and_grows_with_it():
    small = r0.unit_statistics(_unit(0.0), {"RPE1": 1.0, "K562-essential": 1.0}, b=B)
    large = r0.unit_statistics(_unit(0.8), {"RPE1": 1.0, "K562-essential": 1.0}, b=B)
    assert small["component"] < 0.15
    assert large["component"] > 0.4
    assert large["component_lo"] <= large["component"] <= large["component_hi"]


def test_the_scale_giving_the_largest_component_is_used():
    stats = r0.unit_statistics(_unit(0.5), {"RPE1": 0.2, "K562-essential": 5.0}, b=B)
    assert stats["component_scale_used"] == "27-cell label, RPE1"
    assert stats["component"] == pytest.approx(stats["batch_component_raw"] / 0.2)


def test_abs_d_bounds_follow_the_interval():
    stats = r0.unit_statistics(_unit(0.3), {"RPE1": 1.0, "K562-essential": 1.0}, b=B)
    lo, hi = stats["D_lo"], stats["D_hi"]
    assert stats["abs_D_upper"] == max(abs(lo), abs(hi))
    expected_lower = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    assert stats["abs_D_lower"] == expected_lower


def _units(component_hi, component_lo, d_upper, d_lower, genes=20):
    rows = []
    for level in (10, 27):
        for unit in range(3):
            rows.append({"slots": level, "split": "train", "flex_run": unit + 1, "n_target_genes": genes,
                         "enters_reading": genes >= r0.MIN_GENES_PER_UNIT, "component_hi": component_hi,
                         "component_lo": component_lo, "abs_D_upper": d_upper, "abs_D_lower": d_lower})
    return pd.DataFrame(rows)


def test_reading_favours_the_pooled_pair_only_when_every_bound_is_small():
    assert r0.r0_reading(_units(0.05, 0.0, 0.05, 0.0))["reading"] == "favours the pooled pair"
    assert r0.r0_reading(_units(0.12, 0.0, 0.05, 0.0))["reading"] == "favours neither"


def test_the_batch_local_condition_is_read_first():
    units = _units(0.05, 0.0, 0.05, 0.0)
    units.loc[0, ["component_lo", "component_hi"]] = [0.3, 0.6]  # a 10-slot unit meets the batch-local condition
    assert r0.r0_reading(units)["reading"] == "favours the batch-local pair"


def test_reading_favours_the_batch_local_pair_on_any_large_lower_bound():
    units = _units(0.05, 0.0, 0.05, 0.0)
    units.loc[0, ["component_lo", "component_hi"]] = [0.3, 0.6]  # a 10-slot unit
    assert r0.r0_reading(units)["reading"] == "favours the batch-local pair"
    units = _units(0.2, 0.0, 0.3, 0.26)
    assert r0.r0_reading(units)["reading"] == "favours the batch-local pair"


def test_the_ten_slot_component_does_not_count_for_the_pooled_pair():
    units = _units(0.05, 0.0, 0.05, 0.0)
    units.loc[units["slots"] == 10, "component_hi"] = 0.2  # understated, so it cannot hold the reading back
    assert r0.r0_reading(units)["reading"] == "favours the pooled pair"


def test_small_units_do_not_enter_the_reading():
    units = _units(0.05, 0.0, 0.05, 0.0)
    units.loc[0, ["enters_reading", "component_lo"]] = [False, 0.9]
    reading = r0.r0_reading(units)
    assert reading["reading"] == "favours the pooled pair"
    assert reading["units_entering"] == len(units) - 1


def test_construct_pools_keep_ascending_cell_order():
    screen = {"constructs": np.array(["a", "b", "c"]), "pert_code": np.array([1, 0, 1, 2, 0]),
              "pert_own": np.array([10.0, 20.0, 11.0, 30.0, 21.0], dtype=np.float32)}
    pools = r0.construct_pools(screen)
    assert [p.tolist() for p in pools] == [[20.0, 21.0], [10.0, 11.0], [30.0]]


def test_null_labels_give_one_pooled_value_per_gene_and_one_own_value_per_row():
    rng = np.random.default_rng(3)
    n_pool = 1800
    batches = np.repeat(["Flex_1_01", "Flex_2_01"], n_pool // 2)
    train = {
        "targets": np.array(["A", "B"]),
        "nt_batch": batches,
        "nt_values_ext": rng.poisson(1.5, (n_pool, 2)).astype(np.float32),
    }
    rows = pd.DataFrame({"split": "train", "batch": ["Flex_1_01", "Flex_2_01", "Flex_1_01"],
                         "target_gene": ["A", "A", "B"], "n_perturbed_cells": [30, 12, 40],
                         "flex_run": [1, 2, 1], "label_stored": [1.0, 2.0, 3.0]})
    nulls = r0.null_labels(rows, train)
    assert len(nulls[nulls["slots"] == 10]) == 3 and len(nulls[nulls["slots"] == 27]) == 2
    ten = nulls[nulls["slots"] == 10]
    assert ten.loc[ten["target_gene"] == "A", "pooled_draw_null"].nunique() == 1
    draws = endpoint.null_draws(n_pool, np.flatnonzero(batches == "Flex_1_01"), 10, r0.N_CTRL_POOLED)
    expected = endpoint.null_label(train["nt_values_ext"][:, 0], draws)["y"]
    got = ten.loc[(ten["target_gene"] == "A") & (ten["batch"] == "Flex_1_01"), "batch_own_null"].iloc[0]
    assert got == expected
