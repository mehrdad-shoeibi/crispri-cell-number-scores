"""Tests of the E1 label builder against the retained formulas it must reproduce."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import anderson_ksamp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import endpoint  # noqa: E402


def _counts(seed: int = 1, cells: int = 300, genes: int = 400) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = rng.poisson(0.4, (cells, genes)).astype(np.float32)
    block[5] = 0.0  # an empty cell
    return block


def test_external_values_match_the_retained_external_builder():
    block = _counts()
    columns = np.array([0, 7, 399])
    # run_replogle_endpoint.py:111-116: single-precision totals, double-precision values, stored single.
    totals = np.maximum(block.sum(axis=1), 1.0)
    expected = np.log1p(block[:, columns].astype(np.float64) / totals[:, None] * 1e4).astype(np.float32)
    got = endpoint.external_values(block[:, columns], endpoint.dense_totals(block))
    assert got.dtype == np.float32
    assert np.array_equal(got, expected)


def test_external_values_one_column_per_cell():
    block = _counts(2)
    own = np.random.default_rng(3).integers(0, block.shape[1], block.shape[0])
    rows = np.arange(block.shape[0])
    totals = np.maximum(block.sum(axis=1), 1.0)
    expected = np.log1p(block[rows, own].astype(np.float64) / totals * 1e4).astype(np.float32)
    assert np.array_equal(endpoint.external_values(block[rows, own], endpoint.dense_totals(block)), expected)


def test_source_values_match_the_retained_vcc_loader():
    block = _counts(4)
    # src/data/vcc/loader.py:130-143, every step in single precision.
    x = np.asarray(block, dtype=np.float32)
    sums = x.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1.0, 1.0, sums)
    expected = np.log1p(x * (np.float32(1e4) / sums), dtype=np.float32)
    columns = np.array([3, 11, 250])
    got = endpoint.source_values(block[:, columns], endpoint.dense_totals(block))
    assert got.dtype == np.float32
    assert np.array_equal(got, expected[:, columns])


def test_dense_totals_refuses_double_precision():
    with pytest.raises(TypeError):
        endpoint.dense_totals(np.ones((2, 3)))


def test_ad_log1p_matches_scipy_and_handles_degenerate_samples():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.5, 1, 27), rng.normal(0, 1, 788)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = anderson_ksamp([a, b]).statistic
    assert endpoint.ad_log1p(a, b) == pytest.approx(np.log1p(max(0.0, raw)), abs=0)
    assert np.isnan(endpoint.ad_log1p([1.0], b))
    assert np.isnan(endpoint.ad_log1p(np.zeros(5), np.zeros(9)))  # one distinct value in the pool


def test_ad_log1p_is_clamped_at_zero():
    same = np.linspace(0, 1, 50)
    assert endpoint.ad_log1p(same, same) == 0.0


def test_draw_order_perturbed_first_then_controls():
    first, second = endpoint.draw(20260801, 40, 27, 1000, 788)
    rng = np.random.default_rng(20260801)
    assert np.array_equal(first, rng.choice(40, size=27, replace=False))
    assert np.array_equal(second, rng.choice(1000, size=788, replace=False))


def test_control_draw_is_shared_by_pools_larger_than_n_pert():
    for seed in endpoint.SEEDS[:3]:
        reference = endpoint.draw(seed, 28, 27, 11485, 788)[1]
        for pool in (29, 40, 113, 900):
            assert np.array_equal(endpoint.draw(seed, pool, 27, 11485, 788)[1], reference)


def test_summarise_draws_applies_the_valid_draw_rule():
    draws = np.r_[np.arange(19, dtype=float), [np.nan] * 6]
    summary = endpoint.summarise_draws(draws)
    assert summary["n_valid_draws"] == 19
    assert np.isnan(summary["y"])
    assert summary["y_median_valid_draws"] == 9.0
    summary = endpoint.summarise_draws(np.r_[np.arange(20, dtype=float), [np.nan] * 5])
    assert summary["y"] == 9.5


def test_subsampled_label_reproduces_the_retained_resampler():
    rng = np.random.default_rng(6)
    pool = rng.normal(1.0, 1.0, 60).astype(np.float32)
    controls = rng.normal(0.0, 1.0, 2000).astype(np.float32)
    expected = []
    for seed in endpoint.SEEDS:  # run_replogle_endpoint.py:136-144
        g = np.random.default_rng(seed)
        pi = g.choice(pool.size, size=27, replace=False)
        ni = g.choice(controls.size, size=788, replace=False)
        expected.append(endpoint.ad_log1p(pool[pi], controls[ni]))
    summary, draws = endpoint.subsampled_label(pool, controls, 27, 788)
    assert np.array_equal(draws, np.asarray(expected))
    assert summary["y"] == float(np.median(expected))


def test_null_draws_take_slots_first_then_controls_outside_the_slots():
    slot_pool = np.arange(100, 180)
    draws = endpoint.null_draws(500, slot_pool, 10, 300, seeds=(7, 8))
    for (slots, controls), seed in zip(draws, (7, 8)):
        rng = np.random.default_rng(seed)
        expected_slots = slot_pool[rng.choice(slot_pool.size, size=10, replace=False)]
        rest = np.delete(np.arange(500), expected_slots)
        expected_controls = rest[rng.choice(rest.size, size=300, replace=False)]
        assert np.array_equal(slots, expected_slots)
        assert np.array_equal(controls, expected_controls)
        assert not set(slots.tolist()) & set(controls.tolist())


def test_null_draws_share_the_control_positions_across_slot_pools_of_any_size():
    first = endpoint.null_draws(2000, np.arange(0, 707), 27, 788, seeds=(20260801,))[0]
    second = endpoint.null_draws(2000, np.arange(900, 1797), 27, 788, seeds=(20260801,))[0]
    # The same positions are drawn from the remaining cells, which differ by the slot cells only.
    rest_first = np.delete(np.arange(2000), first[0])
    rest_second = np.delete(np.arange(2000), second[0])
    assert np.array_equal(np.searchsorted(rest_first, first[1]), np.searchsorted(rest_second, second[1]))


def test_null_draws_refuse_a_slot_pool_smaller_than_the_slots():
    with pytest.raises(ValueError):
        endpoint.null_draws(100, np.arange(5), 10, 50)


def test_null_label_is_zero_when_slots_and_controls_come_from_one_constant_population():
    values = np.r_[np.zeros(400), np.ones(400)].astype(np.float32)
    draws = endpoint.null_draws(800, np.arange(800), 27, 500, seeds=endpoint.SEEDS)
    summary = endpoint.null_label(values, draws)
    assert summary["n_valid_draws"] == 25
    assert summary["y"] >= 0.0


def test_subsampled_labels_share_the_draws_across_modes():
    rng = np.random.default_rng(11)
    pool = rng.normal(1.0, 1.0, 40).astype(np.float32)
    controls = rng.normal(0.0, 1.0, 900).astype(np.float32)
    (first, draws_first), (second, draws_second) = endpoint.subsampled_labels(
        [pool, pool * 2], [controls, controls * 2], 27, 788
    )
    # A monotone transform of both samples leaves a rank statistic unchanged draw by draw.
    assert np.array_equal(draws_first, draws_second)
    single, draws = endpoint.subsampled_label(pool, controls, 27, 788)
    assert np.array_equal(draws, draws_first) and single == first


def test_subsampled_labels_refuse_modes_of_different_cells():
    with pytest.raises(ValueError):
        endpoint.subsampled_labels([np.ones(30), np.ones(31)], [np.ones(800), np.ones(800)], 27, 788)


def test_own_summaries_follow_their_definitions():
    pert = np.array([0.0, 0.0, 1.0, 2.0])
    ctrl = np.array([1.0, 1.0, 2.0, 2.0, 0.0])
    b = endpoint.own_summaries(pert, ctrl)
    assert b["B0"] == pytest.approx(np.log1p(4))
    assert b["B1"] == pytest.approx(1.2)
    assert b["B2"] == pytest.approx(1.2 - 0.75)
    assert b["B3"] == pytest.approx(0.5 - 0.8)
    assert b["B4"] == pytest.approx(np.log((pert.std(ddof=1) + 0.01) / (ctrl.std(ddof=1) + 0.01)))


def test_half_split_gives_two_disjoint_sorted_halves():
    a, b = endpoint.half_split(11, [20270407, 3])
    assert a.size == 5 and b.size == 6
    assert np.array_equal(np.sort(np.r_[a, b]), np.arange(11))
    assert np.array_equal(a, np.sort(a)) and np.array_equal(b, np.sort(b))
    again = endpoint.half_split(11, [20270407, 3])
    assert np.array_equal(again[0], a)


def test_thin_counts_keeps_cells_with_p_one_and_draws_columns_before_rests():
    counts = np.array([[3.0, 0.0, 5.0], [2.0, 4.0, 1.0]], dtype=np.float32)
    rest = np.array([10.0, 20.0])
    kept, total = endpoint.thin_counts(np.random.default_rng(0), counts, rest, np.array([1.0, 1.0]))
    assert np.array_equal(kept, counts) and np.array_equal(total, np.array([18.0, 27.0], dtype=np.float32))
    rng = np.random.default_rng(5)
    thinned, total = endpoint.thin_counts(rng, counts, rest, np.array([0.5, 0.25]))
    reference = np.random.default_rng(5)
    expected = reference.binomial(np.rint(counts).astype(np.int64), np.array([0.5, 0.25])[:, None])
    expected_rest = reference.binomial(np.rint(rest).astype(np.int64), np.array([0.5, 0.25]))
    assert np.array_equal(thinned, expected.astype(np.float32))
    assert np.array_equal(total, (expected.sum(axis=1) + expected_rest).astype(np.float32))
