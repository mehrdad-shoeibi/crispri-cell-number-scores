"""Unit tests and a synthetic end-to-end check of the diagnostic helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr as _spearmanr


def sps_spearman(a, b):
    return float(_spearmanr(a, b).statistic)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import stats  # noqa: E402
from bioinf2027.features import CLIP_LIMIT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_d1_external import rank_of_target  # noqa: E402


def test_rank_of_target_picks_largest_absolute_value():
    matrix = np.array([[0.1, -5.0, 2.0], [4.0, 1.0, -0.5]])
    # row 0: column 1 is largest in absolute value; row 1: column 0 is.
    assert rank_of_target(matrix, np.array([1, 0])).tolist() == [1.0, 1.0]


def test_rank_of_target_counts_larger_coordinates():
    matrix = np.array([[1.0, -2.0, 3.0, -4.0]])
    # target is column 0, three coordinates are larger, so the rank is four.
    assert rank_of_target(matrix, np.array([0]))[0] == pytest.approx(4.0)


def test_rank_of_target_averages_ties():
    matrix = np.array([[2.0, -2.0, 0.5]])
    # two coordinates tie for largest, so each takes rank 1.5.
    assert rank_of_target(matrix, np.array([0]))[0] == pytest.approx(1.5)


def test_cluster_bootstrap_recovers_a_strong_association():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 3)
    signal = rng.normal(size=60)[groups]
    x = signal + rng.normal(scale=0.05, size=groups.size)
    y = signal + rng.normal(scale=0.05, size=groups.size)
    estimate = stats.cluster_bootstrap_corr(x, y, groups, b=200, seed=1)
    assert estimate.n == groups.size
    assert estimate.n_groups == 60
    assert estimate.spearman > 0.9
    assert estimate.spearman_lo > 0.0
    assert estimate.n_skipped == 0


def test_cluster_bootstrap_interval_covers_zero_for_noise():
    rng = np.random.default_rng(2)
    groups = np.arange(80)
    x = rng.normal(size=80)
    y = rng.normal(size=80)
    estimate = stats.cluster_bootstrap_corr(x, y, groups, b=200, seed=3)
    assert estimate.spearman_lo < 0.0 < estimate.spearman_hi


def test_cluster_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(30), 2)
    x = rng.normal(size=groups.size)
    y = x + rng.normal(scale=0.5, size=groups.size)
    first = stats.cluster_bootstrap_corr(x, y, groups, b=100, seed=7)
    second = stats.cluster_bootstrap_corr(x, y, groups, b=100, seed=7)
    assert first.as_dict() == second.as_dict()


def test_cluster_bootstrap_drops_non_finite_pairs():
    groups = np.arange(10)
    x = np.arange(10, dtype=float)
    y = x.copy()
    y[3] = np.nan
    estimate = stats.cluster_bootstrap_corr(x, y, groups, b=50, seed=5)
    assert estimate.n == 9


def test_cluster_bootstrap_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        stats.cluster_bootstrap_corr(np.zeros(5), np.zeros(4), np.zeros(5))


def test_synthetic_pipeline_detects_a_label_read_from_one_coordinate():
    """A label that is a monotone function of one coordinate must show up.

    This mirrors the diagnostic the scripts run: when the outcome is built from
    a single feature, the rank association with that feature is near one and
    that feature is the largest in absolute value.
    """
    rng = np.random.default_rng(11)
    n, d = 200, 50
    matrix = rng.normal(scale=0.1, size=(n, d))
    target_column = rng.integers(0, d, size=n)
    own = -np.abs(rng.normal(loc=1.5, scale=0.4, size=n))
    matrix[np.arange(n), target_column] = own
    matrix = np.clip(matrix, -CLIP_LIMIT, CLIP_LIMIT)
    label = np.log1p(np.abs(own) * 3.0)

    ranks = rank_of_target(matrix, target_column)
    assert np.mean(ranks <= 1) > 0.9

    estimate = stats.cluster_bootstrap_corr(
        -matrix[np.arange(n), target_column], label, np.arange(n), b=200, seed=13
    )
    assert estimate.spearman > 0.95
    assert estimate.spearman_lo > 0.9


# --------------------------------------------------------------- partial correlation

from bioinf2027.partial import cluster_bootstrap_partial, partial_spearman  # noqa: E402


def test_partial_spearman_removes_a_pure_confound():
    """When the covariate drives both variables, the partial correlation vanishes."""
    rng = np.random.default_rng(21)
    covariate = rng.normal(size=400)
    x = covariate + rng.normal(scale=0.05, size=400)
    y = covariate + rng.normal(scale=0.05, size=400)
    assert partial_spearman(x, y, covariate) == pytest.approx(0.0, abs=0.15)
    assert sps_spearman(x, y) > 0.9


def test_partial_spearman_keeps_a_real_association():
    """A direct link survives adjustment for an unrelated covariate."""
    rng = np.random.default_rng(22)
    signal = rng.normal(size=400)
    x = signal + rng.normal(scale=0.1, size=400)
    y = signal + rng.normal(scale=0.1, size=400)
    covariate = rng.normal(size=400)
    assert partial_spearman(x, y, covariate) > 0.9


def test_partial_spearman_can_uncover_a_masked_association():
    """A covariate pushing the two variables in opposite directions hides a link.

    This is the shape the source endpoint shows: the raw correlation is near
    zero and the adjusted one is strongly positive.
    """
    rng = np.random.default_rng(23)
    covariate = rng.normal(size=600)
    signal = rng.normal(size=600)
    # Equal and opposite covariate loadings cancel in the raw correlation.
    x = signal - covariate
    y = signal + covariate
    raw = sps_spearman(x, y)
    adjusted = partial_spearman(x, y, covariate)
    assert abs(raw) < 0.05
    assert adjusted > 0.8


def test_cluster_bootstrap_partial_is_deterministic_and_bounded():
    rng = np.random.default_rng(24)
    groups = np.repeat(np.arange(50), 3)
    covariate = rng.normal(size=groups.size)
    signal = rng.normal(size=50)[groups]
    x = signal + 0.5 * covariate
    y = signal - 0.5 * covariate
    first = cluster_bootstrap_partial(x, y, covariate, groups, b=200, seed=9)
    second = cluster_bootstrap_partial(x, y, covariate, groups, b=200, seed=9)
    assert first.as_dict() == second.as_dict()
    assert -1.0 <= first.partial_spearman <= 1.0
    assert first.partial_lo <= first.partial_spearman <= first.partial_hi


# ----------------------------------------------------------------- joint bootstrap


def _split_fixture(seed: int = 0, n_groups: int = 120, per_group: int = 3):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    inside = np.repeat(rng.random(n_groups) < 0.3, per_group)
    label = rng.normal(size=groups.size)
    return rng, groups, inside, label


def test_joint_bootstrap_all_subset_matches_the_plain_cluster_bootstrap():
    rng, groups, inside, label = _split_fixture()
    prediction = label + rng.normal(scale=1.0, size=label.size)
    joint = stats.joint_cluster_bootstrap_subsets(
        prediction[:, None], label, groups, inside, {"model": [0]}, b=300, seed=11
    )
    whole = next(r for r in joint if r.subset == "all")
    plain = stats.cluster_bootstrap_corr(prediction, label, groups, b=300, seed=11)
    assert whole.estimate == pytest.approx(plain.spearman, abs=1e-12)
    assert whole.lo == pytest.approx(plain.spearman_lo, abs=1e-12)
    assert whole.hi == pytest.approx(plain.spearman_hi, abs=1e-12)
    assert (whole.n, whole.n_groups) == (plain.n, plain.n_groups)


def test_joint_bootstrap_point_estimates_are_per_subset_spearman():
    rng, groups, inside, label = _split_fixture(1)
    prediction = label + rng.normal(scale=0.7, size=label.size)
    joint = {
        r.subset: r
        for r in stats.joint_cluster_bootstrap_subsets(
            prediction[:, None], label, groups, inside, {"model": [0]}, b=50, seed=3
        )
    }
    assert joint["in"].estimate == pytest.approx(sps_spearman(prediction[inside], label[inside]))
    assert joint["out"].estimate == pytest.approx(sps_spearman(prediction[~inside], label[~inside]))
    assert joint["in_minus_out"].estimate == pytest.approx(
        joint["in"].estimate - joint["out"].estimate
    )
    assert joint["in"].n == int(inside.sum())
    assert joint["out"].n == int((~inside).sum())


def test_joint_bootstrap_detects_signal_confined_to_the_subset():
    rng, groups, inside, label = _split_fixture(2, n_groups=400)
    prediction = np.where(inside, label, 0.0) + rng.normal(scale=0.5, size=label.size)
    joint = {
        r.subset: r
        for r in stats.joint_cluster_bootstrap_subsets(
            prediction[:, None], label, groups, inside, {"model": [0]}, b=400, seed=5
        )
    }
    assert joint["in"].lo > 0.5
    assert joint["out"].lo < 0 < joint["out"].hi
    assert joint["in_minus_out"].lo > 0


def test_joint_bootstrap_difference_covers_zero_when_subsets_behave_alike():
    rng, groups, inside, label = _split_fixture(4, n_groups=400)
    prediction = label + rng.normal(scale=1.0, size=label.size)
    difference = next(
        r
        for r in stats.joint_cluster_bootstrap_subsets(
            prediction[:, None], label, groups, inside, {"model": [0]}, b=400, seed=5
        )
        if r.subset == "in_minus_out"
    )
    assert difference.lo < 0 < difference.hi


def test_joint_bootstrap_family_statistic_is_the_mean_over_its_columns():
    rng, groups, inside, label = _split_fixture(6)
    columns = np.column_stack([label + rng.normal(scale=s, size=label.size) for s in (0.5, 1.0, 2.0)])
    joint = stats.joint_cluster_bootstrap_subsets(
        columns, label, groups, inside, {"family": [0, 1, 2], "first": [0]}, b=30, seed=1
    )
    family = next(r for r in joint if r.family == "family" and r.subset == "all")
    expected = np.mean([sps_spearman(columns[:, j], label) for j in range(3)])
    assert family.estimate == pytest.approx(expected)


def test_joint_bootstrap_is_deterministic_and_counts_undefined_replicates():
    rng, groups, inside, label = _split_fixture(7)
    columns = np.column_stack([label + rng.normal(size=label.size), np.zeros(label.size)])
    kwargs = dict(b=40, seed=9)
    first = stats.joint_cluster_bootstrap_subsets(
        columns, label, groups, inside, {"real": [0], "constant": [1]}, **kwargs
    )
    second = stats.joint_cluster_bootstrap_subsets(
        columns, label, groups, inside, {"real": [0], "constant": [1]}, **kwargs
    )
    assert [r.as_dict() for r in first if r.family == "real"] == [
        r.as_dict() for r in second if r.family == "real"
    ]
    constant = next(r for r in first if r.family == "constant" and r.subset == "all")
    assert np.isnan(constant.estimate) and constant.n_skipped == 40


def test_joint_bootstrap_rejects_a_group_split_between_subsets():
    label = np.arange(6.0)
    groups = np.array([0, 0, 1, 1, 2, 2])
    inside = np.array([True, False, True, True, False, False])
    with pytest.raises(ValueError):
        stats.joint_cluster_bootstrap_subsets(
            label[:, None], label, groups, inside, {"m": [0]}, b=5
        )


def test_table_row_renames_the_sample_sizes_and_keeps_every_other_field():
    estimate = stats.cluster_bootstrap_corr(
        np.arange(12.0), np.arange(12.0) ** 2, np.repeat(np.arange(4), 3), b=20
    )
    row = stats.table_row(estimate)
    assert row["n_rows"] == 12 and row["n_target_genes"] == 4
    assert "n" not in row and "n_groups" not in row
    assert row["spearman"] == estimate.spearman and row["bootstrap_b"] == 20


def test_joint_bootstrap_returns_its_draws_for_derived_statistics():
    rng, groups, inside, label = _split_fixture(8)
    columns = np.column_stack([label + rng.normal(scale=s, size=label.size) for s in (0.5, 1.0)])
    results, point, draws = stats.joint_cluster_bootstrap_subsets(
        columns, label, groups, inside, {"a": [0], "b": [1]}, b=60, seed=2, return_draws=True
    )
    assert point.shape == (2, 4) and draws.shape == (60, 2, 4)
    difference = next(r for r in results if r.family == "a" and r.subset == "in_minus_out")
    lo, hi, skipped = stats.percentile_interval(draws[:, 0, 3])
    assert (lo, hi, skipped) == (difference.lo, difference.hi, difference.n_skipped)
    assert point[0, 3] == pytest.approx(difference.estimate)


def test_paired_screen_difference_uses_shared_groups_and_pairs_the_draws():
    rng = np.random.default_rng(3)
    genes = np.arange(80)
    first_groups = np.repeat(genes, 2)
    second_groups = np.concatenate([np.repeat(genes[:60], 2), np.repeat(np.arange(100, 120), 2)])
    signal_first = rng.normal(size=80)[first_groups]
    signal_second = np.concatenate([rng.normal(size=80)[:60], rng.normal(size=20)])[
        np.searchsorted(np.concatenate([genes[:60], np.arange(100, 120)]), second_groups)
    ]
    first = (signal_first + rng.normal(scale=0.3, size=160), signal_first, first_groups)
    second = (signal_second + rng.normal(scale=2.0, size=160), signal_second, second_groups)
    out = stats.paired_screen_difference(first, second, b=300, seed=4)
    assert out["n_shared_groups"] == 60
    assert out["n_rows_first"] == 120 and out["n_rows_second"] == 120
    shared_first = np.isin(first_groups, genes[:60])
    assert out["first"]["spearman"] == pytest.approx(
        sps_spearman(first[0][shared_first], first[1][shared_first])
    )
    assert out["first_minus_second"]["lo"] > 0


def test_paired_screen_difference_rejects_too_few_shared_groups():
    a = (np.arange(6.0), np.arange(6.0), np.array([0, 0, 1, 1, 2, 2]))
    b = (np.arange(6.0), np.arange(6.0), np.array([2, 2, 3, 3, 4, 4]))
    with pytest.raises(ValueError):
        stats.paired_screen_difference(a, b, b=5)


def test_joint_bootstrap_with_a_covariate_matches_the_partial_module():
    from bioinf2027.partial import cluster_bootstrap_partial, partial_spearman

    rng, groups, inside, label = _split_fixture(9, n_groups=150)
    covariate = label + rng.normal(scale=0.8, size=label.size)
    prediction = covariate + rng.normal(scale=0.6, size=label.size)
    joint = {
        r.subset: r
        for r in stats.joint_cluster_bootstrap_subsets(
            prediction[:, None], label, groups, inside, {"m": [0]}, b=200, seed=13,
            covariate=covariate,
        )
    }
    assert joint["all"].estimate == pytest.approx(partial_spearman(prediction, label, covariate))
    assert joint["in"].estimate == pytest.approx(
        partial_spearman(prediction[inside], label[inside], covariate[inside])
    )
    reference = cluster_bootstrap_partial(prediction, label, covariate, groups, b=200, seed=13)
    assert joint["all"].lo == pytest.approx(reference.partial_lo, abs=1e-12)
    assert joint["all"].hi == pytest.approx(reference.partial_hi, abs=1e-12)


def test_joint_bootstrap_sets_a_constant_family_to_zero_only_when_asked():
    rng, groups, inside, label = _split_fixture(10, n_groups=200)
    coordinate = np.where(inside, label + rng.normal(scale=0.2, size=label.size), 0.0)
    columns = coordinate[:, None]
    plain = {
        r.subset: r
        for r in stats.joint_cluster_bootstrap_subsets(columns, label, groups, inside, {"c": [0]}, b=50, seed=3)
    }
    assert np.isnan(plain["out"].estimate) and plain["out"].n_skipped == 50
    zeroed = {
        r.subset: r
        for r in stats.joint_cluster_bootstrap_subsets(
            columns, label, groups, inside, {"c": [0]}, b=50, seed=3, zero_if_constant={"c"}
        )
    }
    assert zeroed["out"].estimate == 0.0 and zeroed["out"].n_skipped == 0
    assert zeroed["in_minus_out"].estimate == pytest.approx(zeroed["in"].estimate)
    assert zeroed["in"].estimate == pytest.approx(plain["in"].estimate)
    with pytest.raises(ValueError):
        stats.joint_cluster_bootstrap_subsets(
            columns, label, groups, inside, {"c": [0]}, b=5, zero_if_constant={"other"}
        )
