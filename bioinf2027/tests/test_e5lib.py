"""E5 helpers against direct computations (bioinf2027/plans/E5.md, Section 11, check 2, and Section 17)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e5lib  # noqa: E402


def _direct(x, y):
    n, m = len(x), len(y)
    xy = sum(np.linalg.norm(a - b) for a in x for b in y) / (n * m)
    xx = sum(np.linalg.norm(x[i] - x[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    yy = sum(np.linalg.norm(y[i] - y[j]) for i in range(m) for j in range(m) if i != j) / (m * (m - 1))
    return 2 * xy - xx - yy


def test_energy_distance_equals_a_direct_double_loop():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(12, 5)), rng.normal(0.4, 1.2, size=(30, 5))
    assert abs(e5lib.energy_distance(x, y) - _direct(x, y)) < 1e-12
    assert abs(e5lib.energy_distance_gram(x, y) - _direct(x, y)) < 1e-10


def test_energy_distance_is_unbiased_and_may_be_negative():
    rng = np.random.default_rng(2)
    values = [e5lib.energy_distance(rng.normal(size=(8, 3)), rng.normal(size=(40, 3))) for _ in range(400)]
    assert min(values) < 0
    assert abs(np.mean(values)) < 0.05


def test_blocked_within_and_cross_means_match_scipy():
    from scipy.spatial.distance import cdist, pdist

    rng = np.random.default_rng(3)
    y, x = rng.normal(size=(301, 7)), rng.normal(size=(45, 7))
    assert abs(e5lib.mean_within(y, block=64) - pdist(y).mean()) < 1e-10
    assert abs(e5lib.mean_cross(x, y, block=16) - cdist(x, y).mean()) < 1e-10


def test_values_remove_the_target_from_the_total():
    counts = np.array([[1.0, 3.0], [0.0, 2.0]])
    out = e5lib.values(counts, np.array([10.0, 4.0]), np.array([5.0, 3.5]))
    assert np.allclose(out, np.log1p(counts / np.array([[5.0], [1.0]]) * 1e4))


def test_basis_scores_are_centred_principal_components():
    rng = np.random.default_rng(4)
    v = rng.normal(size=(200, 6)) @ np.diag([5, 3, 2, 1, 0.5, 0.1])
    mean, vectors = e5lib.basis(v, 2)
    s = e5lib.scores(v, mean, vectors)
    assert np.allclose(s.mean(axis=0), 0.0, atol=1e-12)
    variances = s.var(axis=0, ddof=1)
    assert variances[0] >= variances[1]
    assert abs(variances[0] - np.linalg.eigvalsh(np.cov(v.T))[-1]) < 1e-9


def test_largest_remainder_sums_and_is_proportional():
    out = e5lib.largest_remainder(np.array([5, 3, 2]), 788)
    assert out.sum() == 788
    assert np.all(np.abs(out - np.array([5, 3, 2]) / 10 * 788) < 1)
    assert e5lib.largest_remainder(np.array([1, 1, 1]), 2).tolist() == [1, 1, 0]


def test_restricted_cubic_spline_is_linear_beyond_the_outer_knots():
    knots = np.quantile(np.linspace(0, 1, 101), e5lib.KNOT_QUANTILES)
    x = np.linspace(0.96, 1.5, 7)
    b = e5lib.rcs_basis(x, knots)
    assert b.shape == (7, 3)
    assert np.allclose(np.diff(b, n=2, axis=0), 0.0, atol=1e-12)
    x_low = np.linspace(-0.5, 0.04, 7)
    assert np.allclose(e5lib.rcs_basis(x_low, knots)[:, 1:], 0.0)


def test_out_of_fold_residuals_remove_a_smooth_function_of_bc():
    rng = np.random.default_rng(5)
    n = 600
    bc = rng.normal(size=(n, 2))
    r = np.column_stack([e5lib.ranks01(bc[:, j]) for j in range(2)])
    y = np.sin(3 * r[:, 0]) + r[:, 1] ** 2 + 0.01 * rng.normal(size=n)
    folds = np.arange(n) % 5
    res = e5lib.out_of_fold_residuals(r, y, folds)
    assert np.nanstd(res) < 0.05 * np.std(y)


def test_within_screen_folds_are_fixed_and_balanced():
    genes = [f"G{i}" for i in range(23)]
    a = e5lib.within_screen_folds(genes)
    assert a == e5lib.within_screen_folds(list(reversed(genes)))
    counts = np.bincount(list(a.values()), minlength=5)
    assert counts.max() - counts.min() <= 1
