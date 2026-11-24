"""E5's fitting harness against sklearn and direct computations (bioinf2027/plans/E5.md, Section 18)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import e5fit  # noqa: E402
import e5lib  # noqa: E402


def test_ridge_path_equals_sklearn_on_standardised_columns():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    x, y, xt = rng.normal(size=(60, 25)), rng.normal(size=60), rng.normal(size=(15, 25))
    x[:, 3] = 2.0  # a constant column is dropped, as StandardScaler gives it zero
    alphas = (0.1, 10.0, 1e4)
    ours = e5fit.ridge_path(x, y, xt, alphas)
    scaler = StandardScaler().fit(x)
    for j, a in enumerate(alphas):
        theirs = Ridge(alpha=a).fit(scaler.transform(x), y).predict(scaler.transform(xt))
        assert np.allclose(ours[:, j], theirs, atol=1e-9)


def test_ridge_path_takes_gesvd_where_gesdd_does_not_converge(monkeypatch):
    rng = np.random.default_rng(3)
    x, y, xt = rng.normal(size=(40, 60)), rng.normal(size=40), rng.normal(size=(10, 60))
    design = np.column_stack([np.ones(40), rng.normal(size=(40, 4))])
    x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]  # rank 35, as a Bc-residualised block with more columns than rows
    direct = e5fit.ridge_path(x, y, xt)
    before = e5fit.SVD_FALLBACKS[0]

    def not_converging(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(e5fit.np.linalg, "svd", not_converging)
    assert np.allclose(e5fit.ridge_path(x, y, xt), direct, rtol=0.0, atol=1e-10)
    assert e5fit.SVD_FALLBACKS[0] == before + 1


def test_the_grid_runs_from_0_01_to_1e6_in_half_decades():
    assert len(e5fit.RIDGE_GRID) == 17
    assert abs(e5fit.RIDGE_GRID[0] - 0.01) < 1e-15 and abs(e5fit.RIDGE_GRID[-1] - 1e6) < 1e-6


def test_gaussian_control_is_fixed_per_gene_and_width():
    a = e5fit.gaussian_vector(512, "ENSG00000141510")
    assert np.array_equal(a, e5fit.gaussian_vector(512, "ENSG00000141510"))
    assert not np.allclose(a[:10], e5fit.gaussian_vector(512, "ENSG00000012048")[:10])
    assert e5fit.gaussian_vector(768, "ENSG00000141510").shape == (768,)


def test_shuffled_control_is_a_fixed_permutation_of_the_vectors():
    vectors = {f"G{i:02d}": np.full(3, float(i)) for i in range(20)}
    s = e5fit.shuffled(vectors, list(vectors))
    assert sorted(v[0] for v in s.values()) == sorted(v[0] for v in vectors.values())
    again = e5fit.shuffled(vectors, list(reversed(list(vectors))))  # the order of the input genes does not matter
    assert all(np.array_equal(s[k], again[k]) for k in s)
    assert sum(np.array_equal(s[k], vectors[k]) for k in s) < 20


def test_fit_predict_uses_the_input_beyond_bc_and_ignores_noise():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(2)
    n, p = 500, 40
    bc = rng.normal(size=(n, 3))
    ranks = np.column_stack([e5lib.ranks01(bc[:, j]) for j in range(3)])
    e = rng.normal(size=(n, p))
    y = ranks[:, 0] + 0.5 * e[:, 0] + 0.1 * rng.normal(size=n)
    genes = np.array([f"G{i}" for i in range(n)])
    train, test = np.arange(n) < 400, np.arange(n) >= 400
    base, _ = e5fit.fit_predict(ranks[train], y[train], genes[train], ranks[test])
    full, record = e5fit.fit_predict(ranks[train], y[train], genes[train], ranks[test], e[train], e[test])
    assert spearmanr(full, y[test])[0] > spearmanr(base, y[test])[0] + 0.1
    noise, record_noise = e5fit.fit_predict(ranks[train], y[train] - 0.5 * e[train, 0], genes[train], ranks[test], rng.normal(size=(400, p)), rng.normal(size=(100, p)))
    assert record_noise["alpha"] >= record["alpha"]


def test_intersection_union_takes_the_larger_p_and_the_smaller_difference():
    rng = np.random.default_rng(3)
    a = (0.10, 0.10 + 0.02 * rng.normal(size=2000))
    b = (0.04, 0.04 + 0.03 * rng.normal(size=2000))
    out = e5fit.iu_statistic(a, b)
    assert out["estimate"] == 0.04
    assert out["p"] == max(out["p_gaussian"], out["p_shuffled"]) == out["p_shuffled"]


def test_padding_of_the_revised_positive_control_is_fixed_per_gene_and_fills_the_width():
    genes = ["ENSG00000141510", "ENSG00000012048"]
    block = e5fit.padding_block(1536, 768, genes)
    assert block.shape == (2, 768)
    assert np.array_equal(block, e5fit.padding_block(1536, 768, genes))
    assert np.array_equal(e5fit.padding_block(1536, 760, genes)[:, :768], block)  # fewer components: the same columns and more
    assert not np.allclose(block[0, :10], e5fit.gaussian_vector(1536, genes[0])[:10])  # not the Gaussian control's values
