"""Helpers of E5 (bioinf2027/plans/E5.md, Sections 4, 5, 7 and 17): the energy distance, the control-cell basis,
the target-free values, the rank spline of Bc, the within-screen gene folds and largest-remainder allocation.

Kept under scripts/ so that no module under bioinf2027/src/ changes while E4's tier 3 runs (E5.md, Section 11,
check 6).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist

SCALE = 1e4
WITHIN_FOLD_SEED = 20271127
N_FOLDS = 5
KNOT_QUANTILES = (0.05, 0.35, 0.65, 0.95)


# --------------------------------------------------------------------------- values and basis


def values(counts: np.ndarray, totals: np.ndarray, removed: np.ndarray | None = None) -> np.ndarray:
    """log1p(count / (total - removed) * 10,000) per cell, in double precision; a denominator below one is one."""
    totals = np.asarray(totals, dtype=np.float64)
    if removed is not None:
        totals = totals - np.asarray(removed, dtype=np.float64)
    totals = np.where(totals < 1.0, 1.0, totals)
    return np.log1p(np.asarray(counts, dtype=np.float64) / totals[:, None] * SCALE)


def basis(control_values: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Control mean and the first n eigenvectors of the control cells' covariance (descending eigenvalues)."""
    v = np.asarray(control_values, dtype=np.float64)
    mean = v.mean(axis=0)
    centred = v - mean
    covariance = centred.T @ centred / (v.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    return mean, eigenvectors[:, order]


def scores(v: np.ndarray, mean: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (np.asarray(v, dtype=np.float64) - mean) @ vectors


# --------------------------------------------------------------------------- energy distance


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """2 mean|X - Y| - U(X) - U(Y), Euclidean, U the mean over distinct pairs; exact (scipy cdist and pdist)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")
    return float(2.0 * cdist(x, y).mean() - pdist(x).mean() - pdist(y).mean())


def _gram_distances(a: np.ndarray, b: np.ndarray, a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    d2 = a2[:, None] + b2[None, :] - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def mean_cross(x: np.ndarray, y: np.ndarray, block: int = 2048) -> float:
    """Mean Euclidean distance between the rows of x and of y, by blocks through the Gram matrix."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2, y2 = (x * x).sum(axis=1), (y * y).sum(axis=1)
    total = 0.0
    for i in range(0, x.shape[0], block):
        total += _gram_distances(x[i:i + block], y, x2[i:i + block], y2).sum()
    return total / (x.shape[0] * y.shape[0])


def mean_within(y: np.ndarray, block: int = 2048) -> float:
    """Mean Euclidean distance over the distinct pairs of rows of y, by blocks through the Gram matrix."""
    y = np.asarray(y, dtype=np.float64)
    m = y.shape[0]
    y2 = (y * y).sum(axis=1)
    total = 0.0
    for i in range(0, m, block):
        j = min(i + block, m)
        d = _gram_distances(y[i:j], y[i:], y2[i:j], y2[i:])
        total += np.triu(d[:, : j - i], k=1).sum() + d[:, j - i:].sum()
    return total / (m * (m - 1) / 2.0)


def energy_distance_gram(x: np.ndarray, y: np.ndarray) -> float:
    """The energy distance through Gram-matrix distances, for wide spaces and large samples."""
    if np.asarray(x).shape[0] < 2 or np.asarray(y).shape[0] < 2:
        return float("nan")
    return float(2.0 * mean_cross(x, y) - mean_within(x) - mean_within(y))


# --------------------------------------------------------------------------- allocation


def largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """Integer allocation of total in proportion to weights; ties of the remainders go to the earlier entry."""
    w = np.asarray(weights, dtype=np.float64)
    if total == 0 or w.sum() <= 0:
        return np.zeros(w.size, dtype=np.int64)
    exact = w / w.sum() * total
    out = np.floor(exact).astype(np.int64)
    rest = total - int(out.sum())
    if rest:
        order = np.lexsort((np.arange(w.size), -(exact - out)))
        out[order[:rest]] += 1
    return out


# --------------------------------------------------------------------------- the rank spline of Bc


def ranks01(values_: np.ndarray) -> np.ndarray:
    """Average ranks divided by the number of finite values; NaN stays NaN."""
    from scipy.stats import rankdata

    v = np.asarray(values_, dtype=np.float64)
    out = np.full(v.shape, np.nan)
    finite = np.isfinite(v)
    out[finite] = rankdata(v[finite]) / finite.sum()
    return out


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline of x with the given knots: x and k - 2 nonlinear columns (Harrell)."""
    x = np.asarray(x, dtype=np.float64)
    t = np.asarray(knots, dtype=np.float64)
    k = t.size
    scale = (t[-1] - t[0]) ** 2
    columns = [x]

    def plus3(u):
        return np.where(u > 0, u, 0.0) ** 3

    for j in range(k - 2):
        term = (plus3(x - t[j])
                - plus3(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + plus3(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        columns.append(term / scale)
    return np.column_stack(columns)


class BcSpline:
    """Least squares of a label on an intercept and the rank splines of the Bc columns, knots from the training rows."""

    def __init__(self, bc_ranks: np.ndarray, train: np.ndarray) -> None:
        self.knots = [np.quantile(bc_ranks[train, j], KNOT_QUANTILES) for j in range(bc_ranks.shape[1])]

    def design(self, bc_ranks: np.ndarray) -> np.ndarray:
        parts = [np.ones((bc_ranks.shape[0], 1))]
        parts += [rcs_basis(bc_ranks[:, j], self.knots[j]) for j in range(bc_ranks.shape[1])]
        return np.column_stack(parts)

    def fit(self, bc_ranks: np.ndarray, y: np.ndarray) -> "BcSpline":
        self.coef, *_ = np.linalg.lstsq(self.design(bc_ranks), y, rcond=None)
        return self

    def predict(self, bc_ranks: np.ndarray) -> np.ndarray:
        return self.design(bc_ranks) @ self.coef


def out_of_fold_residuals(bc_ranks: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """y minus its out-of-fold prediction from the Bc spline; rows with a missing y or Bc stay NaN."""
    usable = np.isfinite(y) & np.isfinite(bc_ranks).all(axis=1)
    out = np.full(y.shape, np.nan)
    for f in np.unique(folds):
        train = usable & (folds != f)
        test = usable & (folds == f)
        if not test.any():
            continue
        model = BcSpline(bc_ranks, train).fit(bc_ranks[train], y[train])
        out[test] = y[test] - model.predict(bc_ranks[test])
    return out


def within_screen_folds(genes) -> dict:
    """Five gene-disjoint folds of one screen's genes: the sorted genes permuted with seed 20271127, fold = position mod 5."""
    ordered = sorted(set(str(g) for g in genes))
    order = np.random.default_rng(WITHIN_FOLD_SEED).permutation(len(ordered))
    return {ordered[i]: int(position % N_FOLDS) for position, i in enumerate(order)}
