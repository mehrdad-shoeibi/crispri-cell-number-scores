"""E5's fitting harness (bioinf2027/plans/E5.md, Sections 7, 8 and 18): ridge on Bc-residualised inputs with the Bc
spline unpenalised, the penalty chosen by gene-grouped inner folds, the Gaussian and shuffled controls, and the
intersection-union statistic. Kept under scripts/ (no module under bioinf2027/src/ changes while E4's tier 3 runs).
"""
from __future__ import annotations

import hashlib

import numpy as np

import e5lib

RIDGE_GRID = tuple(float(10.0 ** e) for e in np.arange(-2.0, 6.0 + 1e-9, 0.5))
INNER_SEED = 20271128
N_INNER = 5
SHUFFLE_SEED = 20271126
REFIT_SEED = 20271129


# --------------------------------------------------------------------------- controls


def gaussian_vector(width: int, ensembl: str) -> np.ndarray:
    """Cₑ for one gene: the seed is the first eight bytes, big-endian, of SHA-256 of "E5 | C | <width> | <Ensembl ID>"."""
    digest = hashlib.sha256(f"E5 | C | {int(width)} | {ensembl}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big")).standard_normal(int(width))


def gaussian_block(width: int, genes) -> np.ndarray:
    return np.vstack([gaussian_vector(width, g) for g in genes])


def padding_block(width: int, components: int, genes) -> np.ndarray:
    """The revised positive control's Gaussian columns (E5.md Section 20): per gene, the first width - components
    values of standard_normal(width), the seed being the first eight bytes, big-endian, of SHA-256 of
    "E5 | PC padding | <width> | <Ensembl ID>"."""
    rows = []
    for g in genes:
        digest = hashlib.sha256(f"E5 | PC padding | {int(width)} | {g}".encode()).digest()
        rows.append(np.random.default_rng(int.from_bytes(digest[:8], "big")).standard_normal(int(width))[: int(width) - int(components)])
    return np.vstack(rows)


def shuffled(vectors: dict, genes) -> dict:
    """Sₑ: the vectors of the given genes permuted among them, genes sorted by identifier, seed 20271126."""
    ordered = sorted(set(str(g) for g in genes))
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(ordered))
    return {ordered[i]: vectors[ordered[order[i]]] for i in range(len(ordered))}


# --------------------------------------------------------------------------- ridge

SVD_FALLBACKS = [0]  # decompositions taken by LAPACK gesvd in this process; the orchestration reads and resets it


def thin_svd(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD by numpy (LAPACK gesdd); where gesdd does not converge, the same decomposition by LAPACK gesvd.

    E5.md Section 19: gesdd fails on some Bc-residualised matrices (rank n minus the spline's columns, with singular
    values at rounding level), depending on the BLAS thread count. Where gesdd converges nothing changes.
    """
    try:
        return np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd

        SVD_FALLBACKS[0] += 1
        return svd(a, full_matrices=False, lapack_driver="gesvd")


def ridge_path(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alphas=RIDGE_GRID) -> np.ndarray:
    """Predictions (test rows x penalties) of ridge with an unpenalised intercept on standardised columns, by SVD."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    keep = sd > 0
    a = (x_train[:, keep] - mean[keep]) / sd[keep]
    b = (x_test[:, keep] - mean[keep]) / sd[keep]
    y_mean = float(np.mean(y_train))
    if a.shape[1] == 0:
        return np.full((x_test.shape[0], len(alphas)), y_mean)
    u, s, vt = thin_svd(a)
    uty = u.T @ (np.asarray(y_train, dtype=np.float64) - y_mean)
    projected = b @ vt.T
    shrink = s[:, None] / (s[:, None] ** 2 + np.asarray(alphas, dtype=np.float64)[None, :])
    return projected @ (shrink * uty[:, None]) + y_mean


def _residualise(design_train, design_test, columns_train, columns_test):
    coef, *_ = np.linalg.lstsq(design_train, columns_train, rcond=None)
    return columns_train - design_train @ coef, columns_test - design_test @ coef


def inner_folds(genes) -> np.ndarray:
    unique = np.array(sorted(set(str(g) for g in genes)))
    order = np.random.default_rng(INNER_SEED).permutation(unique.size)
    fold = {unique[i]: position % N_INNER for position, i in enumerate(order)}
    return np.array([fold[str(g)] for g in genes])


def fit_predict(bc_train, y_train, genes_train, bc_test, e_train=None, e_test=None) -> tuple[np.ndarray, dict]:
    """One outer fold: the Bc spline, and with an input, ridge on its Bc residuals; predictions for the test rows."""
    bc_train = np.asarray(bc_train, dtype=np.float64)
    bc_test = np.asarray(bc_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    spline = e5lib.BcSpline(bc_train, np.ones(bc_train.shape[0], dtype=bool))
    d_train, d_test = spline.design(bc_train), spline.design(bc_test)
    coef, *_ = np.linalg.lstsq(d_train, y_train, rcond=None)
    base_test = d_test @ coef
    if e_train is None:
        return base_test, {}
    r_train = y_train - d_train @ coef
    folds = inner_folds(genes_train)
    sse = np.zeros(len(RIDGE_GRID))
    sst = 0.0
    for f in range(N_INNER):
        test = folds == f
        train = ~test
        if not test.any():
            continue
        inner = e5lib.BcSpline(bc_train[train], np.ones(int(train.sum()), dtype=bool))
        di_train, di_test = inner.design(bc_train[train]), inner.design(bc_train[test])
        ci, *_ = np.linalg.lstsq(di_train, y_train[train], rcond=None)
        e_tr, e_te = _residualise(di_train, di_test, e_train[train], e_train[test])
        pred = di_test @ ci + ridge_path(e_tr, y_train[train] - di_train @ ci, e_te).T
        sse += ((y_train[test][None, :] - pred) ** 2).sum(axis=1)
        sst += float(((y_train[test] - y_train[test].mean()) ** 2).sum())
    r2 = 1.0 - sse / sst
    best = int(np.argmax(r2))  # the first maximum: on a tie the smaller penalty
    e_tr, e_te = _residualise(d_train, d_test, np.asarray(e_train, dtype=np.float64), np.asarray(e_test, dtype=np.float64))
    pred = base_test + ridge_path(e_tr, r_train, e_te, (RIDGE_GRID[best],))[:, 0]
    return pred, {"alpha": RIDGE_GRID[best], "at_grid_edge": best in (0, len(RIDGE_GRID) - 1), "inner_r2": float(r2[best])}


# --------------------------------------------------------------------------- statistics


def iu_statistic(d_gauss: tuple, d_shuffle: tuple) -> dict:
    """Intersection-union: p is the larger of the two p-values, the estimate the smaller of the two differences."""
    from bioinf2027 import resample

    p1, p2 = resample.p_value(d_gauss[1]), resample.p_value(d_shuffle[1])
    return {"estimate": float(min(d_gauss[0], d_shuffle[0])), "p": float(max(p1, p2)), "p_gaussian": p1, "p_shuffled": p2,
            "difference_gaussian": float(d_gauss[0]), "difference_shuffled": float(d_shuffle[0])}
