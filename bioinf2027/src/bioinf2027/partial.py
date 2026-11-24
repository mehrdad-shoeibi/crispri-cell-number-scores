"""Rank-residual partial correlation, with cluster bootstrap intervals.

The convention matches the one the source analyses used: rank every
variable, regress the ranks of the two variables of interest on the ranks of
the covariate, then correlate the residuals.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps

from .stats import BOOTSTRAP_B, BOOTSTRAP_SEED

__all__ = ["PartialEstimate", "partial_spearman", "cluster_bootstrap_partial"]


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y after a simple linear fit on x."""
    centred_x = x - x.mean()
    denominator = float(centred_x @ centred_x)
    if denominator == 0:
        return y - y.mean()
    slope = float(centred_x @ (y - y.mean())) / denominator
    return y - y.mean() - slope * centred_x


def partial_spearman(x: np.ndarray, y: np.ndarray, covariate: np.ndarray) -> float:
    """Spearman correlation of x and y after removing the covariate, on ranks."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covariate = np.asarray(covariate, dtype=np.float64)
    if x.size < 4:
        return float("nan")
    rank_x = sps.rankdata(x)
    rank_y = sps.rankdata(y)
    rank_c = sps.rankdata(covariate)
    res_x = _residual(rank_x, rank_c)
    res_y = _residual(rank_y, rank_c)
    if np.std(res_x) == 0 or np.std(res_y) == 0:
        return float("nan")
    return float(np.corrcoef(res_x, res_y)[0, 1])


@dataclass(frozen=True)
class PartialEstimate:
    n: int
    n_groups: int
    partial_spearman: float
    partial_lo: float
    partial_hi: float
    bootstrap_b: int
    bootstrap_seed: int
    n_skipped: int

    def as_dict(self) -> dict:
        return asdict(self)


def cluster_bootstrap_partial(
    x: np.ndarray,
    y: np.ndarray,
    covariate: np.ndarray,
    groups: np.ndarray,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> PartialEstimate:
    """Partial Spearman with a target-gene cluster bootstrap interval."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covariate = np.asarray(covariate, dtype=np.float64)
    groups = np.asarray(groups)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(covariate)
    x, y, covariate, groups = x[finite], y[finite], covariate[finite], groups[finite]

    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]

    point = partial_spearman(x, y, covariate)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(b):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[p] for p in picked])
        value = partial_spearman(x[index], y[index], covariate[index])
        if np.isfinite(value):
            draws.append(value)

    if draws:
        lo, hi = np.percentile(draws, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return PartialEstimate(
        n=int(x.size),
        n_groups=int(len(unique)),
        partial_spearman=point,
        partial_lo=float(lo),
        partial_hi=float(hi),
        bootstrap_b=int(b),
        bootstrap_seed=int(seed),
        n_skipped=int(b - len(draws)),
    )
