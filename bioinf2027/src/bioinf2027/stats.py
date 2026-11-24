"""Correlation estimates with target-gene cluster bootstrap intervals.

Constructs that share a target gene are dependent, so every interval here
resamples whole target genes rather than individual constructs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps

from .config import BOOTSTRAP_B, BOOTSTRAP_SEED

__all__ = [
    "Estimate",
    "SubsetEstimate",
    "cluster_bootstrap_corr",
    "table_row",
    "joint_cluster_bootstrap_subsets",
    "paired_screen_difference",
    "percentile_interval",
    "BOOTSTRAP_B",
    "BOOTSTRAP_SEED",
]



@dataclass(frozen=True)
class Estimate:
    n: int
    n_groups: int
    spearman: float
    spearman_lo: float
    spearman_hi: float
    pearson: float
    pearson_lo: float
    pearson_hi: float
    bootstrap_b: int
    bootstrap_seed: int
    n_skipped: int

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman and Pearson, returning NaN when either vector is degenerate."""
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    spearman = float(sps.spearmanr(x, y).statistic)
    pearson = float(np.corrcoef(x, y)[0, 1])
    return spearman, pearson


def cluster_bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> Estimate:
    """Correlations between x and y with 95 percent cluster bootstrap intervals.

    Each replicate draws the same number of distinct groups as observed, with
    replacement, and takes every member of a drawn group. Replicates whose
    correlation is undefined are excluded from the interval and counted.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups)
    if not (x.shape == y.shape == groups.shape):
        raise ValueError("x, y and groups must have the same shape")

    finite = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[finite], y[finite], groups[finite]

    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]

    if len(unique) == 0:
        nan = float("nan")
        return Estimate(
            n=0,
            n_groups=0,
            spearman=nan,
            spearman_lo=nan,
            spearman_hi=nan,
            pearson=nan,
            pearson_lo=nan,
            pearson_hi=nan,
            bootstrap_b=int(b),
            bootstrap_seed=int(seed),
            n_skipped=int(b),
        )

    spearman, pearson = _safe_corr(x, y)
    rng = np.random.default_rng(seed)
    draws_s, draws_p = [], []
    for _ in range(b):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[p] for p in picked])
        rs, rp = _safe_corr(x[index], y[index])
        if np.isfinite(rs) and np.isfinite(rp):
            draws_s.append(rs)
            draws_p.append(rp)

    skipped = b - len(draws_s)
    if draws_s:
        s_lo, s_hi = np.percentile(draws_s, [2.5, 97.5])
        p_lo, p_hi = np.percentile(draws_p, [2.5, 97.5])
    else:
        s_lo = s_hi = p_lo = p_hi = float("nan")

    return Estimate(
        n=int(x.size),
        n_groups=int(len(unique)),
        spearman=spearman,
        spearman_lo=float(s_lo),
        spearman_hi=float(s_hi),
        pearson=pearson,
        pearson_lo=float(p_lo),
        pearson_hi=float(p_hi),
        bootstrap_b=int(b),
        bootstrap_seed=int(seed),
        n_skipped=int(skipped),
    )



def table_row(estimate) -> dict:
    """An estimate as table columns, with the sample sizes under explicit names.

    ``n`` becomes ``n_rows`` and ``n_groups`` becomes ``n_target_genes``; every
    interval in these analyses resamples target genes, so the group count is
    always a count of target genes.
    """
    row = estimate.as_dict()
    row["n_rows"] = row.pop("n")
    row["n_target_genes"] = row.pop("n_groups")
    return row


@dataclass(frozen=True)
class SubsetEstimate:
    """One statistic of :func:`joint_cluster_bootstrap_subsets`."""

    family: str
    subset: str
    n: int
    n_groups: int
    estimate: float
    lo: float
    hi: float
    bootstrap_b: int
    bootstrap_seed: int
    n_skipped: int

    def as_dict(self) -> dict:
        return asdict(self)


def _rank_correlations(
    label: np.ndarray, predictions: np.ndarray, covariate: np.ndarray | None = None
) -> np.ndarray:
    """Spearman of every prediction column against the label, NaN where undefined.

    With a covariate the statistic is the rank-residual partial correlation of
    :mod:`bioinf2027.partial`: every variable is ranked, the ranks of the label
    and of each prediction are regressed on the ranks of the covariate, and the
    residuals are correlated.
    """
    k = predictions.shape[1]
    if label.size < (3 if covariate is None else 4):
        return np.full(k, np.nan)
    columns = [label, predictions] + ([covariate] if covariate is not None else [])
    ranks = sps.rankdata(np.column_stack(columns), axis=0)
    ranks = ranks - ranks.mean(axis=0)
    if covariate is not None:
        centred_covariate = ranks[:, -1]
        ranks = ranks[:, :-1]
        denominator = float(centred_covariate @ centred_covariate)
        if denominator > 0:
            slopes = (centred_covariate @ ranks) / denominator
            ranks = ranks - np.outer(centred_covariate, slopes)
    norms = np.sqrt((ranks * ranks).sum(axis=0))
    out = np.full(k, np.nan)
    if norms[0] == 0:
        return out
    valid = norms[1:] > 0
    out[valid] = (ranks[:, 1:][:, valid] * ranks[:, :1]).sum(axis=0) / (norms[1:][valid] * norms[0])
    return out


def joint_cluster_bootstrap_subsets(
    predictions: np.ndarray,
    label: np.ndarray,
    groups: np.ndarray,
    in_subset: np.ndarray,
    families: dict[str, list[int]],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    return_draws: bool = False,
    covariate: np.ndarray | None = None,
    zero_if_constant=(),
):
    """Spearman per family on all rows, inside a subset, outside it, and the difference.

    One replicate draws whole groups with replacement and takes every row of a
    drawn group; the same draw serves every family and every subset, so the
    inside-minus-outside difference is paired within the replicate. A family is
    a set of prediction columns, for example the archived seeds of one model,
    and its statistic is the mean over its columns of the per-column Spearman.
    Intervals are percentile intervals. A replicate in which a statistic is
    undefined is excluded for that statistic and counted.

    ``in_subset`` must be constant within a group, otherwise a group would be
    split between the two subsets and the difference would not be between
    groups.

    With ``covariate`` every Spearman is replaced by the rank-residual partial
    Spearman given that covariate, computed inside each subset of each replicate.

    Families named in ``zero_if_constant`` are predictors that carry no value
    on part of the data, for example a coordinate that is zero filled where it
    is not available. Where all of such a family's columns are constant on a
    subset of a replicate, its statistic there is set to 0 instead of being
    undefined, so that its in-minus-out difference exists.

    With ``return_draws`` the function also returns the point statistics and
    the replicate statistics as arrays of shape (families, 4) and
    (b, families, 4), columns all, in, out, in minus out, so that a statistic
    derived from several families is computed inside the same replicates.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    groups = np.asarray(groups)
    in_subset = np.asarray(in_subset, dtype=bool)
    if predictions.ndim != 2 or predictions.shape[0] != label.shape[0]:
        raise ValueError("predictions must be (rows, columns) with one row per label")
    if not (label.shape == groups.shape == in_subset.shape):
        raise ValueError("label, groups and in_subset must have the same shape")
    if not np.isfinite(predictions).all() or not np.isfinite(label).all():
        raise ValueError("predictions and label must be finite")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=np.float64)
        if covariate.shape != label.shape or not np.isfinite(covariate).all():
            raise ValueError("the covariate must be finite and have one value per label")

    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    for rows in members:
        if in_subset[rows].min() != in_subset[rows].max():
            raise ValueError("in_subset must be constant within a group")

    names = list(families)
    columns = [np.asarray(families[name], dtype=int) for name in names]
    subsets = ("all", "in", "out")
    unknown = set(zero_if_constant) - set(names)
    if unknown:
        raise ValueError("zero_if_constant names unknown families: %s" % sorted(unknown))
    zero_rule = [name in set(zero_if_constant) for name in names]

    def statistics(index: np.ndarray) -> np.ndarray:
        """Rows are families, columns are all, in, out, in minus out."""
        out = np.full((len(names), 4), np.nan)
        inside = in_subset[index]
        for position, rows in enumerate((index, index[inside], index[~inside])):
            per_column = _rank_correlations(
                label[rows], predictions[rows], None if covariate is None else covariate[rows]
            )
            for f, cols in enumerate(columns):
                values = per_column[cols]
                if np.isfinite(values).all():
                    out[f, position] = float(values.mean())
                elif zero_rule[f] and rows.size and np.ptp(predictions[rows][:, cols], axis=0).max() == 0:
                    out[f, position] = 0.0
        out[:, 3] = out[:, 1] - out[:, 2]
        return out

    everything = np.arange(label.shape[0])
    point = statistics(everything)
    sizes = {
        "all": everything,
        "in": everything[in_subset],
        "out": everything[~in_subset],
    }

    rng = np.random.default_rng(seed)
    draws = np.full((b, len(names), 4), np.nan)
    for replicate in range(b):
        picked = rng.integers(0, len(unique), size=len(unique))
        index = np.concatenate([members[p] for p in picked])
        draws[replicate] = statistics(index)

    results = []
    for f, name in enumerate(names):
        for position, subset in enumerate(subsets + ("in_minus_out",)):
            values = draws[:, f, position]
            values = values[np.isfinite(values)]
            if values.size:
                lo, hi = np.percentile(values, [2.5, 97.5])
            else:
                lo = hi = float("nan")
            rows = sizes[subset] if subset in sizes else everything
            results.append(
                SubsetEstimate(
                    family=name,
                    subset=subset,
                    n=int(rows.size),
                    n_groups=int(np.unique(groups[rows]).size),
                    estimate=float(point[f, position]),
                    lo=float(lo),
                    hi=float(hi),
                    bootstrap_b=int(b),
                    bootstrap_seed=int(seed),
                    n_skipped=int(b - values.size),
                )
            )
    if return_draws:
        return results, point, draws
    return results


def percentile_interval(values: np.ndarray) -> tuple[float, float, int]:
    """95 percent percentile interval of the finite values, and how many were not finite."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return float("nan"), float("nan"), int(values.size)
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return float(lo), float(hi), int(values.size - finite.size)


def paired_screen_difference(
    first: tuple[np.ndarray, np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray, np.ndarray],
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Spearman on two screens over their shared groups, and the paired difference.

    Each screen is given as (predictor, label, groups). Only groups present in
    both screens are used. One replicate draws shared groups with replacement
    and takes, in each screen, every row of every drawn group, so the two
    screens are resampled together and the difference first minus second is
    paired within the replicate.
    """
    def members(groups: np.ndarray, shared: np.ndarray) -> list[np.ndarray]:
        order = {g: i for i, g in enumerate(shared.tolist())}
        rows: list[list[int]] = [[] for _ in shared]
        for row, g in enumerate(groups.tolist()):
            if g in order:
                rows[order[g]].append(row)
        return [np.asarray(r, dtype=int) for r in rows]

    arrays = []
    for predictor, label, groups in (first, second):
        predictor = np.asarray(predictor, dtype=np.float64)
        label = np.asarray(label, dtype=np.float64)
        groups = np.asarray(groups)
        if not (predictor.shape == label.shape == groups.shape):
            raise ValueError("predictor, label and groups must have the same shape")
        if not (np.isfinite(predictor).all() and np.isfinite(label).all()):
            raise ValueError("predictor and label must be finite")
        arrays.append((predictor, label, groups))

    shared = np.intersect1d(np.unique(arrays[0][2]), np.unique(arrays[1][2]))
    if shared.size < 3:
        raise ValueError("fewer than three shared groups")
    rows = [members(groups, shared) for _, _, groups in arrays]

    def spearman(screen: int, index: np.ndarray) -> float:
        predictor, label, _ = arrays[screen]
        value = _rank_correlations(label[index], predictor[index][:, None])[0]
        return float(value)

    everything = [np.concatenate(r) for r in rows]
    point = [spearman(k, everything[k]) for k in (0, 1)]
    rng = np.random.default_rng(seed)
    draws = np.full((b, 3), np.nan)
    for replicate in range(b):
        picked = rng.integers(0, shared.size, size=shared.size)
        values = [spearman(k, np.concatenate([rows[k][p] for p in picked])) for k in (0, 1)]
        draws[replicate] = (values[0], values[1], values[0] - values[1])

    out = {
        "n_shared_groups": int(shared.size),
        "n_rows_first": int(everything[0].size),
        "n_rows_second": int(everything[1].size),
        "bootstrap_b": int(b),
        "bootstrap_seed": int(seed),
    }
    for position, name in enumerate(("first", "second", "first_minus_second")):
        estimate = point[0] - point[1] if position == 2 else point[position]
        lo, hi, skipped = percentile_interval(draws[:, position])
        out[name] = {"spearman": float(estimate), "lo": lo, "hi": hi, "n_skipped": skipped}
    return out
