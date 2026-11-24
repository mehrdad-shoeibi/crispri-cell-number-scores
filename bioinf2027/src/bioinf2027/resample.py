"""The joint target-gene cluster bootstrap of the transfer analyses, computed with weights.

A replicate draws as many target genes as there are, with replacement, and takes every row of
every drawn gene, as the retained procedure does (``earlier_revision/pipeline/bootstrap_wiring.py:84-117``:
``np.unique`` of the cluster identifiers, then ``rng.integers(0, G, G)`` per replicate). A row
drawn w times enters the replicate w times, so every statistic of a replicate is a weighted
statistic over the distinct rows with weight w. Spearman correlations use average ranks of the
replicate sample, which the weighted ranks below reproduce exactly, ties included; a test checks
them against ranking the gathered rows directly.

All statistics of one screen come from one weight matrix, so every difference between two of them
is paired within the replicate.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ClusterDraw",
    "cluster_draw",
    "weighted_ranks",
    "weighted_spearman",
    "weighted_pearson",
    "weighted_r2",
    "p_value",
    "holm",
    "holm_state",
]

BLOCK = 1000


@dataclass(frozen=True)
class ClusterDraw:
    """Replicate weights (replicates x rows) and the identity of the draw."""

    weights: np.ndarray
    clusters: np.ndarray
    seed: int
    index_sha256: str

    @property
    def b(self) -> int:
        return int(self.weights.shape[0])


def cluster_draw(groups, b: int, seed: int) -> ClusterDraw:
    """Weights of every row in every replicate of a cluster bootstrap over ``groups``."""
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    n_groups = unique.size
    rng = np.random.default_rng(seed)
    weights = np.empty((int(b), groups.size), dtype=np.float64)
    digest = hashlib.sha256()
    for replicate in range(int(b)):
        picked = rng.integers(0, n_groups, n_groups)
        digest.update(np.ascontiguousarray(picked, dtype=np.int64).tobytes())
        weights[replicate] = np.bincount(picked, minlength=n_groups)[inverse]
    return ClusterDraw(weights=weights, clusters=unique, seed=int(seed), index_sha256=digest.hexdigest())


def weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average ranks of each row in each weighted replicate (replicates x rows).

    For a replicate with weights w, the rows tied at one value occupy the positions after the total
    weight of all smaller values, so each takes that total plus (the tie group's weight + 1) / 2.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    if ordered.size < 2 or bool(np.all(ordered[1:] != ordered[:-1])):
        # no ties: a row's rank is the weight of the rows below it plus (its weight + 1) / 2
        sorted_weights = weights[:, order]
        out = np.empty_like(weights, dtype=np.float64)
        out[:, order] = np.cumsum(sorted_weights, axis=1) - (sorted_weights - 1.0) / 2.0
        return out
    starts = np.concatenate([[0], np.flatnonzero(ordered[1:] != ordered[:-1]) + 1])
    group_of_sorted = np.repeat(np.arange(starts.size), np.diff(np.concatenate([starts, [values.size]])))
    sums = np.add.reduceat(weights[:, order], starts, axis=1)
    before = np.cumsum(sums, axis=1) - sums
    group_rank = before + (sums + 1.0) / 2.0
    out = np.empty_like(weights, dtype=np.float64)
    out[:, order] = group_rank[:, group_of_sorted]
    return out


def _weighted_corr(a: np.ndarray, e: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Pearson correlation of a (replicates x rows) with e (replicates x rows), per replicate."""
    total = weights.sum(axis=1)
    ma = (weights * a).sum(axis=1) / total
    me = (weights * e).sum(axis=1) / total
    da = a - ma[:, None]
    de = e - me[:, None]
    cov = (weights * da * de).sum(axis=1)
    va = (weights * da * da).sum(axis=1)
    ve = (weights * de * de).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = cov / np.sqrt(va * ve)
    out[(va <= 0) | (ve <= 0)] = np.nan
    return out


def weighted_spearman(columns: np.ndarray, label: np.ndarray, draw: ClusterDraw, mask: np.ndarray | None = None) -> np.ndarray:
    """Spearman of every column with the label in every replicate (replicates x columns).

    With ``mask`` only the masked rows enter each replicate, so the statistic is the Spearman inside
    that subset of the replicate sample.
    """
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim == 1:
        columns = columns[:, None]
    weights_all = draw.weights if mask is None else draw.weights * np.asarray(mask, dtype=np.float64)[None, :]
    out = np.full((draw.b, columns.shape[1]), np.nan)
    for start in range(0, draw.b, BLOCK):
        weights = weights_all[start : start + BLOCK]
        label_ranks = weighted_ranks(label, weights)
        for j in range(columns.shape[1]):
            out[start : start + BLOCK, j] = _weighted_corr(weighted_ranks(columns[:, j], weights), label_ranks, weights)
    return out


def weighted_pearson(columns: np.ndarray, other: np.ndarray, draw: ClusterDraw, mask: np.ndarray | None = None) -> np.ndarray:
    """Pearson of every fixed column with a fixed vector in every replicate (replicates x columns)."""
    columns = np.asarray(columns, dtype=np.float64)
    if columns.ndim == 1:
        columns = columns[:, None]
    other = np.asarray(other, dtype=np.float64)
    weights = draw.weights if mask is None else draw.weights * np.asarray(mask, dtype=np.float64)[None, :]
    total = weights.sum(axis=1)
    me = weights @ other / total
    ve = weights @ (other * other) / total - me * me
    out = np.empty((draw.b, columns.shape[1]))
    for j in range(columns.shape[1]):
        a = columns[:, j]
        ma = weights @ a / total
        va = weights @ (a * a) / total - ma * ma
        cov = weights @ (a * other) / total - ma * me
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / np.sqrt(va * ve)
        r[(va <= 1e-15) | (ve <= 1e-15)] = np.nan
        out[:, j] = r
    return out


def weighted_r2(label: np.ndarray, predictions: np.ndarray, draw: ClusterDraw) -> np.ndarray:
    """Pooled R-squared of every prediction column in every replicate (replicates x columns)."""
    label = np.asarray(label, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    weights = draw.weights
    total = weights.sum(axis=1)
    mean = weights @ label / total
    sst = weights @ (label * label) - total * mean * mean
    out = np.empty((draw.b, predictions.shape[1]))
    for j in range(predictions.shape[1]):
        sse = weights @ ((label - predictions[:, j]) ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = 1.0 - sse / sst
    out[sst <= 0] = np.nan
    return out


def p_value(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value: the smallest level at which the percentile interval excludes zero.

    p = min{1, 2 min[1 + #(theta* <= 0), 1 + #(theta* >= 0)] / (B + 1)}, over the finite replicates.
    """
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan")
    below = 1 + int((draws <= 0).sum())
    above = 1 + int((draws >= 0).sum())
    return float(min(1.0, 2.0 * min(below, above) / (draws.size + 1)))


def holm(p_values, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down: which hypotheses are rejected, and each one's threshold at its step.

    A missing p-value is treated as 1. The threshold of the hypothesis at step i (0-based, in the
    order of increasing p) is alpha / (m - i).
    """
    p = np.asarray(p_values, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 1.0)
    m = p.size
    order = np.argsort(p, kind="stable")
    thresholds = np.empty(m)
    rejected = np.zeros(m, dtype=bool)
    stopped = False
    for step, index in enumerate(order):
        thresholds[index] = alpha / (m - step)
        if not stopped and p[index] <= thresholds[index]:
            rejected[index] = True
        else:
            stopped = True
    return rejected, thresholds


def holm_state(estimate: float, rejected: bool) -> str:
    if not rejected or not np.isfinite(estimate):
        return "inconclusive"
    return "Holm-positive" if estimate > 0 else "Holm-negative"
