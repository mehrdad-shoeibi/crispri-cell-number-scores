"""The label builder of E1: one statistic, two normalisations, one draw order.

The statistic is the one of the retained builders
(``scripts/v4/harmonized_endpoint_v1/common.py:41-58`` and
``src/data/vcc/loader.py:146-181``)::

    y = log1p(max(0, anderson_ksamp([a, b]).statistic))

with scipy's default midrank variant. A sample of fewer than two values, a
degenerate sample and a non-finite statistic give NaN.

Two normalisations of a cell's counts, both log1p of the counts scaled to
10,000 per cell, with the total over every gene of the cell and a total below
one replaced by one:

``external`` mode
    the total is the single-precision row sum of the stored counts, the value
    ``log1p(count / total * 1e4)`` is computed in double precision and stored
    in single precision
    (``scripts/v4/harmonized_endpoint_v1/run_replogle_endpoint.py:111-124``).
``source`` mode
    every step in single precision, ``log1p(count * (1e4 / total))``
    (``src/data/vcc/loader.py:130-143``).

Every new label of E1 uses external mode; source mode exists for the builder
checks that tie the builder to the stored VCC label.

Draw order: one generator per seed; the perturbed cells, or the slots of a null
label, are drawn first, and the control cells second, both without replacement
(``run_replogle_endpoint.py:136-144``).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import anderson_ksamp

__all__ = [
    "SEEDS",
    "N_DRAWS",
    "MIN_VALID_DRAWS",
    "SCALE",
    "ad_log1p",
    "dense_totals",
    "external_values",
    "source_values",
    "draw",
    "summarise_draws",
    "subsampled_label",
    "null_draws",
    "null_label",
    "subsampled_labels",
    "own_summaries",
    "half_split",
    "thin_counts",
]

# The seeds of the existing external endpoint (scripts/v4/harmonized_endpoint_v1/common.py:19).
SEEDS = tuple(range(20260801, 20260826))
N_DRAWS = len(SEEDS)
# A label needs this many finite draws (bioinf2027/plans/E1_design.md, Section 7).
MIN_VALID_DRAWS = 20
SCALE = 1e4


def ad_log1p(a, b) -> float:
    """log1p of the clamped two-sample Anderson-Darling statistic, NaN when undefined."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic = anderson_ksamp([a, b]).statistic
    except (ValueError, RuntimeError):
        return float("nan")
    if not np.isfinite(statistic):
        return float("nan")
    return float(np.log1p(max(0.0, float(statistic))))


def dense_totals(block: np.ndarray) -> np.ndarray:
    """Single-precision row sums of a block of single-precision counts."""
    block = np.asarray(block)
    if block.dtype != np.float32:
        raise TypeError("counts must be stored in single precision")
    return block.sum(axis=1)


def external_values(counts, totals) -> np.ndarray:
    """External-mode values of the given counts, one total per row.

    ``counts`` is (cells,) or (cells, columns); ``totals`` holds the
    single-precision total of each cell over all its genes.
    """
    totals = np.maximum(np.asarray(totals, dtype=np.float32), np.float32(1.0))
    counts = np.asarray(counts, dtype=np.float64)
    denominator = totals if counts.ndim == 1 else totals[:, None]
    return np.log1p(counts / denominator * SCALE).astype(np.float32)


def source_values(counts, totals) -> np.ndarray:
    """Source-mode values of the given counts, one total per row, in single precision."""
    totals = np.asarray(totals, dtype=np.float32)
    totals = np.where(totals < np.float32(1.0), np.float32(1.0), totals)
    factor = np.float32(SCALE) / totals
    counts = np.ascontiguousarray(np.asarray(counts, dtype=np.float32))
    scaled = counts * (factor if counts.ndim == 1 else factor[:, None])
    return np.log1p(np.ascontiguousarray(scaled), dtype=np.float32)


def draw(seed: int, n_first: int, k_first: int, n_second: int, k_second: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of one draw: k_first of n_first, then k_second of n_second, one generator."""
    rng = np.random.default_rng(seed)
    first = rng.choice(n_first, size=k_first, replace=False)
    second = rng.choice(n_second, size=k_second, replace=False)
    return first, second


def summarise_draws(draws: np.ndarray) -> dict:
    """Median and spread of the finite draws, the median of the valid draws kept before the 20-draw rule."""
    draws = np.asarray(draws, dtype=np.float64)
    valid = draws[np.isfinite(draws)]
    nan = float("nan")
    if valid.size:
        median_valid = float(np.median(valid))
        q05, q25, q75, q95 = (float(v) for v in np.quantile(valid, [0.05, 0.25, 0.75, 0.95]))
        mean = float(np.mean(valid))
    else:
        median_valid = mean = q05 = q25 = q75 = q95 = nan
    return {
        "y": median_valid if valid.size >= MIN_VALID_DRAWS else nan,
        "y_median_valid_draws": median_valid,
        "y_mean": mean,
        "y_sd": float(np.std(valid, ddof=1)) if valid.size > 1 else nan,
        "y_iqr": q75 - q25,
        "y_p5": q05,
        "y_p95": q95,
        "n_valid_draws": int(valid.size),
    }


def subsampled_label(pert_values, ctrl_values, n_pert: int, n_ctrl: int, seeds=SEEDS) -> tuple[dict, np.ndarray]:
    """The fixed-size label of one row: per seed, n_pert of its perturbed cells against n_ctrl controls."""
    pert_values = np.asarray(pert_values)
    ctrl_values = np.asarray(ctrl_values)
    draws = np.empty(len(seeds), dtype=np.float64)
    for k, seed in enumerate(seeds):
        first, second = draw(seed, pert_values.size, n_pert, ctrl_values.size, n_ctrl)
        draws[k] = ad_log1p(pert_values[first], ctrl_values[second])
    return summarise_draws(draws), draws


def null_draws(n_pool: int, slot_pool: np.ndarray, n_slot: int, n_ctrl: int, seeds=SEEDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slot cells and control cells of a null label, per seed, as positions in the pooled control set.

    ``slot_pool`` holds, in ascending order, the positions the slots are drawn
    from: one batch's control cells for the batch-own null, every pooled control
    cell for the pooled-draw null. Per seed one generator draws the slots first
    and then ``n_ctrl`` control cells from the pooled set without the slot cells,
    taken in ascending order (bioinf2027/plans/E1_design.md, Section 23.4).
    """
    slot_pool = np.asarray(slot_pool, dtype=np.int64)
    if slot_pool.size < n_slot:
        raise ValueError("the slot pool is smaller than the number of slots")
    everything = np.arange(n_pool, dtype=np.int64)
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        slots = slot_pool[rng.choice(slot_pool.size, size=n_slot, replace=False)]
        rest = np.delete(everything, slots)
        controls = rest[rng.choice(rest.size, size=n_ctrl, replace=False)]
        out.append((slots, controls))
    return out


def null_label(values: np.ndarray, draws: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """A null label of one gene: per draw the statistic of its slot cells against its control cells."""
    values = np.asarray(values)
    statistics = np.array([ad_log1p(values[slots], values[controls]) for slots, controls in draws])
    return summarise_draws(statistics)


def subsampled_labels(pools, controls, n_pert: int, n_ctrl: int, seeds=SEEDS) -> list[tuple[dict, np.ndarray]]:
    """Fixed-size labels of one row in several value modes, every mode on the same drawn cells.

    ``pools`` and ``controls`` are sequences of value arrays of equal lengths, one pair per mode (for
    example external and source mode); the draws are made once per seed and applied to every mode.
    """
    pools = [np.asarray(p) for p in pools]
    controls = [np.asarray(c) for c in controls]
    if len({p.size for p in pools}) != 1 or len({c.size for c in controls}) != 1 or len(pools) != len(controls):
        raise ValueError("every mode needs the same cells")
    draws = np.empty((len(pools), len(seeds)), dtype=np.float64)
    for k, seed in enumerate(seeds):
        first, second = draw(seed, pools[0].size, n_pert, controls[0].size, n_ctrl)
        for mode, (pool, control) in enumerate(zip(pools, controls)):
            draws[mode, k] = ad_log1p(pool[first], control[second])
    return [(summarise_draws(draws[mode]), draws[mode]) for mode in range(len(pools))]


def own_summaries(pert_values, ctrl_values) -> dict:
    """B0 to B4 of one row from the values of its pool and of its scheme's control cells.

    B0 = log(1 + cells in the pool); B1 = control mean; B2 = B1 minus the pool mean (the negated own
    delta); B3 = share of pool cells with a non-zero value minus that share among the control cells;
    B4 = log((sd pool + 0.01) / (sd control + 0.01)), sample standard deviations
    (bioinf2027/plans/E1_design.md, Section 24.3).
    """
    pert = np.asarray(pert_values, dtype=np.float64)
    ctrl = np.asarray(ctrl_values, dtype=np.float64)
    b1 = float(ctrl.mean())
    sd_p = float(pert.std(ddof=1)) if pert.size > 1 else float("nan")
    sd_c = float(ctrl.std(ddof=1)) if ctrl.size > 1 else float("nan")
    return {
        "B0": float(np.log1p(pert.size)),
        "B1": b1,
        "B2": b1 - float(pert.mean()),
        "B3": float((pert > 0).mean() - (ctrl > 0).mean()),
        "B4": float(np.log((sd_p + 0.01) / (sd_c + 0.01))),
    }


def half_split(n: int, key) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves of n positions from one permutation; the first holds floor(n / 2), both ascending."""
    order = np.random.default_rng(key).permutation(n)
    return np.sort(order[: n // 2]), np.sort(order[n // 2 :])


def thin_counts(rng: np.random.Generator, counts: np.ndarray, rest: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binomial thinning of the needed columns of each cell and of the rest of the cell, with one p per cell.

    ``counts`` is (cells, columns) or (cells,), ``rest`` the cell's total minus its needed counts. The
    columns are drawn first, row by row in column order, then the rests; the thinned total is the sum
    of the thinned columns and the thinned rest.
    """
    counts = np.asarray(counts)
    p = np.asarray(p, dtype=np.float64)
    flat = counts.reshape(counts.shape[0], -1)
    thinned = rng.binomial(np.rint(flat).astype(np.int64), p[:, None]).astype(np.float32)
    thinned_rest = rng.binomial(np.rint(rest).astype(np.int64), p).astype(np.float32)
    total = thinned.sum(axis=1, dtype=np.float64) + thinned_rest
    return thinned.reshape(counts.shape), total.astype(np.float32)
