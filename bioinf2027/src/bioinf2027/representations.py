"""Feature blocks of the transfer analyses.

The random controls, the magnitude summaries and the mapping of a screen's genes to the 2,000
features repeat the code of the submitted analysis exactly:

- the controls: ``earlier_revision/pipeline/controls.py`` (seeds 20270101 to 20270105, one
  offset per split or screen);
- the magnitude summaries and their training-row z-score: ``src/training/magnitude_augment.py:35-75``;
- the mapping to the features: ``earlier_revision/pipeline/x_cell_accumulator.py:259-276``
  (first column with the symbol, zero filled, non-finite values to 0 or +/-10, clipped to +/-10);
- the block standardisation: ``earlier_revision/pipeline/fm_preprocess.py`` (a scaler fitted on
  training rows only, applied in double precision, returned in single precision).

Tests compare every function here with those files. A realisation k of a control adds 1,000 k to
its base seed (`bioinf2027/plans/E1_design.md`, Section 10, null band); k = 0 is the submitted set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler

__all__ = [
    "SPLIT_OFFSET",
    "C1_SEED",
    "C2_SEED",
    "C3_SEED",
    "C4_SEED",
    "C5_SEED",
    "REALISATION_STEP",
    "c1_features",
    "c2_projection",
    "c2_features",
    "c3_permutation",
    "c4_permutation",
    "c5_gaussian",
    "magnitude",
    "MagnitudeStats",
    "fit_magnitude",
    "mask_target",
    "map_to_features",
    "BlockScaler",
]

# Offsets of the submitted controls: train, validation, test, RPE1, K562-essential.
SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2, "RPE1": 3, "K562-essential": 4}
C1_SEED = 20270101
C2_SEED = 20270102
C3_SEED = 20270103
C4_SEED = 20270104
C5_SEED = 20270105
REALISATION_STEP = 1000
Z_DIM = 768
X_DIM = 2000


def _offset(split: str) -> int:
    if split not in SPLIT_OFFSET:
        raise ValueError("unknown split or screen %r" % split)
    return SPLIT_OFFSET[split]


def c1_features(n_rows: int, split: str, realisation: int = 0, dim: int = Z_DIM) -> np.ndarray:
    """Gaussian features from the row count only: N(0, 1), single precision."""
    rng = np.random.default_rng(C1_SEED + REALISATION_STEP * realisation + _offset(split))
    return rng.standard_normal((int(n_rows), int(dim))).astype(np.float32)


def c2_projection(realisation: int = 0, x_dim: int = X_DIM, out_dim: int = Z_DIM) -> np.ndarray:
    """The fixed random projection, entries N(0, 1 / out_dim), drawn once per realisation."""
    rng = np.random.default_rng(C2_SEED + REALISATION_STEP * realisation)
    return (rng.standard_normal((int(x_dim), int(out_dim))) / np.sqrt(out_dim)).astype(np.float32)


def c2_features(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[1] != matrix.shape[0]:
        raise ValueError("x has %d columns, the projection %d rows" % (x.shape[1], matrix.shape[0]))
    return (x @ matrix).astype(np.float32)


def c3_permutation(n_rows: int, split: str, realisation: int = 0) -> np.ndarray:
    """The row permutation that shuffles z within one split or screen."""
    rng = np.random.default_rng(C3_SEED + REALISATION_STEP * realisation + _offset(split))
    return rng.permutation(int(n_rows))


def c4_permutation(n_rows: int, split: str) -> np.ndarray:
    return np.random.default_rng(C4_SEED + _offset(split)).permutation(int(n_rows))


def c5_gaussian(n_rows: int, split: str) -> np.ndarray:
    rng = np.random.default_rng(C5_SEED + _offset(split))
    return rng.standard_normal((int(n_rows), 4)).astype(np.float32)


def magnitude(x: np.ndarray) -> np.ndarray:
    """[mean |x|, ||x||_2, max |x|, population sd of x] per row, single precision."""
    x = np.asarray(x)
    if x.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    absolute = np.abs(x)
    out = np.empty((x.shape[0], 4), dtype=np.float32)
    out[:, 0] = absolute.mean(axis=1)
    out[:, 1] = np.linalg.norm(x, axis=1)
    out[:, 2] = absolute.max(axis=1)
    out[:, 3] = x.std(axis=1, ddof=0)
    return out


@dataclass(frozen=True)
class MagnitudeStats:
    """Mean and population sd of the four summaries over training rows."""

    mean: np.ndarray
    std: np.ndarray

    def transform(self, summaries: np.ndarray) -> np.ndarray:
        if summaries.shape[0] == 0:
            return summaries.astype(np.float32)
        denominator = np.where(self.std > 0, self.std, 1.0)
        return ((summaries - self.mean) / denominator).astype(np.float32)


def fit_magnitude(train_x: np.ndarray) -> MagnitudeStats:
    summaries = magnitude(np.asarray(train_x, dtype=np.float32))
    if summaries.shape[0] == 0:
        raise ValueError("no training rows")
    return MagnitudeStats(mean=summaries.mean(axis=0).astype(np.float32), std=summaries.std(axis=0, ddof=0).astype(np.float32))


def mask_target(x: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """A copy of x with each row's target coordinate set to zero; -1 marks rows without one."""
    columns = np.asarray(columns, dtype=np.int64)
    if columns.shape[0] != x.shape[0]:
        raise ValueError("one target column per row is needed")
    out = np.array(x, copy=True)
    rows = np.flatnonzero(columns >= 0)
    out[rows, columns[rows]] = 0
    return out


def map_to_features(native: np.ndarray, native_symbols, feature_symbols) -> tuple[np.ndarray, int]:
    """Rows of a screen's native delta matrix in the feature namespace, and the number of mapped features."""
    native = np.asarray(native)
    native_symbols = np.asarray(native_symbols).astype(str)
    feature_symbols = np.asarray(feature_symbols).astype(str)
    if native.shape[1] != native_symbols.size:
        raise ValueError("the native matrix has %d columns and %d symbols" % (native.shape[1], native_symbols.size))
    first: dict[str, int] = {}
    for index, symbol in enumerate(native_symbols.tolist()):
        first.setdefault(symbol, index)
    source = np.array([first.get(symbol, -1) for symbol in feature_symbols.tolist()], dtype=np.int64)
    present = source >= 0
    out = np.zeros((native.shape[0], feature_symbols.size), dtype=np.float32)
    out[:, np.flatnonzero(present)] = np.asarray(native, dtype=np.float32)[:, source[present]]
    out = np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
    out = np.clip(out, -10.0, 10.0)
    return out, int(present.sum())


class BlockScaler:
    """A standard scaler fitted on training rows in double precision; output in single precision."""

    def __init__(self, train_block: np.ndarray) -> None:
        self.scaler = StandardScaler(with_mean=True, with_std=True).fit(np.asarray(train_block, dtype=np.float64))

    def __call__(self, block: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(block, dtype=np.float64)).astype(np.float32)

    def near_constant(self, tolerance: float = 1e-12) -> int:
        return int((np.asarray(self.scaler.var_) <= tolerance).sum())
