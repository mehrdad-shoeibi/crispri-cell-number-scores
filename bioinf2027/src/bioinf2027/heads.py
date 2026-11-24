"""Prediction heads of the transfer analyses, as fixed in `bioinf2027/plans/E1_design.md`, Section 10.

- Primary: the random forest of the submitted analysis (200 trees, maximum depth 20, minimum leaf
  size 2; ``earlier_revision/pipeline/fm_heads.py:25-29``), one thread.
- Secondary: gradient boosting in the submitted configuration (``fm_heads.py:32-37``), and ridge on
  standardised inputs with the penalty chosen by validation R-squared from a fixed grid.
- The positive control of the label construction: boosting with a monotone constraint and no early
  stopping.
- The nuisance model of the deciding statistic R and of the R5 gate (Section 24.3).

Every head runs on one thread: the forest through ``n_jobs=1``, boosting through the OpenMP limit
the calling script sets before scikit-learn is imported, and :func:`single_thread` around each fit.
"""
from __future__ import annotations

import contextlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

__all__ = [
    "FOREST",
    "BOOSTING",
    "MONOTONE",
    "NUISANCE",
    "RIDGE_GRID",
    "forest",
    "boosting",
    "monotone",
    "nuisance",
    "single_thread",
    "pooled_r2",
    "fit_ridge_by_validation",
]

FOREST = {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 2}
BOOSTING = {
    "max_iter": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
}
MONOTONE = {"max_iter": 500, "max_depth": 6, "learning_rate": 0.05, "early_stopping": False}
NUISANCE = {"max_iter": 200, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 20, "early_stopping": False}
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


def forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(**FOREST, n_jobs=1, random_state=int(seed))


def boosting(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**BOOSTING, random_state=int(seed))


def monotone(constraints) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**MONOTONE, monotonic_cst=list(constraints), random_state=0)


def nuisance() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**NUISANCE, random_state=0)


@contextlib.contextmanager
def single_thread():
    """Limit every native thread pool to one thread for the duration of a fit."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        yield
        return
    with threadpool_limits(limits=1):
        yield


def pooled_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return float("nan")
    return float(1.0 - ((y - prediction) ** 2).sum() / total)


def fit_ridge_by_validation(x_train, y_train, x_val, y_val, grid=RIDGE_GRID) -> tuple[Ridge, StandardScaler, dict]:
    """Ridge on inputs standardised with a scaler fitted on the training rows.

    The penalty is the grid value with the largest validation R-squared; ties go to the smaller
    penalty. Returns the refitted model, the scaler and the validation R-squared of every penalty.
    """
    scaler = StandardScaler().fit(np.asarray(x_train, dtype=np.float64))
    xt = scaler.transform(np.asarray(x_train, dtype=np.float64))
    xv = scaler.transform(np.asarray(x_val, dtype=np.float64))
    scores = {}
    models = {}
    for alpha in grid:
        model = Ridge(alpha=float(alpha)).fit(xt, y_train)
        scores[float(alpha)] = pooled_r2(y_val, model.predict(xv))
        models[float(alpha)] = model
    finite = [a for a in scores if np.isfinite(scores[a])]
    if not finite:
        raise ValueError("no penalty gave a finite validation R-squared")
    best = max(finite, key=lambda a: (scores[a], -a))
    return models[best], scaler, {"validation_r2": scores, "chosen": best, "at_grid_edge": best in (min(grid), max(grid))}
