"""Tests of the feature blocks, heads and weighted bootstrap of the transfer analyses.

The controls, the magnitude summaries, the feature mapping and the external bootstrap are checked
against the submitted code, loaded without writing bytecode into the historical tree.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, heads, representations as rep, resample  # noqa: E402

PIPELINE = config.repo_root() / "earlier_revision" / "pipeline"


def _historical(path: Path, name: str):
    if not path.exists():
        pytest.skip("%s is not part of this checkout" % path.name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    # A module defining dataclasses must be registered while it executes.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_controls_equal_the_submitted_generators():
    controls = _historical(PIPELINE / "controls.py", "controls_submitted")
    for mine, theirs in (("train", "train"), ("val", "val"), ("test", "test"), ("RPE1", "RPE1"), ("K562-essential", "K562")):
        assert np.array_equal(rep.c1_features(37, mine), controls.c1_features(37, theirs))
        assert np.array_equal(rep.c3_permutation(41, mine), controls.c3_permutation(41, theirs))
        assert np.array_equal(rep.c4_permutation(41, mine), controls.c4_permutation(41, theirs))
        z = np.arange(41 * 3, dtype=np.float32).reshape(41, 3)
        assert np.array_equal(np.concatenate([z, rep.c5_gaussian(41, mine)], axis=1), controls.c5_features(z, theirs))
    assert np.array_equal(rep.c2_projection(), controls.c2_projection_matrix())
    x = np.random.default_rng(0).normal(size=(5, 2000)).astype(np.float32)
    assert np.array_equal(rep.c2_features(x, rep.c2_projection()), controls.c2_features(x, controls.c2_projection_matrix()))


def test_realisations_shift_the_base_seed():
    assert np.array_equal(rep.c1_features(4, "val", realisation=2), np.random.default_rng(20270101 + 2000 + 1).standard_normal((4, 768)).astype(np.float32))
    assert np.array_equal(rep.c3_permutation(9, "RPE1", realisation=1), np.random.default_rng(20270103 + 1000 + 3).permutation(9))
    assert not np.array_equal(rep.c2_projection(realisation=1), rep.c2_projection())


def test_magnitude_equals_the_submitted_summaries():
    magnitude = _historical(config.repo_root() / "src" / "training" / "magnitude_augment.py", "magnitude_submitted")
    rng = np.random.default_rng(1)
    train = rng.normal(size=(60, 30)).astype(np.float32)
    other = rng.normal(size=(9, 30)).astype(np.float32)
    assert np.array_equal(rep.magnitude(train), magnitude._row_magnitude_features(train))

    class _Shard:
        def __init__(self, x):
            self.x = x

    theirs = magnitude.fit_magnitude_stats([_Shard(train)])
    mine = rep.fit_magnitude(train)
    assert np.array_equal(mine.mean, theirs.mean) and np.array_equal(mine.std, theirs.std)
    assert np.array_equal(mine.transform(rep.magnitude(other)), theirs.transform(magnitude._row_magnitude_features(other)))


def test_mapping_equals_the_submitted_contract():
    accumulator = _historical(PIPELINE / "x_cell_accumulator.py", "x_cell_accumulator_submitted")
    rng = np.random.default_rng(2)
    native_symbols = np.array(["A", "B", "C", "B", "D", "E"])
    features = np.array(["E", "B", "Z", "A", "C"])
    native = rng.normal(scale=6, size=(7, native_symbols.size)).astype(np.float32)
    native[0, 0] = np.nan
    native[1, 2] = np.inf
    mapped, n_mapped = rep.map_to_features(native, native_symbols, features)
    assert n_mapped == 4
    for row in range(native.shape[0]):
        theirs, count = accumulator.map_and_clip_to_vcc_hvg(native[row], native_symbols, features)
        assert count == n_mapped
        assert np.array_equal(mapped[row], theirs)


def test_mask_and_scaler():
    x = np.arange(12, dtype=np.float32).reshape(3, 4) + 1
    masked = rep.mask_target(x, np.array([2, -1, 0]))
    assert masked[0, 2] == 0 and masked[2, 0] == 0 and np.array_equal(masked[1], x[1])
    assert x[0, 2] == 3
    scaler = rep.BlockScaler(x)
    assert np.allclose(scaler(x).mean(axis=0), 0, atol=1e-6)
    assert scaler(x).dtype == np.float32


def _gathered_spearman(values, label, rows):
    a, b = values[rows], label[rows]
    return float(np.corrcoef(sps.rankdata(a), sps.rankdata(b))[0, 1])


def test_weighted_spearman_equals_ranking_the_gathered_rows():
    rng = np.random.default_rng(3)
    n = 80
    groups = rng.integers(0, 25, size=n).astype(str)
    label = np.round(rng.normal(size=n), 1)  # ties
    columns = np.column_stack([np.round(label + rng.normal(scale=0.8, size=n), 1), rng.integers(0, 4, size=n).astype(float)])
    draw = resample.cluster_draw(groups, 40, seed=11)
    got = resample.weighted_spearman(columns, label, draw)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng_check = np.random.default_rng(11)
    for replicate in range(40):
        picked = rng_check.integers(0, unique.size, unique.size)
        rows = np.concatenate([np.flatnonzero(inverse == g) for g in picked])
        for j in range(columns.shape[1]):
            assert got[replicate, j] == pytest.approx(_gathered_spearman(columns[:, j], label, rows), abs=1e-12)
    mask = rng.random(n) < 0.6
    sub = resample.weighted_spearman(columns, label, draw, mask=mask)
    rng_check = np.random.default_rng(11)
    for replicate in range(5):
        picked = rng_check.integers(0, unique.size, unique.size)
        rows = np.concatenate([np.flatnonzero(inverse == g) for g in picked])
        rows = rows[mask[rows]]
        assert sub[replicate, 0] == pytest.approx(_gathered_spearman(columns[:, 0], label, rows), abs=1e-12)


def test_weighted_spearman_reproduces_the_retained_external_bootstrap():
    wiring = _historical(PIPELINE / "bootstrap_wiring.py", "bootstrap_wiring_submitted")
    rng = np.random.default_rng(4)
    n = 120
    clusters = rng.integers(0, 70, size=n).astype(str)
    y = rng.normal(size=n)
    predictions = np.column_stack([y + rng.normal(size=n), rng.normal(size=n), y * 0.2 + rng.normal(size=n)])
    rho_boot, _ = wiring.external_paired_delta_rho(predictions, y, clusters, {"a": [0], "b": [1], "c": [2]}, B=30, seed=20260901)
    draw = resample.cluster_draw(clusters, 30, seed=20260901)
    mine = resample.weighted_spearman(predictions, y, draw)
    assert np.allclose(mine, rho_boot, atol=1e-12)


def test_weighted_pearson_and_r2():
    rng = np.random.default_rng(5)
    n = 50
    groups = rng.integers(0, 20, size=n)
    a, e = rng.normal(size=n), rng.normal(size=n)
    draw = resample.cluster_draw(groups, 10, seed=3)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng_check = np.random.default_rng(3)
    pearson = resample.weighted_pearson(a, e, draw)
    r2 = resample.weighted_r2(e, np.column_stack([a, e * 0.5]), draw)
    for replicate in range(10):
        picked = rng_check.integers(0, unique.size, unique.size)
        rows = np.concatenate([np.flatnonzero(inverse == g) for g in picked])
        assert pearson[replicate, 0] == pytest.approx(np.corrcoef(a[rows], e[rows])[0, 1], abs=1e-10)
        assert r2[replicate, 1] == pytest.approx(heads.pooled_r2(e[rows], e[rows] * 0.5), abs=1e-10)


def test_p_value_and_holm():
    assert resample.p_value(np.ones(9999)) == pytest.approx(2 / 10000)
    assert resample.p_value(np.r_[np.ones(50), -np.ones(50)]) == 1.0
    rejected, thresholds = resample.holm([0.001, 0.04, 0.012, np.nan], alpha=0.05)
    assert rejected.tolist() == [True, False, True, False]
    assert thresholds.tolist() == pytest.approx([0.05 / 4, 0.05 / 2, 0.05 / 3, 0.05 / 1])
    assert resample.holm_state(-0.2, True) == "Holm-negative"
    assert resample.holm_state(0.2, False) == "inconclusive"


def test_ridge_choice_and_heads():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 5))
    y = x @ np.array([1.0, 0.5, 0, 0, 0]) + rng.normal(scale=0.5, size=200)
    model, scaler, record = heads.fit_ridge_by_validation(x[:150], y[:150], x[150:], y[150:])
    assert record["chosen"] in heads.RIDGE_GRID
    assert record["validation_r2"][record["chosen"]] == max(record["validation_r2"].values())
    assert heads.forest(3).get_params()["n_jobs"] == 1
    assert heads.monotone([0, 1]).get_params()["monotonic_cst"] == [0, 1]
