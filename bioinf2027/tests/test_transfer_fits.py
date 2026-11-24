"""Tests of the fitting harness on a small synthetic context."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import heads, representations as rep, transfer  # noqa: E402


def _side(name, split, n, rng, with_release=False):
    x = rng.normal(size=(n, 2000)).astype(np.float32)
    target = rng.integers(-1, 2000, size=n)
    sides_x = {"main": x}
    if with_release:
        sides_x["release"] = (x * 0.5).astype(np.float32)
    b = rng.normal(size=(n, 5))
    return transfer.Side(name=name, control_split=split, z=rng.normal(size=(n, 768)).astype(np.float32), x=sides_x, b=b, target_column=target)


@pytest.fixture()
def context(monkeypatch):
    # The fits write to a temporary folder, which the portable record refuses by design.
    monkeypatch.setattr(transfer, "file_record", lambda path: {"path": str(Path(path).name)})
    monkeypatch.setitem(heads.FOREST, "n_estimators", 3)
    monkeypatch.setitem(heads.BOOSTING, "max_iter", 5)
    rng = np.random.default_rng(7)
    train = _side("train", "train", 60, rng)
    sides = {"val": _side("val", "val", 20, rng), "test": _side("test", "test", 25, rng), "RPE1": _side("RPE1", "RPE1", 30, rng, with_release=True)}
    y = train.b[:, 2] + rng.normal(scale=0.1, size=60)
    labels = {"rebuilt": y, "rebuilt@val": sides["val"].b[:, 2]}
    genes = np.repeat(np.array([f"g{i}" for i in range(12)]), 5)
    meta = {"n_guides": np.ones(60, dtype=int), "genes": genes, "context_index": 3}
    return transfer.Context(name="T", train=train, labels=labels, sides=sides, meta=meta)


def test_forest_fit_saves_every_side_and_is_reused(context, tmp_path):
    spec = transfer.FitSpec("T", "rebuilt", "x", "forest", 0)
    first = transfer.run_fit(context, spec, tmp_path, 20270414, {"RPE1": ("release",)})
    assert not first["reused"]
    _, record, values = transfer.load_predictions(tmp_path / spec.file_name())
    assert set(values) == {"val", "test", "RPE1", "RPE1|release"}
    assert values["RPE1"].shape == (30,) and np.isfinite(values["RPE1"]).all()
    again = transfer.run_fit(context, spec, tmp_path, 20270414, {"RPE1": ("release",)})
    assert again["reused"]
    assert record["training_rows"] == 60


def test_c3_is_the_standardised_z_shuffled_within_the_split(context):
    spec = transfer.FitSpec("T", "rebuilt", "C3", "forest", 0)
    blocks = transfer.Blocks(spec, context.train, np.arange(60))
    scaled = rep.BlockScaler(context.train.z)(context.train.z)
    assert np.array_equal(blocks.matrix(context.train), scaled[rep.c3_permutation(60, "train")])
    external = blocks.matrix(context.sides["RPE1"])
    assert np.array_equal(external, rep.BlockScaler(context.train.z)(context.sides["RPE1"].z)[rep.c3_permutation(30, "RPE1")])


def test_refit_draws_whole_genes_with_replacement(context):
    spec = transfer.FitSpec("T", "rebuilt", "z", "forest", 0, variant="refit 4")
    rows = transfer.training_rows(context, spec, 20270414)
    genes = context.meta["genes"]
    drawn = genes[rows]
    for gene in set(drawn.tolist()):
        assert (drawn == gene).sum() % 5 == 0
    rng = np.random.default_rng([20270414, 4, 3])
    unique = np.unique(genes)
    assert np.array_equal(np.unique(drawn), np.unique(unique[rng.integers(0, unique.size, unique.size)]))


def test_feature_space_baseline_only_where_the_target_is_a_feature(context, tmp_path):
    spec = transfer.FitSpec("T", "rebuilt", "B2 feature space", "forest", 0)
    transfer.run_fit(context, spec, tmp_path, 20270414, {})
    _, record, values = transfer.load_predictions(tmp_path / spec.file_name())
    has = context.sides["test"].target_column >= 0
    assert np.isfinite(values["test"][has]).all() and np.isnan(values["test"][~has]).all()
    assert record["training_rows"] == int((context.train.target_column >= 0).sum())


def test_ridge_records_its_penalty_and_masked_arms_differ(context, tmp_path):
    spec = transfer.FitSpec("T", "rebuilt", "TR", "ridge", 0)
    transfer.run_fit(context, spec, tmp_path, 20270414, {})
    _, record, _ = transfer.load_predictions(tmp_path / spec.file_name())
    assert record["ridge"]["chosen"] in heads.RIDGE_GRID and len(record["ridge"]["validation_r2"]) == len(heads.RIDGE_GRID)
    masked = transfer.Blocks(transfer.FitSpec("T", "rebuilt", "x-", "forest", 0), context.train, np.arange(60)).matrix(context.train)
    plain = transfer.Blocks(transfer.FitSpec("T", "rebuilt", "x", "forest", 0), context.train, np.arange(60)).matrix(context.train)
    has = np.flatnonzero(context.train.target_column >= 0)
    assert not np.allclose(masked[has, context.train.target_column[has]], plain[has, context.train.target_column[has]])


def test_null_realisations_change_the_control(context):
    first = transfer.Blocks(transfer.FitSpec("T", "rebuilt", "C1", "forest", 0), context.train, np.arange(60)).matrix(context.sides["test"])
    other = transfer.Blocks(transfer.FitSpec("T", "rebuilt", "C1", "forest", 0, realisation=5), context.train, np.arange(60)).matrix(context.sides["test"])
    assert not np.allclose(first, other)
