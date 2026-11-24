"""make_e2_null_external_followup.py: the partial Spearman, the fifths of n_t and the interval states."""
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sps

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e2_null_external_followup as fu  # noqa: E402
from bioinf2027 import resample  # noqa: E402


def _partial(p, y, n):
    rp, ry, rn = (sps.rankdata(v) for v in (p, y, n))
    r = np.corrcoef(np.vstack([rp, ry, rn]))
    return (r[0, 1] - r[0, 2] * r[1, 2]) / np.sqrt((1 - r[0, 2] ** 2) * (1 - r[1, 2] ** 2))


def test_partial_spearman_with_unit_weights_equals_the_direct_formula():
    rng = np.random.default_rng(3)
    n = rng.integers(27, 300, 400).astype(float)
    y = -0.002 * n + rng.normal(size=400)
    p = 0.5 * y + 0.01 * n + rng.normal(size=400)
    draw = resample.ClusterDraw(weights=np.ones((1, 400)), clusters=np.array([]), seed=-1, index_sha256="")
    assert abs(fu.partial_spearman(p, y, n, draw)[0] - _partial(p, y, n)) < 1e-12


def test_partial_spearman_with_integer_weights_equals_the_expanded_sample():
    rng = np.random.default_rng(4)
    n = rng.integers(27, 300, 60).astype(float)
    y, p = rng.normal(size=60), rng.normal(size=60)
    w = rng.integers(0, 3, 60).astype(float)
    draw = resample.ClusterDraw(weights=w[None, :], clusters=np.array([]), seed=-1, index_sha256="")
    keep = np.repeat(np.arange(60), w.astype(int))
    assert abs(fu.partial_spearman(p, y, n, draw)[0] - _partial(p[keep], y[keep], n[keep])) < 1e-10


def test_fifths_partition_the_constructs():
    n = np.random.default_rng(5).integers(27, 500, 1932).astype(float)
    masks = fu.fifths(n)
    assert len(masks) == 5
    assert np.array_equal(np.sum(masks, axis=0), np.ones(n.size))
    sizes = [int(m.sum()) for m in masks]
    assert min(sizes) > 0.15 * n.size and max(sizes) < 0.25 * n.size


def test_states():
    assert fu.state(0.01, 0.2) == "above zero"
    assert fu.state(-0.2, -0.01) == "below zero"
    assert fu.state(-0.1, 0.1) == "contains zero"
    assert fu.state(float("nan"), 0.1) == "undefined"


def test_default_label_writes_where_it_did():
    assert fu.LABEL == "rebuilt" and fu.OUT_OVERRIDE is None
    assert fu.out_dir() == fu.OUT
    assert fu.STATUS == fu.STATUS_BY_LABEL["rebuilt"]


def test_existing_label_writes_to_its_own_folder_and_restores():
    try:
        fu.set_label("existing")
        assert fu.out_dir() == fu.OUT / "existing_label"
        assert "06:50, item 4" in fu.STATUS
    finally:
        fu.set_label("rebuilt")
    assert fu.out_dir() == fu.OUT


def test_unknown_label_is_refused():
    import pytest

    with pytest.raises(SystemExit):
        fu.set_label("all")
