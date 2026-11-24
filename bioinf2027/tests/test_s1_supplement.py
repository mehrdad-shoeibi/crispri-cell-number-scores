"""make_s1_supplement.py: the paired correlation against the separate cluster bootstrap, states and the bootstrap mean."""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_s1_supplement as s1  # noqa: E402
from bioinf2027 import stats  # noqa: E402


def _data(seed=7, genes=40, per=6):
    rng = np.random.default_rng(seed)
    g = np.repeat([f"g{i}" for i in range(genes)], per)
    n = rng.integers(10, 120, g.size).astype(float)
    y1 = 0.01 * n + rng.normal(size=g.size)
    y2 = rng.normal(size=g.size)
    return n, y1, y2, g


def test_paired_point_estimates_equal_the_separate_ones():
    n, y1, y2, g = _data()
    old = s1.B
    try:
        s1.B = 300
        value = s1.paired_corr(n, y1, y2, g, seed=11)
        e1 = stats.cluster_bootstrap_corr(y1, n, g, b=300, seed=11)
        e2 = stats.cluster_bootstrap_corr(y2, n, g, b=300, seed=11)
    finally:
        s1.B = old
    assert abs(value["first"] - e1.spearman) < 1e-12 and abs(value["second"] - e2.spearman) < 1e-12
    assert abs(value["estimate"] - (e1.spearman - e2.spearman)) < 1e-12
    assert value["lo"] <= value["estimate"] <= value["hi"]


def test_states():
    assert s1.state(0.1, 0.2) == "above zero" and s1.state(-0.2, -0.1) == "below zero" and s1.state(-0.1, 0.1) == "contains zero"


def test_gene_bootstrap_mean_point():
    rng = np.random.default_rng(3)
    v = rng.normal(size=60)
    g = np.repeat(np.arange(20), 3)
    old = s1.B
    try:
        s1.B = 200
        out = s1.gene_bootstrap_mean(v, g, seed=5)
    finally:
        s1.B = old
    assert abs(out["estimate"] - v.mean()) < 1e-15 and out["target_genes"] == 20


def test_default_writes_to_the_results():
    assert not s1.SMOKE and s1.out_dir() == s1.OUT
