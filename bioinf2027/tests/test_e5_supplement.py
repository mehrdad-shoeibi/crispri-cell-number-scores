"""make_e5_supplement.py: the family-mean feature (leave one out in training) and the interval states."""
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e5_supplement as sup  # noqa: E402


def test_family_feature_leaves_each_training_gene_out():
    prefix_of = {"RPL1": "RPL", "RPL2": "RPL", "RPL3": "RPL", "PSMA1": "PSMA", "ABC": "ABC", "RPL9": "RPL", "XYZ": "XYZ"}
    tr = ["RPL1", "RPL2", "RPL3", "PSMA1", "ABC"]
    y = np.array([1.0, 2.0, 6.0, 4.0, 10.0])
    te = ["RPL9", "XYZ"]
    fn_tr, fn_te = sup.family_feature(tr, te, y, prefix_of)
    assert np.allclose(fn_tr[:3], [(2 + 6) / 2, (1 + 6) / 2, (1 + 2) / 2])
    assert fn_tr[3] == (23.0 - 4.0) / 4 and fn_tr[4] == (23.0 - 10.0) / 4  # no other member: the mean of the other training genes
    assert fn_te[0] == 3.0 and fn_te[1] == 23.0 / 5  # a test gene: the training mean of its prefix, else the training mean


def test_family_feature_of_test_genes_is_the_family_baseline_rule():
    rng = np.random.default_rng(1)
    prefixes = ["A", "B", "C", "D"]
    genes = [f"G{i}" for i in range(40)]
    prefix_of = {g: prefixes[i % 4] for i, g in enumerate(genes)}
    tr, te = genes[:30], genes[30:]
    y = rng.normal(size=30)
    _, fn_te = sup.family_feature(tr, te, y, prefix_of)
    for g, v in zip(te, fn_te):
        members = [y[i] for i, h in enumerate(tr) if prefix_of[h] == prefix_of[g]]
        assert abs(v - np.mean(members)) < 1e-12


def test_states():
    assert sup.state(0.1, 0.2) == "above zero" and sup.state(-0.2, -0.1) == "below zero"
    assert sup.state(-0.1, 0.1) == "contains zero" and sup.state(float("nan"), 1.0) == "undefined"
