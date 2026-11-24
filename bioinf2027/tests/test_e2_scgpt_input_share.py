"""scGPT's input rebuilt without the GPU (make_e2_scgpt_input_share.py): the same genes as cell_input keeps with the
real gene identifiers, every gene of a cell with at most 1,199 expressed genes, exactly 1,199 otherwise."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e2_embed_scgpt as emb  # noqa: E402
import make_e2_scgpt_input_share as m  # noqa: E402


def cell(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 50, size=n).astype(np.float32)


def test_the_positions_kept_are_those_cell_input_keeps_with_the_real_identifiers():
    values = cell(3000)
    ids = np.random.default_rng(7).permutation(60000)[:3000] + 10
    genes, _ = emb.cell_input(values, ids, np.random.default_rng(42), 5, -2.0)
    kept = m.kept_positions(values, np.random.default_rng(42), -2.0)
    assert np.array_equal(ids[kept], genes[1:]) and kept.size == emb.MAX_LENGTH - 1 and np.unique(kept).size == kept.size


def test_a_cell_with_at_most_1199_expressed_genes_keeps_them_all():
    for n in (0, 1, 500, 1199):
        values = cell(n)
        assert np.array_equal(np.sort(m.kept_positions(values, np.random.default_rng(3), -2.0)), np.arange(n))


def test_the_share_before_the_cut_is_the_part_records_and_after_it_is_no_larger():
    values = cell(5000)
    total = float(values.sum()) * 1.25
    before, after = m.cell_shares(values, total, np.random.default_rng(9), -2.0)
    assert before == float(values.sum() / total) and 0 < after < before
    small = cell(800)
    b, a = m.cell_shares(small, float(small.sum()), np.random.default_rng(9), -2.0)
    assert b == a == 1.0
    assert m.cell_shares(small, 0.0, np.random.default_rng(9), -2.0) == (0.0, 0.0)
