"""scGPT's input and the target gene (make_e2_scgpt_target_input.py, e2_scgpt_draws.py; gate G3, item 5 (c)): the
draws of the helper are those of make_e2_scgpt_input_share.kept_positions, and the helper gives the same draws in the
pipeline environment and in scGPT's."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import e2_scgpt_draws as d  # noqa: E402
import make_e2_scgpt_input_share as share  # noqa: E402
import make_e2_scgpt_target_input as m  # noqa: E402


def cells(sizes, seed=5):
    rng = np.random.default_rng(seed)
    values = [rng.integers(1, 60, size=n).astype(np.float32) for n in sizes]
    offsets = np.concatenate([[0], np.cumsum([v.size for v in values])]).astype(np.int64)
    return np.concatenate(values), offsets, np.array([11, 2**63 + 5, 7, 2**40], dtype=np.uint64)[: len(sizes)]


def test_the_helper_keeps_what_the_rebuild_keeps():
    values, offsets, seeds = cells((3000, 1199, 5, 2500))
    kept, _, bounds = d.draws(values, offsets, seeds, -2.0)
    for k in range(seeds.size):
        cell = values[offsets[k]: offsets[k + 1]]
        expected = share.kept_positions(cell, np.random.default_rng(int(seeds[k])), -2.0)
        assert np.array_equal(kept[bounds[k]: bounds[k + 1]], expected)
    assert np.array_equal(np.diff(bounds), [1199, 1199, 5, 1199])


def test_the_helper_gives_the_same_draws_in_scgpt_s_environment(tmp_path):
    try:
        other = m.scgpt_python()
    except m.Stop:
        pytest.skip("scGPT's environment is not beside this one")
    values, offsets, seeds = cells((4000, 900, 1300))
    source = tmp_path / "in.npz"
    np.savez(source, values=values, offsets=offsets, seeds=seeds, pad_value=np.array(-2.0))
    out = {}
    for name, python in (("here", sys.executable), ("scgpt", str(other))):
        subprocess.run([python, "-B", d.__file__, str(source), str(tmp_path / f"{name}.npz")], check=True)
        with np.load(tmp_path / f"{name}.npz") as z:
            out[name] = {k: z[k] for k in z.files}
    assert all(np.array_equal(out["here"][k], out["scgpt"][k]) for k in ("kept", "binned", "bounds"))
    assert str(out["scgpt"]["numpy"]) != "" and str(out["here"]["numpy"]) != ""


def test_the_share_given_among_cells_with_the_target_expressed():
    s = m.shares(np.array([True, True, True, False]), np.array([True, True, False, False]), np.array([True, False, False, False]))
    assert s["share_target_among_the_genes_given_to_scgpt"] == 0.25 and s["share_given_among_cells_with_the_target_expressed"] == 0.5
    assert s["share_target_in_scgpt_vocabulary"] == 0.75 and s["cells_target_given"] == 1
