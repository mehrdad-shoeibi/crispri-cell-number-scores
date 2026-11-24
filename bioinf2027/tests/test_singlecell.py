"""Tests of the single-cell reader and of the chunked group sums."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import singlecell  # noqa: E402

try:
    import h5py
except ImportError:  # the pinned environment has it; other interpreters may not
    h5py = None

needs_h5py = pytest.mark.skipif(h5py is None, reason="h5py is not installed")


def _counts(seed: int = 0, n_cells: int = 60, n_genes: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)


def test_normalised_log1p_scales_each_cell_to_ten_thousand():
    counts = np.array([[1.0, 3.0], [0.0, 5.0]], dtype=np.float32)
    values = singlecell.normalised_log1p(counts)
    assert values.dtype == np.float64
    assert np.expm1(values).sum(axis=1) == pytest.approx([10000.0, 10000.0])


def test_normalised_log1p_gives_zeros_for_an_empty_cell():
    values = singlecell.normalised_log1p(np.zeros((1, 4), dtype=np.float32))
    assert np.array_equal(values, np.zeros((1, 4)))


def test_group_sums_equal_a_direct_computation():
    counts = _counts()
    rows = np.array([0, 1, 4, 5, 9, 20, 21, 22, 40, 59])
    codes = np.array([0, 1, 1, 2, 0, 2, 2, 1, 0, 1])
    state, _ = singlecell.accumulate_group_sums(counts, rows, codes, n_groups=3, chunk=4)
    values = singlecell.normalised_log1p(counts[rows])
    for group in range(3):
        assert state.counts[group] == int((codes == group).sum())
        assert state.sums[group] == pytest.approx(values[codes == group].sum(axis=0), abs=1e-12)
    assert state.n_cells == rows.size
    assert state.n_chunks == 3


def test_group_means_do_not_depend_on_the_chunk_size_beyond_rounding():
    counts = _counts(1, n_cells=200)
    rows = np.arange(0, 200, 2)
    codes = np.arange(rows.size) % 4
    small, _ = singlecell.accumulate_group_sums(counts, rows, codes, n_groups=4, chunk=7)
    large, _ = singlecell.accumulate_group_sums(counts, rows, codes, n_groups=4, chunk=1000)
    for group in range(4):
        assert small.mean(group) == pytest.approx(large.mean(group), abs=1e-12)


def test_snapshot_is_the_state_after_the_requested_chunk():
    counts = _counts(2, n_cells=50)
    rows = np.arange(50)
    codes = np.arange(50) % 2
    full, snapshot = singlecell.accumulate_group_sums(
        counts, rows, codes, n_groups=2, chunk=10, snapshot_after=3
    )
    prefix, _ = singlecell.accumulate_group_sums(counts, rows[:30], codes[:30], n_groups=2, chunk=10)
    assert snapshot.n_chunks == 3 and snapshot.n_cells == 30
    assert np.array_equal(snapshot.sums, prefix.sums)
    assert np.array_equal(snapshot.counts, prefix.counts)
    assert full.n_chunks == 5 and not np.array_equal(full.sums, snapshot.sums)


def test_accumulate_rejects_unsorted_rows_and_bad_codes():
    counts = _counts()
    with pytest.raises(ValueError):
        singlecell.accumulate_group_sums(counts, np.array([3, 1]), np.array([0, 0]), n_groups=1)
    with pytest.raises(ValueError):
        singlecell.accumulate_group_sums(counts, np.array([1, 3]), np.array([0, 2]), n_groups=2)


def test_accumulate_rejects_negative_counts():
    counts = _counts()
    counts[2, 1] = -1.0
    with pytest.raises(ValueError):
        singlecell.accumulate_group_sums(counts, np.arange(5), np.zeros(5, dtype=int), n_groups=1)


@needs_h5py
def test_accumulate_reads_an_h5py_dataset_like_an_array(tmp_path):
    counts = _counts(3, n_cells=40)
    rows = np.array([1, 2, 3, 10, 11, 30, 39])
    codes = np.array([0, 0, 1, 1, 0, 1, 0])
    path = tmp_path / "counts.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=counts)
    with h5py.File(path, "r") as handle:
        from_file, _ = singlecell.accumulate_group_sums(handle["X"], rows, codes, n_groups=2, chunk=3)
    from_array, _ = singlecell.accumulate_group_sums(counts, rows, codes, n_groups=2, chunk=3)
    assert np.array_equal(from_file.sums, from_array.sums)


@needs_h5py
def test_read_annotation_handles_the_three_encodings(tmp_path):
    path = tmp_path / "annotations.h5"
    with h5py.File(path, "w") as handle:
        obs = handle.create_group("obs")
        obs.create_dataset("legacy", data=np.array([1, 0, 1, -1], dtype=np.int16))
        obs.create_group("__categories").create_dataset(
            "legacy", data=np.array([b"control", b"geneA"])
        )
        modern = obs.create_group("modern")
        modern.create_dataset("codes", data=np.array([0, 1, 1, 0], dtype=np.int8))
        modern.create_dataset("categories", data=np.array([b"x", b"y"]))
        obs.create_dataset("plain", data=np.array([b"a", b"b", b"c", b"d"]))
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        assert singlecell.read_annotation(obs, "legacy").tolist() == ["geneA", "control", "geneA", ""]
        assert singlecell.read_annotation(obs, "modern").tolist() == ["x", "y", "y", "x"]
        assert singlecell.read_annotation(obs, "plain").tolist() == ["a", "b", "c", "d"]


def test_read_dtype_casts_each_chunk_before_normalising():
    rng = np.random.default_rng(5)
    counts = rng.gamma(2.0, 1.7, size=(30, 6))  # double precision, not representable in single
    rows, codes = np.arange(30), np.arange(30) % 3
    cast, _ = singlecell.accumulate_group_sums(
        counts, rows, codes, n_groups=3, chunk=7, read_dtype=np.float32
    )
    reference, _ = singlecell.accumulate_group_sums(
        counts.astype(np.float32), rows, codes, n_groups=3, chunk=7
    )
    uncast, _ = singlecell.accumulate_group_sums(counts, rows, codes, n_groups=3, chunk=7)
    assert np.array_equal(cast.sums, reference.sums)
    assert not np.array_equal(uncast.sums, reference.sums)
