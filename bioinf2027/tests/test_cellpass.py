"""Tests of the chunked single-cell reads against dense recomputation."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import cellpass  # noqa: E402


def _counts(seed: int = 0, cells: int = 37, genes: int = 23) -> np.ndarray:
    rng = np.random.default_rng(seed)
    block = rng.poisson(0.7, (cells, genes)).astype(np.float32)
    block[4] = 0.0
    return block


def _write_csr(path: Path, block: np.ndarray) -> None:
    matrix = sp.csr_matrix(block)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = np.array(block.shape)
        group.create_dataset("data", data=matrix.data.astype(np.float32))
        group.create_dataset("indices", data=matrix.indices.astype(np.int32))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))


def _write_dense(path: Path, block: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=block)


@pytest.mark.parametrize("layout", ["csr", "dense"])
def test_chunks_reproduce_totals_columns_pairs_and_masked_totals(tmp_path, layout):
    block = _counts()
    path = tmp_path / "counts.h5"
    (_write_csr if layout == "csr" else _write_dense)(path, block)
    columns = np.array([0, 5, 22])
    mask = np.zeros(block.shape[1], dtype=bool)
    mask[[1, 2, 9, 17]] = True
    with h5py.File(path, "r") as handle:
        chunks = list((cellpass.csr_chunks if layout == "csr" else cellpass.dense_chunks)(handle, chunk_rows=10))
    assert [c.start for c in chunks] == [0, 10, 20, 30]
    for chunk in chunks:
        part = block[chunk.start : chunk.stop]
        assert chunk.totals.dtype == np.float32
        assert np.array_equal(chunk.totals, part.sum(axis=1))
        rows = np.arange(part.shape[0])[::2]
        assert np.array_equal(chunk.columns(rows, columns), part[rows][:, columns])
        own = (rows * 3) % part.shape[1]
        assert np.array_equal(chunk.pairs(rows, own), part[rows, own])
        assert np.array_equal(chunk.masked_totals(rows, mask), part[rows][:, mask].sum(axis=1))


def test_non_integer_counts_are_refused(tmp_path):
    block = _counts()
    block[2, 3] = 0.5
    path = tmp_path / "counts.h5"
    _write_dense(path, block)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="non-integer"):
        list(cellpass.dense_chunks(handle, chunk_rows=10))


def test_a_total_beyond_exact_single_precision_is_refused(tmp_path):
    block = _counts()
    block[1, 0] = 2.0**24
    path = tmp_path / "counts.h5"
    _write_csr(path, block)
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="2\\*\\*24"):
        list(cellpass.csr_chunks(handle, chunk_rows=10))


def test_csr_reader_refuses_another_encoding(tmp_path):
    path = tmp_path / "counts.h5"
    _write_csr(path, _counts())
    with h5py.File(path, "a") as handle:
        handle["X"].attrs["encoding-type"] = "csc_matrix"
    with h5py.File(path, "r") as handle, pytest.raises(ValueError, match="csr_matrix"):
        list(cellpass.csr_chunks(handle))
