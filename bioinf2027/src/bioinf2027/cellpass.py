"""Streaming reads of the single-cell count files, in consecutive row chunks.

Both layouts of the project's files are read with h5py alone: the VCC files
hold a CSR matrix (``X/data``, ``X/indices``, ``X/indptr``), the external
screens a dense single-precision matrix. Every chunk is checked: counts must be
finite, non-negative integers, and every cell total must stay below 2**24, so
that a single-precision total is exact and independent of the order of
summation. The retained builders take single-precision totals, so under this
check the totals here equal theirs value for value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import scipy.sparse as sp

__all__ = ["EXACT_LIMIT", "Chunk", "csr_chunks", "dense_chunks", "matrix_shape"]

EXACT_LIMIT = float(2**24)


@dataclass
class Chunk:
    """One block of consecutive cells, rows ``start`` to ``stop - 1`` of the file."""

    start: int
    stop: int
    matrix: object
    totals: np.ndarray

    @property
    def sparse(self) -> bool:
        return sp.issparse(self.matrix)

    def columns(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Counts of the given local rows at the given columns, single precision."""
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if self.sparse:
            return np.asarray(self.matrix[rows][:, columns].toarray(), dtype=np.float32)
        return np.ascontiguousarray(self.matrix[rows][:, columns], dtype=np.float32)

    def pairs(self, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Count of each local row at its own column, single precision."""
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        if rows.shape != columns.shape:
            raise ValueError("one column per row is required")
        if not rows.size:
            return np.empty(0, dtype=np.float32)
        if self.sparse:
            return np.asarray(self.matrix[rows, columns], dtype=np.float32).ravel()
        return np.asarray(self.matrix[rows, columns], dtype=np.float32)

    def masked_totals(self, rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Single-precision totals of the given local rows over the genes where ``mask`` is true."""
        rows = np.asarray(rows, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        if self.sparse:
            return _csr_totals(self.matrix[rows][:, np.flatnonzero(mask)].tocsr())
        return np.ascontiguousarray(self.matrix[rows][:, mask]).sum(axis=1)


def _check_counts(values: np.ndarray) -> None:
    if values.dtype != np.float32:
        raise TypeError("counts must be stored in single precision, found %s" % values.dtype)
    if not np.isfinite(values).all():
        raise ValueError("non-finite count")
    if (values < 0).any():
        raise ValueError("negative count")
    if not np.array_equal(values, np.floor(values)):
        raise ValueError("non-integer count")


def _csr_totals(matrix) -> np.ndarray:
    data = np.asarray(matrix.data)
    out = np.zeros(matrix.shape[0], dtype=np.float32)
    lengths = np.diff(matrix.indptr)
    filled = lengths > 0
    if data.size:
        out[filled] = np.add.reduceat(data.astype(np.float32, copy=False), matrix.indptr[:-1][filled])
    return out


def _check_totals(totals: np.ndarray) -> None:
    if totals.size and float(totals.max()) >= EXACT_LIMIT:
        raise ValueError("a cell total reaches 2**24; single-precision totals would not be exact")


def matrix_shape(handle) -> tuple[int, int]:
    node = handle["X"]
    if hasattr(node, "keys"):
        return tuple(int(v) for v in node.attrs["shape"])
    return tuple(int(v) for v in node.shape)


def csr_chunks(handle, chunk_rows: int = 4000) -> Iterator[Chunk]:
    """Consecutive row chunks of a CSR count matrix stored under ``X``."""
    group = handle["X"]
    encoding = group.attrs.get("encoding-type")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding != "csr_matrix":
        raise ValueError("expected a csr_matrix, found %s" % encoding)
    n_rows, n_cols = matrix_shape(handle)
    indptr = np.asarray(group["indptr"][()], dtype=np.int64)
    if indptr.size != n_rows + 1:
        raise ValueError("indptr does not match the matrix shape")
    data, indices = group["data"], group["indices"]
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        lo, hi = int(indptr[start]), int(indptr[stop])
        values = np.asarray(data[lo:hi])
        _check_counts(values)
        matrix = sp.csr_matrix(
            (values, np.asarray(indices[lo:hi]), indptr[start : stop + 1] - lo), shape=(stop - start, n_cols)
        )
        totals = _csr_totals(matrix)
        _check_totals(totals)
        yield Chunk(start, stop, matrix, totals)


def dense_chunks(handle, chunk_rows: int = 4000) -> Iterator[Chunk]:
    """Consecutive row chunks of a dense count matrix stored under ``X``."""
    matrix = handle["X"]
    if hasattr(matrix, "keys"):
        raise ValueError("expected a dense matrix")
    n_rows = int(matrix.shape[0])
    for start in range(0, n_rows, chunk_rows):
        stop = min(start + chunk_rows, n_rows)
        block = np.asarray(matrix[start:stop, :])
        _check_counts(block)
        totals = block.sum(axis=1)  # single precision, as run_replogle_endpoint.py:111
        _check_totals(totals)
        yield Chunk(start, stop, block, totals)


