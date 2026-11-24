"""Single-cell count files: annotations, normalisation and group means.

The count files are read with h5py alone. The normalisation and the order of
the summation follow the recorded expression pass exactly, so that a repeated
pass can be compared with its persisted outputs value by value:

    per cell      v = log1p(counts / cell total * 10,000), double precision
    per chunk     rows of one group are summed over the chunk
    across chunks chunk sums are added to a plain double-precision running sum
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SCALE",
    "GroupSums",
    "accumulate_group_sums",
    "normalised_log1p",
    "read_annotation",
]

SCALE = 10000.0


def _as_text(values) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind in ("S", "O"):
        return np.array(
            [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values.tolist()],
            dtype=str,
        )
    return values.astype(str)


def _from_codes(codes, categories: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes).astype(np.int64)
    out = np.full(codes.shape, "", dtype=categories.dtype if categories.size else "<U1")
    known = codes >= 0
    out[known] = categories[codes[known]]
    return out


def read_annotation(group, name: str) -> np.ndarray:
    """One annotation column of a count file as an array of text.

    Three encodings are handled: integer codes whose categories are stored
    under ``__categories`` in the same group, a sub-group holding ``codes`` and
    ``categories``, and a plain dataset of text. A missing category code gives
    an empty string.
    """
    node = group[name]
    if hasattr(node, "keys"):
        return _from_codes(node["codes"][()], _as_text(node["categories"][()]))
    values = node[()]
    if "__categories" in group and name in group["__categories"]:
        return _from_codes(values, _as_text(group["__categories"][name][()]))
    return _as_text(values)


def normalised_log1p(counts) -> np.ndarray:
    """Per-cell log1p of counts scaled to 10,000 per cell, in double precision.

    The total is taken over every gene of the row and a zero total is replaced
    by one, so an empty cell gives zeros instead of a division error.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(counts / total * SCALE)


@dataclass
class GroupSums:
    """Running sums of normalised expression per group."""

    sums: np.ndarray
    counts: np.ndarray
    n_cells: int
    n_chunks: int

    def copy(self) -> "GroupSums":
        return GroupSums(self.sums.copy(), self.counts.copy(), self.n_cells, self.n_chunks)

    def mean(self, group: int) -> np.ndarray:
        if self.counts[group] <= 0:
            raise ValueError("group %d has no cells" % group)
        return self.sums[group] / self.counts[group]


def accumulate_group_sums(
    matrix,
    rows: np.ndarray,
    codes: np.ndarray,
    n_groups: int,
    chunk: int = 5000,
    snapshot_after: int | None = None,
    read_dtype=None,
) -> tuple[GroupSums, GroupSums | None]:
    """Sum normalised expression per group over the given rows, chunk by chunk.

    ``matrix`` is a cells by genes count matrix that supports row slicing, for
    example an open h5py dataset. ``rows`` are ascending row indices and
    ``codes`` gives the group of each of them. The rows are consumed in
    consecutive chunks; inside a chunk the rows of a group are summed in row
    order and the chunk sum is added to that group's running sum.

    When ``snapshot_after`` is given, a copy of the state after that many
    chunks is returned as the second value, so an interrupted earlier pass can
    be compared with this one at the same point.

    ``read_dtype`` casts each chunk of counts on read, before normalisation.
    The recorded expression pass cast to single precision, so a repetition of
    it passes ``numpy.float32``; the cast changes nothing for counts stored in
    single precision or as integers below 2**24.
    """
    rows = np.asarray(rows, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.int64)
    if rows.shape != codes.shape or rows.ndim != 1:
        raise ValueError("rows and codes must be one-dimensional and of equal length")
    if rows.size and (np.diff(rows) <= 0).any():
        raise ValueError("rows must be strictly ascending")
    if codes.size and (codes.min() < 0 or codes.max() >= n_groups):
        raise ValueError("a group code is outside the declared number of groups")

    n_genes = int(matrix.shape[1])
    state = GroupSums(
        sums=np.zeros((n_groups, n_genes), dtype=np.float64),
        counts=np.zeros(n_groups, dtype=np.int64),
        n_cells=0,
        n_chunks=0,
    )
    snapshot = None
    for start in range(0, rows.size, chunk):
        chunk_rows = rows[start : start + chunk]
        chunk_codes = codes[start : start + chunk]
        first, last = int(chunk_rows[0]), int(chunk_rows[-1])
        block = np.asarray(matrix[first : last + 1])
        counts = block[chunk_rows - first]
        if read_dtype is not None:
            counts = counts.astype(read_dtype)
        if not np.isfinite(counts).all():
            raise ValueError("non-finite values in the count matrix")
        if (counts < 0).any():
            raise ValueError("negative values in the count matrix")

        values = normalised_log1p(counts)
        for group in np.unique(chunk_codes):
            selected = chunk_codes == group
            state.sums[group] = state.sums[group] + values[selected].sum(axis=0)
            state.counts[group] += int(selected.sum())
        state.n_cells += int(chunk_rows.size)
        state.n_chunks += 1
        if snapshot_after is not None and state.n_chunks == snapshot_after:
            snapshot = state.copy()
    return state, snapshot
