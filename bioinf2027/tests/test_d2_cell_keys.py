"""Tests of the D2 cell keys and of a unit whose control cells recur with other genes dropped."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_d2_embeddings as d2emb  # noqa: E402

GENES = np.array(["ENSG0", "ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"])
TOKEN0 = 100


class FakeCounts:
    """Rows of a fixed count matrix; refuses rows that are not strictly increasing, as the real reader does."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix
        self.ensembl = GENES
        self.reads: list = []

    def rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if not np.all(np.diff(rows) > 0):
            raise ValueError("rows must be strictly increasing")
        self.reads.append(rows.tolist())
        return self.matrix[rows].astype(np.float32)


class FakeEncoder:
    """Rank encoding: tokens of the expressed genes by decreasing count (ties by column), gene i -> token 100 + i."""

    cls, eos = 2, 3

    def full_rankings(self, counts, var_ensembl, cell_ids, n_counts, work) -> dict:
        out = {}
        for key, row in zip(cell_ids, np.asarray(counts)):
            expressed = np.flatnonzero(row > 0)
            order = expressed[np.argsort(-row[expressed], kind="stable")]
            out[str(key)] = TOKEN0 + order.astype(np.int64)
        return out

    def columns_of(self, var_ensembl, ensembl) -> np.ndarray:
        return np.flatnonzero(np.asarray(var_ensembl) == ensembl)


def _frame(rows: list, genes: list, role: str) -> pd.DataFrame:
    return pd.DataFrame({"file": "vcc_train", "file_row": rows, "cell_id": [f"c{r}" for r in rows], "role": role,
                         "drop_ensembl": genes, "drop_token": [TOKEN0 + int(g[-1]) for g in genes]})


def test_keys_of_perturbed_rows_are_file_and_row():
    frame = _frame([5, 7], ["ENSG1", "ENSG2"], "perturbed")
    assert d2emb.cell_keys(frame) == ["vcc_train:5", "vcc_train:7"]


def test_keys_of_delta_rows_carry_the_dropped_gene():
    frame = _frame([5, 5, 7], ["ENSG1", "ENSG2", "ENSG1"], "delta")
    assert d2emb.cell_keys(frame) == ["vcc_train:5:ENSG1", "vcc_train:5:ENSG2", "vcc_train:7:ENSG1"]


def test_a_repeated_key_is_refused():
    with pytest.raises(RuntimeError):
        d2emb.cell_keys(_frame([5, 5], ["ENSG1", "ENSG1"], "delta"))
    with pytest.raises(RuntimeError):
        d2emb.cell_keys(_frame([5, 5], ["ENSG1", "ENSG2"], "perturbed"))


def test_a_recurring_control_cell_is_read_once_and_dropped_per_row(tmp_path):
    matrix = np.array([[0, 0, 0, 0, 0, 0],
                       [9, 7, 5, 3, 1, 2],
                       [1, 2, 3, 4, 5, 6],
                       [4, 0, 8, 0, 2, 6]], dtype=np.float32)
    reader = FakeCounts(matrix)
    frame = _frame([3, 1, 1, 2, 1], ["ENSG2", "ENSG0", "ENSG3", "ENSG5", "ENSG4"], "delta")
    inputs, records = d2emb.tokenize_unit(FakeEncoder(), frame, {"vcc_train": reader}, tmp_path, drop=True)
    assert reader.reads == [[1, 2, 3]]
    assert records["key"].tolist() == ["vcc_train:1:ENSG0", "vcc_train:1:ENSG3", "vcc_train:1:ENSG4",
                                       "vcc_train:2:ENSG5", "vcc_train:3:ENSG2"]
    encoder = FakeEncoder()
    for key in records["key"]:
        _, row, gene = key.split(":")
        column = int(gene[-1])
        expected = matrix[int(row)].copy()
        expected[column] = 0.0
        ranking = encoder.full_rankings(expected[None, :], GENES, ["k"], None, tmp_path)["k"]
        assert np.array_equal(inputs[key][1:-1], ranking)
        assert TOKEN0 + column not in inputs[key]
    assert records["target_in_input"].all()


def test_perturbed_cells_are_tokenized_as_before(tmp_path):
    matrix = np.array([[0, 0, 0, 0, 0, 0],
                       [9, 7, 5, 3, 1, 2],
                       [1, 2, 3, 4, 5, 6]], dtype=np.float32)
    reader = FakeCounts(matrix)
    frame = _frame([2, 1], ["ENSG1", "ENSG0"], "perturbed")
    inputs, records = d2emb.tokenize_unit(FakeEncoder(), frame, {"vcc_train": reader}, tmp_path, drop=True)
    assert reader.reads == [[1, 2]]
    assert records["key"].tolist() == ["vcc_train:1", "vcc_train:2"]
    assert TOKEN0 + 0 not in inputs["vcc_train:1"] and TOKEN0 + 1 not in inputs["vcc_train:2"]
