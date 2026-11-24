"""Test of the retention table that supports the choice of a perturbed-cell floor."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_d3_vcc_retention import retention_table  # noqa: E402


def test_retention_table_counts_rows_and_genes_over_the_modelled_rows():
    counts = pd.DataFrame(
        {
            "split": ["training"] * 6 + ["validation"] * 2 + ["test"] * 2,
            "target_gene": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "batch": ["b1", "b2"] * 5,
            "n_perturbed_cells": [5, 30, 12, 26, 40, 27, 28, 9, 50, 60],
        }
    )
    table = retention_table(counts, {"A", "E"}).set_index(["split", "perturbed_cell_floor"])

    train10, train27 = table.loc[("train", 10)], table.loc[("train", 27)]
    assert train10["groups_with_any_cells"] == 6
    assert train10["rows_modelled"] == train10["rows_retained"] == 5
    assert train10["target_genes_modelled"] == 3
    assert train27["rows_retained"] == 3  # 30, 40 and 27 cells
    assert train27["rows_retained_share_of_modelled"] == pytest.approx(3 / 5)
    assert train27["rows_retained_share_of_all_groups"] == pytest.approx(3 / 6)
    assert train27["target_genes_retained"] == 2  # A and C; B is lost
    assert train27["rows_retained_target_in_features"] == 1
    assert train27["cells_per_retained_row_min"] == 27
    assert table.loc[("val", 27), "rows_retained"] == 1
    assert table.loc[("test", 27), "target_genes_retained_in_features"] == 1
