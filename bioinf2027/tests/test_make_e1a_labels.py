"""Tests of the E1a label stage on synthetic inputs: count extraction, row pooling, thinning, per-row work."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_e1a_label_extract as second  # noqa: E402
import make_e1a_labels as labels  # noqa: E402
from bioinf2027 import endpoint  # noqa: E402
from test_make_e1a_extract import _screen_file, _vcc_file  # noqa: E402


def test_extract_vcc_counts_keeps_raw_counts_and_totals(tmp_path):
    path = tmp_path / "vcc.h5ad"
    counts, obs, var = _vcc_file(path)
    targets = np.array(["G1", "G4", "G7"])
    shared = {"S": np.isin(np.array(var["gene_id"]), ["ENSG0001", "ENSG0004"])}
    out = second.extract_vcc_counts(path, targets, shared, controls=True, chunk_rows=7)
    control = np.array(obs["target_gene"]) == "non-targeting"
    nt, pert = np.flatnonzero(control), np.flatnonzero(~control)
    assert np.array_equal(out["nt_counts"], counts[nt][:, [1, 4, 7]])
    assert np.array_equal(out["nt_total"], counts[nt].sum(axis=1))
    own = np.array([int(t[1:]) for t in np.array(obs["target_gene"])[pert]])
    assert np.array_equal(out["pert_count"], counts[pert, own])
    assert np.array_equal(out["pert_total"], counts[pert].sum(axis=1))
    assert np.array_equal(out["pert_shared_total_S"], counts[pert][:, shared["S"]].sum(axis=1))
    assert np.array_equal(out["nt_shared_total_S"], counts[nt][:, shared["S"]].sum(axis=1))


def test_extract_screen_counts_keeps_raw_counts_and_totals(tmp_path):
    path = tmp_path / "screen.h5ad"
    counts, obs, table = _screen_file(path)
    out = second.extract_screen_counts("S", path, table, np.array(["ENSG0004"]), chunk_rows=9)
    gene = np.array(obs["gene"])
    nt = np.flatnonzero(gene == "non-targeting")
    assert np.array_equal(out["nt_counts"], counts[nt][:, [2, 9]])
    assert np.array_equal(out["nt_total"], counts[nt].sum(axis=1))
    transcript = np.array(obs["gene_transcript"])
    pert = np.flatnonzero(np.isin(transcript, ["A_t", "C_t"]))
    col = np.where(transcript[pert] == "A_t", 2, 9)
    assert np.array_equal(out["pert_count"], counts[pert, col])
    assert out["constructs"].dtype.kind == "U"


def _rows():
    return pd.DataFrame({
        "split": ["train", "train", "train"],
        "batch": ["Flex_1_01", "Flex_2_01", "Flex_1_01"],
        "target_gene": ["A", "A", "B"],
        "existing_label": [1.0, 3.0, 2.0],
        "positions": [np.array([0, 1, 2]), np.array([5, 6, 7, 8, 9]), np.array([3, 4])],
        "n_pool": [3, 5, 2],
        "n_guides": [1, 2, 1],
        "flex_run": [1, 2, 1],
        "target_is_feature": [True, True, False],
    })


def test_gene_rows_pool_modelled_rows_and_weight_the_existing_label():
    genes = labels.gene_rows(_rows(), n_pert=4)
    assert genes["target_gene"].tolist() == ["A"]
    row = genes.iloc[0]
    assert row["positions"].tolist() == [0, 1, 2, 5, 6, 7, 8, 9]
    assert row["existing_label"] == pytest.approx((3 * 1.0 + 5 * 3.0) / 8)
    assert row["n_guides"] == 2 and row["n_rows_pooled"] == 2


def test_cell_rows_keep_rows_at_the_cell_size():
    kept = labels.cell_rows(_rows(), {"unit": "row", "n_pert": 3})
    assert kept["n_pool"].tolist() == [3, 5]


def _counts(seed=0, n_nt=60, n_pert=40):
    rng = np.random.default_rng(seed)
    nt_counts = rng.poisson(2.0, (n_nt, 2)).astype(np.float32)
    nt_total = (nt_counts.sum(axis=1) + rng.integers(200, 400, n_nt)).astype(np.float32)
    vcc = {"vcc_train": {"nt_counts": nt_counts, "nt_total": nt_total,
                         "nt_shared_total_S": (nt_total * 0.6).astype(np.float32)}}
    for split in ("train", "val", "test"):
        pert_count = rng.poisson(1.0, n_pert).astype(np.float32)
        pert_total = (pert_count + rng.integers(200, 400, n_pert)).astype(np.float32)
        vcc.setdefault("vcc_" + split, {}).update({
            "pert_count": pert_count, "pert_total": pert_total,
            "pert_shared_total_S": (pert_total * 0.6).astype(np.float32),
        })
    return vcc


def test_thin_vcc_leaves_shallow_cells_and_thins_deep_ones_in_the_fixed_order():
    counts = _counts()
    out = labels.thin_vcc(counts, "S", target=1e9)
    assert np.array_equal(out["nt"][0], counts["vcc_train"]["nt_counts"])
    assert np.array_equal(out["nt"][1], counts["vcc_train"]["nt_total"])
    shallow = labels.thin_vcc(counts, "S", target=50.0)
    assert (shallow["nt"][1] < counts["vcc_train"]["nt_total"]).all()
    again = labels.thin_vcc(counts, "S", target=50.0)
    assert np.array_equal(again["test"][0], shallow["test"][0])


def test_vcc_work_matches_a_direct_build_and_bounds_the_simulation(monkeypatch):
    rng = np.random.default_rng(4)
    n_nt = 2000  # halves of 1,000 hold the 788 controls a half label draws
    values = {"nt_ext": rng.gamma(2.0, 1.0, (n_nt, 1)).astype(np.float32)}
    values["nt_src"] = values["nt_ext"]
    pert = rng.gamma(3.0, 1.0, 120).astype(np.float32)
    for split in ("train", "val", "test"):
        values[f"{split}_ext"] = pert
        values[f"{split}_src"] = pert
    rows = pd.DataFrame({"split": ["train", "train"], "batch": ["b0", "b1"], "target_gene": ["A", "A"],
                         "positions": [np.arange(0, 60), np.arange(60, 90)], "n_pool": [60, 30],
                         "existing_label": [1.0, 2.0], "flex_run": [1, 1], "n_guides": [1, 1], "target_is_feature": [True, True]})
    spec = {"n_pert": 27, "controls": "pooled", "n_ctrl": 788, "unit": "row", "norm": "all",
            "screens": (), "halves": (27, 10), "simulate": True}
    monkeypatch.setitem(labels.CELLS, "T", spec)
    data = {"cell_rows": {"T": rows}, "vcc_values": {"all": values}, "target_slot": {"A": 0},
            "pooled_index": np.arange(n_nt), "pooled_halves": endpoint.half_split(n_nt, 20270408)}
    monkeypatch.setattr(labels, "DATA", data)
    records = labels.vcc_work("T", 0, 2)
    direct, _ = endpoint.subsampled_label(pert[:60], values["nt_ext"][:, 0], 27, 788)
    assert records[0]["label"]["y"] == direct["y"]
    assert np.isfinite(records[0]["label"]["y_draws_1_12"]) and np.isfinite(records[0]["label"]["y_draws_13_25"])
    assert {h["level"] for h in records[0]["halves"]} == {27, 10}
    assert {h["level"] for h in records[1]["halves"]} == {10}
    sizes = [s["size"] for s in records[0]["simulation"]]
    assert len(sizes) == 5 and all(27 <= s <= 60 for s in sizes)
    assert records[1]["simulation"] == []  # 30 cells hold fewer than 2 x 27
