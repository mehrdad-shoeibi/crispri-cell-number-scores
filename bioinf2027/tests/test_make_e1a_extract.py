"""Tests of the E1a extraction passes on small synthetic files in the layouts of the real ones."""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import make_e1a_extract as extract  # noqa: E402
from bioinf2027 import endpoint  # noqa: E402


def _text(values) -> np.ndarray:
    return np.array([str(v).encode() for v in values], dtype="S")


def _write_annotations(handle, obs: dict, var: dict) -> None:
    for name, table in (("obs", obs), ("var", var)):
        group = handle.create_group(name)
        group.attrs["_index"] = "_index"
        for key, values in table.items():
            group.create_dataset(key, data=_text(values))


def _vcc_file(path: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(12)]
    targets = rng.choice(["G1", "G4", "G7", "non-targeting"], size=60, p=[0.25, 0.25, 0.2, 0.3])
    batch = rng.choice(["Flex_1_01", "Flex_2_01"], size=60)
    counts = rng.poisson(1.0, (60, 12)).astype(np.float32)
    counts[7] = 0.0
    obs = {"_index": [f"cell{i}" for i in range(60)], "target_gene": targets, "batch": batch, "guide_id": [f"g{t}" for t in targets]}
    var = {"_index": genes, "gene_id": [f"ENSG{i:04d}" for i in range(12)]}
    matrix = sp.csr_matrix(counts)
    with h5py.File(path, "w") as handle:
        _write_annotations(handle, obs, var)
        group = handle.create_group("X")
        group.attrs["encoding-type"] = "csr_matrix"
        group.attrs["shape"] = np.array(counts.shape)
        group.create_dataset("data", data=matrix.data.astype(np.float32))
        group.create_dataset("indices", data=matrix.indices.astype(np.int32))
        group.create_dataset("indptr", data=matrix.indptr.astype(np.int64))
    return counts, obs, var


def _screen_file(path: Path, seed: int = 1):
    rng = np.random.default_rng(seed)
    ids = [f"ENSG{i:04d}" for i in range(3, 15)]
    gene = rng.choice(["A", "B", "C", "non-targeting"], size=50)
    transcript = np.array([f"{g}_t" if g != "non-targeting" else "nt" for g in gene])
    counts = rng.poisson(1.2, (50, 12)).astype(np.float32)
    obs = {"_index": [f"c{i}" for i in range(50)], "gene": gene, "gene_transcript": transcript, "gem_group": rng.integers(1, 4, 50)}
    var = {"_index": ids}
    with h5py.File(path, "w") as handle:
        _write_annotations(handle, obs, var)
        handle.create_dataset("X", data=counts)
    table = pd.DataFrame(
        {"construct": ["A_t", "C_t"], "target_col": [2, 9], "target_ensg": [ids[2], ids[9]], "eligible": [True, True]}
    )
    return counts, obs, table


def test_extract_vcc_values_follow_both_modes(tmp_path):
    path = tmp_path / "vcc.h5ad"
    counts, obs, var = _vcc_file(path)
    targets = np.array(["G1", "G4", "G7"])
    shared = {"S": np.isin(np.array(var["gene_id"]), ["ENSG0003", "ENSG0004", "ENSG0010"])}
    out = extract.extract_vcc(path, targets, shared, controls=True, chunk_rows=7)

    totals = counts.sum(axis=1)
    assert np.array_equal(out["all_totals"], totals)
    control = np.array(obs["target_gene"]) == "non-targeting"
    nt_rows = np.flatnonzero(control)
    pert_rows = np.flatnonzero(~control)
    assert np.array_equal(out["nt_rows"], nt_rows) and np.array_equal(out["pert_rows"], pert_rows)
    cols = np.array([1, 4, 7])
    assert np.array_equal(out["nt_values_ext"], endpoint.external_values(counts[nt_rows][:, cols], totals[nt_rows]))
    assert np.array_equal(out["nt_values_src"], endpoint.source_values(counts[nt_rows][:, cols], totals[nt_rows]))
    own = np.array([int(t[1:]) for t in np.array(obs["target_gene"])[pert_rows]])
    assert np.array_equal(out["pert_own_ext"], endpoint.external_values(counts[pert_rows, own], totals[pert_rows]))
    assert np.array_equal(out["pert_own_src"], endpoint.source_values(counts[pert_rows, own], totals[pert_rows]))
    assert np.array_equal(out["nt_shared_totals_S"], counts[nt_rows][:, shared["S"]].sum(axis=1))
    assert out["pert_guide"].tolist() == [f"g{t}" for t in np.array(obs["target_gene"])[pert_rows]]


def test_extract_vcc_without_controls_skips_the_control_values(tmp_path):
    path = tmp_path / "vcc.h5ad"
    _vcc_file(path)
    out = extract.extract_vcc(path, np.array(["G1"]), {}, controls=False, chunk_rows=11)
    assert "nt_values_ext" not in out and "nt_barcode" in out


def test_extract_vcc_refuses_a_target_outside_the_annotation(tmp_path):
    path = tmp_path / "vcc.h5ad"
    _vcc_file(path)
    with pytest.raises(RuntimeError, match="absent"):
        extract.extract_vcc(path, np.array(["G99"]), {}, controls=True)


def test_extract_screen_values_follow_the_external_builder(tmp_path):
    path = tmp_path / "screen.h5ad"
    counts, obs, table = _screen_file(path)
    out = extract.extract_screen("S", path, table, np.array(["ENSG0004", "ENSG0005"]), chunk_rows=9)

    totals = counts.sum(axis=1)
    gene = np.array(obs["gene"])
    nt_rows = np.flatnonzero(gene == "non-targeting")
    assert np.array_equal(out["nt_rows"], nt_rows)
    assert np.array_equal(out["target_cols"], np.array([2, 9]))
    expected_nt = np.log1p(counts[nt_rows][:, [2, 9]].astype(np.float64) / np.maximum(totals[nt_rows], 1.0)[:, None] * 1e4).astype(np.float32)
    assert np.array_equal(out["nt_values"], expected_nt)
    transcript = np.array(obs["gene_transcript"])
    pert_rows = np.flatnonzero(np.isin(transcript, ["A_t", "C_t"]))
    assert np.array_equal(out["pert_rows"], pert_rows)
    col = np.where(transcript[pert_rows] == "A_t", 2, 9)
    expected_own = np.log1p(counts[pert_rows, col].astype(np.float64) / np.maximum(totals[pert_rows], 1.0) * 1e4).astype(np.float32)
    assert np.array_equal(out["pert_own"], expected_own)
    shared = np.isin(np.array([f"ENSG{i:04d}" for i in range(3, 15)]), ["ENSG0004", "ENSG0005"])
    assert np.array_equal(out["nt_shared_totals"], counts[nt_rows][:, shared].sum(axis=1))
    assert int(out["shared_genes"]) == 2


@pytest.mark.parametrize("kind", ["vcc", "screen"])
def test_every_extracted_array_can_be_cached_without_pickling(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(extract, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(extract.config, "portable", lambda path: "portable/" + Path(path).name)
    if kind == "vcc":
        path = tmp_path / "vcc.h5ad"
        _vcc_file(path)
        out = extract.extract_vcc(path, np.array(["G1", "G4"]), {}, controls=True)
    else:
        path = tmp_path / "screen.h5ad"
        _, _, table = _screen_file(path)
        out = extract.extract_screen("S", path, table, np.array(["ENSG0004"]))
    assert all(np.asarray(v).dtype != object for v in out.values())
    record = extract.save_cache(kind, out)
    with np.load(tmp_path / "cache" / f"{kind}.npz", allow_pickle=False) as store:
        assert set(store.files) == set(out)
    assert record["bytes"] > 0


def test_extract_screen_refuses_target_columns_with_another_identifier(tmp_path):
    path = tmp_path / "screen.h5ad"
    _, _, table = _screen_file(path)
    table.loc[0, "target_ensg"] = "ENSG9999"
    with pytest.raises(RuntimeError, match="identifiers"):
        extract.extract_screen("S", path, table, np.array([]))
