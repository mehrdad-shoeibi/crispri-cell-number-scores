"""The E1 grid cells as E1b fits them: rows, labels, trivial predictors and feature blocks.

Every per-row table of E1a is read from the data root and refused unless it matches the digest in
its cell's committed manifest (``bioinf2027/endpoints/<cell>/manifest.json``). The cells, their row
unit, control scheme and external endpoint are those of `bioinf2027/plans/E1_design.md`, Section
24.2. Features follow Section 10: batch-local x and z in G5 and G6 (and their DT cells), x and z
re-referenced to the pooled controls in G1 to G4 (and the SU and G1-DT cells), and in G3 and G4 the
cell-weighted means of the re-referenced row features over the gene's modelled rows (Section 6).

The VCC rows of a cell are kept in the loader's order (split, then the order of the feature shards),
the order in which the submitted controls were generated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, e1inputs, features
from .provenance import sha256

__all__ = ["CELLS", "PRIMARY", "POOLED_PAIR", "GENE_PAIR", "CellData", "load_tables", "build_cell", "TRIVIAL"]

TRIVIAL = ("B0", "B1", "B2", "B3", "B4")
PRIMARY = ("G5", "G6")
POOLED_PAIR = ("G1", "G2")
GENE_PAIR = ("G3", "G4")


def _cells() -> dict:
    cells = {
        "G1": {"n_pert": 27, "features": "pooled", "unit": "row", "screens": e1inputs.SCREENS},
        "G2": {"n_pert": 10, "features": "pooled", "unit": "row", "screens": e1inputs.SCREENS},
        "G3": {"n_pert": 27, "features": "pooled", "unit": "gene", "screens": e1inputs.SCREENS},
        "G4": {"n_pert": 10, "features": "pooled", "unit": "gene", "screens": e1inputs.SCREENS},
        "G5": {"n_pert": 27, "features": "batch", "unit": "row", "screens": e1inputs.SCREENS},
        "G6": {"n_pert": 10, "features": "batch", "unit": "row", "screens": e1inputs.SCREENS},
    }
    for screen in e1inputs.SCREENS:
        cells[f"G1-SU-{screen}"] = {"n_pert": 27, "features": "pooled", "unit": "row", "screens": (screen,), "base": "G1"}
        for base in ("G1", "G5", "G6"):
            cells[f"{base}-DT-{screen}"] = {**{k: v for k, v in cells[base].items() if k != "screens"}, "screens": (screen,), "base": base}
    return cells


CELLS = _cells()


def endpoint_root():
    return config.storage_root() / "bioinf2027" / "endpoints"


def manifest(cell: str) -> dict:
    return json.loads((config.repo_root() / "bioinf2027" / "endpoints" / cell / "manifest.json").read_text())


def load_tables(cell: str) -> dict[str, pd.DataFrame]:
    """The cell's VCC table and one external table per screen, each checked against its digest."""
    own = manifest(cell)
    out = {}
    path = endpoint_root() / cell / "vcc_labels.csv"
    if sha256(path) != own["tables"]["vcc_labels.csv"]["sha256"]:
        raise RuntimeError("%s/vcc_labels.csv does not match its manifest" % cell)
    out["vcc"] = pd.read_csv(path, float_precision="round_trip")
    for screen in CELLS[cell]["screens"]:
        name = f"external_{screen}.csv"
        source = cell if name in own["tables"] else own.get("external_reused_from")
        if source is None:
            raise RuntimeError("%s has no external table for %s" % (cell, screen))
        record = manifest(source)["tables"][name]
        path = endpoint_root() / source / name
        if sha256(path) != record["sha256"]:
            raise RuntimeError("%s/%s does not match its manifest" % (source, name))
        table = pd.read_csv(path, float_precision="round_trip")
        if not np.array_equal(table["construct"].astype(str).to_numpy(), np.sort(table["construct"].astype(str).to_numpy())):
            raise RuntimeError("%s/%s is not sorted by construct" % (source, name))
        out[screen] = table
        out[f"{screen}_source"] = source
    return out


@dataclass
class CellData:
    """One cell: VCC rows in the loader's order with labels, B0 to B4 and features."""

    name: str
    spec: dict
    vcc: pd.DataFrame
    x: np.ndarray
    z: np.ndarray
    target_column: np.ndarray
    external: dict = field(default_factory=dict)
    external_source: dict = field(default_factory=dict)

    def split(self, name: str) -> np.ndarray:
        return (self.vcc["split"] == name).to_numpy()


def build_cell(cell: str, rows: e1inputs.VccRows, means: e1inputs.ControlMeans, modelled: pd.DataFrame | None = None) -> CellData:
    """Rows, labels and feature blocks of one cell.

    ``modelled`` is the G2 (or G6) table of every modelled row, needed for the gene-level cells: it
    holds each row's perturbed-cell count, the weight of the gene-level means.
    """
    spec = CELLS[cell]
    tables = load_tables(cell)
    vcc = tables["vcc"].copy()
    symbols = features.hvg_symbols()
    position = {s: i for i, s in enumerate(symbols.tolist())}
    rank = {s: i for i, s in enumerate(e1inputs.SPLITS)}
    if spec["unit"] == "row":
        index = rows.index(vcc)
        vcc["_loader"] = rows.shard_position[index]
        vcc = vcc.assign(_index=index).sort_values("_loader", kind="stable").reset_index(drop=True)
        index = vcc["_index"].to_numpy()
        x = rows.x[index].astype(np.float64)
        if spec["features"] == "pooled":
            shift = means.shift()
            x = x + np.stack([shift[b] for b in vcc["batch"]])
            pooled_nt = means.pooled_embedding(rows.batch_mu_nt)
            z = rows.mu_pert[index].astype(np.float64) - pooled_nt[None, :]
        else:
            z = rows.z[index].astype(np.float64)
        existing = rows.existing[index]
        if not np.allclose(existing, vcc["existing_label"].to_numpy(), rtol=0, atol=1e-6):
            raise RuntimeError("%s: the existing label of the shards differs from the E1a table" % cell)
    else:
        if modelled is None:
            raise ValueError("gene-level cells need the table of modelled rows")
        vcc["_order"] = [(rank[s], g) for s, g in zip(vcc["split"], vcc["target_gene"])]
        vcc = vcc.sort_values("_order", kind="stable").drop(columns="_order").reset_index(drop=True)
        index = rows.index(modelled)
        shift = means.shift()
        row_x = rows.x[index].astype(np.float64) + np.stack([shift[b] for b in modelled["batch"]])
        pooled_nt = means.pooled_embedding(rows.batch_mu_nt)
        row_z = rows.mu_pert[index].astype(np.float64) - pooled_nt[None, :]
        weights = modelled["n_pool"].to_numpy(dtype=np.float64)
        x = e1inputs.gene_level(vcc, modelled, row_x, weights)
        z = e1inputs.gene_level(vcc, modelled, row_z, weights)
    target_column = np.array([position.get(g, -1) for g in vcc["target_gene"].astype(str)], dtype=np.int64)
    if not np.array_equal(target_column >= 0, vcc["target_is_feature"].to_numpy(dtype=bool)):
        raise RuntimeError("%s: the target-feature flag disagrees with the feature list" % cell)
    data = CellData(
        name=cell,
        spec=spec,
        vcc=vcc.drop(columns=[c for c in ("_loader", "_index") if c in vcc]),
        x=x.astype(np.float32),
        z=z.astype(np.float32),
        target_column=target_column,
    )
    for screen in spec["screens"]:
        data.external[screen] = tables[screen]
        data.external_source[screen] = tables[f"{screen}_source"]
    return data
