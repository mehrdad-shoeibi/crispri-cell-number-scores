"""Feature inputs of the E1b evaluation, read from the persisted artifacts of the submitted work.

VCC, per (split, batch, target gene) row (`bioinf2027/plans/E1_design.md`, Sections 1 and 10):

- x, the delta over the 2,000 features against the row's batch controls, from the feature shards
  (``IGA_MAIN/cache/vcc/variants/delta_hvg/shards``), unclipped, single precision;
- the existing label, the shards' ``y``;
- the encoder's mean embedding of the row's perturbed cells and of its batch's control cells, and
  their difference z (``geneformer_row_features_full.parquet``);
- the per-batch control means over all genes (``cache/vcc/nt_baseline/mean_log1p.npy``) and the
  per-batch control-cell counts (``nt_per_batch`` of ``subsample_sizes.json``), which re-reference
  x and z to the pooled controls: x + mean_b - mean_pooled, and mu_pert - sum_b n_b mu_nt,b / sum_b n_b.

External screens, per eligible construct in construct order: the stored embedding z (with its
construct index), the cell-level delta in the native gene space, its mapping to the 2,000 features,
and the arrays behind the submitted analysis.

Nothing here computes a label, a prediction or a statistic.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features, representations as rep, singlecell

__all__ = [
    "SPLITS",
    "SCREENS",
    "ENCODER_SPLIT",
    "paths",
    "VccRows",
    "load_vcc_rows",
    "ControlMeans",
    "load_control_means",
    "ScreenInputs",
    "load_screen_inputs",
    "native_symbols",
    "gene_level",
    "vcc_annotation_file",
    "vcc_target_ensembl",
    "release_mapped",
]

SPLITS = ("train", "val", "test")
SCREENS = ("RPE1", "K562-essential")
ENCODER_SPLIT = {"train": "training", "val": "validation", "test": "test"}
SHORT = {"RPE1": "RPE1", "K562-essential": "K562"}


def paths() -> dict[str, Path]:
    root = config.storage_root()
    vcc = root / "IGA_MAIN" / "cache" / "vcc"
    gf = root / "IGA_GENEFORMER_EXT"
    raw = root / "IGA_MAIN" / "data" / "replogle"
    repo = config.repo_root()
    return {
        "shards": vcc / "variants" / "delta_hvg" / "shards",
        "control_means": vcc / "nt_baseline" / "mean_log1p.npy",
        "batch_order": vcc / "nt_baseline" / "batch_order.npy",
        "subsample_sizes": root / "IGA_MAIN" / "results_v4" / "harmonized_endpoint_v1" / "subsample_sizes.json",
        "encoder_rows": root / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2" / "geneformer_row_features_full.parquet",
        "RPE1_native": gf / "x_cell" / "x_cell_rpe1_native.npz",
        "RPE1_mapped_with_ids": gf / "x_cell" / "x_cell_rpe1_mapped.npz",
        "RPE1_submitted_x": gf / "x_cell" / "x_cell_rpe1_1932_mapped.npy",
        "K562-essential_native": gf / "k562_essential" / "x_cell" / "x_cell_k562_native.npz",
        "K562-essential_submitted_x": gf / "k562_essential" / "x_cell" / "x_cell_k562_1903_mapped.npy",
        "RPE1_z": gf / "rpe1_topup" / "z_rpe1_1932.npy",
        "RPE1_z_index": gf / "rpe1_topup" / "rpe1_construct_index.parquet",
        "K562-essential_z": gf / "k562_essential" / "z" / "z_k562_1903.npy",
        "K562-essential_z_index": gf / "k562_essential" / "z" / "k562_construct_index.parquet",
        "RPE1_single_cell": raw / "raw_singlecell" / "rpe1_raw_singlecell_01.h5ad",
        "K562-essential_single_cell": raw / "raw_singlecell" / "K562_essential_raw_singlecell_01.h5ad",
        "RPE1_release": raw / "raw" / "rpe1_normalized_bulk_01.h5ad",
        "K562-essential_release": raw / "raw" / "K562_essential_normalized_bulk_01.h5ad",
        "RPE1_submitted_magnitude": repo / "earlier_revision" / "phase4" / "m_tilde" / "m_tilde_xcell_RPE1.npz",
        "K562-essential_submitted_magnitude": repo / "earlier_revision" / "phase4" / "m_tilde" / "m_tilde_xcell_K562.npz",
    }


# --------------------------------------------------------------------------- VCC


@dataclass
class VccRows:
    """Every modelled VCC row, sorted by split, batch and target gene."""

    keys: pd.DataFrame
    x: np.ndarray
    existing: np.ndarray
    z: np.ndarray
    mu_pert: np.ndarray
    batch_mu_nt: dict
    shard_position: np.ndarray

    def index(self, frame: pd.DataFrame) -> np.ndarray:
        """Positions of the rows of ``frame`` (split, batch, target_gene) in this table."""
        lookup = {key: i for i, key in enumerate(zip(self.keys["split"], self.keys["batch"], self.keys["target_gene"]))}
        try:
            return np.array([lookup[key] for key in zip(frame["split"], frame["batch"], frame["target_gene"])], dtype=np.int64)
        except KeyError as missing:
            raise RuntimeError("row %s is not a modelled VCC row" % (missing,)) from None


def load_vcc_rows() -> VccRows:
    """The shard rows in the loader's order, then sorted, with the encoder rows attached by key."""
    p = paths()
    batch_order = np.load(p["batch_order"], allow_pickle=True).astype(str)
    frames, xs, ys = [], [], []
    for split in SPLITS:
        files = {Path(f).stem: f for f in glob.glob(str(p["shards"] / split / "*.npz"))}
        if sorted(files) != sorted(batch_order.tolist()):
            raise RuntimeError("the %s shards do not cover the 48 batches" % split)
        for batch in batch_order.tolist():
            with np.load(files[batch], allow_pickle=True) as store:
                if str(np.asarray(store["batch_name"]).astype(str).ravel()[0]) != batch:
                    raise RuntimeError("shard %s/%s names another batch" % (split, batch))
                genes = np.asarray(store["sample_ids"]).astype(str)
                frames.append(pd.DataFrame({"split": split, "batch": batch, "target_gene": genes}))
                xs.append(np.asarray(store["x"], dtype=np.float32))
                ys.append(np.asarray(store["y"], dtype=np.float32))
    keys = pd.concat(frames, ignore_index=True)
    # The position of each row in the loader's order: splits in turn, batches in batch order, rows
    # in the order the shard stores them. The submitted controls were generated in this order.
    keys["shard_position"] = np.arange(len(keys))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    rank = {s: i for i, s in enumerate(SPLITS)}
    order = np.array(sorted(range(len(keys)), key=lambda i: (rank[keys.at[i, "split"]], keys.at[i, "batch"], keys.at[i, "target_gene"])))
    keys = keys.iloc[order].reset_index(drop=True)
    x, y = x[order], y[order]

    import pyarrow.parquet as pq

    names = pq.read_schema(p["encoder_rows"]).names
    pert = sorted([n for n in names if n.startswith("gf_mu_pert_")], key=lambda n: int(n.rsplit("_", 1)[1]))
    nt = sorted([n for n in names if n.startswith("gf_mu_nt_")], key=lambda n: int(n.rsplit("_", 1)[1]))
    zc = sorted([n for n in names if n.startswith("gf_z_emb_") and n.rsplit("_", 1)[1].isdigit()], key=lambda n: int(n.rsplit("_", 1)[1]))
    if not (len(pert) == len(nt) == len(zc) == 768):
        raise RuntimeError("the encoder row table does not hold 768 columns per block")
    table = pd.read_parquet(p["encoder_rows"], columns=["split", "batch", "target_gene"] + pert + nt + zc)
    table[["split", "batch", "target_gene"]] = table[["split", "batch", "target_gene"]].astype(str)
    lookup = {key: i for i, key in enumerate(zip(table["split"], table["batch"], table["target_gene"]))}
    if len(lookup) != len(table):
        raise RuntimeError("the encoder row table repeats a row key")
    rows = np.array([lookup[(ENCODER_SPLIT[s], b, t)] for s, b, t in zip(keys["split"], keys["batch"], keys["target_gene"])])
    mu_pert = table[pert].to_numpy(np.float32)[rows]
    mu_nt = table[nt].to_numpy(np.float32)[rows]
    z = table[zc].to_numpy(np.float32)[rows]
    batch_mu_nt = {}
    for batch in sorted(set(keys["batch"])):
        members = mu_nt[(keys["batch"] == batch).to_numpy()]
        if not np.array_equal(members.min(axis=0), members.max(axis=0)):
            raise RuntimeError("the rows of batch %s carry different control-cell means" % batch)
        batch_mu_nt[batch] = members[0].copy()
    return VccRows(
        keys=keys[["split", "batch", "target_gene"]].copy(),
        x=x,
        existing=y.astype(np.float64),
        z=z,
        mu_pert=mu_pert,
        batch_mu_nt=batch_mu_nt,
        shard_position=keys["shard_position"].to_numpy(),
    )


@dataclass
class ControlMeans:
    per_batch: np.ndarray  # (48, 18080) single precision, source mode
    batch_order: np.ndarray
    selected: np.ndarray
    counts: dict

    def pooled(self) -> np.ndarray:
        """The mean over all control cells, from the batch means weighted by their cell counts."""
        weights = np.array([self.counts[b] for b in self.batch_order.tolist()], dtype=np.float64)
        return (weights[:, None] * self.per_batch.astype(np.float64)).sum(axis=0) / weights.sum()

    def shift(self) -> dict:
        """Per batch, mean_b - mean_pooled over the 2,000 features (double precision)."""
        pooled = self.pooled()[self.selected]
        return {b: self.per_batch[i].astype(np.float64)[self.selected] - pooled for i, b in enumerate(self.batch_order.tolist())}

    def pooled_embedding(self, batch_mu_nt: dict) -> np.ndarray:
        weights = np.array([self.counts[b] for b in self.batch_order.tolist()], dtype=np.float64)
        stack = np.stack([batch_mu_nt[b].astype(np.float64) for b in self.batch_order.tolist()])
        return (weights[:, None] * stack).sum(axis=0) / weights.sum()


def load_control_means() -> ControlMeans:
    p = paths()
    per_batch = np.load(p["control_means"], allow_pickle=False)
    batch_order = np.load(p["batch_order"], allow_pickle=True).astype(str)
    selected = np.load(config.vcc_cache()["selected_genes"]).astype(np.int64)
    counts = {str(k): int(v) for k, v in json.loads(p["subsample_sizes"].read_text())["nt_per_batch"].items()}
    if per_batch.shape != (batch_order.size, np.load(config.vcc_cache()["var_gene_names"], allow_pickle=True).size):
        raise RuntimeError("the per-batch control means do not cover every batch and gene")
    if sorted(counts) != sorted(batch_order.tolist()):
        raise RuntimeError("the control-cell counts do not name the 48 batches")
    return ControlMeans(per_batch=per_batch, batch_order=batch_order, selected=selected, counts=counts)


def gene_level(frame: pd.DataFrame, rows_frame: pd.DataFrame, values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Cell-weighted means of row values over each (split, target gene) of ``frame``.

    ``rows_frame`` holds the modelled rows (split, target_gene) aligned with ``values`` and
    ``weights`` (perturbed cells per row).
    """
    out = np.empty((len(frame), values.shape[1]), dtype=np.float64)
    groups = rows_frame.groupby(["split", "target_gene"], sort=False).indices
    for i, (split, gene) in enumerate(zip(frame["split"], frame["target_gene"])):
        members = groups[(split, gene)]
        w = weights[members].astype(np.float64)
        out[i] = (w[:, None] * values[members].astype(np.float64)).sum(axis=0) / w.sum()
    return out


# --------------------------------------------------------------------------- external


def native_symbols(screen: str) -> np.ndarray:
    """Gene symbols of the screen's single-cell matrix, in column order."""
    import h5py

    with h5py.File(paths()[f"{screen}_single_cell"], "r") as handle:
        return singlecell.read_annotation(handle["var"], "gene_name")


@dataclass
class ScreenInputs:
    """One screen's eligible constructs, in construct order, with their features."""

    name: str
    constructs: np.ndarray
    z: np.ndarray
    x: dict
    target_column: np.ndarray
    notes: dict


def _ordered_z(screen: str, constructs: np.ndarray) -> np.ndarray:
    p = paths()
    z = np.load(p[f"{screen}_z"])
    index = pd.read_parquet(p[f"{screen}_z_index"])
    if not np.array_equal(index["row"].to_numpy(), np.arange(len(index))):
        raise RuntimeError("the %s embedding index does not number its rows in order" % screen)
    ids = index["construct"].astype(str).to_numpy()
    if z.shape != (ids.size, 768):
        raise RuntimeError("the %s embedding does not match its index" % screen)
    position = {c: i for i, c in enumerate(ids.tolist())}
    missing = [c for c in constructs.tolist() if c not in position]
    if missing:
        raise RuntimeError("%d %s constructs have no stored embedding" % (len(missing), screen))
    return z[[position[c] for c in constructs.tolist()]].astype(np.float32)


def load_screen_inputs(screen: str, endpoint: pd.DataFrame) -> ScreenInputs:
    """Embedding and expression delta of every construct of ``endpoint`` (sorted by construct).

    ``x`` holds: "native" (the cell-level delta in the screen's gene space), "full_pass" (that delta
    mapped to the 2,000 features), "submitted" (the mapped array behind the submitted analysis) and,
    for RPE1, "mapped_with_ids" (the mapped array stored with construct identifiers).
    """
    p = paths()
    constructs = endpoint["construct"].astype(str).to_numpy()
    if not np.array_equal(constructs, np.sort(constructs)):
        raise RuntimeError("the endpoint is not sorted by construct")
    symbols = features.hvg_symbols()
    with np.load(p[f"{screen}_native"], allow_pickle=True) as store:
        ids = np.asarray(store["ids"]).astype(str)
        native = np.asarray(store["x"], dtype=np.float64)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], constructs):
        raise RuntimeError("the %s native delta does not cover the endpoint constructs" % screen)
    native = native[order]
    gene_symbols = native_symbols(screen)
    full_pass, n_mapped = rep.map_to_features(native, gene_symbols, symbols)
    x = {"native": native, "full_pass": full_pass}
    submitted = np.load(p[f"{screen}_submitted_x"])
    if submitted.shape != (constructs.size, 2000):
        raise RuntimeError("the submitted %s x has shape %s" % (screen, submitted.shape))
    x["submitted"] = submitted.astype(np.float32)
    if screen == "RPE1":
        with np.load(p["RPE1_mapped_with_ids"], allow_pickle=True) as store:
            ids2 = np.asarray(store["ids"]).astype(str)
            mapped = np.asarray(store["x"], dtype=np.float32)
        order2 = np.argsort(ids2)
        if not np.array_equal(ids2[order2], constructs):
            raise RuntimeError("the RPE1 mapped array with identifiers does not cover the constructs")
        x["mapped_with_ids"] = mapped[order2]
    position = {s: i for i, s in enumerate(symbols.tolist())}
    target_column = np.array([position.get(s, -1) for s in endpoint["target_symbol"].astype(str)], dtype=np.int64)
    return ScreenInputs(
        name=screen,
        constructs=constructs,
        z=_ordered_z(screen, constructs),
        x=x,
        target_column=target_column,
        notes={"features_mapped": n_mapped, "native_genes": int(gene_symbols.size)},
    )


def release_mapped(screen: str, constructs: np.ndarray) -> np.ndarray:
    """The release pseudobulk mapped to the 2,000 features, as the submitted secondary basis.

    Rows whose identifier contains "non-targeting" are dropped, each feature takes the first column
    with its symbol, missing features are zero, non-finite values become 0 or +/-10, and every value
    is clipped to +/-10 (``scripts/phase11b_classical_transfer/run.py:87-120``).
    """
    import h5py

    with h5py.File(paths()[f"{screen}_release"], "r") as handle:
        ids = singlecell.read_annotation(handle["obs"], handle["obs"].attrs["_index"])
        gene_name = singlecell.read_annotation(handle["var"], "gene_name")
        matrix = np.asarray(handle["X"][()], dtype=np.float32)
    keep = np.char.find(ids.astype(str), "non-targeting") < 0
    ids, matrix = ids[keep], matrix[keep]
    mapped, _ = rep.map_to_features(matrix, gene_name, features.hvg_symbols())
    row = {c: i for i, c in enumerate(ids.tolist())}
    missing = [c for c in constructs.tolist() if c not in row]
    if missing:
        raise RuntimeError("%d %s constructs are absent from the release file" % (len(missing), screen))
    return mapped[[row[c] for c in constructs.tolist()]]


def vcc_annotation_file() -> Path:
    return config.storage_root() / "IGA_MAIN" / "data" / "vcc" / "train" / "adata_Training.h5ad"


def vcc_target_ensembl(genes) -> tuple[dict, list]:
    """Ensembl identifiers of VCC target genes, from the VCC gene annotation (index symbol, gene_id)."""
    import h5py

    with h5py.File(vcc_annotation_file(), "r") as handle:
        var = handle["var"]
        symbols = singlecell.read_annotation(var, var.attrs["_index"])
        gene_id = singlecell.read_annotation(var, "gene_id")
    lookup = dict(zip(symbols.tolist(), gene_id.tolist()))
    genes = sorted(set(str(g) for g in genes))
    return {g: lookup[g] for g in genes if g in lookup}, [g for g in genes if g not in lookup]
