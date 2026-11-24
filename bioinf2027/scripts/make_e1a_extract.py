"""E1a, first part, extraction: one pass over each single-cell file for R0 and the builder checks.

Design: bioinf2027/plans/E1_design.md, Sections 7, 9 and 21 to 23. By the
author's decision at gate G1 (Section 22, items 2 and 3) no perturbed-cell label
of any grid cell is built in this part of E1a. This script computes no label
and no statistic at all: it reads each of the five single-cell files once and
caches the per-cell values that read-out R0 and the builder checks (a) to (e)
need, under $IGA_STORAGE_ROOT/bioinf2027/cache/E1a, with a manifest.

Per VCC file: the total of every cell, and for every perturbed cell its target
gene's value in external and in source mode (bioinf2027.endpoint). From the
training file, which holds the 38,176 control cells shared by the three
splits, the control cells' values at every target gene of the modelled rows, in
both modes, and their totals over the genes VCC shares with each external
screen. Per external screen: the total of every cell, the control cells' values
at every eligible target column, their totals over the genes the screen shares
with VCC, and every eligible perturbed cell's value at its own target column.

The manifest records the portable path, size and digest of every input, checked
against bioinf2027/plans/E1_input_records.json for the five single-cell files,
and of every cached array. The compute stage, bioinf2027/scripts/make_e1a_r0.py,
refuses a cache that does not match its manifest.
"""
from __future__ import annotations

import glob
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import cellpass, config, endpoint, singlecell  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
INPUT_RECORDS = REPO / "bioinf2027" / "plans" / "E1_input_records.json"
PLAN = "bioinf2027/plans/E1_design.md, Sections 7, 9 and 21 to 23"

SPLITS = {"train": "training", "val": "validation", "test": "test"}
SPLIT_ROWS = {"train": 5740, "val": 2052, "test": 3972}
SCREENS = ("RPE1", "K562-essential")
RECORD_KEYS = {
    "train": "VCC train",
    "val": "VCC validation",
    "test": "VCC test",
    "RPE1": "RPE1",
    "K562-essential": "K562-essential",
}
CONTROL = "non-targeting"
CHUNK_ROWS = 4000
# Fewer shared genes than this means the identifiers of the two annotations do not match.
MIN_SHARED_GENES = 1000


# --------------------------------------------------------------------------- locations


def vcc_files() -> dict[str, Path]:
    data = config.storage_root() / "IGA_MAIN" / "data" / "vcc"
    return {
        "train": data / "train" / "adata_Training.h5ad",
        "val": data / "validation" / "adata_Validation.h5ad",
        "test": data / "test" / "adata_Test.h5ad",
    }


def screen_files() -> dict[str, Path]:
    data = config.storage_root() / "IGA_MAIN" / "data" / "replogle" / "raw_singlecell"
    return {
        "RPE1": data / "rpe1_raw_singlecell_01.h5ad",
        "K562-essential": data / "K562_essential_raw_singlecell_01.h5ad",
    }


def cache_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "cache" / "E1a"


def shard_root() -> Path:
    return config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "variants" / "delta_hvg" / "shards"


def nt_baseline_dir() -> Path:
    return config.storage_root() / "IGA_MAIN" / "cache" / "vcc" / "nt_baseline"


def cell_count_file() -> Path:
    return config.storage_root() / "MADA" / "data" / "metadata" / "vcc" / "final_labels.parquet"


def subsample_sizes_file() -> Path:
    return config.storage_root() / "IGA_MAIN" / "results_v4" / "harmonized_endpoint_v1" / "subsample_sizes.json"


def encoder_dir() -> Path:
    return config.storage_root() / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2"


# --------------------------------------------------------------------------- annotations


def _attr_text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def obs_table(handle, names) -> dict[str, np.ndarray]:
    obs = handle["obs"]
    out = {name: singlecell.read_annotation(obs, name) for name in names}
    out["_index"] = singlecell.read_annotation(obs, _attr_text(obs.attrs["_index"]))
    return out


def var_index(handle) -> np.ndarray:
    var = handle["var"]
    return singlecell.read_annotation(var, _attr_text(var.attrs["_index"]))


def stored_rows() -> pd.DataFrame:
    """Every modelled VCC row: split, batch and target gene, from the feature shards."""
    frames = []
    for split in SPLITS:
        paths = sorted(glob.glob(str(shard_root() / split / "*.npz")))
        if not paths:
            raise RuntimeError("no shards for split %s" % split)
        for path in paths:
            store = np.load(path, allow_pickle=True)
            frames.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "batch": str(np.asarray(store["batch_name"]).astype(str).ravel()[0]),
                        "target_gene": np.asarray(store["sample_ids"]).astype(str),
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True)
    observed = table.groupby("split").size().to_dict()
    if observed != SPLIT_ROWS:
        raise RuntimeError("modelled rows per split %s differ from %s" % (observed, SPLIT_ROWS))
    return table


def eligible(name: str) -> pd.DataFrame:
    table = pd.read_csv(config.screen_paths()[name]["eligibility"])
    return table[table["eligible"].astype(bool)].reset_index(drop=True)


# --------------------------------------------------------------------------- passes


def _rows_in(rows: np.ndarray, start: int, stop: int) -> tuple[int, int]:
    return int(np.searchsorted(rows, start)), int(np.searchsorted(rows, stop))


def extract_vcc(path: Path, targets: np.ndarray, shared: dict[str, np.ndarray], controls: bool, chunk_rows: int = CHUNK_ROWS) -> dict:
    """Per-cell values of one VCC file."""
    import h5py

    with h5py.File(path, "r") as handle:
        obs = obs_table(handle, ("target_gene", "batch", "guide_id"))
        symbols = var_index(handle)
        position = {s: i for i, s in enumerate(symbols.tolist())}
        missing_targets = sorted(set(targets.tolist()) - set(position))
        if missing_targets:
            raise RuntimeError("target genes absent from the gene annotation: %s" % missing_targets[:5])
        target_cols = np.array([position[t] for t in targets.tolist()], dtype=np.int64)

        is_control = obs["target_gene"] == CONTROL
        nt_rows = np.flatnonzero(is_control)
        pert_rows = np.flatnonzero(~is_control)
        pert_target = obs["target_gene"][pert_rows]
        absent = sorted(set(pert_target.tolist()) - set(position))
        if absent:
            raise RuntimeError("perturbed targets absent from the gene annotation: %s" % absent[:5])
        own_col = np.array([position[t] for t in pert_target.tolist()], dtype=np.int64)

        n_rows = cellpass.matrix_shape(handle)[0]
        out = {
            "n_cells": np.int64(n_rows),
            "all_totals": np.zeros(n_rows, dtype=np.float32),
            "pert_rows": pert_rows,
            "pert_target": pert_target,
            "pert_batch": obs["batch"][pert_rows],
            "pert_guide": obs["guide_id"][pert_rows],
            "pert_own_ext": np.zeros(pert_rows.size, dtype=np.float32),
            "pert_own_src": np.zeros(pert_rows.size, dtype=np.float32),
            "nt_rows": nt_rows,
            "nt_barcode": obs["_index"][nt_rows],
            "nt_batch": obs["batch"][nt_rows],
        }
        if controls:
            out["targets"] = targets
            out["target_cols"] = target_cols
            out["nt_values_ext"] = np.zeros((nt_rows.size, targets.size), dtype=np.float32)
            out["nt_values_src"] = np.zeros((nt_rows.size, targets.size), dtype=np.float32)
            for screen in shared:
                out["nt_shared_totals_" + screen] = np.zeros(nt_rows.size, dtype=np.float32)

        started = time.time()
        for chunk in cellpass.csr_chunks(handle, chunk_rows):
            out["all_totals"][chunk.start : chunk.stop] = chunk.totals
            lo, hi = _rows_in(pert_rows, chunk.start, chunk.stop)
            if hi > lo:
                local = pert_rows[lo:hi] - chunk.start
                counts = chunk.pairs(local, own_col[lo:hi])
                totals = chunk.totals[local]
                out["pert_own_ext"][lo:hi] = endpoint.external_values(counts, totals)
                out["pert_own_src"][lo:hi] = endpoint.source_values(counts, totals)
            if controls:
                lo, hi = _rows_in(nt_rows, chunk.start, chunk.stop)
                if hi > lo:
                    local = nt_rows[lo:hi] - chunk.start
                    counts = chunk.columns(local, target_cols)
                    totals = chunk.totals[local]
                    out["nt_values_ext"][lo:hi] = endpoint.external_values(counts, totals)
                    out["nt_values_src"][lo:hi] = endpoint.source_values(counts, totals)
                    for screen, mask in shared.items():
                        out["nt_shared_totals_" + screen][lo:hi] = chunk.masked_totals(local, mask)
        out["pass_seconds"] = np.float64(time.time() - started)
    return out


def extract_screen(name: str, path: Path, table: pd.DataFrame, vcc_gene_ids: np.ndarray, chunk_rows: int = CHUNK_ROWS) -> dict:
    """Per-cell values of one external screen."""
    import h5py

    # A plain text array: pandas 3 returns an object array from a string column, which the cache refuses.
    constructs = np.array(table["construct"].astype(str).tolist(), dtype=str)
    construct_col = table["target_col"].astype(int).to_numpy()
    with h5py.File(path, "r") as handle:
        obs = obs_table(handle, ("gene", "gene_transcript", "gem_group"))
        genes = var_index(handle)
        if not np.array_equal(genes[construct_col], table["target_ensg"].astype(str).to_numpy()):
            raise RuntimeError("%s: eligibility target columns do not carry the target identifiers" % name)

        is_control = np.char.find(np.char.lower(obs["gene"].astype(str)), CONTROL) >= 0
        code_of = {c: i for i, c in enumerate(constructs.tolist())}
        code = np.array([code_of.get(t, -1) for t in obs["gene_transcript"].tolist()], dtype=np.int64)
        code[is_control] = -1
        nt_rows = np.flatnonzero(is_control)
        pert_rows = np.flatnonzero(code >= 0)
        pert_code = code[pert_rows]
        target_cols = np.unique(construct_col)
        col_slot = {int(c): i for i, c in enumerate(target_cols.tolist())}
        shared_mask = np.isin(genes, vcc_gene_ids)

        n_rows = cellpass.matrix_shape(handle)[0]
        out = {
            "n_cells": np.int64(n_rows),
            "constructs": constructs,
            "construct_col": construct_col,
            "construct_slot": np.array([col_slot[int(c)] for c in construct_col], dtype=np.int64),
            "target_cols": target_cols,
            "all_totals": np.zeros(n_rows, dtype=np.float32),
            "nt_rows": nt_rows,
            "nt_gem": obs["gem_group"][nt_rows],
            "nt_values": np.zeros((nt_rows.size, target_cols.size), dtype=np.float32),
            "nt_shared_totals": np.zeros(nt_rows.size, dtype=np.float32),
            "pert_rows": pert_rows,
            "pert_code": pert_code,
            "pert_gem": obs["gem_group"][pert_rows],
            "pert_own": np.zeros(pert_rows.size, dtype=np.float32),
            "shared_genes": np.int64(shared_mask.sum()),
        }
        started = time.time()
        for chunk in cellpass.dense_chunks(handle, chunk_rows):
            out["all_totals"][chunk.start : chunk.stop] = chunk.totals
            lo, hi = _rows_in(nt_rows, chunk.start, chunk.stop)
            if hi > lo:
                local = nt_rows[lo:hi] - chunk.start
                out["nt_values"][lo:hi] = endpoint.external_values(chunk.columns(local, target_cols), chunk.totals[local])
                out["nt_shared_totals"][lo:hi] = chunk.masked_totals(local, shared_mask)
            lo, hi = _rows_in(pert_rows, chunk.start, chunk.stop)
            if hi > lo:
                local = pert_rows[lo:hi] - chunk.start
                cols = construct_col[pert_code[lo:hi]]
                out["pert_own"][lo:hi] = endpoint.external_values(chunk.pairs(local, cols), chunk.totals[local])
        out["pass_seconds"] = np.float64(time.time() - started)
    return out


# --------------------------------------------------------------------------- records


def verify_single_cell_inputs() -> dict:
    """Digest of every single-cell input against the records made before the design was final."""
    recorded = json.loads(INPUT_RECORDS.read_text())["files"]
    out = {}
    for key, path in {**vcc_files(), **screen_files()}.items():
        record = file_record(path)
        expected = recorded[RECORD_KEYS[key]]
        record["matches_input_records"] = bool(
            record.get("sha256") == expected["sha256"] and record.get("bytes") == expected["bytes"]
        )
        if not record["matches_input_records"]:
            raise RuntimeError("%s differs from bioinf2027/plans/E1_input_records.json" % key)
        out[key] = record
    return out


def other_inputs() -> dict:
    """Every other input of E1a's first part, with size and digest (Sections 1 and 21.2, item 6)."""
    import pyarrow.parquet as pq

    encoder = encoder_dir()
    index = pd.read_parquet(encoder / "cell_embedding_index.parquet")
    nt_files = sorted(str(p) for p in (encoder / "nt_embeddings").rglob("*") if p.is_file())
    records = {
        "vcc_feature_shards_and_stored_label": folder_record(glob.glob(str(shard_root() / "*" / "*.npz")), shard_root()),
        "vcc_control_means_per_batch": file_record(nt_baseline_dir() / "mean_log1p.npy"),
        "vcc_control_batch_order": file_record(nt_baseline_dir() / "batch_order.npy"),
        "vcc_cell_counts": file_record(cell_count_file()),
        "vcc_gene_names": file_record(config.vcc_cache()["var_gene_names"]),
        "vcc_selected_genes": file_record(config.vcc_cache()["selected_genes"]),
        "frozen_subsample_sizes": file_record(subsample_sizes_file()),
        "encoder_row_table": file_record(encoder / "geneformer_row_features_full.parquet"),
        "encoder_cell_index": file_record(encoder / "cell_embedding_index.parquet"),
        "encoder_control_cell_embeddings": folder_record(nt_files, encoder / "nt_embeddings"),
    }
    for name in SCREENS:
        paths = config.screen_paths()[name]
        records[f"{name}_endpoint"] = file_record(paths["endpoint"])
        records[f"{name}_eligibility"] = file_record(paths["eligibility"])
    schema = pq.read_schema(encoder / "geneformer_row_features_full.parquet")
    layout = {
        "row_table_columns": len(schema.names),
        "row_table_columns_not_embedding": [n for n in schema.names if not n.startswith(("gf_mu_pert_", "gf_mu_nt_", "gf_z_emb_"))],
        "cell_index_columns": [str(c) for c in index.columns],
        "cell_index_rows": int(len(index)),
        "cell_index_roles": {str(k): int(v) for k, v in index["role"].value_counts().items()} if "role" in index else {},
        "cell_index_head": index.head(3).astype(str).to_dict("records"),
        "control_embedding_files": len(nt_files),
        "control_embedding_file_names_first": [Path(p).name for p in nt_files[:5]],
    }
    return records, layout


def save_cache(name: str, arrays: dict) -> dict:
    folder = cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npz"
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError("%s/%s would need pickling" % (name, key))
    np.savez(path, **arrays)
    return file_record(path)


def main() -> None:
    import h5py

    started = time.time()
    single_cell = verify_single_cell_inputs()
    print("[extract] five single-cell files match their recorded digests", flush=True)
    others, layout = other_inputs()
    rows = stored_rows()
    targets = np.array(sorted(set(rows["target_gene"])))

    with h5py.File(vcc_files()["train"], "r") as handle:
        vcc_ids = singlecell.read_annotation(handle["var"], "gene_id")
    shared = {}
    for name, path in screen_files().items():
        with h5py.File(path, "r") as handle:
            shared[name] = np.isin(vcc_ids, var_index(handle))
        if shared[name].sum() < MIN_SHARED_GENES:
            raise RuntimeError("only %d genes shared by VCC and %s; identifiers do not match" % (shared[name].sum(), name))

    cache = {}
    for split, path in vcc_files().items():
        arrays = extract_vcc(path, targets, shared, controls=(split == "train"))
        cache["vcc_" + split] = save_cache("vcc_" + split, arrays)
        print(f"[extract] VCC {split}: {arrays['pert_rows'].size} perturbed cells, {float(arrays['pass_seconds']):.0f} s", flush=True)
    for name, path in screen_files().items():
        arrays = extract_screen(name, path, eligible(name), vcc_ids)
        cache["screen_" + name] = save_cache("screen_" + name, arrays)
        print(f"[extract] {name}: {arrays['pert_rows'].size} eligible perturbed cells, {float(arrays['pass_seconds']):.0f} s", flush=True)

    manifest = {
        "plan": PLAN,
        "made_by": "bioinf2027/scripts/make_e1a_extract.py",
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "single_cell_inputs": single_cell,
        "other_inputs": others,
        "encoder_cache_layout": layout,
        "cache": cache,
        "target_genes": int(targets.size),
        "genes_shared_with_vcc": {name: int(mask.sum()) for name, mask in shared.items()},
        "counts_check": "every chunk: finite, non-negative, integer counts, and cell totals below 2**24",
        "chunk_rows": CHUNK_ROWS,
        "seconds": time.time() - started,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    path = cache_dir() / "manifest.json"
    path.write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[extract] manifest written, %.0f s in all" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
