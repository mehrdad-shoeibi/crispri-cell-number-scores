"""E1b, input records: path, size, digest and layout of every input E1b reads beyond the E1a tables.

Design: bioinf2027/plans/E1_design.md, Sections 10, 21.1 (item 4), 25.2, 26 and 27. Nothing here
computes a statistic; the record states what each file holds (array keys, shapes and types, table
columns), so that the pre-run checks and the fits can refuse any input that differs.

Output: bioinf2027/results/E1/input_records.json.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, folder_record, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E1" / "input_records.json"


def root() -> Path:
    return config.storage_root()


def inputs() -> dict[str, Path]:
    r = root()
    vcc = r / "IGA_MAIN" / "cache" / "vcc"
    encoder = r / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2"
    gf = r / "IGA_GENEFORMER_EXT"
    raw = r / "IGA_MAIN" / "data" / "replogle"
    return {
        "vcc_control_means_per_batch": vcc / "nt_baseline" / "mean_log1p.npy",
        "vcc_control_batch_order": vcc / "nt_baseline" / "batch_order.npy",
        "vcc_gene_names": vcc / "var_gene_names.npy",
        "vcc_selected_genes": vcc / "variants" / "delta_hvg" / "selected_genes.npy",
        "vcc_subsample_sizes": r / "IGA_MAIN" / "results_v4" / "harmonized_endpoint_v1" / "subsample_sizes.json",
        "encoder_row_table": encoder / "geneformer_row_features_full.parquet",
        "rpe1_native_delta": gf / "x_cell" / "x_cell_rpe1_native.npz",
        "rpe1_mapped_delta": gf / "x_cell" / "x_cell_rpe1_mapped.npz",
        "rpe1_mapped_delta_1932": gf / "x_cell" / "x_cell_rpe1_1932_mapped.npy",
        "k562_native_delta": gf / "k562_essential" / "x_cell" / "x_cell_k562_native.npz",
        "k562_mapped_delta_1903": gf / "k562_essential" / "x_cell" / "x_cell_k562_1903_mapped.npy",
        "rpe1_z": gf / "rpe1_topup" / "z_rpe1_1932.npy",
        "rpe1_z_index": gf / "rpe1_topup" / "rpe1_construct_index.parquet",
        "k562_z": gf / "k562_essential" / "z" / "z_k562_1903.npy",
        "k562_z_index": gf / "k562_essential" / "z" / "k562_construct_index.parquet",
        "rpe1_single_cell": raw / "raw_singlecell" / "rpe1_raw_singlecell_01.h5ad",
        "k562_single_cell": raw / "raw_singlecell" / "K562_essential_raw_singlecell_01.h5ad",
        "rpe1_release_bulk": raw / "raw" / "rpe1_normalized_bulk_01.h5ad",
        "k562_release_bulk": raw / "raw" / "K562_essential_normalized_bulk_01.h5ad",
        "submitted_rpe1_magnitude": REPO / "earlier_revision" / "phase4" / "m_tilde" / "m_tilde_xcell_RPE1.npz",
        "submitted_k562_magnitude": REPO / "earlier_revision" / "phase4" / "m_tilde" / "m_tilde_xcell_K562.npz",
        "submitted_external_predictions": REPO / "earlier_revision" / "phase4" / "phase4_ext_predictions.npz",
        "submitted_external_values": REPO / "earlier_revision" / "phase4" / "phase4_primary.json",
        "submitted_vcc_test_values": REPO / "earlier_revision" / "phase1" / "test_table.json",
        "submitted_vcc_selection": REPO / "earlier_revision" / "phase1" / "frozen_selection.json",
        "submitted_vcc_test_label": REPO / "earlier_revision" / "phase1" / "preds" / "y_test.npy",
        "submitted_rpe1_endpoint": REPO / "earlier_revision" / "endpoint" / "rpe1_endpoint_by_construct.csv",
        "submitted_k562_endpoint": REPO / "earlier_revision" / "endpoint" / "k562_essential_endpoint_by_construct.csv",
    }


def layout(path: Path) -> dict:
    """Keys, shapes and types of an array file, columns of a table, keys of a JSON object."""
    suffix = path.suffix
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as store:
            out = {}
            for key in store.files:
                value = store[key]
                entry = {"shape": list(value.shape), "dtype": str(value.dtype)}
                if value.ndim == 1 and value.dtype.kind in "OUS" and value.size:
                    entry["first"] = [str(v) for v in value[:3]]
                out[key] = entry
            return {"arrays": out}
    if suffix == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False) if path.stat().st_size > 10_000_000 else np.load(path, allow_pickle=True)
        entry = {"shape": list(value.shape), "dtype": str(value.dtype)}
        if value.ndim == 1 and value.dtype.kind in "OUS":
            entry["values"] = [str(v) for v in value.tolist()] if value.size <= 64 else [str(v) for v in value[:5]]
        return {"array": entry}
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        names = list(schema.names)
        prefixes = {}
        others = []
        for name in names:
            head = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1].isdigit() else None
            if head:
                prefixes[head] = prefixes.get(head, 0) + 1
            else:
                others.append(name)
        meta = pq.read_metadata(path)
        out = {"rows": int(meta.num_rows), "numbered_column_groups": prefixes, "other_columns": others}
        if meta.num_rows <= 5000 and others:
            import pandas as pd

            head = pd.read_parquet(path, columns=others).head(3)
            out["first_rows"] = json.loads(head.to_json(orient="records"))
        return out
    if suffix == ".json":
        content = json.loads(path.read_text())
        return {"keys": list(content) if isinstance(content, dict) else type(content).__name__}
    if suffix == ".h5ad":
        import h5py

        with h5py.File(path, "r") as handle:
            var = handle["var"]
            obs = handle["obs"]
            matrix = handle["X"]
            shape = list(matrix.attrs["shape"]) if "shape" in matrix.attrs else list(matrix.shape)
            return {
                "X_shape": [int(v) for v in shape],
                "X_kind": "sparse group" if hasattr(matrix, "keys") else "dense",
                "var_columns": sorted(var.keys()),
                "obs_columns": sorted(obs.keys()),
                "var_index": str(var.attrs.get("_index")),
            }
    if suffix == ".csv":
        with open(path) as handle:
            header = handle.readline().strip().split(",")
        return {"columns": header}
    return {}


def shards() -> dict:
    base = root() / "IGA_MAIN" / "cache" / "vcc" / "variants" / "delta_hvg" / "shards"
    out = {}
    for split in ("train", "val", "test"):
        paths = sorted(glob.glob(str(base / split / "*.npz")))
        first = Path(paths[0])
        out[split] = {
            "files": len(paths),
            "names_first_last": [first.name, Path(paths[-1]).name],
            "first_file_layout": layout(first),
            "folder": folder_record(paths, base / split),
        }
    return out


def main() -> None:
    records = {}
    for name, path in inputs().items():
        records[name] = {**file_record(path), **layout(path)}
        print("[inputs] %s" % name, flush=True)
    out = {
        "purpose": "every E1b input beyond the E1a tables, with its layout",
        "plan": "bioinf2027/plans/E1_design.md, Sections 10, 21.1, 25.2, 26 and 27",
        "inputs": records,
        "vcc_feature_shards": shards(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(out), indent=1, allow_nan=False) + "\n")
    print("[inputs] written %s" % config.portable(OUT), flush=True)


if __name__ == "__main__":
    main()
