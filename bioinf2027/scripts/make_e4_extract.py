"""E4, extraction: raw counts of the 2,000 feature genes for every cell the matched features need.

Design: bioinf2027/plans/E4.md, Sections 3 and 4 (as amended by Sections 10 to 12). One pass over each of
the five single-cell files keeps, for the cells of the E1a label cache (every VCC perturbed cell, the
38,176 VCC control cells, every eligible perturbed cell and every control cell of the two screens), the
raw count at each feature gene: on VCC the 2,000 selected genes by symbol, on a screen the first column
carrying each feature's symbol, as the cell-level input maps them
(earlier_revision/pipeline/x_cell_accumulator.py:259-276), zero where the screen lacks the gene. The
cells and their order are those of the E1a label cache, checked row for row, so that its totals and
draws apply unchanged. No feature value and no statistic is computed here.

Output: $IGA_STORAGE_ROOT/bioinf2027/cache/E4/*.npz with a manifest of digests.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import cellpass, config, features, singlecell  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1a_extract as first  # noqa: E402
import make_e1a_label_extract as second  # noqa: E402

PLAN = "bioinf2027/plans/E4.md, Sections 3 and 4, as amended by Sections 10 to 12"
CHUNK_ROWS = 4000


def cache_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "cache" / "E4"


def label_cache() -> dict:
    """The E1a label cache's row lists, refused unless every file matches its manifest."""
    folder = second.cache_dir()
    manifest = json.loads((folder / "manifest.json").read_text())
    out = {}
    for name, record in manifest["cache"].items():
        path = folder / f"{name}.npz"
        if sha256(path) != record["sha256"]:
            raise RuntimeError("E1a label cache file %s does not match its manifest" % name)
        with np.load(path, allow_pickle=False) as store:
            out[name] = {key: store[key] for key in ("pert_rows", "nt_rows") if key in store.files}
    return {"rows": out, "manifest": file_record(folder / "manifest.json")}


def extract(path: Path, pert_rows: np.ndarray, nt_rows: np.ndarray | None, columns: np.ndarray, present: np.ndarray, sparse: bool) -> dict:
    """Counts of the given rows at the given columns; absent features stay zero."""
    import h5py

    n_features = columns.size
    out = {"pert_counts": np.zeros((pert_rows.size, n_features), dtype=np.float32)}
    if nt_rows is not None:
        out["nt_counts"] = np.zeros((nt_rows.size, n_features), dtype=np.float32)
    used = np.flatnonzero(present)
    started = time.time()
    with h5py.File(path, "r") as handle:
        chunks = cellpass.csr_chunks(handle, CHUNK_ROWS) if sparse else cellpass.dense_chunks(handle, CHUNK_ROWS)
        for chunk in chunks:
            for key, rows in (("pert_counts", pert_rows), ("nt_counts", nt_rows)):
                if rows is None:
                    continue
                lo, hi = second._rows_in(rows, chunk.start, chunk.stop)
                if hi > lo:
                    local = rows[lo:hi] - chunk.start
                    out[key][lo:hi, used] = chunk.columns(local, columns[used])
    out["pass_seconds"] = np.float64(time.time() - started)
    return out


def main() -> None:
    import h5py

    started = time.time()
    single_cell = first.verify_single_cell_inputs()
    cache = label_cache()
    symbols = features.hvg_symbols()
    outputs = {}
    folder = cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for split, path in first.vcc_files().items():
        with h5py.File(path, "r") as handle:
            position = {s: i for i, s in enumerate(first.var_index(handle).tolist())}
        missing = [s for s in symbols.tolist() if s not in position]
        if missing:
            raise RuntimeError("feature genes absent from the VCC %s file: %s" % (split, missing[:5]))
        columns = np.array([position[s] for s in symbols.tolist()], dtype=np.int64)
        rows = cache["rows"]["vcc_" + split]
        arrays = extract(path, rows["pert_rows"], rows["nt_rows"] if split == "train" else None, columns, np.ones(columns.size, bool), sparse=True)
        arrays["pert_rows"] = rows["pert_rows"]
        if split == "train":
            arrays["nt_rows"] = rows["nt_rows"]
        arrays["feature_columns"] = columns
        target = folder / f"vcc_{split}.npz"
        np.savez(target, **arrays)
        outputs["vcc_" + split] = file_record(target)
        print("[e4 extract] VCC %s: %d perturbed cells, %.0f s" % (split, rows["pert_rows"].size, float(arrays["pass_seconds"])), flush=True)
    for name, path in first.screen_files().items():
        with h5py.File(path, "r") as handle:
            gene_name = singlecell.read_annotation(handle["var"], "gene_name")
        first_column: dict = {}
        for index, symbol in enumerate(gene_name.tolist()):
            first_column.setdefault(symbol, index)
        columns = np.array([first_column.get(s, -1) for s in symbols.tolist()], dtype=np.int64)
        present = columns >= 0
        rows = cache["rows"]["screen_" + name]
        arrays = extract(path, rows["pert_rows"], rows["nt_rows"], columns, present, sparse=False)
        arrays["pert_rows"] = rows["pert_rows"]
        arrays["nt_rows"] = rows["nt_rows"]
        arrays["feature_columns"] = columns
        target = folder / f"screen_{name}.npz"
        np.savez(target, **arrays)
        outputs["screen_" + name] = {**file_record(target), "features_mapped": int(present.sum())}
        print("[e4 extract] %s: %d perturbed and %d control cells, %d features mapped, %.0f s" % (
            name, rows["pert_rows"].size, rows["nt_rows"].size, present.sum(), float(arrays["pass_seconds"])), flush=True)
    manifest = {
        "plan": PLAN,
        "made_by": "bioinf2027/scripts/make_e4_extract.py",
        "single_cell_inputs": single_cell,
        "label_cache_manifest": cache["manifest"],
        "feature_symbols": "bioinf2027/src/bioinf2027/features.py, hvg_symbols",
        "cache": outputs,
        "counts_check": "every chunk: finite, non-negative, integer counts, and cell totals below 2**24",
        "seconds": time.time() - started,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    (folder / "manifest.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[e4 extract] manifest written, %.0f s in all" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
