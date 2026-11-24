"""E1a, label stage, extraction: raw counts and totals for every grid cell of E1.

Design: bioinf2027/plans/E1_design.md, Section 24.3. One pass over each of the five single-cell
files keeps what the labels of every grid cell need and that the first E1a pass did not keep: the
raw count of every needed column (VCC control cells: the 282 target genes; VCC perturbed cells: their
own target; external control cells: every eligible target column; external eligible perturbed cells:
their own target), and every cell's total over all genes and over the genes shared with the other
side. Thinning (the DT cells) and shared-gene size factors (G1-SU) need counts, not values. No label
and no statistic is computed here.

Output: $IGA_STORAGE_ROOT/bioinf2027/cache/E1a_labels/*.npz with a manifest of digests, the inputs
checked against bioinf2027/plans/E1_input_records.json.
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

from bioinf2027 import cellpass, config, singlecell  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_e1a_extract as first  # noqa: E402

PLAN = "bioinf2027/plans/E1_design.md, Section 24.3"
CHUNK_ROWS = 4000


def cache_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "cache" / "E1a_labels"


def _rows_in(rows: np.ndarray, start: int, stop: int) -> tuple[int, int]:
    return int(np.searchsorted(rows, start)), int(np.searchsorted(rows, stop))


def extract_vcc_counts(path: Path, targets: np.ndarray, shared: dict, controls: bool, chunk_rows: int = CHUNK_ROWS) -> dict:
    """Raw counts and totals of one VCC file."""
    import h5py

    with h5py.File(path, "r") as handle:
        obs = first.obs_table(handle, ("target_gene", "batch", "guide_id"))
        position = {s: i for i, s in enumerate(first.var_index(handle).tolist())}
        target_cols = np.array([position[t] for t in targets.tolist()], dtype=np.int64)
        is_control = obs["target_gene"] == first.CONTROL
        nt_rows = np.flatnonzero(is_control)
        pert_rows = np.flatnonzero(~is_control)
        pert_target = obs["target_gene"][pert_rows]
        own_col = np.array([position[t] for t in pert_target.tolist()], dtype=np.int64)
        out = {
            "pert_rows": pert_rows,
            "pert_target": pert_target,
            "pert_batch": obs["batch"][pert_rows],
            "pert_guide": obs["guide_id"][pert_rows],
            "pert_count": np.zeros(pert_rows.size, dtype=np.float32),
            "pert_total": np.zeros(pert_rows.size, dtype=np.float32),
            "nt_rows": nt_rows,
            "nt_batch": obs["batch"][nt_rows],
        }
        for screen in shared:
            out["pert_shared_total_" + screen] = np.zeros(pert_rows.size, dtype=np.float32)
        if controls:
            out["targets"] = targets
            out["target_cols"] = target_cols
            out["nt_counts"] = np.zeros((nt_rows.size, targets.size), dtype=np.float32)
            out["nt_total"] = np.zeros(nt_rows.size, dtype=np.float32)
            for screen in shared:
                out["nt_shared_total_" + screen] = np.zeros(nt_rows.size, dtype=np.float32)
        started = time.time()
        for chunk in cellpass.csr_chunks(handle, chunk_rows):
            lo, hi = _rows_in(pert_rows, chunk.start, chunk.stop)
            if hi > lo:
                local = pert_rows[lo:hi] - chunk.start
                out["pert_count"][lo:hi] = chunk.pairs(local, own_col[lo:hi])
                out["pert_total"][lo:hi] = chunk.totals[local]
                for screen, mask in shared.items():
                    out["pert_shared_total_" + screen][lo:hi] = chunk.masked_totals(local, mask)
            if controls:
                lo, hi = _rows_in(nt_rows, chunk.start, chunk.stop)
                if hi > lo:
                    local = nt_rows[lo:hi] - chunk.start
                    out["nt_counts"][lo:hi] = chunk.columns(local, target_cols)
                    out["nt_total"][lo:hi] = chunk.totals[local]
                    for screen, mask in shared.items():
                        out["nt_shared_total_" + screen][lo:hi] = chunk.masked_totals(local, mask)
        out["pass_seconds"] = np.float64(time.time() - started)
    return out


def extract_screen_counts(name: str, path: Path, table, vcc_gene_ids: np.ndarray, chunk_rows: int = CHUNK_ROWS) -> dict:
    """Raw counts and totals of one external screen."""
    import h5py

    constructs = np.array(table["construct"].astype(str).tolist(), dtype=str)
    construct_col = table["target_col"].astype(int).to_numpy()
    with h5py.File(path, "r") as handle:
        obs = first.obs_table(handle, ("gene", "gene_transcript", "gem_group"))
        genes = first.var_index(handle)
        if not np.array_equal(genes[construct_col], np.array(table["target_ensg"].astype(str).tolist())):
            raise RuntimeError("%s: eligibility target columns do not carry the target identifiers" % name)
        is_control = np.char.find(np.char.lower(obs["gene"].astype(str)), first.CONTROL) >= 0
        code_of = {c: i for i, c in enumerate(constructs.tolist())}
        code = np.array([code_of.get(t, -1) for t in obs["gene_transcript"].tolist()], dtype=np.int64)
        code[is_control] = -1
        nt_rows = np.flatnonzero(is_control)
        pert_rows = np.flatnonzero(code >= 0)
        pert_code = code[pert_rows]
        target_cols = np.unique(construct_col)
        col_slot = {int(c): i for i, c in enumerate(target_cols.tolist())}
        shared_mask = np.isin(genes, vcc_gene_ids)
        out = {
            "constructs": constructs,
            "construct_col": construct_col,
            "construct_slot": np.array([col_slot[int(c)] for c in construct_col], dtype=np.int64),
            "target_cols": target_cols,
            "nt_rows": nt_rows,
            "nt_counts": np.zeros((nt_rows.size, target_cols.size), dtype=np.float32),
            "nt_total": np.zeros(nt_rows.size, dtype=np.float32),
            "nt_shared_total": np.zeros(nt_rows.size, dtype=np.float32),
            "pert_rows": pert_rows,
            "pert_code": pert_code,
            "pert_count": np.zeros(pert_rows.size, dtype=np.float32),
            "pert_total": np.zeros(pert_rows.size, dtype=np.float32),
            "pert_shared_total": np.zeros(pert_rows.size, dtype=np.float32),
            "shared_genes": np.int64(shared_mask.sum()),
        }
        started = time.time()
        for chunk in cellpass.dense_chunks(handle, chunk_rows):
            lo, hi = _rows_in(nt_rows, chunk.start, chunk.stop)
            if hi > lo:
                local = nt_rows[lo:hi] - chunk.start
                out["nt_counts"][lo:hi] = chunk.columns(local, target_cols)
                out["nt_total"][lo:hi] = chunk.totals[local]
                out["nt_shared_total"][lo:hi] = chunk.masked_totals(local, shared_mask)
            lo, hi = _rows_in(pert_rows, chunk.start, chunk.stop)
            if hi > lo:
                local = pert_rows[lo:hi] - chunk.start
                out["pert_count"][lo:hi] = chunk.pairs(local, construct_col[pert_code[lo:hi]])
                out["pert_total"][lo:hi] = chunk.totals[local]
                out["pert_shared_total"][lo:hi] = chunk.masked_totals(local, shared_mask)
        out["pass_seconds"] = np.float64(time.time() - started)
    return out


def save(name: str, arrays: dict) -> dict:
    folder = cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError("%s/%s would need pickling" % (name, key))
    path = folder / f"{name}.npz"
    np.savez(path, **arrays)
    return file_record(path)


def main() -> None:
    import h5py

    started = time.time()
    single_cell = first.verify_single_cell_inputs()
    rows = first.stored_rows()
    targets = np.array(sorted(set(rows["target_gene"])))
    with h5py.File(first.vcc_files()["train"], "r") as handle:
        vcc_ids = singlecell.read_annotation(handle["var"], "gene_id")
    shared = {}
    for name, path in first.screen_files().items():
        with h5py.File(path, "r") as handle:
            shared[name] = np.isin(vcc_ids, first.var_index(handle))
    cache = {}
    for split, path in first.vcc_files().items():
        arrays = extract_vcc_counts(path, targets, shared, controls=(split == "train"))
        cache["vcc_" + split] = save("vcc_" + split, arrays)
        print(f"[extract] VCC {split}: {arrays['pert_rows'].size} perturbed cells, {float(arrays['pass_seconds']):.0f} s", flush=True)
    for name, path in first.screen_files().items():
        arrays = extract_screen_counts(name, path, first.eligible(name), vcc_ids)
        cache["screen_" + name] = save("screen_" + name, arrays)
        print(f"[extract] {name}: {arrays['pert_rows'].size} eligible perturbed cells, {float(arrays['pass_seconds']):.0f} s", flush=True)
    manifest = {
        "plan": PLAN,
        "made_by": "bioinf2027/scripts/make_e1a_label_extract.py",
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "single_cell_inputs": single_cell,
        "cache": cache,
        "target_genes": int(targets.size),
        "genes_shared_with_vcc": {name: int(mask.sum()) for name, mask in shared.items()},
        "counts_check": "every chunk: finite, non-negative, integer counts, and cell totals below 2**24",
        "seconds": time.time() - started,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    (cache_dir() / "manifest.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[extract] manifest written, %.0f s in all" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
