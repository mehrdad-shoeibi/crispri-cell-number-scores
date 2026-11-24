"""D2, the GPU pass: the identity check and the token-drop embeddings, in resumable units.

Design: bioinf2027/plans/D2.md, Sections 3, 5, 10, 11, 12 and 13. Runs in the D2 environment
(bioinf2027/requirements_d2.txt), launched through bioinf2027/scripts/run_item.py with that
environment's interpreter. The cell lists come from bioinf2027/scripts/make_d2_units.py and are
refused unless they match bioinf2027/results/D2/units.json.

Each unit of up to 2,000 cells: the cells' raw counts are read from their count file; the tokenizer's
full rank encoding is computed with and without the target's columns set to zero (the total
unchanged); the drop is checked for every cell (Section 11.2, item 4) and any failure stops the job;
the model input is CLS, the first 4,094 genes and EOS; the CLS embedding of layer quant_layers + (-1)
is computed on the GPU in fp32. The unit's embeddings and per-cell records are saved with a digest,
and a unit whose record exists is not repeated. A job stops starting new units after --hours.

Before the job the GPU is checked with nvidia-smi and refused if any compute process runs on it.

  --identity        the identity check: 2,000 cells without any drop against their cached embeddings
  --part NAME ...   one or more parts of the token-drop pass
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "2")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IDENTITY_TOLERANCE = 1e-4
IDENTITY_OF_RECORD = "identity_single_precision_totals"
PLAN = "bioinf2027/plans/D2.md, Sections 3, 5, 10 to 13"


def d2_root() -> Path:
    return config.storage_root() / "bioinf2027" / "D2"


def set_caches() -> None:
    cache = d2_root() / "hf_cache"
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ.pop("TRANSFORMERS_CACHE", None)


# --------------------------------------------------------------------------- GPU check


def gpu_reading(gpu: int) -> dict:
    """The GPU's state and every compute process on it, from nvidia-smi."""
    query = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                           check=True, capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                          check=True, capture_output=True, text=True).stdout.strip()
    processes = [line for line in apps.splitlines() if line.strip()]
    return {"gpu": query, "compute_processes": processes, "free": not processes, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# --------------------------------------------------------------------------- inputs


def stores() -> dict:
    root = config.storage_root()
    return {
        "VCC": root / "MADA" / "cache" / "foundation_models" / "geneformer" / "full_feature_extraction_v2",
        "RPE1 chunks": root / "MADA_data" / "processed" / "phasec" / "rpe1_features" / "full",
        "RPE1 top-up": root / "IGA_GENEFORMER_EXT",
        "K562-essential units": root / "IGA_GENEFORMER_EXT",
    }


def cached_path(store: str, name: str) -> Path:
    base = stores()[store]
    if store == "VCC":
        sub = "nt_embeddings" if name.startswith("nt__") else "perturbed_embeddings"
        return base / sub / f"{name}.npy"
    return base / name


def count_files() -> dict:
    root = config.storage_root() / "IGA_MAIN" / "data"
    return {
        "vcc_train": root / "vcc" / "train" / "adata_Training.h5ad",
        "vcc_val": root / "vcc" / "validation" / "adata_Validation.h5ad",
        "vcc_test": root / "vcc" / "test" / "adata_Test.h5ad",
        "RPE1": root / "replogle" / "raw_singlecell" / "rpe1_raw_singlecell_01.h5ad",
        "K562-essential": root / "replogle" / "raw_singlecell" / "K562_essential_raw_singlecell_01.h5ad",
    }


class Counts:
    """Row reads of one count file: CSR (VCC) or dense (external), with the genes' Ensembl identifiers."""

    def __init__(self, name: str) -> None:
        import h5py

        from bioinf2027 import singlecell

        self.name = name
        self.handle = h5py.File(count_files()[name], "r")
        node = self.handle["X"]
        self.sparse = hasattr(node, "keys")
        if name.startswith("vcc"):
            with h5py.File(count_files()["vcc_train"], "r") as train:
                self.ensembl = singlecell.read_annotation(train["var"], "gene_id")
            symbols = singlecell.read_annotation(self.handle["var"], self.handle["var"].attrs["_index"])
            if symbols.size != self.ensembl.size:
                raise RuntimeError("the VCC files list different numbers of genes")
        else:
            var = self.handle["var"]
            index = var.attrs["_index"]
            self.ensembl = singlecell.read_annotation(var, index.decode() if isinstance(index, bytes) else str(index))
        if self.sparse:
            self.indptr = np.asarray(node["indptr"][()], dtype=np.int64)
            self.data, self.indices = node["data"], node["indices"]
            self.n_genes = int(node.attrs["shape"][1])
        else:
            self.n_genes = int(node.shape[1])

    def rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if not np.all(np.diff(rows) > 0):
            raise ValueError("rows must be strictly increasing")
        if not self.sparse:
            return np.asarray(self.handle["X"][rows, :], dtype=np.float32)
        out = np.zeros((rows.size, self.n_genes), dtype=np.float32)
        for i, r in enumerate(rows.tolist()):
            lo, hi = int(self.indptr[r]), int(self.indptr[r + 1])
            out[i, np.asarray(self.indices[lo:hi])] = np.asarray(self.data[lo:hi], dtype=np.float32)
        return out


# --------------------------------------------------------------------------- one unit


def cell_keys(frame: pd.DataFrame) -> list:
    """The key of each row of a cell list: file:row, and for a delta row file:row:dropped gene.

    The delta samples draw their control cells for each target gene separately, so one control cell
    can stand in several rows of a part, each with another gene dropped; (file, row, gene) is unique.
    Perturbed and identity rows hold each cell once and keep the key file:row.
    """
    out = []
    for f, r, role, gene in zip(frame["file"], frame["file_row"], frame["role"], frame["drop_ensembl"]):
        out.append(f"{f}:{int(r)}:{gene}" if role == "delta" else f"{f}:{int(r)}")
    if len(set(out)) != len(out):
        raise RuntimeError("two rows of the cell list share a key")
    return out


def tokenize_unit(encoder, frame: pd.DataFrame, readers: dict, work: Path, drop: bool) -> tuple[dict, pd.DataFrame]:
    """Model inputs of the unit's cells and a per-cell record of the drop."""
    inputs, records = {}, []
    for file_name, part in frame.groupby("file", sort=True):
        reader = readers.setdefault(file_name, Counts(file_name))
        part = part.sort_values("file_row", kind="stable")
        # a control cell of a delta sample can recur with another gene dropped: each row is read once
        unique_rows, inverse = np.unique(part["file_row"].to_numpy(dtype=np.int64), return_inverse=True)
        counts = reader.rows(unique_rows)[inverse]
        # the float32 row sum of the float32 counts, exact for integer counts below 2**24, as submitted
        n_counts = counts.sum(axis=1, dtype=np.float32)
        keys = cell_keys(part)
        undropped = encoder.full_rankings(counts, reader.ensembl, keys, n_counts, work)
        if drop:
            dropped_counts = counts.copy()
            column_cache: dict = {}
            for i, ens in enumerate(part["drop_ensembl"].astype(str).tolist()):
                if ens not in column_cache:
                    column_cache[ens] = encoder.columns_of(reader.ensembl, ens)
                dropped_counts[i, column_cache[ens]] = 0.0
            dropped = encoder.full_rankings(dropped_counts, reader.ensembl, keys, n_counts, work)
        else:
            dropped = undropped
        from bioinf2027 import tokendrop

        for key, row in zip(keys, part.itertuples()):
            if key not in undropped or key not in dropped:
                raise RuntimeError("cell %s was not tokenized (no gene of the token dictionary?)" % key)
            token = int(row.drop_token) if drop and int(row.drop_token) >= 0 else None
            check = tokendrop.check_drop(undropped[key], dropped[key], token) if drop else {"passed": True}
            if not check["passed"]:
                raise RuntimeError("the drop check failed for %s (target %s): %s" % (key, row.drop_ensembl, check))
            inputs[key] = tokendrop.truncate(dropped[key], encoder.cls, encoder.eos)
            records.append({"key": key, "cell_id": row.cell_id, "file": file_name, "file_row": int(row.file_row),
                            "full_length_undropped": int(undropped[key].size), "full_length_dropped": int(dropped[key].size),
                            "input_length": int(inputs[key].size), "truncated": bool(undropped[key].size > tokendrop.GENES_KEPT),
                            "target_in_input": bool(check.get("target_in_input", False)), "appended": int(check.get("appended", 0))})
    return inputs, pd.DataFrame(records)


def run_unit(encoder, part: str, unit: int, frame: pd.DataFrame, readers: dict, drop: bool) -> dict:
    folder = d2_root() / "embeddings" / part
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / f"unit_{unit:05d}.json"
    array_path = folder / f"unit_{unit:05d}.npz"
    if done.exists():
        record = json.loads(done.read_text())
        if array_path.exists() and sha256(array_path) == record["file"]["sha256"]:
            return {**record, "reused": True}
    started = time.time()
    work = d2_root() / "work" / part
    inputs, records = tokenize_unit(encoder, frame, readers, work, drop)
    tokenized = time.time()
    embeddings = encoder.embed(inputs, work)
    keys = records["key"].tolist()
    matrix = np.stack([embeddings[k] for k in keys]).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("non-finite embedding in %s unit %d" % (part, unit))
    temporary = array_path.with_name(array_path.name + ".tmp.npz")
    np.savez(temporary, key=np.array(keys), embedding=matrix, **{c: records[c].to_numpy() for c in records.columns if c not in ("key", "file")})
    os.replace(temporary, array_path)
    record = {"part": part, "unit": unit, "cells": int(len(keys)), "file": file_record(array_path),
              "tokenize_seconds": round(tokenized - started, 1), "embed_seconds": round(time.time() - tokenized, 1),
              "cells_per_second": round(len(keys) / max(time.time() - tokenized, 1e-9), 2), "drop": drop,
              "drop_check": "passed for every cell" if drop else "no drop"}
    done.write_text(json.dumps(strict_json(record), indent=1) + "\n")
    return {**record, "reused": False}


# --------------------------------------------------------------------------- identity


def identity(encoder, frame: pd.DataFrame, readers: dict, name: str = "identity") -> dict:
    result = run_unit(encoder, name, 0, frame, readers, drop=False)
    with np.load(d2_root() / "embeddings" / name / "unit_00000.npz", allow_pickle=False) as store:
        keys = store["key"].astype(str)
        new = store["embedding"].astype(np.float64)
        lengths = store["input_length"]
    position = {f"{f}:{r}": i for i, (f, r) in enumerate(zip(frame["file"], frame["file_row"]))}
    cached = np.zeros_like(new)
    stores_of = np.empty(keys.size, dtype=object)
    loaded: dict = {}
    for i, key in enumerate(keys.tolist()):
        row = frame.iloc[position[key]]
        path = cached_path(row["store"], row["cached_file"])
        if path not in loaded:
            loaded[path] = np.load(path, mmap_mode="r")
        cached[i] = np.asarray(loaded[path][int(row["cached_row"])], dtype=np.float64)
        stores_of[i] = row["store"]
    error = new - cached
    worst = np.abs(error).max(axis=1)
    deciles = np.quantile(lengths, np.linspace(0, 1, 11))
    by_decile = []
    for lo, hi in zip(deciles[:-1], deciles[1:]):
        mask = (lengths >= lo) & (lengths <= hi)
        by_decile.append({"input_length_from": float(lo), "input_length_to": float(hi), "cells": int(mask.sum()),
                          "per_cell_error_sd_median": float(np.median(error[mask].std(axis=1))) if mask.any() else None})
    per_store = {}
    for store in sorted(set(stores_of.tolist())):
        mask = stores_of == store
        per_store[store] = {"cells": int(mask.sum()), "max_abs_difference": float(np.abs(error[mask]).max()),
                            "cells_above_tolerance": int((worst[mask] > IDENTITY_TOLERANCE).sum())}
    mean_error = error.mean(axis=0)
    out = {"cells": int(keys.size), "tolerance": IDENTITY_TOLERANCE, "max_abs_difference": float(np.abs(error).max()),
           "cells_above_tolerance": int((worst > IDENTITY_TOLERANCE).sum()), "per_store": per_store,
           "mean_error_vector_norm": float(np.linalg.norm(mean_error)), "error_sd_by_input_length_decile": by_decile,
           "median_cached_embedding_norm": float(np.median(np.linalg.norm(cached, axis=1))),
           "passed": bool((worst <= IDENTITY_TOLERANCE).all()), "unit": result}
    np.save(d2_root() / "embeddings" / name / "mean_error_vector.npy", mean_error)
    return out


# --------------------------------------------------------------------------- main


def load_parts(names) -> dict:
    record = json.loads((REPO / "bioinf2027" / "results" / "D2" / "units.json").read_text())
    folder = d2_root() / "units"
    out = {}
    for name in names:
        path = folder / f"{name}.parquet"
        if sha256(path) != record["files"][name]["sha256"]:
            raise RuntimeError("the %s cell list does not match bioinf2027/results/D2/units.json" % name)
        out[name] = pd.read_parquet(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--part", action="append", default=[])
    parser.add_argument("--hours", type=float, default=23.0)
    parser.add_argument("--identity-name", default="identity", help="output name of an identity run (a repeat keeps the first)")
    options = parser.parse_args()
    set_caches()
    reading = gpu_reading(options.gpu)
    print("[d2] nvidia-smi: %s" % json.dumps(reading), flush=True)
    if not reading["free"]:
        print("[d2] GPU %d is in use by another process; nothing is run" % options.gpu, flush=True)
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
    from bioinf2027 import tokendrop

    repository = config.storage_root() / "MADA" / "cache" / "foundation_models" / "geneformer-repo"
    encoder = tokendrop.Encoder(repository)
    encoder.load()
    environment = tokendrop.environment_record()
    print("[d2] environment %s" % json.dumps(environment), flush=True)
    readers: dict = {}
    started = time.time()
    log = {"plan": PLAN, "gpu": reading, "environment": environment, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if options.identity:
        parts = load_parts(["identity"])
        out = identity(encoder, parts["identity"], readers, options.identity_name)
        log["identity"] = out
        path = REPO / "bioinf2027" / "results" / "D2" / f"{options.identity_name}.json"
        path.write_text(json.dumps(strict_json(log), indent=1, allow_nan=False) + "\n")
        print("[d2] identity: max %.3g, cells above %g: %d, passed %s" % (out["max_abs_difference"], IDENTITY_TOLERANCE, out["cells_above_tolerance"], out["passed"]), flush=True)
        return 0
    # The check of record is the repeat with single-precision totals (bioinf2027/plans/D2.md, Section 14);
    # the first run, identity.json, stays on file.
    identity_record = json.loads((REPO / "bioinf2027" / "results" / "D2" / f"{IDENTITY_OF_RECORD}.json").read_text())
    if not identity_record["identity"]["passed"]:
        print("[d2] the identity check has not passed; no token-drop embedding is computed", flush=True)
        return 3
    parts = load_parts(options.part)
    completed, stopped = [], False
    for name, frame in parts.items():
        for unit, cells in frame.groupby("unit", sort=True):
            if time.time() - started > options.hours * 3600:
                stopped = True
                break
            result = run_unit(encoder, name, int(unit), cells, readers, drop=True)
            completed.append(result)
            if not result["reused"]:
                print("[d2] %s unit %d: %d cells, %.1f cells/s, %.0f s since start" % (name, unit, result["cells"], result["cells_per_second"], time.time() - started), flush=True)
        if stopped:
            break
    print("[d2] %d units done (%d reused); stopped at the time limit: %s" % (len(completed), sum(r["reused"] for r in completed), stopped), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
