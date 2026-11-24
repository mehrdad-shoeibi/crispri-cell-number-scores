"""E1: record the single-cell inputs before the design is final.

E1 rebuilds labels from cell-level data. This script records, for every
single-cell file E1 will read, its portable path, size, digest and shape, and
for the VCC files the number of cells, genes and guides per (target gene,
batch) group, so that the design note can cite them. No label, feature or
statistic of E1 is computed.

Output: bioinf2027/plans/E1_input_records.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, singlecell  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "plans" / "E1_input_records.json"


def single_cell_files() -> dict[str, Path]:
    data = config.storage_root() / "IGA_MAIN" / "data"
    return {
        "VCC train": data / "vcc" / "train" / "adata_Training.h5ad",
        "VCC validation": data / "vcc" / "validation" / "adata_Validation.h5ad",
        "VCC test": data / "vcc" / "test" / "adata_Test.h5ad",
        "RPE1": data / "replogle" / "raw_singlecell" / "rpe1_raw_singlecell_01.h5ad",
        "K562-essential": data / "replogle" / "raw_singlecell" / "K562_essential_raw_singlecell_01.h5ad",
    }


def describe(name: str, path: Path) -> dict:
    import h5py

    record = file_record(path)
    with h5py.File(path, "r") as handle:
        matrix = handle["X"]
        if isinstance(matrix, h5py.Group):
            shape = [int(v) for v in matrix.attrs["shape"]]
            record["matrix"] = {"layout": str(matrix.attrs.get("encoding-type")), "shape": shape,
                                "stored_dtype": str(matrix["data"].dtype), "stored_values": int(matrix["data"].shape[0])}
        else:
            record["matrix"] = {"layout": "dense", "shape": [int(v) for v in matrix.shape], "stored_dtype": str(matrix.dtype)}
        obs = handle["obs"]
        record["obs_columns"] = sorted(k for k in obs.keys() if not k.startswith("__"))
        if name.startswith("VCC"):
            target = singlecell.read_annotation(obs, "target_gene")
            batch = singlecell.read_annotation(obs, "batch")
            guide = singlecell.read_annotation(obs, "guide_id")
            control = target == "non-targeting"
            groups = {}
            for t, b, g in zip(target[~control].tolist(), batch[~control].tolist(), guide[~control].tolist()):
                groups.setdefault((t, b), set()).add(g)
            per_group = np.array([len(v) for v in groups.values()])
            record["vcc"] = {
                "control_cells": int(control.sum()),
                "perturbed_cells": int((~control).sum()),
                "batches": int(len(set(batch.tolist()))),
                "target_genes": int(len(set(target[~control].tolist()))),
                "groups": int(len(groups)),
                "guides_per_group": {"min": int(per_group.min()), "median": float(np.median(per_group)),
                                     "max": int(per_group.max()),
                                     "groups_with_more_than_one_guide": int((per_group > 1).sum())},
            }
    return record


def main() -> None:
    records = {name: describe(name, path) for name, path in single_cell_files().items()}
    OUT.write_text(json.dumps(strict_json({
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "purpose": "inputs of E1, recorded before its design is final; no label or statistic computed",
        "files": records,
    }), indent=1, allow_nan=False) + "\n")
    for name, record in records.items():
        print(name, record.get("bytes"), record["matrix"], record.get("vcc", ""))
    print("wrote", config.portable(OUT))


if __name__ == "__main__":
    main()
