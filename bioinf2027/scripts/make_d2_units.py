"""D2, the cell lists of the token-drop pass: every cell to embed, the token it drops, and its cached embedding.

Design: bioinf2027/plans/D2.md, Sections 3 to 5, 10, 11 and 12, with the details of Section 13.
Runs in the pipeline environment; nothing is embedded or tokenized here.

Cells:
- the identity sample: 500 cells from each store (VCC, RPE1 chunks, RPE1 top-up, K562-essential units),
  perturbed and control, drawn with seed 20271103; embedded without any drop;
- every perturbed cell of the modelled VCC rows and of the eligible external constructs, with its own
  target's token dropped;
- the delta samples: on VCC 10 control cells of each of the 48 batches per target gene (seed 20271101),
  externally 100 pooled control cells per target gene (seed 20271102), each with that gene's token
  dropped; their halves (seed 20271104), 5 and 5 within each batch on VCC, 50 and 50 externally.

Checks before any list is written (Section 10.2, item 6): the perturbed cells of every VCC row and
external construct are exactly the cells behind its cached embedding (the stores' cell indices), the
counts equal E1's, and every control cell of the delta samples has a cached embedding.

Outputs: $IGA_STORAGE_ROOT/bioinf2027/D2/units/<part>.parquet and bioinf2027/results/D2/units.json.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, singlecell  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_d2_input_records as d2records  # noqa: E402
import make_e1a_extract as first  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "D2" / "units.json"
IDENTITY_SEED = 20271103
VCC_DELTA_SEED = 20271101
EXTERNAL_DELTA_SEED = 20271102
HALF_SEED = 20271104
IDENTITY_PER_STORE = 500
VCC_DELTA_PER_BATCH = 10
EXTERNAL_DELTA = 100
UNIT = 2000
CONTROL = "non-targeting"
ENCODER_SPLIT = {"train": "training", "val": "validation", "test": "test"}
COLUMNS = ["part", "unit", "store", "file", "file_row", "cell_id", "cached_file", "cached_row", "role",
           "row_split", "row_batch", "row_gene", "construct", "drop_ensembl", "drop_symbol", "gene_index", "batch", "half"]


def units_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "D2" / "units"


def mapping_dict() -> dict:
    with open(d2records.encoder_paths()["ensembl_mapping_dictionary"], "rb") as handle:
        return pickle.load(handle)


def token_dict() -> dict:
    with open(d2records.encoder_paths()["token_dictionary"], "rb") as handle:
        return pickle.load(handle)


# --------------------------------------------------------------------------- VCC


def vcc_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Every cell of the three VCC files, the encoder's cell index, and symbol -> Ensembl."""
    import h5py

    frames = []
    # Only the training file carries gene_id; the other files must list the same genes in the same order.
    with h5py.File(first.vcc_files()["train"], "r") as handle:
        train_symbols = singlecell.read_annotation(handle["var"], handle["var"].attrs["_index"])
        gene_id = singlecell.read_annotation(handle["var"], "gene_id")
    symbol_to_ensembl = dict(zip(train_symbols.tolist(), gene_id.tolist()))
    for split, path in first.vcc_files().items():
        with h5py.File(path, "r") as handle:
            obs = first.obs_table(handle, ("target_gene", "batch", "guide_id"))
            symbols = singlecell.read_annotation(handle["var"], handle["var"].attrs["_index"])
        if not np.array_equal(symbols, train_symbols):
            raise RuntimeError("the VCC %s file lists other genes than the training file" % split)
        frames.append(pd.DataFrame({"file": f"vcc_{split}", "file_row": np.arange(obs["_index"].size), "cell_id": obs["_index"].astype(str),
                                    "target_gene": obs["target_gene"].astype(str), "batch": obs["batch"].astype(str), "split": split}))
    cells = pd.concat(frames, ignore_index=True)
    index = pd.read_parquet(d2records.store_paths()["vcc_cell_index"])
    index["cell_id"] = index["cell_id"].astype(str)
    return cells, index, symbol_to_ensembl


def vcc_perturbed(cells: pd.DataFrame, index: pd.DataFrame, rows: pd.DataFrame, ensembl: dict) -> tuple[pd.DataFrame, dict]:
    pert_index = index[index["role"] == "perturbed"].set_index("cell_id")
    if not pert_index.index.is_unique:
        raise RuntimeError("the VCC perturbed cell index repeats a cell")
    keyed = cells[cells["target_gene"] != CONTROL]
    groups = keyed.groupby(["split", "batch", "target_gene"]).indices
    out, mismatched = [], []
    for row in rows.itertuples():
        members = keyed.iloc[groups.get((row.split, row.batch, row.target_gene), [])]
        if len(members) != int(row.n_pool):
            mismatched.append((row.split, row.batch, row.target_gene, len(members), int(row.n_pool)))
            continue
        found = pert_index.reindex(members["cell_id"])
        if found["chunk_id"].isna().any():
            raise RuntimeError("a perturbed cell of %s has no cached embedding" % ((row.split, row.batch, row.target_gene),))
        if not (found["source_split"] == ENCODER_SPLIT[row.split]).all() or not (found["source_batch"] == row.batch).all():
            raise RuntimeError("cached embeddings of %s come from another split or batch" % ((row.split, row.batch, row.target_gene),))
        out.append(pd.DataFrame({
            "store": "VCC", "file": members["file"].to_numpy(), "file_row": members["file_row"].to_numpy(), "cell_id": members["cell_id"].to_numpy(),
            "cached_file": found["chunk_id"].to_numpy(), "cached_row": found["embedding_row_index"].astype(int).to_numpy(), "role": "perturbed",
            "row_split": row.split, "row_batch": row.batch, "row_gene": row.target_gene, "construct": "",
            "drop_ensembl": ensembl[row.target_gene], "drop_symbol": row.target_gene, "gene_index": -1, "batch": row.batch, "half": ""}))
    if mismatched:
        raise RuntimeError("%d VCC rows hold another number of cells than E1 records, e.g. %s" % (len(mismatched), mismatched[:3]))
    frame = pd.concat(out, ignore_index=True)
    covered = set(frame["cell_id"])
    check = {"rows": int(len(rows)), "cells": int(len(frame)), "index_perturbed_cells": int(len(pert_index)),
             "index_cells_not_in_a_modelled_row": int(len(set(pert_index.index) - covered)),
             "cells_equal_the_index": bool(covered == set(pert_index.index))}
    return frame, check


def vcc_delta(cells: pd.DataFrame, index: pd.DataFrame, genes: list, ensembl: dict) -> tuple[pd.DataFrame, dict]:
    nt_index = index[index["role"] == "NT"].set_index("cell_id")
    controls = cells[(cells["target_gene"] == CONTROL) & (cells["split"] == "train")]
    batches = sorted(set(controls["batch"]))
    members = {b: controls[controls["batch"] == b].sort_values("file_row") for b in batches}
    out = []
    for g, gene in enumerate(genes):
        rng = np.random.default_rng([VCC_DELTA_SEED, g])
        for b_index, batch in enumerate(batches):
            pool = members[batch]
            picked = np.sort(rng.choice(len(pool), VCC_DELTA_PER_BATCH, replace=False))
            chosen = pool.iloc[picked]
            halves = np.random.default_rng([HALF_SEED, g, b_index]).permutation(VCC_DELTA_PER_BATCH)
            half = np.where(np.isin(np.arange(VCC_DELTA_PER_BATCH), halves[: VCC_DELTA_PER_BATCH // 2]), "A", "B")
            found = nt_index.reindex(chosen["cell_id"])
            if found["chunk_id"].isna().any():
                raise RuntimeError("a VCC control cell of the delta sample has no cached embedding")
            out.append(pd.DataFrame({
                "store": "VCC", "file": chosen["file"].to_numpy(), "file_row": chosen["file_row"].to_numpy(), "cell_id": chosen["cell_id"].to_numpy(),
                "cached_file": found["chunk_id"].to_numpy(), "cached_row": found["embedding_row_index"].astype(int).to_numpy(), "role": "delta",
                "row_split": "", "row_batch": "", "row_gene": gene, "construct": "", "drop_ensembl": ensembl[gene], "drop_symbol": gene,
                "gene_index": g, "batch": batch, "half": half}))
    frame = pd.concat(out, ignore_index=True)
    return frame, {"genes": len(genes), "batches": len(batches), "cells": int(len(frame)), "control_cells_in_the_index": int(len(nt_index)),
                   "control_cells_in_the_training_file": int(len(controls))}


# --------------------------------------------------------------------------- external


def external_cells(screen: str) -> pd.DataFrame:
    import h5py

    path = first.screen_files()[screen]
    with h5py.File(path, "r") as handle:
        obs = first.obs_table(handle, ("gene", "gene_transcript"))
    is_control = np.char.find(np.char.lower(obs["gene"].astype(str)), CONTROL) >= 0
    return pd.DataFrame({"file": f"{screen}", "file_row": np.arange(obs["gene"].size), "gene_transcript": obs["gene_transcript"].astype(str),
                         "is_control": is_control})


def external_store(screen: str) -> pd.DataFrame:
    """Every cell of the screen's cached embedding stores: original row, store file and row."""
    paths = d2records.store_paths()
    if screen == "RPE1":
        master = pd.read_parquet(paths["rpe1_master_cell_index"])
        chunks = pd.DataFrame({"orig_obs_idx": master["orig_obs_idx"].astype(int), "cached_file": ["geneformer_chunks/chunk_%04d/emb.npy" % c for c in master["chunk_id"].astype(int)],
                               "cached_row": master["chunk_row_pos"].astype(int), "store": "RPE1 chunks", "role_in_store": master["role"].astype(str),
                               "label_in_store": master["gene_transcript"].astype(str)})
        top = pd.read_parquet(paths["rpe1_topup"] / "rpe1_43_percell_index.parquet")
        topup = pd.DataFrame({"orig_obs_idx": top["orig_obs_idx"].astype(int), "cached_file": "rpe1_topup/rpe1_43_percell_emb.npy",
                              "cached_row": np.arange(len(top)), "store": "RPE1 top-up", "role_in_store": "perturbed", "label_in_store": top["gene_transcript"].astype(str)})
        return pd.concat([chunks, topup], ignore_index=True)
    frames = []
    for path in sorted(paths["k562_units"].glob("unit_*.parquet")):
        frame = pd.read_parquet(path)
        frames.append(pd.DataFrame({"orig_obs_idx": frame["orig_obs_idx"].astype(int), "cached_file": f"k562_essential/emb_units/{path.stem}.npy",
                                    "cached_row": np.arange(len(frame)), "store": "K562-essential units",
                                    "role_in_store": np.where(frame["construct"].astype(str).str.contains("NT_POOL"), "nt", "perturbed"),
                                    "label_in_store": frame["construct"].astype(str)}))
    return pd.concat(frames, ignore_index=True)


def external_perturbed(screen: str, cells: pd.DataFrame, store: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    store_pert = store[store["role_in_store"] == "perturbed"]
    # a cell may appear in the chunks and in the top-up; the top-up carries the 43 constructs the chunks lack
    by_cell = {}
    for row in store_pert.itertuples():
        by_cell.setdefault(int(row.orig_obs_idx), []).append(row)
    out, problems = [], []
    perturbed = cells[~cells["is_control"]]
    groups = perturbed.groupby("gene_transcript").indices
    for row in table.itertuples():
        members = perturbed.iloc[groups.get(row.construct, [])]
        if len(members) != int(row.n_pool):
            problems.append((row.construct, len(members), int(row.n_pool)))
            continue
        cached_file, cached_row, stores = [], [], []
        for orig in members["file_row"].tolist():
            entries = [e for e in by_cell.get(int(orig), []) if e.label_in_store == row.construct]
            if len(entries) != 1:
                raise RuntimeError("%s cell %d has %d cached embeddings" % (row.construct, orig, len(entries)))
            cached_file.append(entries[0].cached_file)
            cached_row.append(int(entries[0].cached_row))
            stores.append(entries[0].store)
        out.append(pd.DataFrame({
            "store": stores, "file": screen, "file_row": members["file_row"].to_numpy(), "cell_id": members["file_row"].astype(str).to_numpy(),
            "cached_file": cached_file, "cached_row": cached_row, "role": "perturbed", "row_split": "", "row_batch": "", "row_gene": "",
            "construct": row.construct, "drop_ensembl": row.target_ensg, "drop_symbol": row.target_symbol, "gene_index": -1, "batch": "", "half": ""}))
    if problems:
        raise RuntimeError("%d %s constructs hold another number of cells than E1 records, e.g. %s" % (len(problems), screen, problems[:3]))
    frame = pd.concat(out, ignore_index=True)
    return frame, {"constructs": int(len(table)), "cells": int(len(frame)), "by_store": frame["store"].value_counts().to_dict()}


def external_delta(screen: str, cells: pd.DataFrame, store: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    controls = cells[cells["is_control"]].sort_values("file_row")
    store_nt = store[store["role_in_store"] == "nt"].drop_duplicates("orig_obs_idx").set_index("orig_obs_idx")
    genes = sorted(set(table["target_ensg"].astype(str)))
    symbol = dict(zip(table["target_ensg"].astype(str), table["target_symbol"].astype(str)))
    found_all = store_nt.reindex(controls["file_row"].astype(int))
    if found_all["cached_file"].isna().any():
        raise RuntimeError("%s: a control cell has no cached embedding" % screen)
    out = []
    for g, gene in enumerate(genes):
        picked = np.sort(np.random.default_rng([EXTERNAL_DELTA_SEED, g]).choice(len(controls), EXTERNAL_DELTA, replace=False))
        chosen = controls.iloc[picked]
        halves = np.random.default_rng([HALF_SEED, g]).permutation(EXTERNAL_DELTA)
        half = np.where(np.isin(np.arange(EXTERNAL_DELTA), halves[: EXTERNAL_DELTA // 2]), "A", "B")
        found = found_all.iloc[picked]
        out.append(pd.DataFrame({
            "store": found["store"].to_numpy(), "file": screen, "file_row": chosen["file_row"].to_numpy(), "cell_id": chosen["file_row"].astype(str).to_numpy(),
            "cached_file": found["cached_file"].to_numpy(), "cached_row": found["cached_row"].astype(int).to_numpy(), "role": "delta",
            "row_split": "", "row_batch": "", "row_gene": "", "construct": "", "drop_ensembl": gene, "drop_symbol": symbol[gene],
            "gene_index": g, "batch": "", "half": half}))
    frame = pd.concat(out, ignore_index=True)
    return frame, {"genes": len(genes), "control_cells": int(len(controls)), "cells": int(len(frame))}


# --------------------------------------------------------------------------- identity sample


def identity_sample(vcc_cells_: pd.DataFrame, index: pd.DataFrame, stores: dict, ext_cells: dict) -> pd.DataFrame:
    """500 cells from each store, perturbed and control, in a fixed order of stores."""
    rng = np.random.default_rng(IDENTITY_SEED)
    out = []
    vcc_keyed = vcc_cells_.drop_duplicates("cell_id").set_index("cell_id")
    picked = index.iloc[np.sort(rng.choice(len(index), IDENTITY_PER_STORE, replace=False))]
    located = vcc_keyed.reindex(picked["cell_id"])
    if located["file"].isna().any():
        raise RuntimeError("a VCC cell of the identity sample is absent from the files")
    out.append(pd.DataFrame({"store": "VCC", "file": located["file"].to_numpy(), "file_row": located["file_row"].astype(int).to_numpy(),
                             "cell_id": picked["cell_id"].to_numpy(), "cached_file": picked["chunk_id"].to_numpy(),
                             "cached_row": picked["embedding_row_index"].astype(int).to_numpy(), "role": "identity"}))
    for screen, name in (("RPE1", "RPE1 chunks"), ("RPE1", "RPE1 top-up"), ("K562-essential", "K562-essential units")):
        store = stores[screen]
        part = store[store["store"] == name].reset_index(drop=True)
        chosen = part.iloc[np.sort(rng.choice(len(part), IDENTITY_PER_STORE, replace=False))]
        out.append(pd.DataFrame({"store": name, "file": screen, "file_row": chosen["orig_obs_idx"].astype(int).to_numpy(),
                                 "cell_id": chosen["orig_obs_idx"].astype(str).to_numpy(), "cached_file": chosen["cached_file"].to_numpy(),
                                 "cached_row": chosen["cached_row"].astype(int).to_numpy(), "role": "identity"}))
    frame = pd.concat(out, ignore_index=True)
    for column in COLUMNS:
        if column not in frame:
            frame[column] = "" if column not in ("gene_index",) else -1
    return frame


# --------------------------------------------------------------------------- main


def with_units(frame: pd.DataFrame, part: str) -> pd.DataFrame:
    frame = frame.sort_values(["file", "file_row"], kind="stable").reset_index(drop=True)
    frame["part"] = part
    frame["unit"] = np.arange(len(frame)) // UNIT
    return frame[COLUMNS]


def tokens(frame: pd.DataFrame, mapping: dict, tokens_: dict) -> pd.DataFrame:
    collapsed = frame["drop_ensembl"].astype(str).str.upper().map(mapping)
    frame["drop_collapsed"] = collapsed.fillna("")
    frame["drop_token"] = collapsed.map(tokens_).fillna(-1).astype(int)
    return frame


def main() -> None:
    started = time.time()
    rows = e1cells.load_tables("G6")["vcc"][["split", "batch", "target_gene", "n_pool"]]
    cells, index, ensembl = vcc_tables()
    mapping, tokens_ = mapping_dict(), token_dict()
    report: dict = {"plan": "bioinf2027/plans/D2.md, Sections 3 to 5, 10 to 13", "unit_size": UNIT}
    parts = {}
    frame, report["vcc_perturbed"] = vcc_perturbed(cells, index, rows, ensembl)
    parts["vcc_perturbed"] = frame
    genes = sorted(set(rows["target_gene"]))
    frame, report["vcc_delta"] = vcc_delta(cells, index, genes, ensembl)
    parts["vcc_delta"] = frame
    stores, ext = {}, {}
    for screen in first.screen_files():
        table = e1cells.load_tables("G1")[screen]
        ext[screen] = external_cells(screen)
        stores[screen] = external_store(screen)
        short = "rpe1" if screen == "RPE1" else "k562"
        frame, report[f"{short}_perturbed"] = external_perturbed(screen, ext[screen], stores[screen], table)
        parts[f"{short}_perturbed"] = frame
        frame, report[f"{short}_delta"] = external_delta(screen, ext[screen], stores[screen], table)
        parts[f"{short}_delta"] = frame
        print("[units] %s done" % screen, flush=True)
    parts["identity"] = identity_sample(cells, index, stores, ext)
    folder = units_dir()
    folder.mkdir(parents=True, exist_ok=True)
    files = {}
    for part, frame in parts.items():
        frame = tokens(with_units(frame, part), mapping, tokens_)
        path = folder / f"{part}.parquet"
        frame.to_parquet(path, index=False)
        files[part] = {**file_record(path), "cells": int(len(frame)), "units": int(frame["unit"].nunique()),
                       "cells_whose_target_has_no_token": int((frame["drop_token"] < 0).sum()) if part != "identity" else None}
        print("[units] %s: %d cells in %d units" % (part, len(frame), frame["unit"].nunique()), flush=True)
    report["files"] = files
    report["cells_in_all_parts_but_identity"] = int(sum(v["cells"] for k, v in files.items() if k != "identity"))
    report["inputs"] = {"d2_input_records": file_record(REPO / "bioinf2027" / "results" / "D2" / "input_records.json"),
                        "g6_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G6" / "manifest.json"),
                        "g1_manifest": file_record(REPO / "bioinf2027" / "endpoints" / "G1" / "manifest.json")}
    report["seconds"] = round(time.time() - started, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(strict_json(report), indent=1, allow_nan=False, default=str) + "\n")
    print("[units] written; %d cells to embed with a drop" % report["cells_in_all_parts_but_identity"], flush=True)


if __name__ == "__main__":
    main()
