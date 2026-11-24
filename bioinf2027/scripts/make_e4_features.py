"""E4, the matched features: x and z from exactly the cells of label draws 1 to 3, with the builder checks.

Design: bioinf2027/plans/E4.md, Section 4 as amended by Sections 10 to 12, with the details of Section 13.
For each cell of the two pairs (G5, G6, G1, G2), each VCC row and each external construct, feature
replicate k (k = 1, 2, 3) uses the perturbed and control cells of draw k of the row's label (seeds
20260801 to 20260803, the label builder's draw order; bioinf2027/src/bioinf2027/endpoint.py, draw):

- x, the mean over the draw's perturbed cells of each feature gene's external-mode value minus the mean
  over the draw's control cells; on a screen zero where the screen lacks the feature, clipped to +/-10;
- z, the mean of the cached CLS embeddings of the draw's perturbed cells minus that of its control cells.

Checks before any feature is written (stop rules of Sections 7 and 10.2): every draw of every row and
construct regenerates the stored E1 label (1e-9); with all cells in place of a draw the builder
reproduces the stored VCC x (source mode, 1e-5) and z (1e-5) and the external x and z (1e-5); every cell
of every draw has a cached embedding. A failed check stops E4.

Also written: the cell-disjoint variant (draw 1 taken from half B of the R5 halves) and the
shared-gene size-factor variant (draws 1 to 3 normalised by the totals over the genes VCC shares with
each screen). No head is fitted.

Outputs: $IGA_STORAGE_ROOT/bioinf2027/E4/features/*.npz; bioinf2027/results/E4/features.json.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, e1inputs, endpoint, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_d2_embeddings as d2emb  # noqa: E402
import make_d2_units as d2units  # noqa: E402
import make_e1a_extract as first  # noqa: E402
import make_e1a_labels as builder  # noqa: E402
import make_e4_extract as extract  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E4"
PLAN = "bioinf2027/plans/E4.md, Section 4 as amended by Sections 10 to 13"
CELLS = ("G5", "G6", "G1", "G2")
DRAW_SEEDS = endpoint.SEEDS[:3]
LABEL_TOLERANCE = 1e-9
FEATURE_TOLERANCE = 1e-5
CLIP = 10.0
DISJOINT_LEVEL = 10
WORKERS = 4
SCREENS = e1inputs.SCREENS

STORE: dict = {}


def text(values) -> np.ndarray:
    """Fixed-width text, not an object array (pandas' to_numpy of strings gives objects)."""
    return np.asarray(values, dtype=str)


def save_npz(path: Path, arrays: dict) -> None:
    """np.savez that refuses object arrays: every reader loads these files with allow_pickle=False.

    The first smoke fits (2026-11-10) failed on construct names written as objects."""
    objects = sorted(k for k, v in arrays.items() if np.asarray(v).dtype.hasobject)
    if objects:
        raise TypeError("object arrays would need pickle to load: %s" % objects)
    np.savez(path, **arrays)


def features_root() -> Path:
    return config.storage_root() / "bioinf2027" / "E4" / "features"


# --------------------------------------------------------------------------- inputs


def load_feature_counts() -> dict:
    """The E4 extraction, refused unless every file matches its manifest and its rows those of the label cache."""
    folder = extract.cache_dir()
    manifest = json.loads((folder / "manifest.json").read_text())
    out = {}
    for name, record in manifest["cache"].items():
        path = folder / f"{name}.npz"
        if sha256(path) != record["sha256"]:
            raise RuntimeError("E4 extraction file %s does not match its manifest" % name)
        with np.load(path, allow_pickle=False) as store:
            out[name] = {key: store[key] for key in store.files}
    return {"arrays": out, "manifest": file_record(folder / "manifest.json")}


def embedding_rows() -> dict:
    """Cached embeddings of every cell the draws can use, aligned with the label cache's cell arrays."""
    loaded: dict = {}

    def fetch(store: str, name: str, rows: np.ndarray) -> np.ndarray:
        path = d2emb.cached_path(store, name)
        if path not in loaded:
            loaded[path] = np.load(path, mmap_mode="r")
        return np.asarray(loaded[path][np.asarray(rows, dtype=np.int64)], dtype=np.float32)

    counts = STORE["counts"]
    units = d2emb.d2_root() / "units"
    record = json.loads((REPO / "bioinf2027" / "results" / "D2" / "units.json").read_text())
    out = {}
    vcc_pert = pd.read_parquet(units / "vcc_perturbed.parquet")
    if sha256(units / "vcc_perturbed.parquet") != record["files"]["vcc_perturbed"]["sha256"]:
        raise RuntimeError("the VCC perturbed cell list differs from its record")
    for split in builder.SPLITS:
        rows = counts["vcc_" + split]["pert_rows"]
        part = vcc_pert[vcc_pert["file"] == f"vcc_{split}"].set_index("file_row")
        # cells of groups below 10 perturbed cells belong to no modelled row and keep no embedding
        located = part.reindex(rows)
        emb = np.full((rows.size, 768), np.nan, dtype=np.float32)
        for name in located["cached_file"].dropna().unique():
            positions = np.flatnonzero((located["cached_file"] == name).to_numpy())
            emb[positions] = fetch("VCC", name, located["cached_row"].to_numpy()[positions].astype(np.int64))
        out["vcc_" + split] = emb
    cells, index, _ = d2units.vcc_tables()
    train_cells = cells[cells["file"] == "vcc_train"].set_index("file_row")
    nt = index[index["role"] == "NT"].set_index("cell_id")
    nt_rows = counts["vcc_train"]["nt_rows"]
    ids = train_cells.loc[nt_rows, "cell_id"].to_numpy()
    found = nt.reindex(ids)
    if found["chunk_id"].isna().any():
        raise RuntimeError("a VCC control cell has no cached embedding")
    emb = np.empty((nt_rows.size, 768), dtype=np.float32)
    for name, group in found.reset_index().groupby("chunk_id"):
        emb[group.index.to_numpy()] = fetch("VCC", name, group["embedding_row_index"].to_numpy())
    out["vcc_nt"] = emb
    for screen in SCREENS:
        short = "rpe1" if screen == "RPE1" else "k562"
        listed = pd.read_parquet(units / f"{short}_perturbed.parquet")
        store = d2units.external_store(screen)
        arrays = counts["screen_" + screen]
        by_row = listed.set_index("file_row")
        pert_rows = arrays["pert_rows"]
        emb = np.empty((pert_rows.size, 768), dtype=np.float32)
        located = by_row.loc[pert_rows]
        for (store_name, name), group in located.reset_index().groupby(["store", "cached_file"]):
            emb[group.index.to_numpy()] = fetch(store_name, name, group["cached_row"].to_numpy())
        out[f"{screen}_pert"] = emb
        nt_store = store[store["role_in_store"] == "nt"].drop_duplicates("orig_obs_idx").set_index("orig_obs_idx")
        found = nt_store.reindex(arrays["nt_rows"])
        if found["cached_file"].isna().any():
            raise RuntimeError("%s: a control cell has no cached embedding" % screen)
        emb = np.empty((arrays["nt_rows"].size, 768), dtype=np.float32)
        for (store_name, name), group in found.reset_index().groupby(["store", "cached_file"]):
            emb[group.index.to_numpy()] = fetch(store_name, name, group["cached_row"].to_numpy())
        out[f"{screen}_nt"] = emb
    return out


def values(counts: np.ndarray, totals: np.ndarray, mode: str = "external") -> np.ndarray:
    return endpoint.external_values(counts, totals) if mode == "external" else endpoint.source_values(counts, totals)


# --------------------------------------------------------------------------- features of one set of cells


def vcc_features(split: str, pert: np.ndarray, ctrl: np.ndarray, norm: str = "all", mode: str = "external") -> tuple[np.ndarray, np.ndarray]:
    fc = STORE["features"]
    counts = STORE["counts"]
    pert_total = counts["vcc_" + split]["pert_total"] if norm == "all" else counts["vcc_" + split]["pert_shared_total_" + norm]
    nt_total = counts["vcc_train"]["nt_total"] if norm == "all" else counts["vcc_train"]["nt_shared_total_" + norm]
    p = values(fc["vcc_" + split]["pert_counts"][pert], pert_total[pert], mode).astype(np.float64).mean(axis=0)
    c = values(fc["vcc_train"]["nt_counts"][ctrl], nt_total[ctrl], mode).astype(np.float64).mean(axis=0)
    ep = STORE["emb"]["vcc_" + split][pert].astype(np.float64)
    if np.isnan(ep).any():
        raise RuntimeError("a perturbed cell of a draw has no cached embedding")
    z = ep.mean(axis=0) - STORE["emb"]["vcc_nt"][ctrl].astype(np.float64).mean(axis=0)
    return p - c, z


def screen_features(screen: str, pert: np.ndarray, ctrl: np.ndarray, norm: str = "all") -> tuple[np.ndarray, np.ndarray]:
    fc = STORE["features"]["screen_" + screen]
    arrays = STORE["counts"]["screen_" + screen]
    pert_total = arrays["pert_total"] if norm == "all" else arrays["pert_shared_total"]
    nt_total = arrays["nt_total"] if norm == "all" else arrays["nt_shared_total"]
    p = values(fc["pert_counts"][pert], pert_total[pert]).astype(np.float64).mean(axis=0)
    c = values(fc["nt_counts"][ctrl], nt_total[ctrl]).astype(np.float64).mean(axis=0)
    x = p - c
    x[fc["feature_columns"] < 0] = 0.0
    x = np.clip(np.nan_to_num(x, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)
    z = STORE["emb"][f"{screen}_pert"][pert].astype(np.float64).mean(axis=0) - STORE["emb"][f"{screen}_nt"][ctrl].astype(np.float64).mean(axis=0)
    return x, z


def control_set(cell: str, batch: str) -> np.ndarray:
    spec = builder.CELLS[cell]
    return builder.DATA["pooled_index"] if spec["controls"] == "pooled" else builder.DATA["batch_members"][batch]


# --------------------------------------------------------------------------- work


def label_work(cell: str, start: int, stop: int) -> list[float]:
    """Largest difference between the regenerated and the stored label, per VCC row of a chunk."""
    spec = builder.CELLS[cell]
    rows = builder.DATA["cell_rows"][cell]
    vals = builder.DATA["vcc_values"]["all"]
    slot = builder.DATA["target_slot"]
    out = []
    for k in range(start, stop):
        row = rows.iloc[k]
        t = slot[row.target_gene]
        index = control_set(cell, row.batch)
        summary, _ = endpoint.subsampled_label(vals[f"{row.split}_ext"][row.positions], vals["nt_ext"][index, t], spec["n_pert"], spec["n_ctrl"])
        out.append(summary["y"])
    return out


def external_label_work(cell: str, screen: str, start: int, stop: int) -> list[float]:
    spec = builder.CELLS[cell]
    arrays = builder.DATA["screens"][screen]
    vals = builder.DATA["screen_values"][(screen, "all")]
    out = []
    for k in range(start, stop):
        summary, _ = endpoint.subsampled_label(vals["pert"][arrays["pools"][k]], vals["nt"][:, arrays["construct_slot"][k]], spec["n_pert"], spec["n_ctrl"])
        out.append(summary["y"])
    return out


def feature_work(cell: str, side: str, start: int, stop: int, variant: str) -> dict:
    """Features of rows or constructs start to stop of a cell under one variant."""
    spec = builder.CELLS[cell]
    n_pert, n_ctrl = spec["n_pert"], spec["n_ctrl"]
    xs, zs, bs, keep = [], [], [], []
    for k in range(start, stop):
        if side == "VCC":
            row = builder.DATA["cell_rows"][cell].iloc[k]
            pool, controls = row.positions, control_set(cell, row.batch)
            split = row.split
        else:
            arrays = builder.DATA["screens"][side]
            pool, controls = arrays["pools"][k], np.arange(STORE["counts"]["screen_" + side]["nt_rows"].size)
        if variant.startswith("draw ") or variant.startswith("shared "):
            seed = DRAW_SEEDS[int(variant.split()[-1]) - 1]
            first_, second = endpoint.draw(seed, pool.size, n_pert, controls.size, n_ctrl)
            cells_p, cells_c = pool[first_], controls[second]
            norm = variant.split()[1] if variant.startswith("shared ") else "all"
        elif variant == "cell-disjoint":
            level = DISJOINT_LEVEL if side == "VCC" else n_pert
            if pool.size < 2 * level:
                xs.append(None)
                zs.append(None)
                bs.append(None)
                keep.append(False)
                continue
            _, half_b = endpoint.half_split(pool.size, [builder.HALF_CELLS_SEED, k])
            if side == "VCC":
                if spec["controls"] == "pooled":
                    ctrl_b, n_c = builder.DATA["pooled_halves"][1], n_ctrl
                else:
                    ctrl_b, n_c = builder.DATA["batch_halves"][row.batch][1], builder.BATCH_HALF_CONTROLS
            else:
                ctrl_b, n_c = builder.DATA["screens"][side]["control_halves"][1], n_ctrl
            first_, second = endpoint.draw(DRAW_SEEDS[0], half_b.size, level, ctrl_b.size, n_c)
            cells_p, cells_c = pool[half_b][first_], ctrl_b[second]
            norm = "all"
        else:
            raise ValueError(variant)
        if side == "VCC":
            if norm != "all" and norm not in SCREENS:
                raise ValueError(norm)
            x, z = vcc_features(split, cells_p, cells_c, norm)
            vals = builder.DATA["vcc_values"]["all" if norm == "all" else f"shared {norm}"]
            t = builder.DATA["target_slot"][row.target_gene]
            own = endpoint.own_summaries(vals[f"{split}_ext"][cells_p], vals["nt_ext"][cells_c, t])
        else:
            x, z = screen_features(side, cells_p, cells_c, "all" if norm == "all" else "shared")
            vals = builder.DATA["screen_values"][(side, "all" if norm == "all" else "shared")]
            own = endpoint.own_summaries(vals["pert"][cells_p], vals["nt"][cells_c, builder.DATA["screens"][side]["construct_slot"][k]])
        xs.append(x.astype(np.float32))
        zs.append(z.astype(np.float32))
        bs.append(np.array([own[name] for name in ("B0", "B1", "B2", "B3", "B4")], dtype=np.float64))
        keep.append(True)
    return {"x": xs, "z": zs, "b": bs, "keep": keep}


def all_cell_work(side: str, start: int, stop: int) -> dict:
    """The builder checks (i) and (ii): features with every cell of the row or construct."""
    worst_x, worst_z = 0.0, 0.0
    if side == "VCC":
        rows = builder.DATA["cell_rows"]["G6"]
        stored = STORE["stored_vcc"]
        for k in range(start, stop):
            row = rows.iloc[k]
            x, z = vcc_features(row.split, row.positions, builder.DATA["batch_members"][row.batch], mode="source")
            key = (row.split, row.batch, row.target_gene)
            sx, sz = stored[key]
            worst_x = max(worst_x, float(np.abs(x - sx).max()))
            worst_z = max(worst_z, float(np.abs(z - sz).max()))
    else:
        arrays = builder.DATA["screens"][side]
        stored_x, stored_z = STORE["stored_external"][side]
        controls = np.arange(STORE["counts"]["screen_" + side]["nt_rows"].size)
        xs, zs = [], []
        for k in range(start, stop):
            x, z = screen_features(side, arrays["pools"][k], controls)
            worst_x = max(worst_x, float(np.abs(x - stored_x[k]).max()))
            worst_z = max(worst_z, float(np.abs(z - stored_z[k]).max()))
            xs.append(x.astype(np.float32))
            zs.append(z)
        # the cache-built all-cell external features (Section 15, items 1 and 2)
        return {"x": worst_x, "z": worst_z, "x_rows": xs, "z_rows": zs}
    return {"x": worst_x, "z": worst_z}


def run(function, args_list: list[tuple]) -> list:
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(function, *args) for args in args_list]
        return [f.result() for f in futures]


def chunks(n: int, size: int = 400) -> list[tuple[int, int]]:
    return [(s, min(s + size, n)) for s in range(0, n, size)]


# --------------------------------------------------------------------------- main


def stored_features() -> None:
    """The stored all-cell features the checks compare with: VCC shards and encoder rows, external x and z."""
    rows = e1inputs.load_vcc_rows()
    STORE["stored_vcc"] = {key: (rows.x[i].astype(np.float64), rows.z[i].astype(np.float64))
                           for i, key in enumerate(zip(rows.keys["split"], rows.keys["batch"], rows.keys["target_gene"]))}
    STORE["stored_external"] = {}
    for screen in SCREENS:
        table = builder.DATA["screens"][screen]["table"]
        inputs = e1inputs.load_screen_inputs(screen, table)
        STORE["stored_external"][screen] = (inputs.x["full_pass"].astype(np.float64), inputs.z.astype(np.float64))


def main() -> None:
    started = time.time()
    counts, manifest = builder.load_counts()
    STORE["counts"] = counts
    builder.prepare(counts)
    fc = load_feature_counts()
    STORE["features"] = fc["arrays"]
    for split in builder.SPLITS:
        if not np.array_equal(fc["arrays"]["vcc_" + split]["pert_rows"], counts["vcc_" + split]["pert_rows"]):
            raise RuntimeError("the E4 extraction and the label cache list different VCC %s cells" % split)
    for screen in SCREENS:
        for key in ("pert_rows", "nt_rows"):
            if not np.array_equal(fc["arrays"]["screen_" + screen][key], counts["screen_" + screen][key]):
                raise RuntimeError("the E4 extraction and the label cache list different %s cells" % screen)
    STORE["emb"] = embedding_rows()
    stored_features()
    report: dict = {"plan": PLAN, "inputs": {"label_cache": manifest["manifest"], "e4_extraction": fc["manifest"],
                                             "d2_cell_lists": file_record(REPO / "bioinf2027" / "results" / "D2" / "units.json")}}

    # check: every draw regenerates the stored label
    regeneration = {}
    for cell in CELLS:
        table = e1cells.load_tables(cell)
        n = len(builder.DATA["cell_rows"][cell])
        regenerated = np.array([v for part in run(label_work, [(cell, s, e) for s, e in chunks(n)]) for v in part])
        keys = builder.DATA["cell_rows"][cell][["split", "batch", "target_gene"]].astype(str)
        stored = table["vcc"].set_index(["split", "batch", "target_gene"])["y"].loc[list(map(tuple, keys.to_numpy()))].to_numpy()
        entry = {"vcc_rows": int(n), "vcc_max_abs_difference": float(np.nanmax(np.abs(regenerated - stored)))}
        for screen in SCREENS:
            m = len(builder.DATA["screens"][screen]["pools"])
            ext = np.array([v for part in run(external_label_work, [(cell, screen, s, e) for s, e in chunks(m)]) for v in part])
            entry[f"{screen}_max_abs_difference"] = float(np.nanmax(np.abs(ext - table[screen]["y"].to_numpy())))
        entry["passed"] = bool(max(v for k, v in entry.items() if k.endswith("difference")) <= LABEL_TOLERANCE)
        regeneration[cell] = entry
        print("[e4] labels of %s regenerated: %s" % (cell, entry), flush=True)
    report["label_regeneration"] = regeneration

    # checks (i) and (ii): all-cell features against the stored ones
    n = len(builder.DATA["cell_rows"]["G6"])
    parts = run(all_cell_work, [("VCC", s, e) for s, e in chunks(n)])
    builder_checks = {"VCC": {"x_max_abs_difference": max(p["x"] for p in parts), "z_max_abs_difference": max(p["z"] for p in parts)}}
    builder_checks["VCC"]["passed"] = bool(builder_checks["VCC"]["x_max_abs_difference"] <= FEATURE_TOLERANCE
                                           and builder_checks["VCC"]["z_max_abs_difference"] <= FEATURE_TOLERANCE)
    all_cell_external = {}
    for screen in SCREENS:
        m = len(builder.DATA["screens"][screen]["pools"])
        parts = run(all_cell_work, [(screen, s, e) for s, e in chunks(m)])
        built_z = np.stack([v for p in parts for v in p["z_rows"]])
        all_cell_external[f"{screen}_x"] = np.stack([v for p in parts for v in p["x_rows"]])
        all_cell_external[f"{screen}_z"] = built_z.astype(np.float32)
        all_cell_external[f"{screen}_constructs"] = text(builder.DATA["screens"][screen]["table"]["construct"].astype(str))
        # check (ii) for the external z as amended by Section 15, item 1: within the tolerance after the
        # shift shared by the screen's constructs, the shift recorded
        difference = built_z - STORE["stored_external"][screen][1]
        shift = np.median(difference, axis=0)
        after = float(np.abs(difference - shift).max())
        coordinate = int(np.argmax(np.abs(shift)))
        builder_checks[screen] = {"x_max_abs_difference": max(p["x"] for p in parts), "z_max_abs_difference": max(p["z"] for p in parts),
                                  "z_shared_shift": {"norm": float(np.linalg.norm(shift)), "largest_coordinate": float(shift[coordinate]),
                                                     "coordinate_index": coordinate},
                                  "z_max_abs_difference_after_the_shared_shift": after,
                                  "z_rule": "within the tolerance after the shift shared by the screen's constructs (E4 plan, Section 15, item 1)"}
        builder_checks[screen]["passed"] = bool(builder_checks[screen]["x_max_abs_difference"] <= FEATURE_TOLERANCE and after <= FEATURE_TOLERANCE)
    report["builder_checks"] = builder_checks
    report["first_run"] = ("bioinf2027/results/E4/features.json at 7e60a33: check (ii) of the external z failed before its amendment "
                           "(E4 plan, Sections 14 and 15; bioinf2027/diag/E4_external_z.json)")
    report["second_run"] = ("bioinf2027/results/E4/features.json at 167863f: every check passed, but the construct names were written as "
                            "object arrays, which the fits cannot load without pickle (run record bioinf2027/runs/E4.json, label "
                            "e4-fits-smoke); the same computation is rerun with the names as text")
    report["coverage"] = "every perturbed cell of a modelled VCC row and of an eligible construct and every control cell has a cached embedding (checked on load)"
    print("[e4] builder checks: %s" % builder_checks, flush=True)
    passed = all(v["passed"] for v in regeneration.values()) and all(v["passed"] for v in builder_checks.values())
    report["checks_passed"] = passed
    OUT.mkdir(parents=True, exist_ok=True)
    if not passed:
        (OUT / "features.json").write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
        print("[e4] a check failed; no feature is written", flush=True)
        raise SystemExit(1)

    # the features
    folder = features_root()
    folder.mkdir(parents=True, exist_ok=True)
    variants = [f"draw {k}" for k in (1, 2, 3)] + ["cell-disjoint"] + [f"shared {s} {k}" for s in SCREENS for k in (1, 2, 3)]
    written = {}
    for cell in CELLS:
        rows = builder.DATA["cell_rows"][cell]
        for variant in variants:
            sides = ["VCC"] + list(SCREENS)
            if variant.startswith("shared "):
                sides = ["VCC", variant.split()[1]]
            arrays = {}
            for side in sides:
                n = len(rows) if side == "VCC" else len(builder.DATA["screens"][side]["pools"])
                parts = run(feature_work, [(cell, side, s, e, variant) for s, e in chunks(n)])
                keep = np.array([k for p in parts for k in p["keep"]])
                x = np.zeros((n, 2000), dtype=np.float32)
                z = np.zeros((n, 768), dtype=np.float32)
                b = np.full((n, 5), np.nan)
                xs = [v for p in parts for v in p["x"]]
                zs = [v for p in parts for v in p["z"]]
                b_list = [v for p in parts for v in p["b"]]
                for i in np.flatnonzero(keep):
                    x[i], z[i], b[i] = xs[i], zs[i], b_list[i]
                arrays[f"{side}_x"], arrays[f"{side}_z"], arrays[f"{side}_b"], arrays[f"{side}_kept"] = x, z, b, keep
            arrays["VCC_keys"] = text(["|".join(k) for k in rows[["split", "batch", "target_gene"]].astype(str).to_numpy()])
            for screen in SCREENS:
                arrays[f"{screen}_constructs"] = text(builder.DATA["screens"][screen]["table"]["construct"].astype(str))
            name = f"{cell}_{variant.replace(' ', '_')}.npz"
            save_npz(folder / name, arrays)
            written[f"{cell} | {variant}"] = file_record(folder / name)
            print("[e4] %s %s written, %.0f s" % (cell, variant, time.time() - started), flush=True)
    save_npz(folder / "external_all-cell.npz", all_cell_external)
    written["external | all-cell"] = file_record(folder / "external_all-cell.npz")
    report["features"] = written
    report["draw_level_trivial_predictors"] = "B0 to B4 of each draw's cells (endpoint.own_summaries; B0 is log(1 + n_pert)), stored as <side>_b"
    report["variants"] = {"draw k": "the cells of label draw k, seeds %s" % list(DRAW_SEEDS),
                          "cell-disjoint": "draw 1 (seed 20260801) taken from half B of the R5 halves; VCC at 10 cells, externally at the cell's n_pert; rows below twice that are not kept",
                          "shared S k": "draw k with every cell normalised by its total over the genes VCC shares with screen S",
                          "external | all-cell": "every cell of each construct and every control cell, from the per-cell cache (Section 15, item 2)"}
    report["seconds"] = round(time.time() - started, 1)
    (OUT / "features.json").write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
    print("[e4] done in %.0f s" % (time.time() - started), flush=True)


# --------------------------------------------------------------------------- depth-thinned stage


HYPERGEOMETRIC_SEED = 20271106
DT_CELLS = ("G5", "G6")


def thinned_feature_counts(screen: str) -> dict:
    """Feature counts of every VCC cell thinned to the screen's depth, conditionally on the label pass.

    The label pass thinned each cell's needed columns (control cells: the 282 targets; perturbed cells:
    their own target) and the rest of the cell (seed 20270405; make_e1a_labels.thin_vcc). Here the
    thinned rest is distributed over the cell's other genes by one multivariate hypergeometric draw per
    cell, collapsed to the feature genes that are not needed columns and one category for everything
    else (one generator per screen, seed 20271106; control cells of the training file, then the
    perturbed cells of the training, validation and test files, each in file row order); feature genes
    that are needed columns keep the label pass's thinned counts (Section 10.2, item 9).
    """
    counts = STORE["counts"]
    fc = STORE["features"]
    depth = float(np.median(counts["screen_" + screen]["nt_shared_total"]))
    thinned = builder.thin_vcc(counts, screen, depth)
    symbols = features.hvg_symbols()
    targets = counts["vcc_train"]["targets"].astype(str)
    slot_of_target = {t: i for i, t in enumerate(targets.tolist())}
    feature_slot = np.array([slot_of_target.get(s, -1) for s in symbols.tolist()], dtype=np.int64)
    rng = np.random.default_rng(HYPERGEOMETRIC_SEED)
    out = {"depth": depth}

    def draw_cell(feature_counts: np.ndarray, needed: np.ndarray, needed_thinned: np.ndarray, total: float, needed_total: float,
                  thinned_total: float, thinned_needed_total: float) -> np.ndarray:
        other = np.flatnonzero(needed < 0)
        colors = np.rint(feature_counts[other]).astype(np.int64)
        rest = int(round(total - needed_total - colors.sum()))
        if rest < 0:
            raise RuntimeError("a cell's feature counts exceed its total")
        nsample = int(round(thinned_total - thinned_needed_total))
        drawn = rng.multivariate_hypergeometric(np.append(colors, rest), nsample)
        result = np.zeros(feature_counts.size, dtype=np.float32)
        result[other] = drawn[:-1]
        have = np.flatnonzero(needed >= 0)
        result[have] = needed_thinned[needed[have]]
        return result

    nt_counts, nt_total = thinned["nt"]
    original = counts["vcc_train"]
    fx = fc["vcc_train"]["nt_counts"]
    nt_out = np.zeros_like(fx)
    for i in range(fx.shape[0]):
        nt_out[i] = draw_cell(fx[i], feature_slot, nt_counts[i], float(original["nt_total"][i]), float(original["nt_counts"][i].sum()),
                              float(nt_total[i]), float(nt_counts[i].sum()))
    out["nt"] = (nt_out, nt_total)
    for split in builder.SPLITS:
        arrays = counts["vcc_" + split]
        own_counts, own_total = thinned[split]
        fx = fc["vcc_" + split]["pert_counts"]
        position = {s: j for j, s in enumerate(symbols.tolist())}
        own_feature = np.array([position.get(t, -1) for t in arrays["pert_target"].astype(str).tolist()], dtype=np.int64)
        pert_out = np.zeros_like(fx)
        for i in range(fx.shape[0]):
            needed = np.full(symbols.size, -1, dtype=np.int64)
            if own_feature[i] >= 0:
                needed[own_feature[i]] = 0
            pert_out[i] = draw_cell(fx[i], needed, np.array([own_counts[i]], dtype=np.float32), float(arrays["pert_total"][i]),
                                    float(arrays["pert_count"][i]), float(own_total[i]), float(own_counts[i]))
        out[split] = (pert_out, own_total)
    return out


def dt_main() -> None:
    started = time.time()
    counts, manifest = builder.load_counts()
    STORE["counts"] = counts
    builder.prepare(counts)
    fc = load_feature_counts()
    STORE["features"] = fc["arrays"]
    main_record = json.loads((OUT / "features.json").read_text())
    if not main_record.get("checks_passed"):
        raise SystemExit("the main builder checks have not passed; no thinned feature is built")
    folder = features_root()
    written, checks = {}, {}
    for screen in SCREENS:
        thinned = thinned_feature_counts(screen)
        nt_counts, nt_total = thinned["nt"]
        for base in DT_CELLS:
            cell = f"{base}-DT-{screen}"
            if not all(np.array_equal(a, b) for a, b in zip(builder.DATA["cell_rows"][cell]["positions"], builder.DATA["cell_rows"][base]["positions"])):
                raise RuntimeError("%s and %s draw from different pools" % (cell, base))
            spec = builder.CELLS[base]
            rows = builder.DATA["cell_rows"][base]
            for k in (1, 2, 3):
                x = np.zeros((len(rows), 2000), dtype=np.float32)
                for i, row in enumerate(rows.itertuples()):
                    controls = control_set(base, row.batch)
                    first_, second = endpoint.draw(DRAW_SEEDS[k - 1], row.positions.size, spec["n_pert"], controls.size, spec["n_ctrl"])
                    pert_counts, pert_total = thinned[row.split]
                    cp, cc = row.positions[first_], controls[second]
                    p = endpoint.external_values(pert_counts[cp], pert_total[cp]).astype(np.float64).mean(axis=0)
                    c = endpoint.external_values(nt_counts[cc], nt_total[cc]).astype(np.float64).mean(axis=0)
                    x[i] = (p - c).astype(np.float32)
                name = f"{cell}_draw_{k}.npz"
                save_npz(folder / name, {"VCC_x": x, "VCC_keys": text(["|".join(r) for r in rows[["split", "batch", "target_gene"]].astype(str).to_numpy()])})
                written[f"{cell} | draw {k}"] = file_record(folder / name)
                print("[e4 dt] %s draw %d written, %.0f s" % (cell, k, time.time() - started), flush=True)
        checks[screen] = {"depth_target": thinned["depth"], "pools_equal_to_the_unthinned_cells": True,
                          "thinned_totals": "the label pass's thinned totals, by construction"}
    record = {"plan": PLAN + "; depth-thinned stage (Sections 10.1, item 3b, and 10.2, item 9)", "features": written, "checks": checks,
              "inputs": {"label_cache": manifest["manifest"], "e4_extraction": fc["manifest"], "main_features": file_record(OUT / "features.json")},
              "seeds": {"label-pass thinning": builder.THIN_SEED, "hypergeometric": HYPERGEOMETRIC_SEED}, "seconds": round(time.time() - started, 1)}
    (OUT / "features_thinned.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e4 dt] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--thinned"]:
        dt_main()
    elif sys.argv[1:]:
        raise SystemExit("unknown arguments: %s" % sys.argv[1:])
    else:
        main()
