"""E1a, label stage: the rebuilt labels of every grid cell, with B0 to B4, the R5 halves and the R2
pool-size simulation.

Design: bioinf2027/plans/E1_design.md, Sections 7 to 9, 21, 23 and 24 (the grid of Section 24.2).
Inputs: the count cache of bioinf2027/scripts/make_e1a_label_extract.py, refused unless every array
matches its manifest, and checked to reproduce the values of the first E1a pass bit for bit. Every
cell's protocol is written, flushed and read back before any label value is computed. No prediction
head is fitted.

Per-row tables go under $IGA_STORAGE_ROOT/bioinf2027/endpoints/<cell>/ (they are large); the
repository holds each cell's protocol and a manifest with the tables' sizes and digests, under
bioinf2027/endpoints/<cell>/.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import json  # noqa: E402
import multiprocessing  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1a_extract as first  # noqa: E402
import make_e1a_label_extract as second  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E1_design.md, Sections 7 to 9, 21, 23 and 24"
SPLITS = first.SPLITS
SCREENS = first.SCREENS
WORKERS = 14
CHUNK = 150
THIN_SEED = 20270405
HALF_CELLS_SEED = 20270407
HALF_CONTROLS_SEED = 20270408
SIMULATION_SEED = 20270406
SIMULATED_SIZES = 5
BATCH_HALF_CONTROLS = 350

# The grid of Section 24.2. "controls" is the VCC scheme; externally the controls are always pooled.
CELLS = {
    "G1": {"n_pert": 27, "controls": "pooled", "n_ctrl": 788, "unit": "row", "norm": "all", "screens": SCREENS, "halves": (27, 10), "simulate": True},
    "G2": {"n_pert": 10, "controls": "pooled", "n_ctrl": 788, "unit": "row", "norm": "all", "screens": SCREENS, "halves": (10,), "simulate": True},
    "G3": {"n_pert": 27, "controls": "pooled", "n_ctrl": 788, "unit": "gene", "norm": "all", "screens": SCREENS, "halves": (), "simulate": False, "external_from": "G1"},
    "G4": {"n_pert": 10, "controls": "pooled", "n_ctrl": 788, "unit": "gene", "norm": "all", "screens": SCREENS, "halves": (), "simulate": False, "external_from": "G2"},
    "G5": {"n_pert": 27, "controls": "batch", "n_ctrl": 700, "unit": "row", "norm": "all", "screens": SCREENS, "halves": (27, 10), "simulate": True},
    "G6": {"n_pert": 10, "controls": "batch", "n_ctrl": 700, "unit": "row", "norm": "all", "screens": SCREENS, "halves": (10,), "simulate": True},
}
for _screen in SCREENS:
    CELLS[f"G1-SU-{_screen}"] = {"n_pert": 27, "controls": "pooled", "n_ctrl": 788, "unit": "row", "norm": "shared " + _screen, "screens": (_screen,), "halves": (), "simulate": False}
    for _base in ("G1", "G5", "G6"):
        _spec = CELLS[_base]
        CELLS[f"{_base}-DT-{_screen}"] = {
            "n_pert": _spec["n_pert"], "controls": _spec["controls"], "n_ctrl": _spec["n_ctrl"], "unit": "row",
            "norm": "thinned " + _screen, "screens": (_screen,), "halves": (), "simulate": False, "external_from": _base,
        }

DATA: dict = {}


# --------------------------------------------------------------------------- inputs


def endpoint_root() -> Path:
    return config.storage_root() / "bioinf2027" / "endpoints"


def load_counts() -> tuple[dict, dict]:
    folder = second.cache_dir()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    arrays = {}
    for name, record in manifest["cache"].items():
        path = folder / f"{name}.npz"
        if sha256(path) != record["sha256"]:
            raise RuntimeError("cache file %s does not match its manifest" % name)
        with np.load(path, allow_pickle=False) as store:
            arrays[name] = {key: store[key] for key in store.files}
    return arrays, {"manifest": file_record(manifest_path), "content": manifest}


def check_against_first_pass(counts: dict) -> dict:
    """The values recomputed from the counts equal the values the first E1a pass cached, bit for bit."""
    folder = first.cache_dir()
    manifest = json.loads((folder / "manifest.json").read_text())
    report = {"first_pass_manifest": file_record(folder / "manifest.json")}
    for name, record in manifest["cache"].items():
        path = folder / f"{name}.npz"
        if sha256(path) != record["sha256"]:
            raise RuntimeError("first-pass cache file %s does not match its manifest" % name)
    for split in SPLITS:
        with np.load(folder / f"vcc_{split}.npz") as old:
            new = counts["vcc_" + split]
            checks = {
                "pert_own_ext": np.array_equal(endpoint.external_values(new["pert_count"], new["pert_total"]), old["pert_own_ext"]),
                "pert_own_src": np.array_equal(endpoint.source_values(new["pert_count"], new["pert_total"]), old["pert_own_src"]),
                "pert_rows": np.array_equal(new["pert_rows"], old["pert_rows"]),
                "pert_totals": np.array_equal(new["pert_total"], old["all_totals"][old["pert_rows"]]),
            }
            if split == "train":
                checks["nt_values_ext"] = np.array_equal(endpoint.external_values(new["nt_counts"], new["nt_total"]), old["nt_values_ext"])
                checks["nt_values_src"] = np.array_equal(endpoint.source_values(new["nt_counts"], new["nt_total"]), old["nt_values_src"])
                for screen in SCREENS:
                    checks["nt_shared_total_" + screen] = np.array_equal(new["nt_shared_total_" + screen], old["nt_shared_totals_" + screen])
            report["vcc_" + split] = {k: bool(v) for k, v in checks.items()}
    for screen in SCREENS:
        with np.load(folder / f"screen_{screen}.npz") as old:
            new = counts["screen_" + screen]
            checks = {
                "nt_values": np.array_equal(endpoint.external_values(new["nt_counts"], new["nt_total"]), old["nt_values"]),
                "pert_own": np.array_equal(endpoint.external_values(new["pert_count"], new["pert_total"]), old["pert_own"]),
                "nt_shared_total": np.array_equal(new["nt_shared_total"], old["nt_shared_totals"]),
                "constructs": np.array_equal(new["constructs"], old["constructs"]),
            }
            report["screen_" + screen] = {k: bool(v) for k, v in checks.items()}
    report["passed"] = bool(all(all(v.values()) for k, v in report.items() if isinstance(v, dict) and "manifest" not in k and "path" not in v))
    if not report["passed"]:
        raise RuntimeError("the counts do not reproduce the first pass: %s" % report)
    return report


# --------------------------------------------------------------------------- value variants


def thin_vcc(counts: dict, screen: str, target: float) -> dict:
    """VCC counts thinned to the screen's depth (Section 24.3), one generator per screen."""
    rng = np.random.default_rng(THIN_SEED)
    train = counts["vcc_train"]
    p = np.minimum(1.0, target / np.maximum(train["nt_shared_total_" + screen].astype(np.float64), 1.0))
    rest = train["nt_total"].astype(np.float64) - train["nt_counts"].astype(np.float64).sum(axis=1)
    nt_counts, nt_total = endpoint.thin_counts(rng, train["nt_counts"], rest, p)
    out = {"nt": (nt_counts, nt_total), "p_controls_median": float(np.median(p))}
    for split in SPLITS:
        arrays = counts["vcc_" + split]
        p = np.minimum(1.0, target / np.maximum(arrays["pert_shared_total_" + screen].astype(np.float64), 1.0))
        rest = arrays["pert_total"].astype(np.float64) - arrays["pert_count"].astype(np.float64)
        out[split] = endpoint.thin_counts(rng, arrays["pert_count"], rest, p)
    return out


def vcc_values(counts: dict, norm: str, thinned: dict) -> dict:
    """Control and perturbed values of VCC in external and source mode, for one normalisation."""
    train = counts["vcc_train"]
    if norm == "all":
        nt = (train["nt_counts"], train["nt_total"])
        pert = {s: (counts["vcc_" + s]["pert_count"], counts["vcc_" + s]["pert_total"]) for s in SPLITS}
    elif norm.startswith("shared "):
        screen = norm.split(" ", 1)[1]
        nt = (train["nt_counts"], train["nt_shared_total_" + screen])
        pert = {s: (counts["vcc_" + s]["pert_count"], counts["vcc_" + s]["pert_shared_total_" + screen]) for s in SPLITS}
    else:
        screen = norm.split(" ", 1)[1]
        nt = thinned[screen]["nt"]
        pert = {s: thinned[screen][s] for s in SPLITS}
    return {
        "nt_ext": endpoint.external_values(*nt),
        "nt_src": endpoint.source_values(*nt),
        **{f"{s}_ext": endpoint.external_values(*pert[s]) for s in SPLITS},
        **{f"{s}_src": endpoint.source_values(*pert[s]) for s in SPLITS},
    }


def screen_values(counts: dict, screen: str, norm: str) -> dict:
    arrays = counts["screen_" + screen]
    total_nt = arrays["nt_shared_total"] if norm.startswith("shared") else arrays["nt_total"]
    total_pert = arrays["pert_shared_total"] if norm.startswith("shared") else arrays["pert_total"]
    return {
        "nt": endpoint.external_values(arrays["nt_counts"], total_nt),
        "pert": endpoint.external_values(arrays["pert_count"], total_pert),
    }


# --------------------------------------------------------------------------- rows


def vcc_row_table(counts: dict) -> pd.DataFrame:
    """Every modelled row with its pool positions, control members and existing label."""
    import glob

    frames = []
    for split in SPLITS:
        for path in sorted(glob.glob(str(first.shard_root() / split / "*.npz"))):
            store = np.load(path, allow_pickle=True)
            frames.append(pd.DataFrame({
                "split": split,
                "batch": str(np.asarray(store["batch_name"]).astype(str).ravel()[0]),
                "target_gene": np.asarray(store["sample_ids"]).astype(str),
                "existing_label": np.asarray(store["y"]).astype(np.float64),
            }))
    rows = pd.concat(frames, ignore_index=True).sort_values(["split", "batch", "target_gene"]).reset_index(drop=True)
    positions, guides, guide_ids = [], [], []
    for split in SPLITS:
        arrays = counts["vcc_" + split]
        keys = pd.DataFrame({"t": arrays["pert_target"], "b": arrays["pert_batch"], "g": arrays["pert_guide"]})
        keys["pos"] = np.arange(len(keys))
        grouped = {key: part for key, part in keys.groupby(["t", "b"], sort=False)}
        for row in rows[rows["split"] == split].itertuples():
            part = grouped[(row.target_gene, row.batch)]
            positions.append(np.sort(part["pos"].to_numpy()))
            guides.append(int(part["g"].nunique()))
            guide_ids.append(";".join(sorted(set(part["g"].astype(str)))))
    order = rows.index[rows["split"] == "train"].tolist() + rows.index[rows["split"] == "val"].tolist() + rows.index[rows["split"] == "test"].tolist()
    rows = rows.loc[order].reset_index(drop=True)
    rows["positions"] = positions
    rows["n_pool"] = [p.size for p in positions]
    rows["n_guides"] = guides
    rows["guides"] = guide_ids
    rows["flex_run"] = rows["batch"].str.split("_").str[1].astype(int)
    rows = rows.sort_values(["split", "batch", "target_gene"]).reset_index(drop=True)
    symbols = set(features.hvg_symbols().tolist())
    rows["target_is_feature"] = rows["target_gene"].isin(symbols)
    return rows


def gene_rows(rows: pd.DataFrame, n_pert: int) -> pd.DataFrame:
    """One row per (split, target gene), pooling every perturbed cell of the gene's modelled rows."""
    out = []
    for (split, gene), part in rows.groupby(["split", "target_gene"], sort=True):
        positions = np.sort(np.concatenate(part["positions"].tolist()))
        if positions.size < n_pert:
            continue
        weights = part["n_pool"].to_numpy(dtype=float)
        out.append({
            "split": split, "batch": "all modelled rows", "target_gene": gene, "flex_run": 0,
            "positions": positions, "n_pool": int(positions.size), "n_rows_pooled": int(len(part)),
            "n_guides": int(part["n_guides"].max()),
            "existing_label": float(np.sum(weights * part["existing_label"].to_numpy()) / weights.sum()),
            "target_is_feature": bool(part["target_is_feature"].iloc[0]),
        })
    return pd.DataFrame(out)


def cell_rows(rows: pd.DataFrame, spec: dict) -> pd.DataFrame:
    if spec["unit"] == "gene":
        return gene_rows(rows, spec["n_pert"])
    return rows[rows["n_pool"] >= spec["n_pert"]].reset_index(drop=True)


# --------------------------------------------------------------------------- protocols


def protocol(cell: str, spec: dict, inputs: dict) -> dict:
    return {
        "cell": cell,
        "plan": PLAN,
        "spec": {k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
        "statistic": "log1p(max(0, anderson_ksamp([a, b]).statistic)), scipy default midrank variant",
        "seeds": list(endpoint.SEEDS),
        "minimum_valid_draws": endpoint.MIN_VALID_DRAWS,
        "modes": "external mode for every label; on VCC also source mode from the same draws (Section 24.1, item 3): every primary statistic is repeated with the source-mode VCC labels against the unchanged external endpoints",
        "thinning": {"seed": THIN_SEED, "order": "controls of the training file (every needed column, rows then columns, then every rest), then the perturbed cells of the training, validation and test files (own counts, then rests)"} if spec["norm"].startswith("thinned") else None,
        "halves": {"cells_seed": HALF_CELLS_SEED, "controls_seed": HALF_CONTROLS_SEED, "levels": list(spec["halves"]), "batch_half_controls": BATCH_HALF_CONTROLS} if spec["halves"] else None,
        "simulation": {"seed": SIMULATION_SEED, "sizes_per_row": SIMULATED_SIZES} if spec["simulate"] else None,
        "inputs": inputs,
    }


def write_protocol(cell: str, content: dict) -> dict:
    folder = REPO / "bioinf2027" / "endpoints" / cell
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "protocol.json"
    text = json.dumps(strict_json(content), indent=1, allow_nan=False) + "\n"
    with open(path, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(path.read_text()) != json.loads(text):
        raise RuntimeError("protocol of %s did not read back unchanged" % cell)
    return file_record(path)


# --------------------------------------------------------------------------- work


def _label_fields(summary: dict, prefix: str = "", draws=None) -> dict:
    keys = ("y", "y_median_valid_draws", "y_mean", "y_sd", "y_iqr", "y_p5", "y_p95", "n_valid_draws")
    out = {prefix + k: summary[k] for k in keys}
    if draws is not None:
        # Medians of draws 1 to 12 and 13 to 25, for the split-half reliability over draws of R5.
        for name, part in (("y_draws_1_12", draws[:12]), ("y_draws_13_25", draws[12:])):
            part = part[np.isfinite(part)]
            out[prefix + name] = float(np.median(part)) if part.size else float("nan")
    return out


def _controls(spec: dict, values: dict, t: int, batch: str):
    if spec["controls"] == "pooled":
        index = DATA["pooled_index"]
    else:
        index = DATA["batch_members"][batch]
    return index, values["nt_ext"][index, t], values["nt_src"][index, t]


def vcc_work(cell: str, start: int, stop: int) -> list[dict]:
    spec = CELLS[cell]
    rows = DATA["cell_rows"][cell]
    values = DATA["vcc_values"][spec["norm"]]
    slot = DATA["target_slot"]
    sizes = rows["n_pool"].to_numpy()
    out = []
    for k in range(start, stop):
        row = rows.iloc[k]
        t = slot[row.target_gene]
        pos = row.positions
        pert_ext = values[f"{row.split}_ext"][pos]
        pert_src = values[f"{row.split}_src"][pos]
        index, ctrl_ext, ctrl_src = _controls(spec, values, t, row.batch)
        (ext, ext_draws), (src, _) = endpoint.subsampled_labels([pert_ext, pert_src], [ctrl_ext, ctrl_src], spec["n_pert"], spec["n_ctrl"])
        record = {"label": {**_label_fields(ext, draws=ext_draws), **_label_fields(src, "src_"), **endpoint.own_summaries(pert_ext, ctrl_ext)}, "halves": [], "simulation": []}
        b1_full = record["label"]["B1"]
        for level in spec["halves"]:
            if pos.size < 2 * level:
                continue
            first_half, second_half = endpoint.half_split(pos.size, [HALF_CELLS_SEED, k])
            versions = []
            if spec["controls"] == "pooled":
                versions.append(("pooled control halves", DATA["pooled_halves"], spec["n_ctrl"]))
            else:
                versions.append(("deciding: 350 controls from each batch half", DATA["batch_halves"][row.batch], BATCH_HALF_CONTROLS))
                whole = DATA["batch_members"][row.batch]
                versions.append(("upper bound: 700 controls of the whole batch", (whole, whole), spec["n_ctrl"]))
            for version, (ctrl_a, ctrl_b), n_ctrl in versions:
                for half, cells, ctrl in (("A", first_half, ctrl_a), ("B", second_half, ctrl_b)):
                    hv = pert_ext[cells]
                    hc = values["nt_ext"][ctrl, t]
                    summary, _ = endpoint.subsampled_label(hv, hc, level, n_ctrl)
                    record["halves"].append({
                        "level": level, "version": version, "half": half, "n_half": int(cells.size),
                        **_label_fields(summary), **endpoint.own_summaries(hv, hc),
                        "B2_against_full_controls": float(b1_full - hv.astype(np.float64).mean()),
                    })
        if spec["simulate"] and pos.size >= 2 * spec["n_pert"]:
            rng = np.random.default_rng([SIMULATION_SEED, k])
            candidates = sizes[(sizes >= spec["n_pert"]) & (sizes <= pos.size)]
            for j, size in enumerate(rng.choice(candidates, size=SIMULATED_SIZES, replace=True)):
                subset = np.sort(rng.choice(pos.size, size=int(size), replace=False))
                summary, _ = endpoint.subsampled_label(pert_ext[subset], ctrl_ext, spec["n_pert"], spec["n_ctrl"])
                record["simulation"].append({"simulation": j, "size": int(size), "y": summary["y"], "n_valid_draws": summary["n_valid_draws"], "y_mean": summary["y_mean"]})
        out.append(record)
    return out


def external_work(cell: str, screen: str, start: int, stop: int) -> list[dict]:
    spec = CELLS[cell]
    arrays = DATA["screens"][screen]
    values = DATA["screen_values"][(screen, "shared" if spec["norm"].startswith("shared") else "all")]
    out = []
    for k in range(start, stop):
        cells = arrays["pools"][k]
        pool = values["pert"][cells]
        controls = values["nt"][:, arrays["construct_slot"][k]]
        summary, draws = endpoint.subsampled_label(pool, controls, spec["n_pert"], spec["n_ctrl"])
        record = {"label": {**_label_fields(summary, draws=draws), **endpoint.own_summaries(pool, controls)}, "halves": []}
        level = spec["n_pert"]
        if spec["halves"] and cells.size >= 2 * level:
            first_half, second_half = endpoint.half_split(cells.size, [HALF_CELLS_SEED, k])
            ctrl_a, ctrl_b = arrays["control_halves"]
            for half, part, ctrl in (("A", first_half, ctrl_a), ("B", second_half, ctrl_b)):
                hv, hc = pool[part], controls[ctrl]
                half_summary, _ = endpoint.subsampled_label(hv, hc, level, spec["n_ctrl"])
                record["halves"].append({"level": level, "version": "pooled control halves", "half": half,
                                         "n_half": int(part.size), **_label_fields(half_summary), **endpoint.own_summaries(hv, hc)})
        out.append(record)
    return out


def run_parallel(function, args_list: list[tuple]) -> list:
    context = multiprocessing.get_context("fork")
    results = [None] * len(args_list)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = {pool.submit(function, *args): i for i, args in enumerate(args_list)}
        for future, i in futures.items():
            results[i] = future.result()
    return [record for chunk in results for record in chunk]


# --------------------------------------------------------------------------- tables


def write_table(frame: pd.DataFrame, folder: Path, name: str) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return file_record(path)


def vcc_tables(cell: str, records: list[dict]) -> dict[str, pd.DataFrame]:
    rows = DATA["cell_rows"][cell]
    keys = rows[[c for c in ("split", "batch", "target_gene", "flex_run", "n_pool", "n_guides", "guides", "n_rows_pooled", "existing_label", "target_is_feature") if c in rows]].reset_index(drop=True)
    labels = pd.concat([keys, pd.DataFrame([r["label"] for r in records])], axis=1)
    halves = [{**keys.iloc[i].to_dict(), **h} for i, r in enumerate(records) for h in r["halves"]]
    simulation = [{**keys.iloc[i][["split", "batch", "target_gene", "n_pool"]].to_dict(), **s} for i, r in enumerate(records) for s in r["simulation"]]
    tables = {"vcc_labels.csv": labels}
    if halves:
        tables["vcc_halves.csv"] = pd.DataFrame(halves)
    if simulation:
        tables["pool_size_simulation.csv"] = pd.DataFrame(simulation)
    return tables


def external_tables(cell: str, screen: str, records: list[dict]) -> dict[str, pd.DataFrame]:
    arrays = DATA["screens"][screen]
    keys = arrays["table"][["construct", "target_symbol", "target_ensg", "y_primary"]].rename(columns={"y_primary": "stored_y_primary"}).reset_index(drop=True)
    keys["n_pool"] = [p.size for p in arrays["pools"]]
    keys["target_is_feature"] = keys["target_symbol"].isin(set(features.hvg_symbols().tolist()))
    labels = pd.concat([keys, pd.DataFrame([r["label"] for r in records])], axis=1)
    halves = [{**keys.iloc[i][["construct", "target_ensg", "n_pool"]].to_dict(), **h} for i, r in enumerate(records) for h in r["halves"]]
    tables = {f"external_{screen}.csv": labels}
    if halves:
        tables[f"external_{screen}_halves.csv"] = pd.DataFrame(halves)
    return tables


# --------------------------------------------------------------------------- main


def prepare(counts: dict) -> dict:
    """Shared inputs of every worker: values, rows, control members and halves."""
    rows = vcc_row_table(counts)
    train = counts["vcc_train"]
    targets = train["targets"].tolist()
    DATA["target_slot"] = {t: i for i, t in enumerate(targets)}
    n_pool = train["nt_rows"].size
    DATA["pooled_index"] = np.arange(n_pool)
    batches = sorted(set(train["nt_batch"].tolist()))
    DATA["batch_members"] = {b: np.flatnonzero(train["nt_batch"] == b) for b in batches}
    a, b = endpoint.half_split(n_pool, HALF_CONTROLS_SEED)
    DATA["pooled_halves"] = (a, b)
    DATA["batch_halves"] = {}
    for index, batch in enumerate(batches):
        members = DATA["batch_members"][batch]
        ha, hb = endpoint.half_split(members.size, [HALF_CONTROLS_SEED, index])
        DATA["batch_halves"][batch] = (members[ha], members[hb])
    depth = {}
    for screen in SCREENS:
        depth[screen] = float(np.median(counts["screen_" + screen]["nt_shared_total"]))
    thinned = {screen: thin_vcc(counts, screen, depth[screen]) for screen in SCREENS}
    norms = sorted({spec["norm"] for spec in CELLS.values()})
    DATA["vcc_values"] = {norm: vcc_values(counts, norm, thinned) for norm in norms}
    DATA["cell_rows"] = {cell: cell_rows(rows, spec) for cell, spec in CELLS.items()}
    DATA["screens"], DATA["screen_values"] = {}, {}
    for screen in SCREENS:
        arrays = counts["screen_" + screen]
        table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        order = np.argsort(arrays["pert_code"], kind="stable")
        codes = arrays["pert_code"][order]
        bounds = np.searchsorted(codes, np.arange(arrays["constructs"].size + 1))
        pools = [order[bounds[i] : bounds[i + 1]] for i in range(arrays["constructs"].size)]
        by_construct = {c: i for i, c in enumerate(arrays["constructs"].tolist())}
        index = np.array([by_construct[c] for c in table["construct"].astype(str)])
        DATA["screens"][screen] = {
            "table": table,
            "pools": [pools[i] for i in index],
            "construct_slot": arrays["construct_slot"][index],
            "control_halves": endpoint.half_split(arrays["nt_rows"].size, HALF_CONTROLS_SEED),
        }
        for norm in ("all", "shared"):
            DATA["screen_values"][(screen, norm)] = screen_values(counts, screen, norm)
    return {
        "depth_targets": depth,
        "thinning_median_p_controls": {s: thinned[s]["p_controls_median"] for s in SCREENS},
        "thinned_control_total_medians": {s: float(np.median(thinned[s]["nt"][1])) for s in SCREENS},
        "rows": {cell: int(len(frame)) for cell, frame in DATA["cell_rows"].items()},
    }


def main() -> None:
    started = time.time()
    counts, manifest = load_counts()
    identity = check_against_first_pass(counts)
    print("[labels] counts reproduce the first pass bit for bit", flush=True)
    facts = prepare(counts)
    inputs = {"count_cache_manifest": manifest["manifest"], "first_pass_check": identity}
    protocols = {cell: write_protocol(cell, protocol(cell, spec, {**inputs, "rows": facts["rows"][cell]})) for cell, spec in CELLS.items()}
    print("[labels] %d protocols written and read back" % len(protocols), flush=True)

    root = endpoint_root()
    manifests = {}
    for cell, spec in CELLS.items():
        t0 = time.time()
        folder = root / cell
        records = {}
        n = len(DATA["cell_rows"][cell])
        tables = vcc_tables(cell, run_parallel(vcc_work, [(cell, s, min(s + CHUNK, n)) for s in range(0, n, CHUNK)]))
        source = spec.get("external_from")
        for screen in spec["screens"]:
            if source is not None:
                records[screen] = {"reused_from": source}
                continue
            m = len(DATA["screens"][screen]["pools"])
            tables.update(external_tables(cell, screen, run_parallel(external_work, [(cell, screen, s, min(s + CHUNK, m)) for s in range(0, m, CHUNK)])))
        written = {name: write_table(frame, folder, name) for name, frame in tables.items()}
        manifests[cell] = {"protocol": protocols[cell], "tables": written, "external_reused_from": source, "seconds": time.time() - t0}
        (REPO / "bioinf2027" / "endpoints" / cell / "manifest.json").write_text(json.dumps(strict_json(manifests[cell]), indent=1, allow_nan=False) + "\n")
        print(f"[labels] {cell}: {n} VCC rows, {time.time() - t0:.0f} s", flush=True)

    g1 = {s: pd.read_csv(root / "G1" / f"external_{s}.csv", float_precision="round_trip") for s in SCREENS}
    check_a = {s: float(np.nanmax(np.abs(g1[s]["y_median_valid_draws"] - g1[s]["stored_y_primary"]))) for s in SCREENS}
    summary = {
        "plan": PLAN,
        "paths_relative_to": "$" + config.STORAGE_VARIABLE + " for data-root files; the repository root otherwise",
        "facts": facts,
        "first_pass_check": identity,
        "g1_external_against_stored_endpoint_max_abs_difference": check_a,
        "cells": manifests,
        "seconds": time.time() - started,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    (REPO / "bioinf2027" / "endpoints" / "labels_summary.json").write_text(json.dumps(strict_json(summary), indent=1, allow_nan=False) + "\n")
    print("[labels] done in %.0f s; G1 external against the stored endpoint: %s" % (time.time() - started, check_a), flush=True)


if __name__ == "__main__":
    main()
