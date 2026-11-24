"""E1a, first part, compute: read-out R0 and the builder checks (a) to (e).

Design: bioinf2027/plans/E1_design.md, Sections 7, 9 and 21 to 23; Section 23.4
fixes R0 in full. By the author's decision at gate G1 (Section 22, items 2 and
3) no perturbed-cell label of any grid cell is built here. The builder checks
recompute the existing labels, which already exist: the 27-cell external
endpoints (check a) and the stored all-cells VCC label (checks b and c). R0 uses
control cells only.

Inputs: the per-cell cache written by bioinf2027/scripts/make_e1a_extract.py,
refused unless every array matches the digest its manifest records, and the
files named in that manifest. The protocol file is written, flushed and re-read
before the first value is computed. Every output goes to bioinf2027/endpoints/,
the provenance last.
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1a_extract as extract  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "bioinf2027" / "endpoints"
PLAN = "bioinf2027/plans/E1_design.md, Sections 7, 9 and 21 to 23 (Section 23.4 fixes R0)"

SPLITS = extract.SPLITS
SCREENS = extract.SCREENS
N_CTRL_POOLED = 788
N_CTRL_BATCH = 700
N_PERT_EXTERNAL = 27
SLOT_LEVELS = (10, 27)
TOLERANCE = {"a": 1e-9, "b": 2e-6, "c": 1e-4, "e_z": 1e-9, "e_nt": 1e-5}
MIN_GENES_PER_UNIT = 15
POOLED_BELOW = 0.10
BATCH_LOCAL_AT_LEAST = 0.25
DEPTH_FACTOR = 1.5
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20270301
QUANTILES = (0.25, 0.5, 0.75)


# --------------------------------------------------------------------------- inputs


def load_cache() -> tuple[dict, dict]:
    """Every cached array, refused unless its digest equals the one in the manifest."""
    folder = extract.cache_dir()
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


def stored_rows() -> pd.DataFrame:
    """Every modelled VCC row with its stored label and perturbed-cell count."""
    import glob

    frames = []
    for split in SPLITS:
        for path in sorted(glob.glob(str(extract.shard_root() / split / "*.npz"))):
            store = np.load(path, allow_pickle=True)
            label = np.asarray(store["y"])
            frames.append(
                pd.DataFrame(
                    {
                        "split": split,
                        "batch": str(np.asarray(store["batch_name"]).astype(str).ravel()[0]),
                        "target_gene": np.asarray(store["sample_ids"]).astype(str),
                        "label_stored": label.astype(np.float64),
                    }
                )
            )
    table = pd.concat(frames, ignore_index=True)
    counts = pd.read_parquet(extract.cell_count_file(), columns=["split", "target_gene", "batch", "n_perturbed_cells"])
    counts = counts.assign(split=counts["split"].map({v: k for k, v in SPLITS.items()}))
    table = table.merge(counts, on=["split", "target_gene", "batch"], how="left", validate="one_to_one")
    if table["n_perturbed_cells"].isna().any():
        raise RuntimeError("cell-count join left rows unmatched")
    if table.groupby("split").size().to_dict() != extract.SPLIT_ROWS:
        raise RuntimeError("modelled rows per split differ from the recorded counts")
    table["n_perturbed_cells"] = table["n_perturbed_cells"].astype(int)
    table["flex_run"] = table["batch"].str.split("_").str[1].astype(int)
    return table


# --------------------------------------------------------------------------- protocol


def protocol(manifest: dict) -> dict:
    return {
        "item": "E1a, first part: R0 and the builder checks (a) to (e)",
        "plan": PLAN,
        "no_perturbed_cell_label_of_any_grid_cell": True,
        "cache_manifest": manifest["manifest"],
        "script_sha256": sha256(Path(__file__)),
        "statistic": "log1p(max(0, anderson_ksamp([a, b]).statistic)), scipy default midrank variant",
        "seeds": list(endpoint.SEEDS),
        "minimum_valid_draws": endpoint.MIN_VALID_DRAWS,
        "tolerances": TOLERANCE,
        "check_a": "external mode, 27 perturbed against 788 pooled control cells, median of the valid draws before the 20-draw rule, against y_primary",
        "check_b": "source mode, all perturbed cells of a row against all control cells of its batch, against the stored label",
        "check_c": "external mode against source mode, same cells",
        "check_d": "control indices drawn after the perturbed draw, every pool size from n_pert + 1 to the largest pool of each side, n_pert 27 and 10",
        "check_e": "gf_mu_pert minus gf_mu_nt against gf_z_emb on every row and dimension; mean of each batch's cached control-cell embeddings against gf_mu_nt",
        "r0": {
            "slot_levels": list(SLOT_LEVELS),
            "controls_per_draw": N_CTRL_POOLED,
            "draw_order": "per seed one generator: slots first (the row's batch's control cells, ascending, or all pooled control cells, ascending), then 788 control cells from the pooled set without the slot cells, ascending",
            "rows": "10 slots: all modelled rows; 27 slots: rows with at least 27 perturbed cells",
            "units": "split by Flex run, at each slot level; units with fewer than %d target genes are descriptive" % MIN_GENES_PER_UNIT,
            "bootstrap": {"replicates": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "unit": "target gene, within each unit", "interval": "95 % percentile, 2.5th and 97.5th"},
            "component": "sqrt(max(0, SD^2 batch-own - SD^2 pooled-draw)) / scale; scales: SD of the existing VCC label on the unit's rows (recomputed per replicate), SD of the existing 27-cell label on RPE1 and on K562-essential (fixed); the scale giving the largest point component is kept in every replicate",
            "ten_slot_component": "understated (no 10-cell label exists); enters only the batch-local condition",
            "D": "Spearman(batch-own null, B1) - Spearman(pooled-draw null, B1); B1 = pooled control mean of the target gene, external mode",
            "abs_D_bounds": "upper = max(|lo|, |hi|); lower = 0 if the interval contains 0, else min(|lo|, |hi|)",
            "reading": {
                "favours the batch-local pair": "read first: any unit that enters, either level: lower bound of the component or of |D| at least %.2f" % BATCH_LOCAL_AT_LEAST,
                "favours the pooled pair": "the batch-local condition does not hold, and in every unit that enters: upper bound of |D| below %.2f at both levels and upper bound of the 27-slot component below %.2f" % (POOLED_BELOW, POOLED_BELOW),
                "favours neither": "otherwise",
            },
            "depth_trigger": "control cells, totals over the genes VCC shares with the screen (Ensembl identifiers), medians compared; G1-DT is built for a screen if the larger median exceeds the smaller by more than a factor of %.1f" % DEPTH_FACTOR,
        },
    }


def write_protocol(content: dict) -> dict:
    """Write the protocol, flush it to disk and read it back before any value is computed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "protocol_R0.json"
    text = json.dumps(strict_json(content), indent=1, allow_nan=False) + "\n"
    with open(path, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(path.read_text()) != json.loads(text):
        raise RuntimeError("the protocol file did not read back unchanged")
    return file_record(path)


# --------------------------------------------------------------------------- check (a)


def construct_pools(screen: dict) -> list[np.ndarray]:
    """Own values of each eligible construct's cells, in ascending cell order."""
    order = np.argsort(screen["pert_code"], kind="stable")
    codes = screen["pert_code"][order]
    values = screen["pert_own"][order]
    bounds = np.searchsorted(codes, np.arange(screen["constructs"].size + 1))
    return [values[bounds[i] : bounds[i + 1]] for i in range(screen["constructs"].size)]


def _gap(x: np.ndarray, y: np.ndarray, tolerance: float) -> dict:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    both_nan = np.isnan(x) & np.isnan(y)
    diff = np.where(both_nan, 0.0, np.abs(x - y))
    diff = np.where(np.isnan(diff), np.inf, diff)
    return {"max_abs_difference": float(diff.max()) if diff.size else 0.0,
            "rows_above_tolerance": int((diff > tolerance).sum())}


def check_a(name: str, screen: dict) -> tuple[dict, pd.DataFrame]:
    table = features.read_endpoint(config.screen_paths()[name]["endpoint"]).set_index("construct")
    pools = construct_pools(screen)
    n_nt = int(screen["nt_rows"].size)
    rows = []
    for ci, construct in enumerate(screen["constructs"].tolist()):
        controls = screen["nt_values"][:, screen["construct_slot"][ci]]
        summary, _ = endpoint.subsampled_label(pools[ci], controls, N_PERT_EXTERNAL, N_CTRL_POOLED)
        stored = table.loc[construct]
        rows.append(
            {
                "screen": name,
                "construct": construct,
                "pool": int(pools[ci].size),
                "stored_pool": int(stored["n_perturbed_pool"]),
                "stored_nt_pool": int(stored["n_nt_pool"]),
                "y": summary["y_median_valid_draws"],
                "stored_y": float(stored["y_primary"]),
                "n_valid": summary["n_valid_draws"],
                "stored_n_valid": int(stored["n_valid_draws"]),
                "y_mean": summary["y_mean"],
                "stored_y_mean": float(stored["y_mean"]),
                "y_pooled": endpoint.ad_log1p(pools[ci], controls),
                "stored_y_pooled": float(stored["y_pooled"]),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(table) or set(frame["construct"]) != set(table.index):
        raise RuntimeError("%s: eligible constructs and endpoint constructs differ" % name)
    primary = _gap(frame["y"], frame["stored_y"], TOLERANCE["a"])
    report = {
        "screen": name,
        "constructs": int(len(frame)),
        "control_cells": n_nt,
        "pools_equal_stored": bool((frame["pool"] == frame["stored_pool"]).all()),
        "control_pool_equals_stored": bool((frame["stored_nt_pool"] == n_nt).all()),
        "valid_draw_counts_equal_stored": bool((frame["n_valid"] == frame["stored_n_valid"]).all()),
        "y_primary": primary,
        "y_mean": _gap(frame["y_mean"], frame["stored_y_mean"], TOLERANCE["a"]),
        "y_pooled": _gap(frame["y_pooled"], frame["stored_y_pooled"], TOLERANCE["a"]),
        "tolerance": TOLERANCE["a"],
    }
    report["passed"] = bool(primary["rows_above_tolerance"] == 0 and report["pools_equal_stored"] and report["control_pool_equals_stored"])
    return report, frame


# --------------------------------------------------------------------------- checks (b) and (c)


def row_positions(split_arrays: dict, rows: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Positions, in a split's perturbed-cell arrays, of each stored row's cells, ascending."""
    keys = pd.DataFrame({"target_gene": split_arrays["pert_target"], "batch": split_arrays["pert_batch"]})
    keys["position"] = np.arange(len(keys))
    grouped = keys.groupby(["target_gene", "batch"], sort=False)["position"]
    lookup = {key: np.sort(values.to_numpy()) for key, values in grouped}
    return {(t, b): lookup[(t, b)] for t, b in zip(rows["target_gene"], rows["batch"])}


def check_bc(rows: pd.DataFrame, arrays: dict) -> tuple[dict, dict, pd.DataFrame]:
    train = arrays["vcc_train"]
    slot = {t: i for i, t in enumerate(train["targets"].tolist())}
    members = {b: np.flatnonzero(train["nt_batch"] == b) for b in np.unique(train["nt_batch"])}
    recorded = json.loads(extract.subsample_sizes_file().read_text())["nt_per_batch"]
    out = []
    for split in SPLITS:
        split_arrays = arrays["vcc_" + split]
        subset = rows[rows["split"] == split]
        positions = row_positions(split_arrays, subset)
        for row in subset.itertuples():
            pos = positions[(row.target_gene, row.batch)]
            t, batch = slot[row.target_gene], members[row.batch]
            out.append(
                {
                    "split": split,
                    "batch": row.batch,
                    "target_gene": row.target_gene,
                    "n_cells": int(pos.size),
                    "n_cells_recorded": int(row.n_perturbed_cells),
                    "n_controls": int(batch.size),
                    "n_controls_recorded": int(recorded[row.batch]),
                    "label_stored": float(row.label_stored),
                    "y_source_mode": endpoint.ad_log1p(split_arrays["pert_own_src"][pos], train["nt_values_src"][batch, t]),
                    "y_external_mode": endpoint.ad_log1p(split_arrays["pert_own_ext"][pos], train["nt_values_ext"][batch, t]),
                }
            )
    frame = pd.DataFrame(out)
    counts_ok = bool((frame["n_cells"] == frame["n_cells_recorded"]).all())
    controls_ok = bool((frame["n_controls"] == frame["n_controls_recorded"]).all())
    b = {"rows": int(len(frame)), **_gap(frame["y_source_mode"], frame["label_stored"], TOLERANCE["b"]),
         "tolerance": TOLERANCE["b"], "cell_counts_equal_recorded": counts_ok, "control_counts_equal_recorded": controls_ok}
    b["passed"] = bool(b["rows_above_tolerance"] == 0 and counts_ok and controls_ok)
    c = {"rows": int(len(frame)), **_gap(frame["y_external_mode"], frame["y_source_mode"], TOLERANCE["c"]),
         "rows_that_differ_at_all": int((frame["y_external_mode"] != frame["y_source_mode"]).sum()),
         "tolerance": TOLERANCE["c"]}
    c["passed"] = bool(c["rows_above_tolerance"] == 0)
    return b, c, frame


def shared_control_pool(arrays: dict) -> dict:
    """The control cells of the validation and test files are those of the training file."""
    train = arrays["vcc_train"]
    reference = dict(zip(train["nt_barcode"].tolist(), train["nt_batch"].tolist()))
    out = {"training_control_cells": int(train["nt_rows"].size)}
    for split in ("val", "test"):
        other = arrays["vcc_" + split]
        same = dict(zip(other["nt_barcode"].tolist(), other["nt_batch"].tolist())) == reference
        out[split] = {"control_cells": int(other["nt_rows"].size), "same_cells_and_batches": bool(same)}
    out["passed"] = bool(all(out[s]["same_cells_and_batches"] for s in ("val", "test")))
    return out


# --------------------------------------------------------------------------- check (d)


def check_d(rows: pd.DataFrame, arrays: dict) -> dict:
    train = arrays["vcc_train"]
    n_pooled = int(train["nt_rows"].size)
    batch_sizes = sorted({int((train["nt_batch"] == b).sum()) for b in np.unique(train["nt_batch"])})
    row_max = int(rows["n_perturbed_cells"].max())
    gene_max = int(rows.groupby(["split", "target_gene"])["n_perturbed_cells"].sum().max())
    configs = [("VCC, pooled controls", n_pooled, N_CTRL_POOLED, max(row_max, gene_max))]
    configs += [("VCC, batch of %d control cells" % s, s, N_CTRL_BATCH, row_max) for s in batch_sizes]
    exact = {"VCC rows": {str(k): int((rows["n_perturbed_cells"] == k).sum()) for k in SLOT_LEVELS}}
    for name in SCREENS:
        screen = arrays["screen_" + name]
        pools = np.bincount(screen["pert_code"], minlength=screen["constructs"].size)
        n_nt = int(screen["nt_rows"].size)
        configs += [(name, n_nt, N_CTRL_POOLED, int(pools.max())), (name + ", 700 controls", n_nt, N_CTRL_BATCH, int(pools.max()))]
        exact[name] = {str(k): int((pools == k).sum()) for k in SLOT_LEVELS}
    results = []
    for label, n_control, n_draw, largest in configs:
        for n_pert in SLOT_LEVELS:
            differ, exact_shares = 0, 0
            first = []
            for seed in endpoint.SEEDS:
                reference = endpoint.draw(seed, n_pert + 1, n_pert, n_control, n_draw)[1]
                for pool in range(n_pert + 2, largest + 1):
                    if not np.array_equal(endpoint.draw(seed, pool, n_pert, n_control, n_draw)[1], reference):
                        differ += 1
                        if len(first) < 10:
                            first.append([int(seed), pool])
                exact_shares += int(np.array_equal(endpoint.draw(seed, n_pert, n_pert, n_control, n_draw)[1], reference))
            results.append({"side": label, "control_pool": n_control, "controls_drawn": n_draw, "n_pert": n_pert,
                            "pool_sizes": [n_pert + 1, largest], "seed_pool_pairs_that_differ": differ,
                            "first_differences": first, "seeds_where_a_pool_of_exactly_n_pert_shares_the_draw": exact_shares})
    return {"configurations": results, "pools_of_exactly_n_pert": exact,
            "passed": bool(all(r["seed_pool_pairs_that_differ"] == 0 for r in results))}


# --------------------------------------------------------------------------- check (e)


EMBEDDING_DIMS = 768
ENCODER_SPLITS = {"training": "train", "validation": "val", "test": "test"}


def check_e(rows: pd.DataFrame, layout: dict) -> dict:
    """Arithmetic of the cached encoder rows and the per-batch control-cell embeddings.

    The row table holds, per (target gene, batch) row, the mean embedding of the
    perturbed cells, that of the batch's control cells and their difference z.
    The control-cell embeddings are stored one file per batch,
    nt_embeddings/nt__batch_<batch>.npy, as the cell index records
    (chunk_id "nt__batch_<batch>", role "NT").
    """
    folder = extract.encoder_dir()
    table = pd.read_parquet(folder / "geneformer_row_features_full.parquet")
    columns = {kind: [f"{kind}_{j}" for j in range(EMBEDDING_DIMS)] for kind in ("gf_mu_pert", "gf_mu_nt", "gf_z_emb")}
    pert = table[columns["gf_mu_pert"]].to_numpy(dtype=np.float64)
    control = table[columns["gf_mu_nt"]].to_numpy(dtype=np.float64)
    z = table[columns["gf_z_emb"]].to_numpy(dtype=np.float64)
    z_difference = float(np.abs(pert - control - z).max())

    keys = set(zip(table["split"].map(ENCODER_SPLITS), table["target_gene"].astype(str), table["batch"].astype(str)))
    stored = set(zip(rows["split"], rows["target_gene"], rows["batch"]))
    keys_match = bool(len(table) == len(rows) and keys == stored)

    index = pd.read_parquet(folder / "cell_embedding_index.parquet")
    index_controls = index[index["role"] == "NT"].groupby("source_batch").size()
    recorded = json.loads(extract.subsample_sizes_file().read_text())["nt_per_batch"]
    per_batch, worst = [], 0.0
    for batch, part in table.groupby("batch", sort=True):
        cells = np.load(folder / "nt_embeddings" / f"nt__batch_{batch}.npy")
        mean = cells.astype(np.float64).mean(axis=0)
        difference = float(np.abs(control[part.index.to_numpy()] - mean[None, :]).max())
        worst = max(worst, difference)
        per_batch.append(
            {
                "batch": str(batch),
                "control_cells_in_file": int(cells.shape[0]),
                "control_cells_in_index": int(index_controls.get(batch, 0)),
                "control_cells_recorded": int(recorded[str(batch)]),
                "rows": int(len(part)),
                "max_abs_difference": difference,
            }
        )
    counts_match = all(
        b["control_cells_in_file"] == b["control_cells_in_index"] == b["control_cells_recorded"] for b in per_batch
    )
    report = {
        "rows": int(len(table)),
        "keys_equal_the_stored_rows": keys_match,
        "z_equals_mu_pert_minus_mu_nt_max_abs_difference": z_difference,
        "batch_control_mean_max_abs_difference": worst,
        "batches": len(per_batch),
        "control_cell_counts_agree": bool(counts_match),
        "per_batch": per_batch,
        "tolerance_z": TOLERANCE["e_z"],
        "tolerance_control_mean": TOLERANCE["e_nt"],
        "layout": {k: layout[k] for k in ("cell_index_roles", "control_embedding_files")},
    }
    report["passed"] = bool(
        keys_match and counts_match and z_difference <= TOLERANCE["e_z"] and worst <= TOLERANCE["e_nt"]
    )
    return report


# --------------------------------------------------------------------------- R0


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(sps.rankdata(x), sps.rankdata(y))[0, 1])


def null_labels(rows: pd.DataFrame, train: dict) -> pd.DataFrame:
    """Batch-own and pooled-draw null labels of every row, at both slot levels."""
    values = train["nt_values_ext"]
    n_pool = values.shape[0]
    slot = {t: i for i, t in enumerate(train["targets"].tolist())}
    members = {b: np.flatnonzero(train["nt_batch"] == b) for b in np.unique(train["nt_batch"])}
    pooled_mean = values.astype(np.float64).mean(axis=0)
    out = []
    for level in SLOT_LEVELS:
        frame = rows if level == 10 else rows[rows["n_perturbed_cells"] >= 27]
        pooled_draws = endpoint.null_draws(n_pool, np.arange(n_pool), level, N_CTRL_POOLED)
        pooled_null = {t: endpoint.null_label(values[:, slot[t]], pooled_draws) for t in sorted(set(frame["target_gene"]))}
        for batch, part in frame.groupby("batch", sort=True):
            own_draws = endpoint.null_draws(n_pool, members[batch], level, N_CTRL_POOLED)
            for row in part.itertuples():
                t = slot[row.target_gene]
                own = endpoint.null_label(values[:, t], own_draws)
                pooled = pooled_null[row.target_gene]
                out.append(
                    {
                        "split": row.split,
                        "batch": batch,
                        "flex_run": int(row.flex_run),
                        "target_gene": row.target_gene,
                        "n_perturbed_cells": int(row.n_perturbed_cells),
                        "slots": level,
                        "batch_own_null": own["y"],
                        "batch_own_valid_draws": own["n_valid_draws"],
                        "batch_own_mean": own["y_mean"],
                        "pooled_draw_null": pooled["y"],
                        "pooled_draw_valid_draws": pooled["n_valid_draws"],
                        "pooled_draw_mean": pooled["y_mean"],
                        "B1_pooled_control_mean": float(pooled_mean[t]),
                        "batch_control_mean": float(values[members[batch], t].astype(np.float64).mean()),
                        "existing_label": float(row.label_stored),
                    }
                )
    return pd.DataFrame(out)


def external_scales() -> dict[str, float]:
    out = {}
    for name in SCREENS:
        table = features.read_endpoint(config.screen_paths()[name]["endpoint"])
        out[name] = float(np.std(table["y_primary"].dropna().to_numpy(dtype=float), ddof=1))
    return out


def _unit_point(bo, pd_, b1, label):
    sd_bo, sd_pd, sd_label = (float(np.std(v, ddof=1)) for v in (bo, pd_, label))
    raw = float(np.sqrt(max(0.0, sd_bo**2 - sd_pd**2)))
    return raw, sd_bo, sd_pd, sd_label, _spearman(bo, b1) - _spearman(pd_, b1)


def unit_statistics(part: pd.DataFrame, scales: dict[str, float], b: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    """Batch component and D of one unit, with target-gene cluster-bootstrap intervals."""
    keep = part["batch_own_null"].notna() & part["pooled_draw_null"].notna()
    part = part[keep]
    bo = part["batch_own_null"].to_numpy(float)
    pd_ = part["pooled_draw_null"].to_numpy(float)
    b1 = part["B1_pooled_control_mean"].to_numpy(float)
    label = part["existing_label"].to_numpy(float)
    genes, inverse = np.unique(part["target_gene"].to_numpy(), return_inverse=True)
    groups = [np.flatnonzero(inverse == g) for g in range(genes.size)]

    raw, sd_bo, sd_pd, sd_label, d = _unit_point(bo, pd_, b1, label)
    candidates = {"existing VCC label on the unit's rows": sd_label, **{f"27-cell label, {k}": v for k, v in scales.items()}}
    components = {k: (raw / v if v > 0 else float("nan")) for k, v in candidates.items()}
    chosen = max(components, key=lambda k: components[k] if np.isfinite(components[k]) else -np.inf)

    rng = np.random.default_rng(seed)
    comp_draws = np.full(b, np.nan)
    d_draws = np.full(b, np.nan)
    for r in range(b):
        index = np.concatenate([groups[g] for g in rng.integers(0, genes.size, size=genes.size)])
        r_raw, _, _, r_label, r_d = _unit_point(bo[index], pd_[index], b1[index], label[index])
        scale = r_label if chosen.startswith("existing") else candidates[chosen]
        comp_draws[r] = r_raw / scale if scale > 0 else np.nan
        d_draws[r] = r_d
    c_lo, c_hi = (float(v) for v in np.nanpercentile(comp_draws, [2.5, 97.5]))
    d_lo, d_hi = (float(v) for v in np.nanpercentile(d_draws, [2.5, 97.5]))
    return {
        "n_rows": int(len(part)),
        "n_target_genes": int(genes.size),
        "rows_without_both_nulls": int((~keep).sum()),
        "median_batch_own": float(np.median(bo)),
        "median_pooled_draw": float(np.median(pd_)),
        "sd_batch_own": sd_bo,
        "sd_pooled_draw": sd_pd,
        "sd_existing_label": sd_label,
        "spearman_batch_own_B1": _spearman(bo, b1),
        "spearman_pooled_draw_B1": _spearman(pd_, b1),
        "batch_component_raw": raw,
        **{"component_scale_" + k: v for k, v in components.items()},
        "component_scale_used": chosen,
        "component": components[chosen],
        "component_lo": c_lo,
        "component_hi": c_hi,
        "component_replicates_undefined": int(np.isnan(comp_draws).sum()),
        "D": d,
        "D_lo": d_lo,
        "D_hi": d_hi,
        "D_replicates_undefined": int(np.isnan(d_draws).sum()),
        "abs_D_upper": max(abs(d_lo), abs(d_hi)),
        "abs_D_lower": 0.0 if d_lo <= 0 <= d_hi else min(abs(d_lo), abs(d_hi)),
    }


def r0_units(nulls: pd.DataFrame, scales: dict[str, float], b: int = BOOTSTRAP_B) -> pd.DataFrame:
    out = []
    for (level, split, run), part in nulls.groupby(["slots", "split", "flex_run"], sort=True):
        entry = {"slots": int(level), "split": split, "flex_run": int(run), **unit_statistics(part, scales, b=b)}
        entry["enters_reading"] = bool(entry["n_target_genes"] >= MIN_GENES_PER_UNIT)
        out.append(entry)
    return pd.DataFrame(out)


def r0_reading(units: pd.DataFrame) -> dict:
    entering = units[units["enters_reading"]]
    at27 = entering[entering["slots"] == 27]
    # The batch-local condition is read first (bioinf2027/plans/E1_design.md, Section 23.4).
    batch_local = bool(((entering["component_lo"] >= BATCH_LOCAL_AT_LEAST) | (entering["abs_D_lower"] >= BATCH_LOCAL_AT_LEAST)).any())
    pooled = bool(
        not batch_local
        and len(entering) > 0
        and len(at27) > 0
        and (entering["abs_D_upper"] < POOLED_BELOW).all()
        and (at27["component_hi"] < POOLED_BELOW).all()
    )
    reading = "favours the batch-local pair" if batch_local else "favours the pooled pair" if pooled else "favours neither"
    return {
        "reading": reading,
        "units": int(len(units)),
        "units_entering": int(len(entering)),
        "units_not_entering": units.loc[~units["enters_reading"], ["slots", "split", "flex_run", "n_target_genes"]].to_dict("records"),
        "largest_upper_bound_of_abs_D": float(entering["abs_D_upper"].max()) if len(entering) else float("nan"),
        "largest_upper_bound_of_27_slot_component": float(at27["component_hi"].max()) if len(at27) else float("nan"),
        "largest_lower_bound_of_component": float(entering["component_lo"].max()) if len(entering) else float("nan"),
        "largest_lower_bound_of_abs_D": float(entering["abs_D_lower"].max()) if len(entering) else float("nan"),
        "consequence_under_section_22_item_3": "batch-local pair primary unless the reading favours the pooled pair; the author fixes the pair",
    }


def control_mean_spread(rows: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    """Per target gene: spread over batches of its control mean, against its own delta."""
    train = arrays["vcc_train"]
    values = train["nt_values_ext"].astype(np.float64)
    slot = {t: i for i, t in enumerate(train["targets"].tolist())}
    batches = np.unique(train["nt_batch"])
    by_batch = np.vstack([values[train["nt_batch"] == b].mean(axis=0) for b in batches])
    pooled = values.mean(axis=0)
    out = []
    for split in SPLITS:
        split_arrays = arrays["vcc_" + split]
        subset = rows[rows["split"] == split]
        modelled = set(zip(subset["target_gene"], subset["batch"]))
        in_rows = np.array([(t, b) in modelled for t, b in zip(split_arrays["pert_target"], split_arrays["pert_batch"])])
        for gene in sorted(set(subset["target_gene"])):
            t = slot[gene]
            cells = in_rows & (split_arrays["pert_target"] == gene)
            own = float(split_arrays["pert_own_ext"][cells].astype(np.float64).mean() - pooled[t])
            spread = float(np.std(by_batch[:, t], ddof=1))
            out.append({"split": split, "target_gene": gene, "pooled_control_mean": float(pooled[t]),
                        "sd_over_batches_of_control_mean": spread, "own_delta_gene_level": own,
                        "perturbed_cells": int(cells.sum()),
                        "abs_own_delta_over_batch_sd": abs(own) / spread if spread > 0 else float("nan")})
    return pd.DataFrame(out)


def _quartiles(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES} | {"n_cells": int(values.size)}


def depth_table(rows: pd.DataFrame, arrays: dict) -> tuple[pd.DataFrame, dict]:
    train = arrays["vcc_train"]
    out = [{"side": "VCC", "cells": "control", "genes": "all", **_quartiles(train["all_totals"][train["nt_rows"]])}]
    for split in SPLITS:
        split_arrays = arrays["vcc_" + split]
        subset = rows[rows["split"] == split]
        modelled = set(zip(subset["target_gene"], subset["batch"]))
        in_rows = np.array([(t, b) in modelled for t, b in zip(split_arrays["pert_target"], split_arrays["pert_batch"])])
        out.append({"side": "VCC " + split, "cells": "perturbed, modelled rows", "genes": "all",
                    **_quartiles(split_arrays["all_totals"][split_arrays["pert_rows"][in_rows]])})
    trigger = {}
    for name in SCREENS:
        screen = arrays["screen_" + name]
        out.append({"side": name, "cells": "control", "genes": "all", **_quartiles(screen["all_totals"][screen["nt_rows"]])})
        out.append({"side": name, "cells": "perturbed, eligible constructs", "genes": "all", **_quartiles(screen["all_totals"][screen["pert_rows"]])})
        vcc_shared = train["nt_shared_totals_" + name]
        ext_shared = screen["nt_shared_totals"]
        out.append({"side": "VCC", "cells": "control", "genes": "shared with " + name, **_quartiles(vcc_shared)})
        out.append({"side": name, "cells": "control", "genes": "shared with VCC", **_quartiles(ext_shared)})
        m_vcc, m_ext = float(np.median(vcc_shared)), float(np.median(ext_shared))
        ratio = max(m_vcc, m_ext) / min(m_vcc, m_ext)
        trigger[name] = {"vcc_median": m_vcc, "screen_median": m_ext, "ratio_larger_to_smaller": ratio,
                         "factor": DEPTH_FACTOR, "build_G1_DT": bool(ratio > DEPTH_FACTOR),
                         "deeper_side": "VCC" if m_vcc > m_ext else name}
    return pd.DataFrame(out), trigger


def guides_table(rows: pd.DataFrame, arrays: dict) -> pd.DataFrame:
    out = []
    for split in SPLITS:
        split_arrays = arrays["vcc_" + split]
        frame = pd.DataFrame({"target_gene": split_arrays["pert_target"], "batch": split_arrays["pert_batch"],
                              "guide": split_arrays["pert_guide"]})
        per_group = frame.groupby(["target_gene", "batch"])["guide"].nunique()
        subset = rows[rows["split"] == split]
        modelled = per_group.reindex(pd.MultiIndex.from_arrays([subset["target_gene"], subset["batch"]]))
        for scope, values in (("modelled rows", modelled.to_numpy()), ("all groups", per_group.to_numpy())):
            out.append({"split": split, "scope": scope, "groups": int(values.size),
                        "one_guide": int((values == 1).sum()), "two_guides": int((values == 2).sum()),
                        "more_than_two": int((values > 2).sum())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- main


def write_table(frame: pd.DataFrame, name: str) -> dict:
    path = OUT_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)
    return file_record(path)


def main() -> None:
    started = time.time()
    arrays, manifest = load_cache()
    protocol_record = write_protocol(protocol(manifest))
    print("[compute] protocol written and read back", flush=True)
    rows = stored_rows()

    checks = {"shared_control_pool": shared_control_pool(arrays)}
    a_frames = []
    for name in SCREENS:
        checks[f"a_{name}"], frame = check_a(name, arrays["screen_" + name])
        a_frames.append(frame)
        print(f"[compute] check (a) {name}: max difference {checks[f'a_{name}']['y_primary']['max_abs_difference']:.3g}", flush=True)
    checks["b"], checks["c"], bc_frame = check_bc(rows, arrays)
    print(f"[compute] checks (b) and (c): {checks['b']['max_abs_difference']:.3g}, {checks['c']['max_abs_difference']:.3g}", flush=True)
    checks["d"] = check_d(rows, arrays)
    print(f"[compute] check (d): passed {checks['d']['passed']}", flush=True)
    checks["e"] = check_e(rows, manifest["content"]["encoder_cache_layout"])
    print(f"[compute] check (e): passed {checks['e']['passed']}", flush=True)

    nulls = null_labels(rows, arrays["vcc_train"])
    scales = external_scales()
    units = r0_units(nulls, scales)
    reading = r0_reading(units)
    print(f"[compute] R0: {reading['reading']}", flush=True)
    spread = control_mean_spread(rows, arrays)
    depth, trigger = depth_table(rows, arrays)
    guides = guides_table(rows, arrays)

    outputs = {
        "builder_check_a_constructs.csv": write_table(pd.concat(a_frames, ignore_index=True), "builder_check_a_constructs.csv"),
        "builder_check_bc_rows.csv": write_table(bc_frame, "builder_check_bc_rows.csv"),
        "null_labels.csv": write_table(nulls, "null_labels.csv"),
        "R0_units.csv": write_table(units, "R0_units.csv"),
        "R0_control_mean_spread.csv": write_table(spread, "R0_control_mean_spread.csv"),
        "R0_depth.csv": write_table(depth, "R0_depth.csv"),
        "R0_guides.csv": write_table(guides, "R0_guides.csv"),
    }
    summary = {
        "checks": checks,
        "all_builder_checks_passed": bool(all(v.get("passed", False) for v in checks.values())),
        "r0_reading": reading,
        "external_scales": scales,
        "depth_trigger": trigger,
        "missing_null_rows": {
            str(level): {
                "batch_own": int(nulls.loc[nulls["slots"] == level, "batch_own_null"].isna().sum()),
                "pooled_draw": int(nulls.loc[nulls["slots"] == level, "pooled_draw_null"].isna().sum()),
            }
            for level in SLOT_LEVELS
        },
    }
    for name, payload in (("builder_checks.json", {k: summary[k] for k in ("checks", "all_builder_checks_passed")}),
                          ("R0_reading.json", {k: summary[k] for k in ("r0_reading", "external_scales", "depth_trigger", "missing_null_rows")})):
        path = OUT_DIR / name
        path.write_text(json.dumps(strict_json(payload), indent=1, allow_nan=False) + "\n")
        outputs[name] = file_record(path)
    provenance = {
        "plan": PLAN,
        "paths_relative_to": "$" + config.STORAGE_VARIABLE,
        "protocol": protocol_record,
        "cache_manifest": manifest["manifest"],
        "single_cell_inputs": manifest["content"]["single_cell_inputs"],
        "other_inputs": manifest["content"]["other_inputs"],
        "cache": manifest["content"]["cache"],
        "outputs": outputs,
        "seconds": time.time() - started,
        "peak_memory_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    (OUT_DIR / "R0_provenance.json").write_text(json.dumps(strict_json(provenance), indent=1, allow_nan=False) + "\n")
    print("[compute] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    main()
