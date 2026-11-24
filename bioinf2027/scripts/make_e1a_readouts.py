"""E1a, read-outs R1 to R9 of every grid cell, the R2 stop rule and the R5 gate.

Design: bioinf2027/plans/E1_design.md, Sections 9, 12, 21, 23 and 24 (Section 24.4 fixes the
read-out details). Inputs: the per-row tables of bioinf2027/scripts/make_e1a_labels.py, refused
unless each matches the digest in its cell's committed manifest, and R0's null labels. The R5
residualisation fits nuisance models on B0 to B4 only; no prediction head is fitted.

Outputs: bioinf2027/endpoints/readouts/*.csv and readouts.json, the provenance last.

With --smoke every stage runs with 20 bootstrap replicates and writes to
$IGA_STORAGE_ROOT/bioinf2027/scratch/E1a_readouts_smoke; it tests the code path on the real tables
and its values are never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, stats  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1a_labels as builder  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "endpoints" / "readouts"
PLAN = "bioinf2027/plans/E1_design.md, Sections 9, 12, 21, 23 and 24"
SPLITS = tuple(builder.SPLITS)
SCREENS = builder.SCREENS
B = stats.BOOTSTRAP_B
SEED = stats.BOOTSTRAP_SEED
FOLD_SEED = 20270409
WORKERS = 14
EXPECTATION = {27: (0.80, 0.97), 10: (0.65, 0.92), "gene": (0.85, 0.98)}
R4_REFERENCE = {"RPE1": 0.174, "K562-essential": 0.236}
R2_LIMIT = 0.3
GATE = 0.2
ROW_FLOOR = 1000
GENE_FLOOR = 60
TRIVIAL = ("B0", "B1", "B2", "B3", "B4")


# --------------------------------------------------------------------------- inputs


def load_tables() -> tuple[dict, dict]:
    """Every per-row table, checked against the digest its cell's manifest records."""
    tables, records = {}, {}
    for cell in builder.CELLS:
        manifest = json.loads((REPO / "bioinf2027" / "endpoints" / cell / "manifest.json").read_text())
        records[cell] = {"manifest": file_record(REPO / "bioinf2027" / "endpoints" / cell / "manifest.json")}
        tables[cell] = {}
        for name, record in manifest["tables"].items():
            path = builder.endpoint_root() / cell / name
            if sha256(path) != record["sha256"]:
                raise RuntimeError("%s/%s does not match its manifest" % (cell, name))
            tables[cell][name] = pd.read_csv(path, float_precision="round_trip")
        tables[cell]["_external_from"] = manifest.get("external_reused_from")
    return tables, records


def external_table(tables: dict, cell: str, screen: str) -> pd.DataFrame:
    source = tables[cell]["_external_from"] or cell
    return tables[source][f"external_{screen}.csv"]


# --------------------------------------------------------------------------- statistics


def corr_row(x, y, groups, **keys) -> dict:
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    keep = np.isfinite(x) & np.isfinite(y)
    estimate = stats.cluster_bootstrap_corr(x[keep], y[keep], groups[keep], b=B, seed=SEED)
    return {**keys, **stats.table_row(estimate), "descriptive": bool(estimate.n_groups < 50)}


def centred(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"v": values, "g": groups})
    return (frame["v"] - frame.groupby("g")["v"].transform("mean")).to_numpy()


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(sps.rankdata(x), sps.rankdata(y))[0, 1])


# --------------------------------------------------------------------------- tasks


def task_r1_r3_r6(cell: str, split: str) -> list[dict]:
    frame = TABLES[cell]["vcc_labels.csv"]
    part = frame[frame["split"] == split]
    genes = part["target_gene"].to_numpy()
    spec = builder.CELLS[cell]
    level = "gene" if spec["unit"] == "gene" else spec["n_pert"]
    out = []
    for label, column in (("median, external mode", "y"), ("median, source mode", "src_y"), ("mean across draws", "y_mean")):
        out.append(corr_row(part[column], part["existing_label"], genes, cell=cell, split=split, readout="R1" if column != "y_mean" else "R6, R1", label=label))
        for subset, mask in (("all rows", np.ones(len(part), bool)), ("target is a feature", part["target_is_feature"].to_numpy()), ("target is not a feature", ~part["target_is_feature"].to_numpy())):
            if mask.sum() < 3:
                continue
            row = corr_row(part[column].to_numpy()[mask], part["B2"].to_numpy()[mask], genes[mask], cell=cell, split=split,
                           readout="R3" if column != "y_mean" else "R6, R3", label=label, subset=subset)
            if subset == "all rows" and column == "y":
                lo, hi = EXPECTATION[level]
                row["expected_range"] = f"{lo:+.2f} to {hi:+.2f}"
                row["outside_expectation"] = bool(not (lo <= row["spearman"] <= hi))
            out.append(row)
    return out


def task_external(cell: str, screen: str) -> list[dict]:
    frame = external_table(TABLES, cell, screen)
    genes = frame["target_ensg"].to_numpy()
    out = [corr_row(frame["y"], frame["B2"], genes, cell=cell, screen=screen, readout="R3", label="median", subset="all constructs")]
    out.append(corr_row(frame["y"], np.log(frame["n_pool"]), genes, cell=cell, screen=screen, readout="R2", label="between constructs: label against log pool size", subset="all constructs"))
    out.append(corr_row(frame["y"], frame["stored_y_primary"], genes, cell=cell, screen=screen, readout="agreement with the stored 27/788 endpoint", label="median", subset="all constructs"))
    return out


def task_r2(cell: str) -> dict:
    """Between- and within-gene association with pool size, the mechanical part and the stop rule."""
    labels = TABLES[cell]["vcc_labels.csv"]
    sim = TABLES[cell]["pool_size_simulation.csv"]
    out = {"cell": cell, "between_gene": []}
    for split in SPLITS + ("all",):
        part = labels if split == "all" else labels[labels["split"] == split]
        out["between_gene"].append(corr_row(part["y"], part["n_pool"], part["target_gene"], cell=cell, split=split, label="rebuilt label against pool size"))
        out["between_gene"].append(corr_row(part["existing_label"], part["n_pool"], part["target_gene"], cell=cell, split=split, label="existing label against pool size, same rows"))
    single = labels[(labels["n_guides"] == 1) & labels["y"].notna()].copy()
    single["group"] = single["target_gene"] + "|" + single["guides"].astype(str)
    sizes = single.groupby("group")["y"].transform("size")
    single = single[sizes >= 3]
    single["c_y"] = centred(single["y"].to_numpy(), single["group"].to_numpy())
    single["c_n"] = centred(single["n_pool"].to_numpy(float), single["group"].to_numpy())
    sim = sim[sim["n_valid_draws"] >= 20].copy()
    sim["row"] = sim["split"] + "|" + sim["batch"] + "|" + sim["target_gene"]
    sim["c_y"] = centred(sim["y"].to_numpy(), sim["row"].to_numpy())
    sim["c_n"] = centred(sim["size"].to_numpy(float), sim["row"].to_numpy())

    def excess_for(obs_frame, sim_frame):
        obs = _spearman(obs_frame["c_n"].to_numpy(), obs_frame["c_y"].to_numpy())
        mech = _spearman(sim_frame["c_n"].to_numpy(), sim_frame["c_y"].to_numpy())
        return obs, mech

    per_split = {}
    for split in SPLITS + ("all",):
        o = single if split == "all" else single[single["split"] == split]
        s_ = sim if split == "all" else sim[sim["split"] == split]
        obs, mech = excess_for(o, s_)
        sign = 1.0 if obs >= 0 else -1.0
        genes = np.unique(np.r_[o["target_gene"].to_numpy(), s_["target_gene"].to_numpy()])
        members_o = {g: np.flatnonzero(o["target_gene"].to_numpy() == g) for g in genes}
        members_s = {g: np.flatnonzero(s_["target_gene"].to_numpy() == g) for g in genes}
        rng = np.random.default_rng(SEED)
        draws = np.full((B, 3), np.nan)
        on, oy, sn, sy = o["c_n"].to_numpy(), o["c_y"].to_numpy(), s_["c_n"].to_numpy(), s_["c_y"].to_numpy()
        for r in range(B):
            picked = genes[rng.integers(0, genes.size, size=genes.size)]
            io = np.concatenate([members_o[g] for g in picked])
            is_ = np.concatenate([members_s[g] for g in picked])
            ob = _spearman(on[io], oy[io]) if io.size else np.nan
            me = _spearman(sn[is_], sy[is_]) if is_.size else np.nan
            draws[r] = (ob, me, sign * (ob - me))
        lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
        per_split[split] = {
            "observed_within_gene_and_guide": obs, "observed_lo": lo[0], "observed_hi": hi[0],
            "mechanical": mech, "mechanical_lo": lo[1], "mechanical_hi": hi[1],
            "sign_of_observed": sign, "excess": sign * (obs - mech), "excess_lo": lo[2], "excess_hi": hi[2],
            "observed_rows": int(len(o)), "observed_groups": int(o["group"].nunique()) if len(o) else 0,
            "simulated_labels": int(len(s_)), "simulated_rows": int(s_["row"].nunique()) if len(s_) else 0,
            "target_genes": int(genes.size), "replicates_undefined": int(np.isnan(draws[:, 2]).sum()),
        }
    out["within_and_mechanical"] = per_split
    out["stop_rule_fires"] = bool(per_split["all"]["excess_lo"] > R2_LIMIT)
    out["stop_rule"] = "the lower bound of the 95 %% interval of the excess, all rows with splits pooled, above %.1f" % R2_LIMIT
    return out


def _rank01(values: np.ndarray) -> np.ndarray:
    return sps.rankdata(values) / (values.size + 1)


def residuals(frame: pd.DataFrame, columns, kind: str) -> np.ndarray:
    """Out-of-fold residuals of the rank-scaled label on the given columns, five folds grouped by gene."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression

    y = _rank01(frame["y"].to_numpy())
    x = frame[list(columns)].to_numpy(dtype=float)
    genes = np.array(sorted(set(frame["gene"])))
    fold_of = dict(zip(genes[np.random.default_rng(FOLD_SEED).permutation(genes.size)], np.arange(genes.size) % 5))
    folds = frame["gene"].map(fold_of).to_numpy()
    out = np.full(y.size, np.nan)
    for f in range(5):
        test = folds == f
        if not test.any() or test.all():
            continue
        if kind == "boosting":
            model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20,
                                                  early_stopping=False, random_state=0)
        else:
            model = LinearRegression()
        model.fit(x[~test], y[~test])
        out[test] = y[test] - model.predict(x[test])
    return out


def _icc1(values: np.ndarray, groups: np.ndarray) -> float:
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    counts = frame.groupby("g")["v"].transform("size")
    frame = frame[counts >= 2]
    if frame["g"].nunique() < 3:
        return float("nan")
    k = frame.groupby("g").size()
    means = frame.groupby("g")["v"].mean()
    grand = frame["v"].mean()
    msb = float((k * (means - grand) ** 2).sum() / (len(k) - 1))
    msw = float(((frame["v"] - frame["g"].map(means)) ** 2).sum() / (len(frame) - len(k)))
    k0 = (k.sum() - (k ** 2).sum() / k.sum()) / (len(k) - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def task_r5(cell: str, side: str) -> list[dict]:
    """Residual reliability of two cell-disjoint halves on VCC training rows or on one screen."""
    if side == "VCC":
        halves = TABLES[cell]["vcc_halves.csv"]
        halves = halves[halves["split"] == "train"].copy()
        halves["key"] = halves["batch"] + "|" + halves["target_gene"]
        halves["gene"] = halves["target_gene"]
    else:
        halves = TABLES[cell][f"external_{side}_halves.csv"].copy()
        halves["key"] = halves["construct"]
        halves["gene"] = halves["target_ensg"]
    out = []
    for (level, version), part in halves.groupby(["level", "version"], sort=True):
        wide = {h: g.set_index("key") for h, g in part.groupby("half")}
        if set(wide) != {"A", "B"}:
            continue
        keys = wide["A"].index.intersection(wide["B"].index)
        a, b = wide["A"].loc[keys], wide["B"].loc[keys]
        ok = a["y"].notna().to_numpy() & b["y"].notna().to_numpy()
        a, b = a[ok].reset_index(), b[ok].reset_index()
        row = {"cell": cell, "side": side, "level": int(level), "version": version, "rows": int(len(a)), "target_genes": int(a["gene"].nunique())}
        if len(a) < 50:
            out.append({**row, "note": "fewer than 50 rows with both half labels"})
            continue
        res_a, res_b = residuals(a, TRIVIAL, "boosting"), residuals(b, TRIVIAL, "boosting")
        row["residual_reliability"] = _spearman(res_a, res_b)
        row["gate_passed"] = bool(row["residual_reliability"] >= GATE)
        lin_a, lin_b = residuals(a, ("B0", "B1", "B2"), "linear"), residuals(b, ("B0", "B1", "B2"), "linear")
        row["residual_reliability_linear_B0_B2"] = _spearman(lin_a, lin_b)
        row["half_label_spearman"] = _spearman(a["y"].to_numpy(), b["y"].to_numpy())
        out.append(row)
    return out


def task_r5_extras(cell: str) -> dict:
    """ICC(1) of full-label residuals within groups, and the split-half reliability over draws."""
    labels = TABLES[cell]["vcc_labels.csv"]
    train = labels[(labels["split"] == "train") & labels["y"].notna()].copy()
    train["gene"] = train["target_gene"]
    res = residuals(train, TRIVIAL, "boosting")
    groups = (train["target_gene"] + "|" + train["guides"].astype(str)).to_numpy()
    r = _spearman(train["y_draws_1_12"].to_numpy(), train["y_draws_13_25"].to_numpy())
    out = {"cell": cell, "vcc_training_rows": int(len(train)),
           "icc1_residuals_within_gene_and_guide": _icc1(res, groups),
           "split_half_over_draws_spearman": r, "split_half_over_draws_spearman_brown": 2 * r / (1 + r) if np.isfinite(r) else float("nan")}
    for screen in builder.CELLS[cell]["screens"]:
        ext = external_table(TABLES, cell, screen).copy()
        ext = ext[ext["y"].notna()]
        ext["gene"] = ext["target_ensg"]
        res_e = residuals(ext, TRIVIAL, "boosting")
        out[f"icc1_residuals_between_constructs_of_a_gene_{screen}"] = _icc1(res_e, ext["target_ensg"].to_numpy())
        r = _spearman(ext["y_draws_1_12"].to_numpy(), ext["y_draws_13_25"].to_numpy())
        out[f"split_half_over_draws_{screen}"] = r
    return out


def task_r8(cell: str) -> list[dict]:
    halves = TABLES[cell]["vcc_halves.csv"]
    part = halves[(halves["level"] == 10) & halves["version"].str.startswith(("pooled", "deciding"))]
    wide = {h: g.set_index(["split", "batch", "target_gene"]) for h, g in part.groupby("half")}
    keys = wide["A"].index.intersection(wide["B"].index)
    a, b = wide["A"].loc[keys].reset_index(), wide["B"].loc[keys]
    out = []
    for split in SPLITS:
        mask = (a["split"] == split).to_numpy()
        out.append(corr_row(a["y"].to_numpy()[mask], b["B2_against_full_controls"].to_numpy()[mask], a["target_gene"].to_numpy()[mask],
                            cell=cell, split=split, readout="R8", label="half-A label against half-B negated own delta"))
    return out


TABLES: dict = {}


def run(function, args_list):
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        futures = [pool.submit(function, *args) for args in args_list]
        return [f.result() for f in futures]


# --------------------------------------------------------------------------- simple read-outs


def r4_r7_missing(tables: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r4, r7, missing = [], [], []
    for cell, spec in builder.CELLS.items():
        labels = tables[cell]["vcc_labels.csv"]
        scopes = [(f"VCC {s}", labels[labels["split"] == s]) for s in SPLITS]
        scopes += [(screen, external_table(tables, cell, screen)) for screen in spec["screens"]]
        for scope, part in scopes:
            sd = float(np.nanstd(part["y"], ddof=1))
            r4.append({"cell": cell, "scope": scope, "median_within_row_iqr": float(np.nanmedian(part["y_iqr"])),
                       "between_row_sd": sd, "ratio": float(np.nanmedian(part["y_iqr"]) / sd) if sd > 0 else np.nan,
                       "reference": R4_REFERENCE.get(scope, np.nan)})
            entry = {"cell": cell, "scope": scope, "rows": int(len(part))}
            for column in ("y",) + TRIVIAL + ("n_pool",):
                q = np.nanquantile(part[column].to_numpy(float), [0.25, 0.5, 0.75])
                entry.update({f"{column}_q25": q[0], f"{column}_median": q[1], f"{column}_q75": q[2]})
            r7.append(entry)
            gone = part["y"].isna()
            missing.append({"cell": cell, "scope": scope, "rows": int(len(part)), "missing_labels": int(gone.sum()),
                            "B1_median_missing": float(part.loc[gone, "B1"].median()) if gone.any() else np.nan,
                            "B1_median_present": float(part.loc[~gone, "B1"].median())})
        if spec["controls"] == "batch":
            counts = labels.groupby("batch")["batch"].transform("size")
            sizes = json.loads(builder.first.subsample_sizes_file().read_text())["nt_per_batch"]
            control = labels["batch"].map(sizes)
            bins = pd.qcut(control, 4, duplicates="drop")
            for interval, part in labels.groupby(bins, observed=True):
                sd = float(np.nanstd(part["y"], ddof=1))
                r4.append({"cell": cell, "scope": f"VCC, batch control cells {interval}", "median_within_row_iqr": float(np.nanmedian(part["y_iqr"])),
                           "between_row_sd": sd, "ratio": float(np.nanmedian(part["y_iqr"]) / sd) if sd > 0 else np.nan, "reference": np.nan})
            del counts
    return pd.DataFrame(r4), pd.DataFrame(r7), pd.DataFrame(missing)


def r9(tables: dict) -> pd.DataFrame:
    out = []
    for pair, ten in (("pooled pair", "G2"), ("batch-local pair", "G6")):
        labels = tables[ten]["vcc_labels.csv"]
        dropped = labels["n_pool"] < 27
        train = labels[labels["split"] == "train"]
        kept_genes = set(train.loc[train["n_pool"] >= 27, "target_gene"])
        lost = ~train["target_gene"].isin(kept_genes)
        for scope, frame, mask in (("rows of every split", labels, dropped.to_numpy()), ("training target genes", train, lost.to_numpy())):
            for group, m in (("dropped at 27", mask), ("retained", ~mask)):
                part = frame[m]
                entry = {"pair": pair, "cell": ten, "scope": scope, "group": group, "rows": int(len(part)), "target_genes": int(part["target_gene"].nunique())}
                for column in ("y", "B1", "B2"):
                    q = np.nanquantile(part[column].to_numpy(float), [0.25, 0.5, 0.75]) if len(part) else [np.nan] * 3
                    entry.update({f"{column}_q25": q[0], f"{column}_median": q[1], f"{column}_q75": q[2]})
                out.append(entry)
    return pd.DataFrame(out)


def batch_own_baseline(tables: dict) -> pd.DataFrame:
    nulls = pd.read_csv(REPO / "bioinf2027" / "endpoints" / "null_labels.csv", float_precision="round_trip")
    out = []
    for cell, level in (("G1", 27), ("G2", 10)):
        labels = tables[cell]["vcc_labels.csv"]
        merged = labels.merge(nulls[nulls["slots"] == level][["split", "batch", "target_gene", "batch_own_null"]],
                              on=["split", "batch", "target_gene"], how="left", validate="one_to_one")
        for split in SPLITS:
            part = merged[merged["split"] == split]
            out.append(corr_row(part["batch_own_null"], part["y"], part["target_gene"], cell=cell, split=split, level="rows"))
            counts = part.groupby("target_gene")["y"].transform("size")
            within = part[counts >= 3]
            out.append(corr_row(centred(within["batch_own_null"].to_numpy(), within["target_gene"].to_numpy()),
                                centred(within["y"].to_numpy(), within["target_gene"].to_numpy()),
                                within["target_gene"], cell=cell, split=split, level="within gene"))
    return pd.DataFrame(out)


def floors(tables: dict) -> pd.DataFrame:
    out = []
    for cell, spec in builder.CELLS.items():
        labels = tables[cell]["vcc_labels.csv"]
        train = labels[(labels["split"] == "train") & labels["y"].notna()]
        rows, genes = int(len(train)), int(train["target_gene"].nunique())
        gene_level = spec["unit"] == "gene"
        out.append({"cell": cell, "training_rows": rows, "training_target_genes": genes,
                    "floor": "60 target genes" if gene_level else "1,000 rows and 60 target genes",
                    "underpowered": bool(genes < GENE_FLOOR if gene_level else (rows < ROW_FLOOR or genes < GENE_FLOOR)),
                    "genes_whose_pool_mixes_two_guides": int((labels["n_guides"] > 1).sum()) if gene_level else np.nan})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- main


def write(frame: pd.DataFrame, name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frame.to_csv(path, index=False)
    return file_record(path)


def main() -> None:
    started = time.time()
    tables, inputs = load_tables()
    TABLES.update(tables)
    print("[readouts] %d cells loaded and verified" % len(tables), flush=True)
    outputs = {}
    floor = floors(tables)
    outputs["floors.csv"] = write(floor, "floors.csv")

    vcc_tasks = [(cell, split) for cell in builder.CELLS for split in SPLITS]
    rows = [r for chunk in run(task_r1_r3_r6, vcc_tasks) for r in chunk]
    ext_tasks = [(cell, screen) for cell, spec in builder.CELLS.items() for screen in spec["screens"]]
    rows += [r for chunk in run(task_external, ext_tasks) for r in chunk]
    outputs["R1_R3_R6.csv"] = write(pd.DataFrame(rows), "R1_R3_R6.csv")
    print("[readouts] R1, R3, R6 done", flush=True)

    pairs = ("G1", "G2", "G5", "G6")
    r2 = run(task_r2, [(cell,) for cell in pairs])
    between = pd.DataFrame([row for item in r2 for row in item["between_gene"]])
    outputs["R2_between.csv"] = write(between, "R2_between.csv")
    within = pd.DataFrame([{"cell": item["cell"], "split": split, **values, "stop_rule_fires": item["stop_rule_fires"] if split == "all" else np.nan}
                           for item in r2 for split, values in item["within_and_mechanical"].items()])
    outputs["R2_within_and_mechanical.csv"] = write(within, "R2_within_and_mechanical.csv")
    print("[readouts] R2 done", flush=True)

    r5_tasks = [(cell, side) for cell in pairs for side in ("VCC",) + SCREENS]
    r5 = pd.DataFrame([r for chunk in run(task_r5, r5_tasks) for r in chunk])
    outputs["R5_gate.csv"] = write(r5, "R5_gate.csv")
    extras = pd.DataFrame(run(task_r5_extras, [(cell,) for cell in pairs]))
    outputs["R5_descriptive.csv"] = write(extras, "R5_descriptive.csv")
    print("[readouts] R5 done", flush=True)

    r8 = pd.DataFrame([r for chunk in run(task_r8, [("G2",), ("G6",)]) for r in chunk])
    outputs["R8.csv"] = write(r8, "R8.csv")
    r4, r7, missing = r4_r7_missing(tables)
    outputs["R4.csv"] = write(r4, "R4.csv")
    outputs["R7.csv"] = write(r7, "R7.csv")
    outputs["missing_labels.csv"] = write(missing, "missing_labels.csv")
    outputs["R9.csv"] = write(r9(tables), "R9.csv")
    outputs["batch_own_null_baseline.csv"] = write(batch_own_baseline(tables), "batch_own_null_baseline.csv")

    deciding = r5[~r5["version"].astype(str).str.startswith("upper bound")]
    gate = {}
    for cell in pairs:
        for side in ("VCC",) + SCREENS:
            part = deciding[(deciding["cell"] == cell) & (deciding["side"] == side)]
            values = part["residual_reliability"].to_numpy(float) if "residual_reliability" in part else np.array([])
            gate[f"{cell} | {side}"] = {"versions": int(len(part)), "smallest_reliability": float(np.nanmin(values)) if values.size else np.nan,
                                        "passed": bool(values.size and np.all(values >= GATE))}
    summary = {
        "plan": PLAN,
        "inputs": inputs,
        "outputs": outputs,
        "r2_stop_rule": {item["cell"]: {"fires": item["stop_rule_fires"], **{k: item["within_and_mechanical"]["all"][k] for k in ("excess", "excess_lo", "excess_hi")}} for item in r2},
        "r5_gate": gate,
        "underpowered_cells": floor.loc[floor["underpowered"], "cell"].tolist(),
        "bootstrap": {"replicates": B, "seed": SEED, "unit": "target gene"},
        "seconds": time.time() - started,
    }
    (OUT / "readouts.json").write_text(json.dumps(strict_json(summary), indent=1, allow_nan=False) + "\n")
    print("[readouts] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        B = 20
        OUT = config.storage_root() / "bioinf2027" / "scratch" / "E1a_readouts_smoke"
        PLAN = PLAN + "; smoke run, 20 replicates, never reported"
    elif sys.argv[1:]:
        raise SystemExit("unknown arguments: %s" % sys.argv[1:])
    main()
