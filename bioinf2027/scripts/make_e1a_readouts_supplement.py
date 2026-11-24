"""E1a, read-out supplement: the planned read-outs the first read-out pass left out, the external
baseline check of Section 7, and descriptive analyses added after the first results review.

Design: bioinf2027/plans/E1_design.md, Sections 7, 9 and 24, and bioinf2027/reviews/E1a_results_r1.md
(non-blocking 11, items a to f, and 12; missing analyses 1 to 6).

Planned, left out of the first pass:
  (a) R2 between genes in G3, G4, G1-SU and the DT cells;
  (b) R6 for R2: the mean-across-draws label, between genes in every cell and within genes with its
      mechanical part in G1, G2, G5 and G6;
  (c) the within-(gene, guide) association on the rows that enter the simulation;
  (d) the external baseline check: the external B1 and B2 against the stored control means (read
      from their staged copies under the data root, bioinf2027/scripts/stage_external_control_means.py)
      and the target coordinate of the native delta matrices;
  (e) the external split-half over draws, Spearman-Brown corrected;
  (f) external R3 by feature availability.
Descriptive, added after the first results review; they decide nothing:
  1. the label against B1 alone, B3 and B2/B1, and the rank partial of label and B2 given B1, and
     given B1 and B0;
  2. the ceiling of every label under complete knockdown: each draw's AD statistic with the n_pert
     perturbed values set to zero against the same control draw, summarised like the label;
  3. common support: external constructs whose label, B1 or B2 lies outside the VCC training range;
  4. each base cell against its SU or DT version on the same rows;
  5. each R5 residual reliability with a target-gene cluster-bootstrap interval (residuals held
     fixed), its range over the fold seeds 20270410 to 20270413 and the nuisance model's
     out-of-fold R-squared per half;
  6. the share of each label's replicable rank variance that the flexible B0 to B4 model explains,
     1 - r_res (1 - R2) / r_label, with r_res the residual reliability, r_label the half-label
     Spearman and R2 the mean out-of-fold R-squared of the two halves;
  12. the number of gene-level rows (G3, G4) whose modelled rows carry more than one guide.

No prediction head is fitted. Every table is checked against its manifest before use. Outputs:
bioinf2027/endpoints/readouts/supplement/*.csv and supplement.json; per-row ceilings under
$IGA_STORAGE_ROOT/bioinf2027/endpoints/supplement/. With --smoke every stage runs with 20 bootstrap
replicates and writes to $IGA_STORAGE_ROOT/bioinf2027/scratch/E1a_supplement_smoke; its values are
never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, endpoint, features, stats  # noqa: E402
from bioinf2027.provenance import file_record, strict_json  # noqa: E402

import make_e1a_readouts as readouts  # noqa: E402

builder = readouts.builder
REPO = readouts.REPO
OUT = REPO / "bioinf2027" / "endpoints" / "readouts" / "supplement"
ROWS_OUT = config.storage_root() / "bioinf2027" / "endpoints" / "supplement"
PLAN = ("bioinf2027/plans/E1_design.md, Sections 7, 9 and 24; bioinf2027/reviews/E1a_results_r1.md, "
        "non-blocking 11 and 12 and missing analyses 1 to 6")
PAIRS = ("G1", "G2", "G5", "G6")
OTHER_CELLS = tuple(cell for cell in builder.CELLS if cell not in PAIRS)
DERIVED = {cell: spec.get("external_from") or "G1" for cell, spec in builder.CELLS.items() if "-SU-" in cell or "-DT-" in cell}
FOLD_SEEDS = (20270409, 20270410, 20270411, 20270412, 20270413)
NEAR_CEILING = 0.1
STAGED = config.storage_root() / "bioinf2027" / "cache" / "external_control_means"
STORED_CONTROL_MEANS = {"RPE1": STAGED / "mu_nt_rpe1.npy", "K562-essential": STAGED / "mu_nt_k562_essential.npy"}
STAGING_RECORD = REPO / "bioinf2027" / "endpoints" / "external_control_means.json"
TABLES = readouts.TABLES


# --------------------------------------------------------------------------- statistics


def r2_within(cell: str, column: str) -> dict:
    """Task R2's within-(gene, guide), mechanical and excess statistics for one label column."""
    labels = TABLES[cell]["vcc_labels.csv"]
    sim = TABLES[cell]["pool_size_simulation.csv"]
    single = labels[(labels["n_guides"] == 1) & labels[column].notna()].copy()
    single["group"] = single["target_gene"] + "|" + single["guides"].astype(str)
    sizes = single.groupby("group")[column].transform("size")
    single = single[sizes >= 3]
    single["c_y"] = readouts.centred(single[column].to_numpy(), single["group"].to_numpy())
    single["c_n"] = readouts.centred(single["n_pool"].to_numpy(float), single["group"].to_numpy())
    sim = sim[(sim["n_valid_draws"] >= 20) & sim[column].notna()].copy()
    sim["row"] = sim["split"] + "|" + sim["batch"] + "|" + sim["target_gene"]
    sim["c_y"] = readouts.centred(sim[column].to_numpy(), sim["row"].to_numpy())
    sim["c_n"] = readouts.centred(sim["size"].to_numpy(float), sim["row"].to_numpy())
    per_split = {}
    for split in readouts.SPLITS + ("all",):
        o = single if split == "all" else single[single["split"] == split]
        s_ = sim if split == "all" else sim[sim["split"] == split]
        obs = readouts._spearman(o["c_n"].to_numpy(), o["c_y"].to_numpy())
        mech = readouts._spearman(s_["c_n"].to_numpy(), s_["c_y"].to_numpy())
        sign = 1.0 if obs >= 0 else -1.0
        genes = np.unique(np.r_[o["target_gene"].to_numpy(), s_["target_gene"].to_numpy()])
        members_o = {g: np.flatnonzero(o["target_gene"].to_numpy() == g) for g in genes}
        members_s = {g: np.flatnonzero(s_["target_gene"].to_numpy() == g) for g in genes}
        rng = np.random.default_rng(readouts.SEED)
        draws = np.full((readouts.B, 3), np.nan)
        on, oy, sn, sy = o["c_n"].to_numpy(), o["c_y"].to_numpy(), s_["c_n"].to_numpy(), s_["c_y"].to_numpy()
        for r in range(readouts.B):
            picked = genes[rng.integers(0, genes.size, size=genes.size)]
            io = np.concatenate([members_o[g] for g in picked])
            is_ = np.concatenate([members_s[g] for g in picked])
            ob = readouts._spearman(on[io], oy[io]) if io.size else np.nan
            me = readouts._spearman(sn[is_], sy[is_]) if is_.size else np.nan
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
    return per_split


def residuals_with_seed(frame: pd.DataFrame, columns, seed: int) -> tuple[np.ndarray, float]:
    """The deciding residualisation of R5 with a given fold seed; returns the out-of-fold
    residuals of the rank-scaled label and the out-of-fold R-squared."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    y = readouts._rank01(frame["y"].to_numpy())
    x = frame[list(columns)].to_numpy(dtype=float)
    genes = np.array(sorted(set(frame["gene"])))
    fold_of = dict(zip(genes[np.random.default_rng(seed).permutation(genes.size)], np.arange(genes.size) % 5))
    folds = frame["gene"].map(fold_of).to_numpy()
    out = np.full(y.size, np.nan)
    for f in range(5):
        test = folds == f
        if not test.any() or test.all():
            continue
        model = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20,
                                              early_stopping=False, random_state=0)
        model.fit(x[~test], y[~test])
        out[test] = y[test] - model.predict(x[test])
    keep = np.isfinite(out)
    r2 = float(1.0 - np.sum(out[keep] ** 2) / np.sum((y[keep] - y[keep].mean()) ** 2))
    return out, r2


def rank_partial(frame: pd.DataFrame, x: str, y: str, given: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Rank-scaled x and y, each with its least-squares fit on the rank-scaled covariates removed."""
    def ranks(values):
        return readouts._rank01(np.asarray(values, float))
    design = np.column_stack([np.ones(len(frame))] + [ranks(frame[c]) for c in given])
    out = []
    for column in (x, y):
        v = ranks(frame[column])
        beta, *_ = np.linalg.lstsq(design, v, rcond=None)
        out.append(v - design @ beta)
    return out[0], out[1]


def replicable_share(r_res: float, r_label: float, r2_mean: float) -> float:
    """Share of the replicable rank variance of a label that the trivial-predictor model explains."""
    if not (np.isfinite(r_res) and np.isfinite(r_label) and np.isfinite(r2_mean)) or r_label <= 0:
        return float("nan")
    return float(1.0 - r_res * (1.0 - r2_mean) / r_label)


def guide_union_counts(rows: pd.DataFrame, genes: set) -> int:
    """Genes whose modelled rows, taken together, carry more than one guide."""
    union = {}
    for row in rows.itertuples():
        if row.target_gene in genes:
            union.setdefault((row.split, row.target_gene), set()).update(str(row.guides).split(";"))
    return int(sum(len(v) > 1 for v in union.values()))


# --------------------------------------------------------------------------- tasks


def task_between(cell: str) -> list[dict]:
    labels = TABLES[cell]["vcc_labels.csv"]
    columns = [("mean-across-draws label against pool size", "y_mean", "b")]
    if cell in OTHER_CELLS:
        columns = [("rebuilt label against pool size", "y", "a"),
                   ("existing label against pool size, same rows", "existing_label", "a")] + columns
    out = []
    for split in readouts.SPLITS + ("all",):
        part = labels if split == "all" else labels[labels["split"] == split]
        for label, column, item in columns:
            out.append(readouts.corr_row(part[column], part["n_pool"], part["target_gene"], cell=cell, split=split, item=item, label=label))
    return out


def task_within_mean(cell: str) -> list[dict]:
    return [{"cell": cell, "split": split, "item": "b", "label": "mean across draws", **values,
             "stop_rule_reading": "reported only; the rule is read on the median label"}
            for split, values in r2_within(cell, "y_mean").items()]


def task_within_simulated_rows(cell: str) -> list[dict]:
    labels = TABLES[cell]["vcc_labels.csv"]
    entering = labels["n_pool"] >= 2 * builder.CELLS[cell]["n_pert"]
    single = labels[entering & (labels["n_guides"] == 1) & labels["y"].notna()].copy()
    single["group"] = single["target_gene"] + "|" + single["guides"].astype(str)
    single = single[single.groupby("group")["y"].transform("size") >= 3]
    single["c_y"] = readouts.centred(single["y"].to_numpy(), single["group"].to_numpy())
    single["c_n"] = readouts.centred(single["n_pool"].to_numpy(float), single["group"].to_numpy())
    out = []
    for split in readouts.SPLITS + ("all",):
        part = single if split == "all" else single[single["split"] == split]
        out.append({**readouts.corr_row(part["c_n"], part["c_y"], part["target_gene"], cell=cell, split=split, item="c",
                                        label="within (gene, guide), rows that enter the simulation"),
                    "groups": int(part["group"].nunique())})
    return out


def task_descriptive_one(cell: str, scope: str) -> list[dict]:
    if scope in readouts.SPLITS:
        labels = TABLES[cell]["vcc_labels.csv"]
        part = labels[(labels["split"] == scope) & labels["y"].notna()].copy()
        genes = part["target_gene"].to_numpy()
    else:
        part = readouts.external_table(TABLES, cell, scope)
        part = part[part["y"].notna()].copy()
        genes = part["target_ensg"].to_numpy()
    part["B2_over_B1"] = np.where(part["B1"] > 0, part["B2"] / part["B1"].where(part["B1"] > 0), np.nan)
    keys = {"cell": cell, "scope": scope, "item": "1"}
    out = [readouts.corr_row(part["y"], part[c], genes, **keys, label=f"label against {name}")
           for c, name in (("B1", "B1"), ("B3", "B3"), ("B2_over_B1", "B2 / B1"))]
    for given in (("B1",), ("B1", "B0")):
        rx, ry = rank_partial(part, "y", "B2", given)
        row = readouts.corr_row(rx, ry, genes, **keys, label="rank partial of label and B2 given " + " and ".join(given))
        row["partial_is_the_pearson_of_rank_residuals"] = True
        out.append(row)
    return out


def task_r5_detail(cell: str, side: str) -> list[dict]:
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
        if len(a) < 50:
            continue
        reliabilities, r2 = {}, {}
        for seed in FOLD_SEEDS:
            res_a, r2_a = residuals_with_seed(a, readouts.TRIVIAL, seed)
            res_b, r2_b = residuals_with_seed(b, readouts.TRIVIAL, seed)
            reliabilities[seed] = readouts._spearman(res_a, res_b)
            if seed == FOLD_SEEDS[0]:
                deciding = (res_a, res_b)
                r2 = {"A": r2_a, "B": r2_b}
        estimate = stats.cluster_bootstrap_corr(deciding[0], deciding[1], a["gene"].to_numpy(), b=readouts.B, seed=readouts.SEED)
        row = stats.table_row(estimate)
        label_r = readouts._spearman(a["y"].to_numpy(), b["y"].to_numpy())
        others = [reliabilities[s] for s in FOLD_SEEDS[1:]]
        out.append({
            "cell": cell, "side": side, "level": int(level), "version": version, "item": "5 and 6",
            "rows": int(len(a)), "target_genes": int(a["gene"].nunique()),
            "residual_reliability": reliabilities[FOLD_SEEDS[0]],
            "interval_lo": row["spearman_lo"], "interval_hi": row["spearman_hi"],
            "fold_seed_min": float(np.min(others)), "fold_seed_max": float(np.max(others)),
            "oof_r2_half_A": r2["A"], "oof_r2_half_B": r2["B"], "half_label_spearman": label_r,
            "replicable_share_explained": replicable_share(reliabilities[FOLD_SEEDS[0]], label_r, 0.5 * (r2["A"] + r2["B"])),
        })
    return out


# --------------------------------------------------------------------------- ceilings


def ceiling_of(pool_size: int, controls: np.ndarray, n_pert: int, n_ctrl: int) -> float:
    """The label of a pool whose perturbed values are all zero. The draws depend only on the pool
    size and the seeds, so each draw uses the same control cells as the row's own label."""
    summary, _ = endpoint.subsampled_label(np.zeros(pool_size, dtype=np.float32), controls, n_pert, n_ctrl)
    return float(summary["y"])


def vcc_ceiling(cell: str, start: int, stop: int) -> list[float]:
    """Each row's label with its perturbed values set to zero, against the same control draws."""
    spec = builder.CELLS[cell]
    rows = builder.DATA["cell_rows"][cell]
    values = builder.DATA["vcc_values"][spec["norm"]]
    slot = builder.DATA["target_slot"]
    out = []
    for k in range(start, stop):
        row = rows.iloc[k]
        _, ctrl_ext, _ = builder._controls(spec, values, slot[row.target_gene], row.batch)
        out.append(ceiling_of(row.positions.size, ctrl_ext, spec["n_pert"], spec["n_ctrl"]))
    return out


def external_ceiling(cell: str, screen: str, start: int, stop: int) -> list[float]:
    spec = builder.CELLS[cell]
    arrays = builder.DATA["screens"][screen]
    values = builder.DATA["screen_values"][(screen, "all")]
    out = []
    for k in range(start, stop):
        controls = values["nt"][:, arrays["construct_slot"][k]]
        out.append(ceiling_of(arrays["pools"][k].size, controls, spec["n_pert"], spec["n_ctrl"]))
    return out


def ceiling_rows(scope: str, frame: pd.DataFrame, ceiling: np.ndarray, cell: str, genes: np.ndarray, vcc: bool) -> list[dict]:
    gap = ceiling - frame["y"].to_numpy(float)
    finite = np.isfinite(gap)
    near = finite & (gap <= NEAR_CEILING)
    away = finite & (gap > NEAR_CEILING)
    entry = {"cell": cell, "scope": scope, "item": "2", "rows": int(finite.sum()),
             "ceiling_median": float(np.nanmedian(ceiling)), "gap_median": float(np.nanmedian(gap)),
             "gap_q25": float(np.nanquantile(gap, 0.25)), "gap_q75": float(np.nanquantile(gap, 0.75)),
             "share_within_0.1_of_ceiling": float(near.sum() / max(finite.sum(), 1)),
             "rows_further_than_0.1": int(away.sum())}
    out = [entry]
    if away.sum() >= 3:
        part = frame[away]
        out.append(readouts.corr_row(part["y"], part["B2"], genes[away], cell=cell, scope=scope, item="2",
                                     label="R3 on rows further than 0.1 below the ceiling"))
        if vcc:
            out.append(readouts.corr_row(part["y"], part["existing_label"], genes[away], cell=cell, scope=scope, item="2",
                                         label="R1 on rows further than 0.1 below the ceiling"))
    return out


# --------------------------------------------------------------------------- simple items


def baseline_check() -> tuple[list[dict], dict]:
    """(d): the E1a external B1 and B2 against the stored control means and native deltas."""
    out, records = [], {"staging_record": file_record(STAGING_RECORD)}
    staged = json.loads(STAGING_RECORD.read_text())["arrays"]
    for screen in readouts.SCREENS:
        if file_record(STORED_CONTROL_MEANS[screen])["sha256"] != staged[screen]["staged"]["sha256"]:
            raise RuntimeError("the staged control means of %s differ from their staging record" % screen)
        data = features.load_screen(screen)
        paths = config.screen_paths()[screen]
        records[screen] = {"stored_control_means": file_record(STORED_CONTROL_MEANS[screen]),
                           "endpoint": file_record(paths["endpoint"]), "native_delta": file_record(paths["x_native"]),
                           "mapped_delta_read_for_the_row_alignment_check": file_record(paths["x_mapped"])}
        mu = np.load(STORED_CONTROL_MEANS[screen]).astype(np.float64)
        ext = readouts.external_table(TABLES, "G1", screen).set_index("construct")
        constructs = data.endpoint["construct"].astype(str)
        if set(constructs) != set(ext.index.astype(str)):
            raise RuntimeError("the E1a external constructs differ from the endpoint's on %s" % screen)
        column = data.endpoint["target_col"].astype(int).to_numpy()
        b1 = ext.loc[constructs, "B1"].to_numpy(float)
        b2 = ext.loc[constructs, "B2"].to_numpy(float)
        stored_delta = data.x_native[np.arange(len(constructs)), column].astype(np.float64)
        for name, mine, stored in (("B1 against the stored control mean", b1, mu[column]),
                                   ("negated B2 against the native delta", -b2, stored_delta)):
            d = np.abs(mine - stored)
            out.append({"screen": screen, "item": "d", "comparison": name, "constructs": int(d.size),
                        "max_abs_difference": float(d.max()), "median_abs_difference": float(np.median(d)),
                        "p99_abs_difference": float(np.quantile(d, 0.99)), "above_1e-4": int((d > 1e-4).sum())})
    return out, records


def split_half_corrected() -> list[dict]:
    out = []
    for cell in PAIRS:
        for screen in readouts.SCREENS:
            ext = readouts.external_table(TABLES, cell, screen)
            ext = ext[ext["y"].notna()]
            r = readouts._spearman(ext["y_draws_1_12"].to_numpy(), ext["y_draws_13_25"].to_numpy())
            out.append({"cell": cell, "screen": screen, "item": "e", "split_half_over_draws": r,
                        "spearman_brown": 2 * r / (1 + r) if np.isfinite(r) else np.nan})
    return out


def external_r3_by_availability() -> list[dict]:
    out = []
    for cell in PAIRS + tuple(c for c in builder.CELLS if "-SU-" in c):
        for screen in builder.CELLS[cell]["screens"]:
            ext = readouts.external_table(TABLES, cell, screen)
            ext = ext[ext["y"].notna()]
            for subset, mask in (("target is a feature", ext["target_is_feature"].to_numpy(bool)),
                                 ("target is not a feature", ~ext["target_is_feature"].to_numpy(bool))):
                part = ext[mask]
                out.append(readouts.corr_row(part["y"], part["B2"], part["target_ensg"], cell=cell, screen=screen, item="f",
                                             label="R3", subset=subset))
    return out


def common_support() -> list[dict]:
    out = []
    for cell in PAIRS:
        labels = TABLES[cell]["vcc_labels.csv"]
        train = labels[(labels["split"] == "train") & labels["y"].notna()]
        for screen in readouts.SCREENS:
            ext = readouts.external_table(TABLES, cell, screen)
            ext = ext[ext["y"].notna()]
            for column in ("y", "B1", "B2"):
                lo, hi = float(train[column].min()), float(train[column].max())
                v = ext[column].to_numpy(float)
                out.append({"cell": cell, "screen": screen, "item": "3", "quantity": "label" if column == "y" else column,
                            "vcc_training_min": lo, "vcc_training_max": hi, "constructs": int(v.size),
                            "share_below": float(np.mean(v < lo)), "share_above": float(np.mean(v > hi)),
                            "share_outside": float(np.mean((v < lo) | (v > hi)))})
    return out


def task_derived(derived: str) -> list[dict]:
    base = DERIVED[derived]
    keys = ["split", "batch", "target_gene"]
    merged = TABLES[base]["vcc_labels.csv"][keys + ["y"]].merge(
        TABLES[derived]["vcc_labels.csv"][keys + ["y"]], on=keys, suffixes=("_base", "_derived"), validate="one_to_one")
    out = []
    for split in readouts.SPLITS:
        part = merged[merged["split"] == split]
        row = readouts.corr_row(part["y_base"], part["y_derived"], part["target_gene"], cell=derived, base=base, split=split,
                                item="4", label="derived cell's label against its base cell's label, same rows")
        row["median_base"] = float(part["y_base"].median())
        row["median_derived"] = float(part["y_derived"].median())
        out.append(row)
    return out


def guide_mixing() -> list[dict]:
    rows = TABLES["G2"]["vcc_labels.csv"]
    out = []
    for cell in ("G3", "G4"):
        genes = set(TABLES[cell]["vcc_labels.csv"]["target_gene"])
        out.append({"cell": cell, "item": "12", "gene_rows": int(len(TABLES[cell]["vcc_labels.csv"])),
                    "mixing_by_largest_row_count": int((TABLES[cell]["vcc_labels.csv"]["n_guides"] > 1).sum()),
                    "mixing_by_union_of_guides": guide_union_counts(rows, genes)})
    return out


# --------------------------------------------------------------------------- main


def write(frame: pd.DataFrame, name: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    frame.to_csv(path, index=False)
    return file_record(path)


def main() -> None:
    started = time.time()
    tables, inputs = readouts.load_tables()
    TABLES.update(tables)
    print("[supplement] %d cells loaded and verified" % len(tables), flush=True)
    outputs = {}

    rows = [r for chunk in readouts.run(task_between, [(c,) for c in builder.CELLS]) for r in chunk]
    outputs["a_b_R2_between.csv"] = write(pd.DataFrame(rows), "a_b_R2_between.csv")
    rows = [r for chunk in readouts.run(task_within_mean, [(c,) for c in PAIRS]) for r in chunk]
    outputs["b_R2_within_mean_label.csv"] = write(pd.DataFrame(rows), "b_R2_within_mean_label.csv")
    rows = [r for chunk in readouts.run(task_within_simulated_rows, [(c,) for c in PAIRS]) for r in chunk]
    outputs["c_R2_within_simulated_rows.csv"] = write(pd.DataFrame(rows), "c_R2_within_simulated_rows.csv")
    print("[supplement] R2 items done", flush=True)

    check, check_inputs = baseline_check()
    outputs["d_external_baseline_check.csv"] = write(pd.DataFrame(check), "d_external_baseline_check.csv")
    outputs["e_external_split_half_corrected.csv"] = write(pd.DataFrame(split_half_corrected()), "e_external_split_half_corrected.csv")
    outputs["f_external_R3_by_availability.csv"] = write(pd.DataFrame(external_r3_by_availability()), "f_external_R3_by_availability.csv")
    outputs["3_common_support.csv"] = write(pd.DataFrame(common_support()), "3_common_support.csv")
    outputs["12_guide_mixing.csv"] = write(pd.DataFrame(guide_mixing()), "12_guide_mixing.csv")
    rows = [r for chunk in readouts.run(task_derived, [(c,) for c in DERIVED]) for r in chunk]
    outputs["4_base_against_derived.csv"] = write(pd.DataFrame(rows), "4_base_against_derived.csv")
    tasks = [(cell, scope) for cell in PAIRS for scope in readouts.SPLITS + readouts.SCREENS]
    rows = [r for chunk in readouts.run(task_descriptive_one, tasks) for r in chunk]
    outputs["1_label_against_B1_and_partials.csv"] = write(pd.DataFrame(rows), "1_label_against_B1_and_partials.csv")
    rows = [r for chunk in readouts.run(task_r5_detail, [(c, s) for c in PAIRS for s in ("VCC",) + readouts.SCREENS]) for r in chunk]
    detail = pd.DataFrame(rows)
    gate = pd.read_csv(readouts.OUT / "R5_gate.csv") if readouts.OUT.exists() else None
    if gate is not None:
        merged = detail.merge(gate[["cell", "side", "level", "version", "residual_reliability"]], on=["cell", "side", "level", "version"],
                              suffixes=("", "_gate_file"), how="left")
        detail["same_as_gate_file"] = np.isclose(merged["residual_reliability"], merged["residual_reliability_gate_file"], rtol=0, atol=1e-12)
    outputs["5_6_R5_detail.csv"] = write(detail, "5_6_R5_detail.csv")
    print("[supplement] read-out items done", flush=True)

    counts, count_manifest = builder.load_counts()
    builder.prepare(counts)
    print("[supplement] counts loaded for the ceilings", flush=True)
    ceiling_out, row_files = [], {}
    ROWS_OUT.mkdir(parents=True, exist_ok=True)
    for cell in PAIRS:
        n = len(builder.DATA["cell_rows"][cell])
        values = [v for chunk in readouts.run(vcc_ceiling, [(cell, s, min(s + builder.CHUNK, n)) for s in range(0, n, builder.CHUNK)]) for v in chunk]
        labels = TABLES[cell]["vcc_labels.csv"].copy()
        keys = builder.DATA["cell_rows"][cell][["split", "batch", "target_gene"]].reset_index(drop=True)
        if not np.array_equal(keys.astype(str).to_numpy(), labels[["split", "batch", "target_gene"]].astype(str).to_numpy()):
            raise RuntimeError("row order of %s differs from its label table" % cell)
        labels["ceiling"] = values
        frames = {"VCC": labels}
        for split in readouts.SPLITS:
            part = labels[labels["split"] == split]
            ceiling_out += ceiling_rows(split, part, part["ceiling"].to_numpy(float), cell, part["target_gene"].to_numpy(), True)
        for screen in readouts.SCREENS:
            m = len(builder.DATA["screens"][screen]["pools"])
            ext_values = [v for chunk in readouts.run(external_ceiling, [(cell, screen, s, min(s + builder.CHUNK, m)) for s in range(0, m, builder.CHUNK)]) for v in chunk]
            ext = readouts.external_table(TABLES, cell, screen).copy()
            if not np.array_equal(ext["construct"].astype(str).to_numpy(), builder.DATA["screens"][screen]["table"]["construct"].astype(str).to_numpy()):
                raise RuntimeError("construct order of %s on %s differs" % (cell, screen))
            ext["ceiling"] = ext_values
            frames[screen] = ext
            ceiling_out += ceiling_rows(screen, ext, ext["ceiling"].to_numpy(float), cell, ext["target_ensg"].to_numpy(), False)
        for scope, frame in frames.items():
            name = f"ceiling_{cell}_{scope}.csv"
            keep = [c for c in ("split", "batch", "target_gene", "construct", "target_ensg", "y", "ceiling") if c in frame]
            frame[keep].to_csv(ROWS_OUT / name, index=False)
            row_files[name] = file_record(ROWS_OUT / name)
        print(f"[supplement] ceilings of {cell} done", flush=True)
    outputs["2_ceiling.csv"] = write(pd.DataFrame(ceiling_out), "2_ceiling.csv")

    summary = {
        "plan": PLAN,
        "inputs": {**inputs, "count_cache_manifest": count_manifest["manifest"], "external_baseline_check": check_inputs,
                   "r5_gate_file": file_record(readouts.OUT / "R5_gate.csv")},
        "outputs": outputs,
        "per_row_ceilings": row_files,
        "settings": {"bootstrap": {"replicates": readouts.B, "seed": readouts.SEED, "unit": "target gene"},
                     "r5_residualisation": {"fold_seed": FOLD_SEEDS[0], "fold_seed_sensitivity": list(FOLD_SEEDS[1:]),
                                            "model": "HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20, early_stopping=False, random_state=0)"},
                     "near_ceiling": NEAR_CEILING},
        "seconds": time.time() - started,
    }
    (OUT / "supplement.json").write_text(json.dumps(strict_json(summary), indent=1, allow_nan=False) + "\n")
    print("[supplement] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        readouts.B = 20
        OUT = config.storage_root() / "bioinf2027" / "scratch" / "E1a_supplement_smoke"
        ROWS_OUT = OUT / "rows"
        PLAN = PLAN + "; smoke run, 20 replicates, never reported"
    elif sys.argv[1:]:
        raise SystemExit("unknown arguments: %s" % sys.argv[1:])
    main()
