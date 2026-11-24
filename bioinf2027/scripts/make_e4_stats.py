"""E4, the statistics: the label- and cell-number-matched features against E1b's all-cell pipeline, and the
replacement for the submitted Section 6.4.

Design: bioinf2027/plans/E4.md, Sections 5 and 10 as amended by Section 11 (11.1 governs the readings).
Functions shared with E1b (the residuals of R, the interval and p-value summary, the digest-checked
predictions) are E1b's own (bioinf2027/scripts/make_e1b_stats.py), so that both are computed alike.

- F, per arm (z, x, m, [x; m]), cell and screen: the external Spearman under matched features minus that
  under all-cell features, same heads, rows and seeds. Matched values are averaged over the 3 feature
  replicates x 3 head seeds inside each replicate of E1b's joint target-gene cluster bootstrap (10,000
  replicates, seeds 20260901 and 20260902), the all-cell ones over the 3 head seeds. The family: F in G5
  and G6 on both screens, 16 statistics, Holm at 0.05; a statement needs the same state in both cells; a
  Holm-resolved F whose three per-replicate values do not all share its sign is "not robust to the
  feature draw". Cells excluded by E1's label positive control and arms failing E1's precondition stay in
  the family, computed and not read.
- Secondary and unadjusted: F in G1 and G2, on the target-is and is-not-a-feature subsets, without the
  constructs whose pool holds exactly n_pert cells (scoring side only), under the secondary heads, of R,
  prediction-averaged, per refit, with the existing and source-mode VCC labels; the 2 x 2 of training- and
  scoring-side features; the matched T values (no statement); z - C2 and z - x; the positive control for
  F; the trivial baselines at both cell numbers; the manipulation check; the size-factor variant; the
  depth contrast; the cell-disjoint variant; VCC test R-squared and Spearman under both pipelines.
- The Section 6.4 replacement (Sections 10.4 and 11.1): ordinary least squares on B0, on the four
  magnitude summaries and on both (and on B1, on B2 native, and the magnitude increment over B0 to B2),
  fitted on training rows and scored on test, in four corners; dR2_count, dR2_increment, their total
  change, the two attribution paths and the interaction, target-gene cluster bootstrap (10,000
  replicates, seed 20270301), unadjusted. First the printed validation values are reproduced to 1e-4; a
  failure stops this part (gate G3).

Under Section 15 of the plan the all-cell side of F of z is E1b's heads (the corner fits) scored on the
cache-built external z; F of z against E1b's stored-z predictions is a secondary row, and the E1b z
sensitivity (E1 note, Section 30) is written beside.

Markers on F (E4 plan, Section 10.2, item 7, as E1b defines them, bioinf2027/scripts/make_e1b_stats.py):
"not robust to the label mode" when a resolved F's state under the source-mode VCC labels, against the
same Holm threshold, differs; "not robust to head fitting" when fewer than 80 % of the batch-resampled
refits (replicate 1, head seed 0) share a resolved F's sign.

Passes (E4 plan, Section 16): --pass family reads the predictions of tiers 1 and 2
(predictions_manifest_tier2.json) and writes the family, its readings, the refits behind the head-fitting
marker, z - C2 and the corner check of the fits present to bioinf2027/results/E4/family_pass/; --pass full
reads every tier (predictions_manifest_tier3.json), writes everything, and checks that its values equal
the family pass's (the precondition_matched columns, which need tier 3, excluded); a failed check is
written to stats.json and stops the run (gate G3).

Outputs under bioinf2027/results/E4/: family.csv, readings.csv, secondary.csv, refits.csv,
e1b_z_sensitivity.csv, section_6_4.csv, section_6_4_readings.csv, stats.json. With --smoke: 20 replicates, the smoke
predictions, outputs to the scratch folder, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, representations as rep, resample, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e1b_fits as e1b  # noqa: E402
import fit_tiers  # noqa: E402
import make_e1b_stats as e1s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E4"
E1 = REPO / "bioinf2027" / "results" / "E1"
PLAN = "bioinf2027/plans/E4.md, Sections 5 and 10 as amended by Section 11"
B = 10000
EXT_SEED = e1s.EXT_SEED
VCC_SEED = e1s.VCC_SEED
FOLD_SEED = e1s.FOLD_SEEDS[0]
SEEDS = (0, 1, 2)
REPLICATES = (1, 2, 3)
ALPHA = 0.05
POSITIVE_LIMIT = 0.80
CELLS = ("G5", "G6", "G1", "G2")
PRIMARY = tuple(e1cells.PRIMARY)
SCREENS = tuple(e1cells.e1inputs.SCREENS)
FAMILY = transfer.FAMILY
MASKED = transfer.MASKED
MATCHED_ARMS = FAMILY + MASKED + transfer.CONTROLS + ("TR",)
TRIVIAL = tuple(e1cells.TRIVIAL)
ALL = e1s.ALL_ROWS
MATCHED = "E4 | {cell} | draw {k}"
N_PERT = {"G5": 27, "G6": 10, "G1": 27, "G2": 10}
MAGNITUDE = ("mean |x|", "norm of x", "max |x|", "sd of x")
PRINTED = {"count": 0.4325, "magnitude": 0.2589, "both": 0.4342, "increment": 0.0017}
TOLERANCE = 1e-4
CORNERS = ("existing-label", "labels-matched", "matched", "fourth")
VCC_COLUMNS = {"rebuilt": "y", "source": "src_y", "mean": "y_mean", "existing": "existing_label"}
SMOKE = False
PASS = "full"
EQUALITY_EXCLUDED = ("precondition_matched_passed", "precondition_matched_estimate", "precondition_matched_lo")
Z_CONTRASTS = ("C2", "x")

DATA: dict = {}
summarise = e1s.summarise


# --------------------------------------------------------------------------- predictions


def head_seeds(head: str) -> tuple:
    return (0,) if head in ("ridge", "monotone") or SMOKE else SEEDS


def collect(source: str, contexts: list, label: str, model: str, head: str = "forest", variant: str = ALL, seeds=None) -> list | None:
    """Per-fit predictions on the named sides, one vector per (context, seed); None if a fit is missing."""
    out = []
    for context, side in contexts:
        for seed in head_seeds(head) if seeds is None else seeds:
            values = DATA[source].get(context, label, model, head, seed, variant)
            if values is None or side not in values:
                if SMOKE:
                    continue  # the smoke fits are a subset
                return None
            out.append(np.asarray(values[side], dtype=np.float64))
    return out or None


def all_cell(cell, model, side, label="rebuilt", head="forest", variant=ALL, seeds=None):
    return collect("e1b", [(cell, side)], label, model, head, variant, seeds)


def matched(cell, model, side, label="rebuilt", head="forest", replicates=REPLICATES, template=MATCHED, variant=ALL, seeds=None):
    return collect("e4", [(template.format(cell=cell, k=k), side) for k in replicates], label, model, head, variant, seeds)


CORNER = "E4 | {cell} | all-cell heads on matched sides"
STORED_Z_NOTE = "all-cell side on the stored z, which carries the shared shift (no cache-built counterpart for this label, head or refit)"


def corner_cache(cell, model, screen):
    """E1b's heads (the corner fits) scored on the cache-built all-cell external z (Section 15, item 3)."""
    return collect("e4", [(CORNER.format(cell=cell), f"{screen} cache-built z")], "rebuilt", model)


def all_side(cell, model, side, label="rebuilt", head="forest", variant=ALL, seeds=None):
    """The all-cell side of F: for z on a screen, the corner fits on the cache-built z (Section 15, item 4); otherwise E1b's predictions."""
    if model == "z" and side in SCREENS and label == "rebuilt" and head == "forest" and variant == ALL and seeds is None:
        return corner_cache(cell, model, side)
    return all_cell(cell, model, side, label, head, variant, seeds)


def feature_arrays(cell: str, variant: str) -> dict:
    key = f"{cell} | {variant}"
    if key not in DATA["arrays"]:
        entry = DATA["features"][key]
        path = config.storage_root() / entry["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError("the E4 features of %s differ from their record" % key)
        with np.load(path, allow_pickle=False) as store:
            DATA["arrays"][key] = {k: store[k] for k in store.files}
    return DATA["arrays"][key]


def vcc_order(cell: str, arrays: dict) -> np.ndarray:
    data = e1b.DATA["cells"][cell]
    position = {k: i for i, k in enumerate(np.asarray(arrays["VCC_keys"]).tolist())}
    return np.array([position["|".join(k)] for k in data.vcc[["split", "batch", "target_gene"]].astype(str).to_numpy()])


# --------------------------------------------------------------------------- statistics


def ext_table(cell: str, screen: str) -> pd.DataFrame:
    return e1b.DATA["cells"][cell].external[screen]


def ext_label(cell: str, screen: str, column: str = "y") -> np.ndarray:
    return ext_table(cell, screen)[column].to_numpy(dtype=np.float64)


def screens_of(cell: str) -> list:
    return list(e1b.DATA["cells"][cell].spec["screens"])


def rho(screen: str, vectors, label: np.ndarray, mask=None):
    """Spearman of each vector with the label, averaged over the vectors inside each replicate."""
    if not vectors:
        return None
    columns = np.column_stack(vectors)
    draws = DATA["draws"][screen]
    point = float(resample.weighted_spearman(columns, label, draws["point"], mask=mask)[0].mean())
    return point, resample.weighted_spearman(columns, label, draws["draw"], mask=mask).mean(axis=1)


def diff(a, b):
    if a is None or b is None:
        return None
    return a[0] - b[0], a[1] - b[1]


def residual(cell: str, screen: str, vector: np.ndarray) -> np.ndarray:
    table = ext_table(cell, screen)
    return e1s.oof_residuals(vector, table[list(TRIVIAL)].to_numpy(dtype=np.float64), table["target_ensg"].astype(str).to_numpy(), FOLD_SEED, "boosting")


def r_value(cell: str, screen: str, vectors, mask=None):
    """R as E1b computes it (boosting nuisance on B0 to B4, five gene folds, seed 20270409), averaged over the vectors."""
    if not vectors:
        return None
    key = (cell, screen)
    if key not in DATA["label_residuals"]:
        DATA["label_residuals"][key] = residual(cell, screen, ext_label(cell, screen))
    e_y = DATA["label_residuals"][key]
    draws = DATA["draws"][screen]
    points, replicates = [], []
    for vector in vectors:
        e_p = residual(cell, screen, vector)
        points.append(resample.weighted_pearson(e_p, e_y, draws["point"], mask=mask)[0, 0])
        replicates.append(resample.weighted_pearson(e_p, e_y, draws["draw"], mask=mask)[:, 0])
    return float(np.mean(points)), np.mean(replicates, axis=0)


def vcc_draw(cell: str) -> dict:
    if cell not in DATA["vcc_draws"]:
        data = e1b.DATA["cells"][cell]
        groups = data.vcc[data.split("test")]["target_gene"].astype(str).to_numpy()
        DATA["vcc_draws"][cell] = {"draw": resample.cluster_draw(groups, B, VCC_SEED), "point": e1s._ones(groups.size),
                                   "genes": int(np.unique(groups).size)}
    return DATA["vcc_draws"][cell]


def vcc_label(cell: str, label: str = "rebuilt", split: str = "test") -> np.ndarray:
    data = e1b.DATA["cells"][cell]
    return data.vcc[data.split(split)][VCC_COLUMNS[label]].to_numpy(dtype=np.float64)


def vcc_stats(cell: str, vectors, label: str = "rebuilt") -> dict | None:
    if not vectors:
        return None
    y = vcc_label(cell, label)
    draws = vcc_draw(cell)
    p = np.column_stack(vectors)
    return {"r2": (float(resample.weighted_r2(y, p, draws["point"])[0].mean()), resample.weighted_r2(y, p, draws["draw"]).mean(axis=1)),
            "spearman": (float(resample.weighted_spearman(p, y, draws["point"])[0].mean()), resample.weighted_spearman(p, y, draws["draw"]).mean(axis=1))}


def precondition(cell: str, arm: str, pipeline: str) -> dict:
    """E1's precondition (VCC test R-squared of the arm minus that of C1, lower bound above zero) under one pipeline."""
    get = matched if pipeline == "matched" else all_cell
    a, c = vcc_stats(cell, get(cell, arm, "test")), vcc_stats(cell, get(cell, "C1", "test"))
    if a is None or c is None:
        return {"passes": None, "estimate": float("nan"), "lo": float("nan"), "hi": float("nan")}
    s = summarise(diff(a["r2"], c["r2"]))
    return {"passes": bool(s["lo"] > 0), "estimate": s["estimate"], "lo": s["lo"], "hi": s["hi"]}


def per_replicate(cell: str, screen: str, arm: str, label: np.ndarray, reference, **kw) -> str:
    values = []
    for k in REPLICATES:
        v = diff(rho(screen, matched(cell, arm, screen, replicates=(k,), **kw), label), reference)
        values.append(float("nan") if v is None else v[0])
    return ";".join("%.4f" % v for v in values)


def r5_passed(cell: str, screen: str) -> bool:
    gate = DATA["r5"]
    return bool(gate[f"{cell} | VCC"]["passed"] and gate[f"{cell} | {screen}"]["passed"])


# --------------------------------------------------------------------------- the family and its readings


def e1_flags() -> dict:
    """E1b's label positive control and precondition per cell, screen and arm, when E1b's statistics exist."""
    out = {"available": False, "positive": {}, "precondition": {}}
    positive, pre = E1 / "positive_controls.csv", E1 / "preconditions.csv"
    if SMOKE or not (positive.exists() and pre.exists()):
        return out
    pc = pd.read_csv(positive)
    pc = pc[pc["control"].str.startswith("label construction") & (pc["subset"] == "all")]
    for _, row in pc.iterrows():
        out["positive"][(row["cell"], row["screen"])] = not bool(row["below_0.80"])
    pr = pd.read_csv(pre)
    for _, row in pr[pr["label"] == "rebuilt"].iterrows():
        out["precondition"][(row["cell"], row["arm"])] = bool(row["passes"])
    out.update({"available": True, "files": {"positive_controls": file_record(positive), "preconditions": file_record(pre)}})
    return out


def family_rows(flags: dict) -> pd.DataFrame:
    rows = []
    for cell in PRIMARY:
        for screen in SCREENS:
            label = ext_label(cell, screen)
            for arm in FAMILY:
                reference = rho(screen, all_side(cell, arm, screen), label)
                f = diff(rho(screen, matched(cell, arm, screen), label), reference)
                positive = flags["positive"].get((cell, screen))
                pre_e1 = flags["precondition"].get((cell, arm))
                pre_matched = precondition(cell, arm, "matched")
                f_source = diff(rho(screen, matched(cell, arm, screen, label="source"), label),
                                rho(screen, all_cell(cell, arm, screen, label="source"), label))
                refits = refit_f_values(cell, screen, arm)
                if not flags["available"]:
                    read = "unknown: E1b's statistics not found"
                elif positive is False:
                    read = "no: E1's label positive control is below +0.80 in this cell and screen"
                elif pre_e1 is False:
                    read = "no: the arm fails E1's precondition in this cell"
                else:
                    read = "yes"
                rows.append({"cell": cell, "screen": screen, "arm": arm, "statistic": "F", **summarise(f),
                             "per_replicate": per_replicate(cell, screen, arm, label, reference),
                             "label_positive_control_E1_passed": positive, "precondition_E1_passed": pre_e1,
                             "precondition_matched_passed": pre_matched["passes"], "precondition_matched_estimate": pre_matched["estimate"],
                             "precondition_matched_lo": pre_matched["lo"], "read": read,
                             "all_cell_side": ("E1b's heads on the cache-built external z (E4 plan, Section 15)" if arm == "z"
                                               else "E1b's saved predictions"),
                             "source_mode_values": f_source, "refit_values": refits})
    frame = pd.DataFrame(rows)
    rejected, thresholds = resample.holm(frame["p"].to_numpy(), ALPHA)
    frame["holm_threshold"] = thresholds
    frame["rejected"] = rejected
    frame["state"] = [resample.holm_state(e, r) for e, r in zip(frame["estimate"], rejected)]
    # E1's markers, inherited by F (Section 10.2, item 7), as bioinf2027/scripts/make_e1b_stats.py defines them
    source_state, share = [], []
    for row in frame.itertuples():
        s = row.source_mode_values
        source_state.append(resample.holm_state(s[0], bool(np.isfinite(resample.p_value(s[1])) and resample.p_value(s[1]) <= row.holm_threshold))
                            if s is not None else "not fitted")
        values = row.refit_values
        share.append(float((np.sign(values) == np.sign(row.estimate)).mean()) if values.size and np.isfinite(row.estimate) and row.estimate != 0
                     else float("nan"))
    frame["state_source_mode_labels"] = source_state
    frame["source_mode_all_cell_side"] = [STORED_Z_NOTE if a == "z" else "E1b's saved source-mode predictions" for a in frame["arm"]]
    frame["refits"] = [int(v.size) for v in frame["refit_values"]]
    frame["share_of_refits_with_the_sign"] = share
    frame["not_robust_to_label_mode"] = [r.state != "inconclusive" and r.state != s for r, s in zip(frame.itertuples(), source_state)]
    frame["not_robust_to_head_fitting"] = [r.state != "inconclusive" and np.isfinite(sh) and sh < 0.8 for r, sh in zip(frame.itertuples(), share)]
    frame = frame.drop(columns=["source_mode_values", "refit_values"])
    frame["not_robust_to_the_feature_draw"] = [
        s != "inconclusive" and not all(np.sign(float(v)) == np.sign(e) for v in p.split(";"))
        for s, p, e in zip(frame["state"], frame["per_replicate"], frame["estimate"])]
    frame["family"] = "E4, 16 statistics, Holm at family-wise 0.05; at most 0.25 across E1, E4, D2 and E6"
    return frame


def reading_rows(family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for screen in SCREENS:
        for arm in FAMILY:
            part = family[(family["screen"] == screen) & (family["arm"] == arm)].set_index("cell")
            states = [part.loc[c, "state"] for c in PRIMARY]
            not_read = [f"{c} ({part.loc[c, 'read']})" for c in PRIMARY if part.loc[c, "read"] != "yes"]
            fragile = [c for c in PRIMARY if bool(part.loc[c, "not_robust_to_the_feature_draw"])]
            label_mode = [c for c in PRIMARY if bool(part.loc[c, "not_robust_to_label_mode"])]
            head_fitting = [c for c in PRIMARY if bool(part.loc[c, "not_robust_to_head_fitting"])]
            if not_read:
                statement = "computed and reported, not read: " + "; ".join(not_read)
            elif states[0] != states[1]:
                statement = "undetermined: the cells differ"
            elif states[0] == "inconclusive":
                statement = ("no change from the feature-side cell number is resolved; the interval and the smallest detectable F "
                             "(2.80 times its standard error) are stated, not 'no effect'")
            else:
                statement = ("under the label- and cell-number-matched pipeline the arm's external Spearman is %s than under all-cell "
                             "features; the transfer statements of the paper stay with E1's pipeline" % ("higher" if states[0] == "Holm-positive" else "lower"))
            if fragile:
                statement += "; not robust to the feature draw in " + ", ".join(fragile)
            if label_mode:
                statement += "; not robust to the label mode in " + ", ".join(label_mode)
            if head_fitting:
                statement += "; not robust to head fitting in " + ", ".join(head_fitting)
            rows.append({"screen": screen, "arm": arm, **{c: s for c, s in zip(PRIMARY, states)},
                         **{f"{c}_estimate": part.loc[c, "estimate"] for c in PRIMARY}, **{f"{c}_lo": part.loc[c, "lo"] for c in PRIMARY},
                         **{f"{c}_hi": part.loc[c, "hi"] for c in PRIMARY}, **{f"{c}_detectable_2.80se": part.loc[c, "detectable_2.80se"] for c in PRIMARY},
                         "statement": statement,
                         "family_error": "F (16 statistics, Holm 0.05); summed with E1, D2 and E6 at most 0.25",
                         "qualifier": "the features and the label share cells (Section 10.2, item 5); per-cell depth, chemistry, size-factor gene "
                                      "universe, feature namespace, guide design and survivor selection stay unmatched; the embedding is not depth-matched"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- secondary


def secondary_rows() -> pd.DataFrame:
    rows = []

    def add(cell, screen, quantity, value, **extra):
        rows.append({"cell": cell, "screen": screen, "quantity": quantity, **summarise(value), "adjustment": "secondary, unadjusted", **extra})

    def add_point(cell, screen, quantity, estimate, **extra):
        rows.append({"cell": cell, "screen": screen, "quantity": quantity, "estimate": float(estimate), "adjustment": "secondary, descriptive", **extra})

    for cell in CELLS:
        data = e1b.DATA["cells"][cell]
        draws = [feature_arrays(cell, f"draw {k}") for k in REPLICATES]
        orders = [vcc_order(cell, a) for a in draws]
        train = data.split("train")
        b0_train = data.vcc["B0"].to_numpy(dtype=np.float64)[train]
        # manipulation check on VCC training rows (Section 10.3, item 1)
        all_summaries = rep.magnitude(np.asarray(data.x, dtype=np.float64))[train]
        matched_summaries = [rep.magnitude(np.asarray(a["VCC_x"], dtype=np.float64)[o])[train] for a, o in zip(draws, orders)]
        for j, name in enumerate(MAGNITUDE):
            add_point(cell, "VCC training rows", f"manipulation check: Spearman of {name} with B0, all-cell features",
                      sps.spearmanr(all_summaries[:, j], b0_train).statistic)
            add_point(cell, "VCC training rows", f"manipulation check: Spearman of {name} with B0, matched features (mean over 3 draws)",
                      np.mean([sps.spearmanr(s[:, j], b0_train).statistic for s in matched_summaries]))
        add_point(cell, "VCC training rows", "manipulation check: Spearman of the norm of z with B0, all-cell features",
                  sps.spearmanr(np.linalg.norm(np.asarray(data.z, dtype=np.float64)[train], axis=1), b0_train).statistic)
        add_point(cell, "VCC training rows", "manipulation check: Spearman of the norm of z with B0, matched features (mean over 3 draws)",
                  np.mean([sps.spearmanr(np.linalg.norm(np.asarray(a["VCC_z"], dtype=np.float64)[o][train], axis=1), b0_train).statistic
                           for a, o in zip(draws, orders)]))
        # VCC test under both pipelines (Section 5)
        for arm in MATCHED_ARMS:
            a, m = vcc_stats(cell, all_cell(cell, arm, "test")), vcc_stats(cell, matched(cell, arm, "test"))
            for metric in ("r2", "spearman"):
                add(cell, "VCC test", f"VCC test {metric} of {arm}, all-cell features", None if a is None else a[metric])
                add(cell, "VCC test", f"VCC test {metric} of {arm}, matched features", None if m is None else m[metric])
                add(cell, "VCC test", f"VCC test {metric} of {arm}, matched minus all-cell", None if a is None or m is None else diff(m[metric], a[metric]))
        for screen in screens_of(cell):
            table = ext_table(cell, screen)
            label = ext_label(cell, screen)
            feature = table["target_is_feature"].to_numpy(dtype=bool)
            above = table["n_pool"].to_numpy(dtype=np.float64) > N_PERT[cell]
            gate = r5_passed(cell, screen)
            c1_all = rho(screen, all_cell(cell, "C1", screen), label)
            c1_matched = rho(screen, matched(cell, "C1", screen), label)
            for arm in MATCHED_ARMS:
                a_vec, m_vec = all_side(cell, arm, screen), matched(cell, arm, screen)
                a, m = rho(screen, a_vec, label), rho(screen, m_vec, label)
                add(cell, screen, f"Spearman of {arm}, all-cell features", a)
                add(cell, screen, f"Spearman of {arm}, matched features", m)
                if cell not in PRIMARY or arm not in FAMILY:
                    add(cell, screen, f"F of {arm}", diff(m, a), per_replicate=per_replicate(cell, screen, arm, label, a),
                        note=("C1 does not depend on cells: F of C1 measures the fits alone" if arm == "C1"
                              else STORED_Z_NOTE if arm == "C3" else ""))
                if arm != "C1":
                    add(cell, screen, f"T of {arm}, all-cell features", diff(a, c1_all))
                    add(cell, screen, f"T of {arm}, matched features (carries no statement)", diff(m, c1_matched),
                        precondition_matched_passed=precondition(cell, arm, "matched")["passes"] if arm in FAMILY else None)
                if arm == "z":
                    add(cell, screen, "F of z against E1b's stored-z predictions (includes the shared shift)",
                        diff(m, rho(screen, all_cell(cell, "z", screen), label)), note="secondary row of Section 15, item 4")
                if arm not in FAMILY:
                    continue
                for subset, mask in (("target is a feature", feature), ("target is not a feature", ~feature),
                                     (f"pool above n_pert ({N_PERT[cell]} cells), scoring side only", above)):
                    add(cell, screen, f"F of {arm}, {subset}", diff(rho(screen, m_vec, label, mask), rho(screen, a_vec, label, mask)), constructs=int(mask.sum()))
                for head in ("boosting", "ridge"):
                    add(cell, screen, f"F of {arm}, head {head}", diff(rho(screen, matched(cell, arm, screen, head=head), label),
                                                                     rho(screen, all_cell(cell, arm, screen, head=head), label)),
                        note=STORED_Z_NOTE if arm == "z" else "")
                add(cell, screen, f"F of R of {arm}", diff(r_value(cell, screen, m_vec), r_value(cell, screen, a_vec)), r5_gate_passed=gate,
                    note="" if gate else "not answerable: E1's R5 gate fails in this cell and screen")
                averaged = None if m_vec is None or a_vec is None else diff(rho(screen, [np.mean(m_vec, axis=0)], label),
                                                                            rho(screen, [np.mean(a_vec, axis=0)], label))
                add(cell, screen, f"F of {arm}, prediction-averaged (uses up to 3 x n_pert cells)", averaged)
                corner = collect("e4", [(f"E4 | {cell} | all-cell heads on matched sides", f"{screen} matched draw {k}") for k in REPLICATES], "rebuilt", arm)
                add(cell, screen, f"{arm}: all-cell heads on matched external features minus all-cell", diff(rho(screen, corner, label), a))
                add(cell, screen, f"{arm}: matched heads on all-cell external features minus all-cell",
                    diff(rho(screen, matched(cell, arm, f"{screen} all-cell"), label), a))
                for vcc_label_name in ("existing", "source"):
                    add(cell, screen, f"F of {arm}, {vcc_label_name} VCC label" + (" (the fourth corner)" if vcc_label_name == "existing" else ""),
                        diff(rho(screen, matched(cell, arm, screen, label=vcc_label_name), label),
                             rho(screen, all_cell(cell, arm, screen, label=vcc_label_name), label)),
                        note=STORED_Z_NOTE if arm == "z" else "")
            for quantity, value in z_contrasts(cell, screen, label):
                add(cell, screen, quantity, value)
            # trivial baselines at both cell numbers and the positive control for F (Sections 10.2, item 4, and 10.3, item 3)
            pool_b = table[list(TRIVIAL)].to_numpy(dtype=np.float64)
            draw_b = [a[f"{screen}_b"].astype(np.float64) for a in draws]
            for j, name in enumerate(TRIVIAL):
                add(cell, screen, f"trivial baseline {name}, pool level", rho(screen, [pool_b[:, j]], label))
                if name in ("B2", "B3", "B4"):
                    add(cell, screen, f"trivial baseline {name}, draw level (3 feature draws)", rho(screen, [b[:, j] for b in draw_b], label))
            add(cell, screen, "positive control for F: draw-level minus pool-level B2 native",
                diff(rho(screen, [b[:, 2] for b in draw_b], label), rho(screen, [pool_b[:, 2]], label)))
            add(cell, screen, "positive control for F: [C1; B2] with draw-level B2 minus all-cell [C1; B2]",
                diff(rho(screen, matched(cell, "[C1; B2]", screen, template=MATCHED + " | draw-level B2"), label),
                     rho(screen, all_cell(cell, "[C1; B2]", screen), label)))
            # manipulation check on the screen (Section 10.3, item 1)
            b0 = pool_b[:, 0]
            all_m = rep.magnitude(np.asarray(e1b.DATA["external_x"][screen], dtype=np.float64))
            draw_m = [rep.magnitude(a[f"{screen}_x"].astype(np.float64)) for a in draws]
            for j, name in enumerate(MAGNITUDE):
                add(cell, screen, f"manipulation check: Spearman of {name} with B0, all-cell features", rho(screen, [all_m[:, j]], b0))
                add(cell, screen, f"manipulation check: Spearman of {name} with B0, matched features (3 draws)", rho(screen, [s[:, j] for s in draw_m], b0))
            add(cell, screen, "manipulation check: Spearman of the norm of z with B0, all-cell features",
                rho(screen, [np.linalg.norm(np.asarray(e1b.DATA["screens"][screen].z, dtype=np.float64), axis=1)], b0))
            add(cell, screen, "manipulation check: Spearman of the norm of z with B0, matched features (3 draws)",
                rho(screen, [np.linalg.norm(a[f"{screen}_z"].astype(np.float64), axis=1) for a in draws], b0))
            # the size-factor variant (Section 11.2, item 8)
            template = "E4 | {cell} | shared " + screen + " {k}"
            for arm in ("x", "m", "[x; m]"):
                if cell == "G1":
                    su = f"G1-SU-{screen}"
                    su_label = ext_label(su, screen)
                    v = rho(screen, matched(cell, arm, screen, label=f"rebuilt:{su}", template=template), su_label)
                    add(cell, screen, f"{arm}, shared-gene size factors on both sides, {su} label", v)
                    add(cell, screen, f"{arm}, shared-gene size factors minus E1b's {su} cell (all-cell features, same label)",
                        diff(v, rho(screen, all_cell(su, arm, screen), su_label)))
                else:
                    v = rho(screen, matched(cell, arm, screen, template=template), label)
                    add(cell, screen, f"{arm}, shared-gene size factors in the features only (the label keeps all-gene totals)", v)
                    add(cell, screen, f"{arm}, shared-gene size factors in the features minus matched", diff(v, rho(screen, matched(cell, arm, screen), label)))
        # the cell-disjoint variant (Section 11.2, item 5)
        disjoint = feature_arrays(cell, "cell-disjoint")
        for screen in screens_of(cell):
            label_a, finite = DATA["halves"][cell][screen]
            kept = np.asarray(disjoint[f"{screen}_kept"], dtype=bool) & finite
            values = {arm: rho(screen, collect("e4", [(f"E4 | {cell} | cell-disjoint", screen)], "half A", arm), label_a, kept) for arm in FAMILY + ("C1",)}
            for arm in FAMILY + ("C1",):
                add(cell, screen, f"{arm}, cell-disjoint (half B features, half A label)", values[arm], constructs=int(kept.sum()))
                if arm != "C1":
                    add(cell, screen, f"T of {arm}, cell-disjoint", diff(values[arm], values["C1"]), constructs=int(kept.sum()))
    # the depth contrast (Section 10.1, item 3b)
    for base in PRIMARY:
        for screen in SCREENS:
            dt = f"{base}-DT-{screen}"
            if dt not in e1b.DATA["cells"]:
                continue
            label = ext_label(dt, screen)
            for arm in ("x", "m", "[x; m]"):
                thin = matched(dt, arm, screen, template="E4 | {cell} | draw {k} | thinned VCC features")
                unthin = matched(dt, arm, screen, template="E4 | {cell} | draw {k} | unthinned VCC features")
                add(dt, screen, f"depth contrast of {arm}: thinned minus unthinned matched VCC features, DT label", diff(rho(screen, thin, label), rho(screen, unthin, label)))
    return pd.DataFrame(rows)


def z_contrasts(cell: str, screen: str, label: np.ndarray, others=Z_CONTRASTS) -> list:
    """z - C2 and z - x under both pipelines (Section 10.3); one code path for both passes."""
    out = []
    for other in others:
        for pipeline, get in (("matched", matched), ("all-cell", all_side)):
            out.append((f"z minus {other}, {pipeline} features", diff(rho(screen, get(cell, "z", screen), label), rho(screen, get(cell, other, screen), label))))
    return out


def z_minus_c2_rows() -> pd.DataFrame:
    """The family pass's z - C2 (E4 plan, Section 16, item 1), as secondary_rows computes it."""
    rows = []
    for cell in PRIMARY:
        for screen in screens_of(cell):
            for quantity, value in z_contrasts(cell, screen, ext_label(cell, screen), others=("C2",)):
                rows.append({"cell": cell, "screen": screen, "quantity": quantity, **summarise(value), "adjustment": "secondary, unadjusted"})
    return pd.DataFrame(rows)


def refit_f_values(cell: str, screen: str, arm: str) -> np.ndarray:
    """F per batch-resampled refit (replicate 1, head seed 0, E1b's refit draws), in refit order until one is missing."""
    label = ext_label(cell, screen)
    values = []
    r = 0
    while True:
        m = matched(cell, arm, screen, replicates=(1,), variant=f"refit {r}", seeds=(0,))
        a = all_cell(cell, arm, screen, variant=f"refit {r}", seeds=(0,))
        if m is None or a is None:
            break
        values.append(float(sps.spearmanr(m[0], label).statistic - sps.spearmanr(a[0], label).statistic))
        r += 1
    return np.array(values)


def refit_rows(family: pd.DataFrame) -> pd.DataFrame:
    """F per batch-resampled refit (replicate 1, head seed 0, E1b's refit draws), against the full-data F."""
    rows = []
    for cell in CELLS:
        for screen in screens_of(cell):
            for arm in FAMILY:
                values = refit_f_values(cell, screen, arm)
                if not values.size:
                    continue
                full = family[(family["cell"] == cell) & (family["screen"] == screen) & (family["arm"] == arm)]
                estimate = float(full["estimate"].iloc[0]) if len(full) else float("nan")
                rows.append({"cell": cell, "screen": screen, "arm": arm, "refits": int(values.size), "F": estimate,
                             "note": STORED_Z_NOTE if arm == "z" else "",
                             "p2.5": float(np.percentile(values, 2.5)), "p97.5": float(np.percentile(values, 97.5)),
                             "share_with_the_sign_of_F": float((np.sign(values) == np.sign(estimate)).mean()) if np.isfinite(estimate) else float("nan"),
                             "adjustment": "secondary, unadjusted"})
    return pd.DataFrame(rows)


def e1b_z_sensitivity() -> pd.DataFrame:
    """E1 note, Section 30: T(z) and R(z) of E1b's z heads on the cache-built external z beside the stored z."""
    rows = []
    for cell in CELLS:
        for screen in screens_of(cell):
            label = ext_label(cell, screen)
            c1 = rho(screen, all_cell(cell, "C1", screen), label)
            stored_vec, cache_vec = all_cell(cell, "z", screen), corner_cache(cell, "z", screen)
            stored, cache = rho(screen, stored_vec, label), rho(screen, cache_vec, label)
            t_stored, t_cache = diff(stored, c1), diff(cache, c1)
            r_stored, r_cache = r_value(cell, screen, stored_vec), r_value(cell, screen, cache_vec)
            gate = r5_passed(cell, screen)
            for quantity, value, is_r in (("Spearman of z, stored z", stored, False), ("Spearman of z, cache-built z", cache, False),
                                          ("T(z), stored z", t_stored, False), ("T(z), cache-built z", t_cache, False),
                                          ("T(z), cache-built minus stored", diff(t_cache, t_stored), False),
                                          ("R(z), stored z", r_stored, True), ("R(z), cache-built z", r_cache, True),
                                          ("R(z), cache-built minus stored", diff(r_cache, r_stored), True)):
                rows.append({"cell": cell, "screen": screen, "quantity": quantity, **summarise(value),
                             "r5_gate_passed": gate if is_r else None,
                             "adjustment": "secondary, unadjusted (E1 note, Section 30; E4 plan, Section 15, item 5)",
                             "heads": "E4's corner fits: E1b's z heads, seeds 0 to 2, checked against E1b's saved predictions"})
    return pd.DataFrame(rows)


def corner_check() -> dict:
    """The corner fits' predictions on the all-cell sides against E1b's saved predictions (the docstring of make_e4_fits.py)."""
    largest = 0.0
    compared = 0
    for cell in CELLS:
        for arm in FAMILY + ("C1",):
            for seed in head_seeds("forest"):
                corner = DATA["e4"].get(f"E4 | {cell} | all-cell heads on matched sides", "rebuilt", arm, "forest", seed)
                saved = DATA["e1b"].get(cell, "rebuilt", arm, "forest", seed)
                if corner is None or saved is None:
                    continue
                for side in screens_of(cell) + ["test"]:
                    largest = max(largest, float(np.max(np.abs(np.asarray(corner[side], dtype=np.float64) - np.asarray(saved[side], dtype=np.float64)))))
                    compared += 1
    return {"compared_vectors": compared, "max_abs_difference": largest, "tolerance": 1e-12, "passed": bool(compared > 0 and largest <= 1e-12)}


# --------------------------------------------------------------------------- the Section 6.4 replacement


def corner_designs(cell: str, corner: str, k: int | None = None) -> tuple[dict, np.ndarray]:
    """Design matrices of every OLS model and the label of one corner, in the cell's row order."""
    data = e1b.DATA["cells"][cell]
    if corner in ("existing-label", "labels-matched"):
        x = np.asarray(data.x, dtype=np.float64)
    else:
        arrays = feature_arrays(cell, f"draw {k}")
        x = np.asarray(arrays["VCC_x"], dtype=np.float64)[vcc_order(cell, arrays)]
    summaries = rep.fit_magnitude(x[data.split("train")]).transform(rep.magnitude(x)).astype(np.float64)
    b = data.vcc[list(TRIVIAL)].to_numpy(dtype=np.float64)
    designs = {"B0": b[:, [0]], "magnitude": summaries, "B0 and magnitude": np.column_stack([b[:, [0]], summaries]), "B1": b[:, [1]],
               "B2 native": b[:, [2]], "B0 to B2": b[:, :3], "B0 to B2 and magnitude": np.column_stack([b[:, :3], summaries])}
    column = "existing_label" if corner in ("existing-label", "fourth") else "y"
    return designs, data.vcc[column].to_numpy(dtype=np.float64)


def point_r2(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(1.0 - ((y - prediction) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def printed_reproduction() -> dict:
    """The printed validation values: existing label, all-cell features, every modelled training row (G6's rows), scored on validation."""
    data = e1b.DATA["cells"]["G6"]
    designs, y = corner_designs("G6", "existing-label")
    train, val = data.split("train"), data.split("val")
    values = {}
    for name, design in (("count", "B0"), ("magnitude", "magnitude"), ("both", "B0 and magnitude")):
        model = LinearRegression().fit(designs[design][train], y[train])
        values[name] = point_r2(y[val], model.predict(designs[design][val]))
    values["increment"] = values["both"] - values["count"]
    differences = {n: values[n] - PRINTED[n] for n in PRINTED}
    return {"values": values, "printed": PRINTED, "differences": differences, "tolerance": TOLERANCE,
            "reproduced": bool(all(abs(d) <= TOLERANCE for d in differences.values())),
            "rows": {"train": int(train.sum()), "validation": int(val.sum())},
            "printed_source": "earlier_revision/v7/main.tex:609-611; earlier_revision/w1_assets/tables/table_cellcount_confound.tex:23-25"}


def section_6_4() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, readings = [], []
    for cell in CELLS:
        data = e1b.DATA["cells"][cell]
        train, val, test = data.split("train"), data.split("val"), data.split("test")
        draw = vcc_draw(cell)
        feature = data.vcc["target_is_feature"].to_numpy(dtype=bool)[test]
        subsets = {"all": None, "target is a feature": feature, "target is not a feature": ~feature}
        r2 = {}
        for corner in CORNERS:
            ks = (None,) if corner in ("existing-label", "labels-matched") else REPLICATES
            predictions, validation, y_test = {}, {}, None
            for k in ks:
                designs, y = corner_designs(cell, corner, k)
                y_test = y[test]
                for name, design in designs.items():
                    model = LinearRegression().fit(design[train], y[train])
                    predictions.setdefault(name, []).append(model.predict(design[test]))
                    validation.setdefault(name, []).append(point_r2(y[val], model.predict(design[val])))
            for name in predictions:
                columns = np.column_stack(predictions[name])
                for subset, mask in subsets.items():
                    if mask is None:
                        sub_draw, sub_point, yy, cc = draw["draw"], draw["point"], y_test, columns
                    else:
                        sub_draw = resample.ClusterDraw(weights=draw["draw"].weights[:, mask], clusters=draw["draw"].clusters, seed=VCC_SEED,
                                                        index_sha256=draw["draw"].index_sha256)
                        sub_point, yy, cc = e1s._ones(int(mask.sum())), y_test[mask], columns[mask]
                    r2[(corner, name, subset)] = (float(resample.weighted_r2(yy, cc, sub_point)[0].mean()), resample.weighted_r2(yy, cc, sub_draw).mean(axis=1))
                rows.append({"cell": cell, "corner": corner, "quantity": f"R2 of {name}", "subset": "all", **summarise(r2[(corner, name, "all")]),
                             "validation_r2_descriptive": float(np.mean(validation[name])), "feature_replicates": len(ks)})
                for subset in ("target is a feature", "target is not a feature"):
                    rows.append({"cell": cell, "corner": corner, "quantity": f"R2 of {name}", "subset": subset, **summarise(r2[(corner, name, subset)]),
                                 "feature_replicates": len(ks)})
        quantities = {"count": ("B0", "magnitude"), "increment": ("B0 and magnitude", "B0"), "increment over B0 to B2": ("B0 to B2 and magnitude", "B0 to B2")}
        values = {}
        for corner in CORNERS:
            for q, (a, b) in quantities.items():
                for subset in subsets:
                    values[(corner, q, subset)] = diff(r2[(corner, a, subset)], r2[(corner, b, subset)])
                    rows.append({"cell": cell, "corner": corner, "quantity": f"dR2_{q}", "subset": subset, **summarise(values[(corner, q, subset)])})
        for q in quantities:
            v = {c: values[(c, q, "all")] for c in CORNERS}
            steps = {"total change (matched minus existing-label)": diff(v["matched"], v["existing-label"]),
                     "path 1, label step (labels-matched minus existing-label)": diff(v["labels-matched"], v["existing-label"]),
                     "path 1, feature step (matched minus labels-matched)": diff(v["matched"], v["labels-matched"]),
                     "path 2, feature step at the existing label (fourth minus existing-label)": diff(v["fourth"], v["existing-label"]),
                     "path 2, label step at matched features (matched minus fourth)": diff(v["matched"], v["fourth"])}
            steps["interaction (path 1 feature step minus path 2 feature step)"] = diff(steps["path 1, feature step (matched minus labels-matched)"],
                                                                                     steps["path 2, feature step at the existing label (fourth minus existing-label)"])
            summaries = {name: summarise(value) for name, value in steps.items()}
            for name, s in summaries.items():
                rows.append({"cell": cell, "corner": "steps", "quantity": f"dR2_{q}: {name}", "subset": "all", **s})
            if q in ("count", "increment"):
                readings.append({"cell": cell, "quantity": f"dR2_{q}", "matched_corner": state_of(summarise(v["matched"])),
                                 "total_change": state_of(summaries["total change (matched minus existing-label)"]),
                                 "attribution": attribution(summaries), **mechanical(cell)})
    frame = pd.DataFrame(rows)
    frame["adjustment"] = "unadjusted; the designated statistics are dR2_count and dR2_increment in the matched corner, their total change and steps, in G5 and G6"
    readings_frame = pd.DataFrame(readings)
    pair = readings_frame[readings_frame["cell"].isin(PRIMARY)]
    agreement = {}
    for q, group in pair.groupby("quantity"):
        for column in ("matched_corner", "total_change", "attribution"):
            states = group[column].tolist()
            agreement[(q, column)] = states[0] if len(states) == 2 and states[0] == states[1] else "undetermined: G5 and G6 differ"
    readings_frame["statement_in_the_pair"] = [
        "; ".join(f"{c}: {agreement[(r.quantity, c)]}" for c in ("matched_corner", "total_change", "attribution")) if r.cell in PRIMARY else "descriptive"
        for r in readings_frame.itertuples()]
    return frame, readings_frame


def state_of(s: dict) -> str:
    if not np.isfinite(s["lo"]):
        return "not computed"
    return "positive" if s["lo"] > 0 else ("negative" if s["hi"] < 0 else "not resolved")


def attribution(s: dict) -> str:
    """Section 11.1: a step is credited only if resolved with the same sign on both paths."""
    pairs = (("label step", "path 1, label step (labels-matched minus existing-label)", "path 2, label step at matched features (matched minus fourth)"),
             ("feature step", "path 1, feature step (matched minus labels-matched)", "path 2, feature step at the existing label (fourth minus existing-label)"))
    credited, one_path = [], []
    for step, first, second in pairs:
        a, b = state_of(s[first]), state_of(s[second])
        if a == b and a in ("positive", "negative"):
            credited.append(f"{step} ({a})")
        elif a in ("positive", "negative") or b in ("positive", "negative"):
            one_path.append(step)
    if credited:
        return "credited to the " + " and the ".join(credited)
    interaction = s["interaction (path 1 feature step minus path 2 feature step)"]
    if one_path:
        return "not attributable (resolved on one path only: %s); interaction %.4f [%.4f, %.4f]" % (
            ", ".join(one_path), interaction["estimate"], interaction["lo"], interaction["hi"])
    if state_of(s["total change (matched minus existing-label)"]) in ("positive", "negative"):
        return "resolved, not attributable to one step"
    return "no change resolved"


def mechanical(cell: str) -> dict:
    """E1's R2 mechanical association in the cell (all splits), for the reading of a remaining count association."""
    frame = pd.read_csv(REPO / "bioinf2027" / "endpoints" / "readouts" / "R2_within_and_mechanical.csv")
    row = frame[(frame["cell"] == cell) & (frame["split"] == "all")]
    if row.empty:
        return {"mechanical_association_excludes_zero": None}
    lo, hi = float(row["mechanical_lo"].iloc[0]), float(row["mechanical_hi"].iloc[0])
    return {"mechanical_association_excludes_zero": bool(lo > 0 or hi < 0), "mechanical_lo": lo, "mechanical_hi": hi,
            "mechanical_note": "if it excludes zero, part of a remaining count association is read as the estimator's residual dependence on pool size; "
                               "guide representation and, for fitness genes, the knockdown's effect on cell recovery are candidate explanations, untested"}


# --------------------------------------------------------------------------- main


def load() -> dict:
    folder = OUT if not SMOKE else config.storage_root() / "bioinf2027" / "scratch" / "E4_fits_smoke"
    e1_folder = E1 if not SMOKE else config.storage_root() / "bioinf2027" / "scratch" / "E1b_fits_smoke"
    DATA["e1b"] = e1s.Predictions(e1_folder / "predictions_manifest.json")
    e4_manifest = folder / ("predictions_manifest_tier2.json" if PASS == "family" else "predictions_manifest_tier3.json")
    DATA["e4"] = e1s.Predictions(e4_manifest)
    cells = list(CELLS) + [f"G1-SU-{s}" for s in SCREENS] + [f"{b}-DT-{s}" for b in PRIMARY for s in SCREENS]
    inputs = e1b.load_inputs(cells)
    record = json.loads((OUT / "features.json").read_text())
    if not record.get("checks_passed"):
        raise SystemExit("the E4 builder checks have not passed (bioinf2027/results/E4/features.json)")
    DATA["features"] = record["features"]
    DATA["arrays"] = {}
    DATA["label_residuals"] = {}
    DATA["vcc_draws"] = {}
    DATA["draws"] = {}
    for screen in SCREENS:
        genes = ext_table("G5", screen)["target_ensg"].astype(str).to_numpy()
        for cell in CELLS:
            if not np.array_equal(ext_table(cell, screen)["target_ensg"].astype(str).to_numpy(), genes):
                raise RuntimeError("the external %s tables of G5 and %s list different constructs" % (screen, cell))
        DATA["draws"][screen] = {"draw": resample.cluster_draw(genes, B, EXT_SEED[screen]), "point": e1s._ones(genes.size)}
    DATA["r5"] = json.loads((REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json").read_text())["r5_gate"]
    DATA["halves"] = {}
    for cell in CELLS:
        manifest = e1cells.manifest(cell)
        DATA["halves"][cell] = {}
        for screen in screens_of(cell):
            name = f"external_{screen}_halves.csv"
            path = e1cells.endpoint_root() / cell / name
            if sha256(path) != manifest["tables"][name]["sha256"]:
                raise RuntimeError("%s/%s does not match its manifest" % (cell, name))
            halves = pd.read_csv(path, float_precision="round_trip")
            a = halves[halves["half"] == "A"].set_index("construct")["y"]
            label = ext_table(cell, screen)["construct"].astype(str).map(a).to_numpy(dtype=np.float64)
            DATA["halves"][cell][screen] = (np.nan_to_num(label), np.isfinite(label))
    return {"e1b_fits": inputs, "e1b_predictions": file_record(e1_folder / "predictions_manifest.json"),
            "e4_predictions": file_record(e4_manifest), "features": file_record(OUT / "features.json"),
            "r5_gate": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json"),
            "mechanical": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "R2_within_and_mechanical.csv")}


def write(frame: pd.DataFrame, name: str) -> dict:
    frame.to_csv(OUT_DIR / name, index=False)
    return file_record(OUT_DIR / name)


def family_pass_check() -> dict:
    """The full pass's family values against the family pass's (E4 plan, Section 16, item 5)."""
    folder = OUT_DIR / "family_pass"
    if not (folder / "stats.json").exists():
        return {"family_pass": config.portable(folder), "passed": False, "note": "no family pass on file"}
    pairs = {"family.csv": ("family.csv", ["cell", "screen", "arm"], EQUALITY_EXCLUDED),
             "readings.csv": ("readings.csv", ["screen", "arm"], ()),
             "refits.csv": ("refits.csv", ["cell", "screen", "arm"], ()),
             "z_minus_c2.csv": ("secondary.csv", ["cell", "screen", "quantity"], ())}
    return {"family_pass": config.portable(folder), **fit_tiers.compare_passes(folder, OUT_DIR, pairs)}


def main() -> None:
    started = time.time()
    inputs = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flags = e1_flags()
    family = family_rows(flags)
    outputs = {"family.csv": write(family, "family.csv"), "readings.csv": write(reading_rows(family), "readings.csv")}
    outputs["refits.csv"] = write(refit_rows(family), "refits.csv")
    stats = {"plan": PLAN, "pass": PASS, "replicates": B, "seeds": {"external": EXT_SEED, "vcc": VCC_SEED, "nuisance folds": FOLD_SEED}}
    if PASS == "family":
        outputs["z_minus_c2.csv"] = write(z_minus_c2_rows(), "z_minus_c2.csv")
        corner = corner_check()
        corner["note"] = "the corner fits present in tiers 1 and 2: the corner z fits of G5 and G6 (E4 plan, Section 16, item 5)"
        stats.update({"e1_flags": {k: v for k, v in flags.items() if k in ("available", "files")}, "corner_check": corner,
                      "inputs": inputs, "outputs": outputs, "seconds": round(time.time() - started, 1)})
        (OUT_DIR / "stats.json").write_text(json.dumps(strict_json(stats), indent=1, allow_nan=False) + "\n")
        print("[e4 stats] family pass done in %.0f s; corner check %s (%d vectors)" % (
            time.time() - started, corner["passed"], corner["compared_vectors"]), flush=True)
        return
    outputs["secondary.csv"] = write(secondary_rows(), "secondary.csv")
    outputs["e1b_z_sensitivity.csv"] = write(e1b_z_sensitivity(), "e1b_z_sensitivity.csv")
    reproduction = printed_reproduction()
    if reproduction["reproduced"]:
        table, readings = section_6_4()
        outputs["section_6_4.csv"] = write(table, "section_6_4.csv")
        outputs["section_6_4_readings.csv"] = write(readings, "section_6_4_readings.csv")
    else:
        print("[e4 stats] the printed Section 6.4 values are not reproduced: %s; this part stops (gate G3)" % reproduction["differences"], flush=True)
    e1_stats = E1 / "stats.json"
    theirs = json.loads(e1_stats.read_text())["replicate_index_sha256"] if e1_stats.exists() else {}
    digests = {s: {"E4": DATA["draws"][s]["draw"].index_sha256, "E1b": theirs.get(s), "equal": DATA["draws"][s]["draw"].index_sha256 == theirs.get(s)}
               for s in SCREENS}
    check = family_pass_check()
    stats.update({"replicate_index_digests": digests, "e1_flags": {k: v for k, v in flags.items() if k in ("available", "files")},
                  "corner_check": corner_check(), "printed_reproduction": reproduction, "family_pass_check": check, "inputs": inputs,
                  "outputs": outputs, "seconds": round(time.time() - started, 1)})
    (OUT_DIR / "stats.json").write_text(json.dumps(strict_json(stats), indent=1, allow_nan=False) + "\n")
    print("[e4 stats] done in %.0f s; corner check %s; printed values reproduced %s; family pass equal %s" % (
        time.time() - started, stats["corner_check"]["passed"], reproduction["reproduced"], check["passed"]), flush=True)
    if not check["passed"]:
        print("[e4 stats] the full pass differs from the family pass: gate G3 (E4 plan, Section 16, item 5)", flush=True)
        raise SystemExit(3)


OUT_DIR = OUT

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--pass" in arguments:
        i = arguments.index("--pass")
        if i + 1 >= len(arguments) or arguments[i + 1] not in ("family", "full"):
            raise SystemExit("--pass takes family or full")
        PASS = arguments[i + 1]
        arguments = arguments[:i] + arguments[i + 2:]
    else:
        raise SystemExit("a statistics run takes --pass family or --pass full (bioinf2027/plans/E4.md, Section 16)")
    if arguments == ["--smoke"]:
        SMOKE = True
        B = 20
        resample.BLOCK = 20
        OUT_DIR = config.storage_root() / "bioinf2027" / "scratch" / "E4_stats_smoke"
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    if PASS == "family":
        OUT_DIR = OUT_DIR / "family_pass"
    main()
