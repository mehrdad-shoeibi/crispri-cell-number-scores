"""E6, the statistics, from the saved predictions of bioinf2027/scripts/make_e6_fits.py.

Design: bioinf2027/plans/E6.md, Sections 4, 6 and 7 as amended by Sections 13 to 15.

- One joint target-gene cluster bootstrap per target screen, with E1b's seeds (20260901 RPE1, 20260902
  K562-essential), 10,000 replicates, the target screen's constructs in construct order and target_ensg as
  the cluster; its replicate index digest is compared with E1b's.
- Gene-disjoint predictions: each target construct takes the prediction of the fold that holds its gene.
- The family of 24 at the primary endpoint (27/788) under the forest, Holm at 0.05: T and R of z, x, m,
  [x; m], R- of the masked arms and R(z-), in both directions. R(z-) is entered with p = 1 until D2's z-
  exists. Everything else is secondary and unadjusted.

- Markers of Section 7, as E1b defines them (bioinf2027/scripts/make_e1b_stats.py): "not robust to the
  gene-fold assignment" (the two alternative fold assignments at head seed 0 against the primary assignment
  at head seed 0, Section 13.1, item 2; departure 7 of bioinf2027/reports/E6.md, fixed before the full
  pass, so this column is not compared with the family pass and both values are listed); "not
  robust to fold assignment" (R, R- and R(z-) with E1b's other four nuisance fold seeds); "not robust to
  head fitting" (T, fewer than 80 % of the refits with its sign).
- Paired E6 - E1 (Section 6, item 6, with Section 13.1, items 3 and 4): the gene-disjoint predictions at
  all four endpoints against E1b's matching cell; at the 700-control endpoints, where E6's seed rule
  leaves head seed 0, E6's seed 0 is paired with E1b's seed 0 (Section 16, item 3); E1b's x at 27/788; the
  training-size-matched arm against G1 and G2.

Passes (Section 16): --pass family reads the predictions of tiers 1 and 2 (predictions_manifest_tier2.json)
and writes the family with its markers, the validity controls at 27/788, the readings, the refits, the
paired differences and the secondary rows the readings use (the raw B1 and B2 contrasts, z - C2) to
bioinf2027/results/E6/family_pass/; --pass full reads every tier (predictions_manifest_tier3.json), writes
everything and checks that its values equal the family pass's; a failed check is written to stats.json
and stops the run (gate G3).

- One-sided p-values of Section 4.1 (R >= 0.10, the [TR; arm] increment >= 0.03), where T is Holm-positive,
  R is not and the R5 gate passes; secondary, unadjusted, no statement (departure 8, added before the full
  pass): one_sided.csv.

Outputs under bioinf2027/results/E6/: family.csv, readings.csv, transfer.csv, secondary.csv,
size_matched.csv, oracle.csv, paired_with_E1.csv, refits.csv, folds.csv, non_shared.csv, one_sided.csv,
stats.json.
With --smoke: 20 replicates, the smoke predictions, outputs to the scratch folder, never reported.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, heads, resample, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.stats import percentile_interval  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import fit_tiers  # noqa: E402
import make_e6_fits as fits  # noqa: E402
import make_e6_input_records as input_records  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E6"
PLAN = "bioinf2027/plans/E6.md, Sections 4, 6 and 7 as amended by Sections 13 to 15"
B = 10000
SEED = {"RPE1": 20260901, "K562-essential": 20260902}
FOLD_SEED_R = 20270409
ALPHA = 0.05
POSITIVE_LIMIT = 0.80
DETECTABLE = 2.80
SCREENS = fits.SCREENS
FAMILY = transfer.FAMILY
MASKED = transfer.MASKED
SEEDS = fits.SEEDS
SMOKE = False
PASS = "full"
NUISANCE_SEEDS = (20270410, 20270411, 20270412, 20270413)  # E1b's other nuisance fold seeds (make_e1b_stats.py, FOLD_SEEDS)
MARGIN_R, MARGIN_INCREMENT = 0.10, 0.03  # E1_design.md, Section 23.2, as make_e1b_stats.py
MARKER_CHANGED_BY_THE_FIX = "not_robust_to_the_gene_fold_assignment"  # departure 7: its rule differs between the passes

DATA: dict = {}


# --------------------------------------------------------------------------- predictions


def load_predictions() -> None:
    folder = (fits.predictions_root().parent if SMOKE else OUT)
    path = folder / ("predictions_manifest_tier2.json" if PASS == "family" else "predictions_manifest_tier3.json")
    manifest = json.loads(path.read_text())
    DATA["files"] = manifest["files"]
    DATA["cache"] = {}
    DATA["r_cache"] = {}
    DATA["manifest"] = file_record(path)


def prediction(spec: FitSpec) -> dict | None:
    key = spec.key()
    if key in DATA["cache"]:
        return DATA["cache"][key]
    record = DATA["files"].get(key)
    if record is None:
        return None
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise RuntimeError("prediction file of %s does not match the manifest" % key)
    _, fit_record, values = transfer.load_predictions(path)
    values["_record"] = fit_record
    DATA["cache"][key] = values
    return values


def assembled(source: str, endpoint: str, model: str, head: str, seed: int, label: str = "rebuilt", basis: str = "primary",
              assignment: int = fits.FOLD_SEED, variant: str = "all training rows", draw: int | None = None, matched: str | None = None,
              side_suffix: str = "", c2_as_z: bool = False, suffix: str = "") -> np.ndarray | None:
    """Gene-disjoint predictions on every target construct (NaN where none), from the five folds."""
    target = fits.other(source)
    n = DATA["n"][target]
    out = np.full(n, np.nan)
    folds = DATA["folds"][assignment]
    genes = DATA["genes"][target]
    found = 0
    for k in range(5):
        name = fits.context_name(source, endpoint, f"fold {k}", basis, assignment, draw, matched) + (" | C2 as z" if c2_as_z else "") + suffix
        values = prediction(FitSpec(name, label, model, head, seed, variant))
        key = target + side_suffix
        if values is None or key not in values:
            if SMOKE:
                continue  # the smoke fits hold fold 0 only: the other folds stay missing
            return None
        found += 1
        rows = np.flatnonzero(folds_of(genes, folds) == k)
        if side_suffix:
            kept = DATA["disjoint_kept"][target][rows]
            out[rows[kept]] = values[key]
        else:
            out[rows] = values[key]
    return out if found else None


def whole_screen(source: str, endpoint: str, model: str, head: str, seed: int) -> np.ndarray | None:
    values = prediction(FitSpec(fits.context_name(source, endpoint, "whole screen"), "rebuilt", model, head, seed))
    return None if values is None else values[fits.other(source)]


def folds_of(genes: np.ndarray, folds: dict) -> np.ndarray:
    return np.array([folds[g] for g in genes])


# --------------------------------------------------------------------------- statistics


def rho(screen: str, vectors: list, label: np.ndarray, mask: np.ndarray | None = None):
    """Mean over seeds of the Spearman, point and replicates; None if a vector is missing."""
    if not vectors or any(v is None for v in vectors):
        return None
    usable = np.isfinite(np.column_stack(vectors)).all(axis=1)
    mask = usable if mask is None else (usable & mask)
    columns = np.column_stack([np.where(np.isfinite(v), v, 0.0) for v in vectors])
    point = resample.weighted_spearman(columns, label, DATA["draws"][screen]["point"], mask=mask)[0].mean()
    draws = resample.weighted_spearman(columns, label, DATA["draws"][screen]["draw"], mask=mask).mean(axis=1)
    return float(point), draws


def residual(values: np.ndarray, covariates: np.ndarray, genes: np.ndarray, seed: int = FOLD_SEED_R) -> np.ndarray:
    y = sps.rankdata(values) / (values.size + 1.0)
    unique = np.array(sorted(set(genes.tolist())))
    fold_of = dict(zip(unique[np.random.default_rng(seed).permutation(unique.size)], np.arange(unique.size) % 5))
    folds = np.array([fold_of[g] for g in genes])
    out = np.full(y.size, np.nan)
    with heads.single_thread():
        for f in range(5):
            test = folds == f
            model = heads.nuisance().fit(covariates[~test], y[~test])
            out[test] = y[test] - model.predict(covariates[test])
    return out


def r_stat(screen: str, vectors: list, label: np.ndarray, covariates: np.ndarray, genes: np.ndarray, seed: int = FOLD_SEED_R,
           mask: np.ndarray | None = None):
    if not vectors or any(v is None for v in vectors):
        return None
    use = np.isfinite(np.column_stack(vectors)).all(axis=1) & np.isfinite(label)
    if mask is not None:
        use &= mask
    e_y = np.full(label.size, 0.0)
    e_y[use] = residual(label[use], covariates[use], genes[use], seed)
    points, draws = [], []
    for v in vectors:
        e_p = np.full(label.size, 0.0)
        e_p[use] = residual(v[use], covariates[use], genes[use], seed)
        points.append(resample.weighted_pearson(e_p, e_y, DATA["draws"][screen]["point"], mask=use)[0, 0])
        draws.append(resample.weighted_pearson(e_p, e_y, DATA["draws"][screen]["draw"], mask=use)[:, 0])
    return float(np.mean(points)), np.mean(draws, axis=0)


def contrast(a, b):
    if a is None or b is None:
        return None
    return a[0] - b[0], a[1] - b[1]


def summary(value) -> dict:
    if value is None:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "se": float("nan"), "detectable_2.80se": float("nan")}
    lo, hi, _ = percentile_interval(value[1])
    se = float(np.nanstd(value[1], ddof=1))
    return {"estimate": value[0], "lo": lo, "hi": hi, "p": resample.p_value(value[1]), "se": se, "detectable_2.80se": DETECTABLE * se}


def label_of(screen: str, endpoint: str, column: str = "y") -> np.ndarray:
    return DATA["tables"][endpoint][screen][column].to_numpy(dtype=np.float64)


def covariates_of(screen: str, endpoint: str) -> np.ndarray:
    return DATA["tables"][endpoint][screen][list(e1cells.TRIVIAL)].to_numpy(dtype=np.float64)


def seeds_of(head: str, endpoint: str = fits.PRIMARY) -> tuple:
    if head in ("ridge-cv", "monotone") or SMOKE:
        return (0,)
    return tuple(DATA["seeds_700"]) if endpoint in ("27/700", "10/700") else SEEDS


def arm_rho(source, endpoint, model, head="forest", column="y", mask=None, **kw):
    target = fits.other(source)
    vectors = [assembled(source, endpoint, model, head, s, **kw) for s in seeds_of(head, endpoint)]
    return rho(target, vectors, label_of(target, endpoint, column), mask)


def arm_r(source, endpoint, model, head="forest", seed=FOLD_SEED_R, mask=None, **kw):
    target = fits.other(source)
    key = None if mask is not None else json.dumps([source, endpoint, model, head, seed, kw], sort_keys=True)
    if key is not None and key in DATA["r_cache"]:
        return DATA["r_cache"][key]
    vectors = [assembled(source, endpoint, model, head, s, **kw) for s in seeds_of(head, endpoint)]
    value = r_stat(target, vectors, label_of(target, endpoint), covariates_of(target, endpoint), DATA["genes"][target], seed, mask)
    if key is not None:
        DATA["r_cache"][key] = value
    return value


# --------------------------------------------------------------------------- tables


def family_table() -> pd.DataFrame:
    rows = []
    for source in SCREENS:
        direction = f"{source} to {fits.other(source)}"
        c1 = arm_rho(source, fits.PRIMARY, "C1")
        for arm in FAMILY:
            rows.append({"direction": direction, "statistic": "T", "arm": arm, **summary(contrast(arm_rho(source, fits.PRIMARY, arm), c1))})
            rows.append({"direction": direction, "statistic": "R", "arm": arm, **summary(arm_r(source, fits.PRIMARY, arm))})
        for arm in MASKED:
            rows.append({"direction": direction, "statistic": "R-", "arm": arm, **summary(arm_r(source, fits.PRIMARY, arm))})
        z_minus = arm_r(source, fits.PRIMARY, "z", suffix=" | z- of D2")
        entry = {"direction": direction, "statistic": "R(z-)", "arm": "z-", **summary(z_minus)}
        if z_minus is None:
            entry["p"] = 1.0
            entry["note"] = "D2's z- not yet available: entered with p = 1 (Section 4.1)"
        rows.append(entry)
    frame = pd.DataFrame(rows)
    rejected, thresholds = resample.holm(frame["p"].to_numpy(), ALPHA)
    frame["holm_threshold"] = thresholds
    frame["rejected"] = rejected
    frame["state"] = [resample.holm_state(e, r) for e, r in zip(frame["estimate"], rejected)]
    gate = DATA["r5"]
    frame["r5_gate_passed"] = [gate[f"G1 | {d.split(' to ')[1]}"]["passed"] for d in frame["direction"]]
    frame["read"] = [not (s.startswith("R") and not g) for s, g in zip(frame["statistic"], frame["r5_gate_passed"])]
    frame["family"] = "E6, 24 statistics, Holm at family-wise 0.05; at most 0.25 across E1, E4, D2 and E6"
    # gene-fold assignment sensitivity (Section 13.1, item 2): the primary assignment recomputed at head seed 0 and the
    # two other assignments at head seed 0, each state set against the Holm threshold of the statistic's step in the
    # primary ordering. The marker compares the other assignments' states with the primary assignment's state at head
    # seed 0, not with the family state over three head seeds (departure 7 of bioinf2027/reports/E6.md, fixed before
    # the full pass; the family pass, run before the fix, compared with the family state).
    primary_estimates, primary_states, other_states, other_estimates = [], [], [], []
    for row in frame.itertuples():
        source = row.direction.split(" to ")[0]
        primary = gene_fold_value(source, row.statistic, row.arm, fits.FOLD_SEED)
        others = [gene_fold_value(source, row.statistic, row.arm, a) for a in fits.FOLD_SENSITIVITY]
        primary_estimates.append(float("nan") if primary is None else primary[0])
        primary_states.append(assignment_state(primary, row.holm_threshold))
        other_states.append(";".join(assignment_state(v, row.holm_threshold) for v in others))
        other_estimates.append(";".join("not fitted" if v is None else repr(float(v[0])) for v in others))
    frame["estimate_primary_gene_fold_assignment_head_seed_0"] = primary_estimates
    frame["state_primary_gene_fold_assignment_head_seed_0"] = primary_states
    frame["states_other_gene_fold_assignments"] = other_states
    frame["estimates_other_gene_fold_assignments"] = other_estimates
    frame["not_robust_to_the_gene_fold_assignment"] = [r.state != "inconclusive" and p != "not fitted" and any(s not in (p, "not fitted") for s in o.split(";"))
                                                       for r, p, o in zip(frame.itertuples(), primary_states, other_states)]
    # fold assignment of R's nuisance model and head fitting (Section 7), as E1b defines them
    fold_states, shares = [], []
    for row in frame.itertuples():
        source = row.direction.split(" to ")[0]
        if row.statistic in ("R", "R-", "R(z-)"):
            states = []
            for seed in NUISANCE_SEEDS:
                if row.statistic == "R(z-)":
                    v = arm_r(source, fits.PRIMARY, "z", seed=seed, suffix=" | z- of D2")
                else:
                    v = arm_r(source, fits.PRIMARY, row.arm, seed=seed)
                if v is None:
                    states.append("not fitted")
                else:
                    p = resample.p_value(v[1])
                    states.append(resample.holm_state(v[0], bool(np.isfinite(p) and p <= row.holm_threshold)))
            fold_states.append(";".join(states))
            shares.append(float("nan"))
        else:
            fold_states.append("")
            values = refit_t_values(source, row.arm)
            shares.append(float((np.sign(values) == np.sign(row.estimate)).mean())
                          if values.size and np.isfinite(row.estimate) and row.estimate != 0 else float("nan"))
    frame["states_other_nuisance_fold_seeds"] = fold_states
    frame["share_of_refits_with_the_sign"] = shares
    frame["not_robust_to_fold_assignment"] = [r.state != "inconclusive" and fs != "" and any(x != r.state for x in fs.split(";"))
                                              for r, fs in zip(frame.itertuples(), fold_states)]
    frame["not_robust_to_head_fitting"] = [r.state != "inconclusive" and np.isfinite(sh) and sh < 0.8 for r, sh in zip(frame.itertuples(), shares)]
    return frame


def gene_fold_value(source, statistic, arm, assignment):
    """T or R of a family arm under one gene-fold assignment at head seed 0; None for R-, R(z-) and the masked arms."""
    if statistic == "T":
        return contrast(rho_seed0(source, arm, assignment), rho_seed0(source, "C1", assignment))
    if statistic == "R" and arm in FAMILY:
        return r_seed0(source, arm, assignment)
    return None


def assignment_state(value, threshold) -> str:
    """The state of a value against a given Holm threshold (Section 13.1, item 2)."""
    if value is None:
        return "not fitted"
    p = resample.p_value(value[1])
    return resample.holm_state(value[0], bool(p <= threshold))


def one_sided(value, margin: float) -> float:
    """p = (1 + #{theta*_b >= margin}) / (B + 1), as make_e1b_stats.reduces_to (E1_design.md, Section 23.2)."""
    if value is None:
        return float("nan")
    return (1 + int((value[1] >= margin).sum())) / (value[1].size + 1)


def one_sided_rows(family: pd.DataFrame) -> pd.DataFrame:
    """Section 4.1: where T is Holm-positive, R is not and the R5 gate passes, the one-sided p-values of E1_design.md,
    Section 23.2 (R >= 0.10; the external Spearman increment of [TR; arm] over TR >= 0.03), secondary and unadjusted,
    with no statement attached (departure 8 of bioinf2027/reports/E6.md, added before the full pass)."""
    state = {(r.direction, r.statistic, r.arm): r.state for r in family.itertuples()}
    gate = {r.direction: bool(r.r5_gate_passed) for r in family.itertuples()}
    rows = []
    for source in SCREENS:
        direction = f"{source} to {fits.other(source)}"
        for arm in FAMILY:
            t_positive = state[(direction, "T", arm)] == "Holm-positive"
            r_positive = state[(direction, "R", arm)] == "Holm-positive"
            considered = t_positive and not r_positive and gate[direction]
            entry = {"direction": direction, "arm": arm, "considered": considered,
                     "why_not_considered": "" if considered else ("the R5 gate fails" if not gate[direction]
                                                                  else "T not Holm-positive" if not t_positive else "R Holm-positive")}
            if considered:
                r = arm_r(source, fits.PRIMARY, arm)
                increment = contrast(arm_rho(source, fits.PRIMARY, f"[TR; {arm}]"), arm_rho(source, fits.PRIMARY, "TR"))
                entry.update({"R_estimate": float("nan") if r is None else r[0], f"p_R_at_least_{MARGIN_R:.2f}": one_sided(r, MARGIN_R),
                              "increment_estimate": float("nan") if increment is None else increment[0],
                              f"p_increment_at_least_{MARGIN_INCREMENT:.2f}": one_sided(increment, MARGIN_INCREMENT)})
            entry["adjustment"] = "secondary, unadjusted; no statement attached (bioinf2027/plans/E6.md, Section 4.1)"
            rows.append(entry)
    return pd.DataFrame(rows)


def rho_seed0(source, model, assignment):
    target = fits.other(source)
    return rho(target, [assembled(source, fits.PRIMARY, model, "forest", 0, assignment=assignment)], label_of(target, fits.PRIMARY))


def r_seed0(source, model, assignment):
    target = fits.other(source)
    v = assembled(source, fits.PRIMARY, model, "forest", 0, assignment=assignment)
    return r_stat(target, [v], label_of(target, fits.PRIMARY), covariates_of(target, fits.PRIMARY), DATA["genes"][target])


def positive_controls(endpoints=None) -> pd.DataFrame:
    rows = []
    for source in SCREENS:
        target = fits.other(source)
        for endpoint in endpoints or fits.ENDPOINTS:
            pc = arm_rho(source, endpoint, "label positive control", "monotone")
            entry = {"direction": f"{source} to {target}", "endpoint": endpoint, "control": "label construction", **summary(pc)}
            vector = assembled(source, endpoint, "label positive control", "monotone", 0)
            if vector is not None:
                finite = np.isfinite(vector)
                slope, intercept, r, _, _ = sps.linregress(vector[finite], label_of(target, endpoint)[finite])
                entry.update({"calibration_slope": slope, "calibration_intercept": intercept, "calibration_r2": r * r})
            entry["below_0.80"] = bool(np.isfinite(entry["estimate"]) and entry["estimate"] < POSITIVE_LIMIT)
            rows.append(entry)
            for name in ("[C1; B2]", "[z; B2]"):
                v = arm_rho(source, endpoint, name)
                rows.append({"direction": f"{source} to {target}", "endpoint": endpoint, "control": f"pipeline {name}", **summary(v),
                             "below_0.80": bool(v is not None and v[0] < POSITIVE_LIMIT)})
    return pd.DataFrame(rows)


MASKED_OF = {"x": "x-", "m": "m-", "[x; m]": "[x; m]-"}
CACHE_NOTE = ("z of E4's builder, from the per-cell cache; the stored z of the main analysis differ from it by one shift shared by "
              "every construct (E4 plan, Section 15; bioinf2027/diag/E4_external_z.json)")
UNMASKED_OF = {v: k for k, v in MASKED_OF.items()}


def readings(family: pd.DataFrame, positive: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Section 7, per direction: validity rules 2 to 4, then the table per arm, z read separately (stop rules: Section 8, recorded apart)."""
    rows = []
    state = {(r.direction, r.statistic, r.arm): r.state for r in family.itertuples()}
    fragile = {(r.direction, r.statistic, r.arm): bool(r.not_robust_to_the_gene_fold_assignment) for r in family.itertuples()}
    fold_marker = {(r.direction, r.statistic, r.arm): bool(r.not_robust_to_fold_assignment) for r in family.itertuples()}
    head_marker = {(r.direction, r.statistic, r.arm): bool(r.not_robust_to_head_fitting) for r in family.itertuples()}
    for source in SCREENS:
        direction = f"{source} to {fits.other(source)}"
        pc = positive[(positive.direction == direction) & (positive.endpoint == fits.PRIMARY)]

        def below(control):
            part = pc[pc.control == control]
            return bool(part["below_0.80"].iloc[0]) if len(part) else None

        label_ok = below("label construction") is False
        head_limited = bool(below("pipeline [C1; B2]"))
        z_masks = bool(below("pipeline [z; B2]")) and not head_limited
        gate = bool(DATA["r5"][f"G1 | {fits.other(source)}"]["passed"])
        z_minus = state[(direction, "R(z-)", "z-")]
        for arm in FAMILY:
            t, r = state[(direction, "T", arm)], state[(direction, "R", arm)]
            notes = []
            if not label_ok:
                statement = "no reading: the monotone positive control is below +0.80 (or missing) in this direction"
            elif t == "Holm-negative":
                statement = ("read as limited by the head ([C1; B2 native] below +0.80)" if head_limited
                             else "the arm ranks the target constructs below the Gaussian control")
            elif t == "inconclusive":
                statement = "neither transfer nor its failure is resolved; the interval bounds it" + (
                    "; read as limited by the head ([C1; B2 native] below +0.80)" if head_limited else "")
            elif not gate:
                statement = "transfer above the Gaussian control is resolved; whether it goes beyond the trivial predictors is not answerable on this endpoint"
            elif r == "Holm-positive":
                if arm == "z" and z_minus != "Holm-positive":
                    statement = ("transfer above the Gaussian control is resolved; R is Holm-positive but R(z-) is not, so the statement "
                                 "beyond the trivial predictors is not made for z")
                else:
                    statement = ("trained on the source screen, the arm ranks the target screen's constructs of untrained target genes above "
                                 "the Gaussian control and beyond the specified trivial predictors")
                    if arm in MASKED_OF:
                        masked = state[(direction, "R-", MASKED_OF[arm])]
                        statement += ("; R- of the masked arm is also Holm-positive" if masked == "Holm-positive" else
                                      "; not evidence against reading the target's own coordinate, since R- of the masked arm is not Holm-positive")
            elif r == "Holm-negative":
                statement = "transfer above the Gaussian control is resolved; the prediction runs against the label once the trivial predictors are removed"
            else:
                statement = ("transfer above the Gaussian control is resolved; it is not shown to exceed the specified trivial predictors "
                             "(R with its interval and 2.80 times its standard error)")
            if arm == "z" and z_masks:
                notes.append("[z; B2 native] is below +0.80 while [C1; B2 native] is not: z masks the signal under this head")
            for statistic in ("T", "R"):
                if fragile.get((direction, statistic, arm)):
                    notes.append(f"{statistic} not robust to the gene-fold assignment")
            if fold_marker.get((direction, "R", arm)):
                notes.append("R not robust to fold assignment")
            if arm == "z" and fold_marker.get((direction, "R(z-)", "z-")):
                notes.append("R(z-) not robust to fold assignment")
            if head_marker.get((direction, "T", arm)):
                notes.append("T not robust to head fitting")
            for baseline in ("B1", "B2"):
                part = secondary[(secondary.direction == direction) & (secondary.contrast == f"{arm} minus the target screen's raw {baseline}")]
                if len(part) and part["estimate"].iloc[0] <= 0:
                    notes.append(f"the untrained {baseline} of the target screen is at least as high: no 'transfer works' wording")
            rows.append({"direction": direction, "arm": arm, "T": t, "R": r, "R(z-)": z_minus if arm == "z" else "",
                         "label_positive_control_passed": label_ok, "head_limited": head_limited, "r5_gate_passed": gate,
                         "statement": statement, "markers": "; ".join(notes)})
        for masked in MASKED:
            r_minus, t_unmasked = state[(direction, "R-", masked)], state[(direction, "T", UNMASKED_OF[masked])]
            if not label_ok:
                statement = "no reading: the monotone positive control is below +0.80 (or missing) in this direction"
            elif not gate:
                statement = "not answerable on this endpoint (R5 gate)"
            elif r_minus == "Holm-positive":
                statement = ("information beyond the trivial predictors without a resolved transfer; no transfer statement"
                             if t_unmasked != "Holm-positive" else "R- Holm-positive; read with the unmasked arm's statement")
            elif r_minus == "Holm-negative":
                statement = "the masked arm's prediction runs against the label once the trivial predictors are removed"
            else:
                statement = "not resolved"
            rows.append({"direction": direction, "arm": masked, "T": "", "R": r_minus, "R(z-)": "", "label_positive_control_passed": label_ok,
                         "head_limited": head_limited, "r5_gate_passed": gate, "statement": statement,
                         "markers": "; ".join([m for m, on in (("R- not robust to the gene-fold assignment", fragile.get((direction, "R-", masked))),
                                                                  ("R- not robust to fold assignment", fold_marker.get((direction, "R-", masked)))) if on])})
    return pd.DataFrame(rows)


def transfer_rows() -> pd.DataFrame:
    rows = []
    for source in SCREENS:
        target = fits.other(source)
        feature = DATA["target_is_feature"][target]
        for endpoint in fits.ENDPOINTS:
            for model in fits.MODELS:
                for head in ("forest", "boosting", "ridge-cv"):
                    masks = {"all": None, "target is a feature": feature, "target is not a feature": ~feature,
                             "target gene not held by the source screen": DATA["non_shared"][target]} if head == "forest" else {"all": None}
                    for subset, mask in masks.items():
                        v = arm_rho(source, endpoint, model, head, mask=mask)
                        if v is None:
                            continue
                        c1 = arm_rho(source, endpoint, "C1", head, mask=mask)
                        entry = {"direction": f"{source} to {target}", "endpoint": endpoint, "model": model, "head": head, "subset": subset,
                                 "spearman": v[0], **{f"spearman_{k}": x for k, x in summary(v).items() if k in ("lo", "hi")}}
                        if model != "C1":
                            entry.update({f"minus_C1_{k}": x for k, x in summary(contrast(v, c1)).items()})
                        if head == "forest" and subset == "all":
                            w = [whole_screen(source, endpoint, model, head, s) for s in seeds_of(head, endpoint)]
                            whole = rho(target, w, label_of(target, endpoint))
                            if whole is not None:
                                entry["whole_screen_spearman"] = whole[0]
                                entry.update({f"gene_identity_contribution_{k}": x for k, x in summary(contrast(whole, v)).items()})
                        entry["role"] = "family" if (endpoint == fits.PRIMARY and model in FAMILY + ("C1",) and head == "forest" and subset == "all") else "secondary, unadjusted"
                        rows.append(entry)
    return pd.DataFrame(rows)


FAMILY_PASS_SECONDARY = tuple(f"{arm} minus the target screen's raw {b}" for arm in FAMILY for b in ("B1", "B2")) + ("z minus C2",)


def secondary_rows(family_pass: bool = False) -> pd.DataFrame:
    """Every secondary contrast; with family_pass, only those the readings use and z - C2 (Section 16)."""
    rows = []

    def add(direction, name, value, **extra):
        if family_pass and name not in FAMILY_PASS_SECONDARY:
            return
        if value is not None:
            rows.append({"direction": direction, "contrast": name, **summary(value), "adjustment": "unadjusted", **extra})

    for source in SCREENS:
        target = fits.other(source)
        direction = f"{source} to {target}"
        label = label_of(target, fits.PRIMARY)
        for arm in FAMILY:
            a = arm_rho(source, fits.PRIMARY, arm)
            add(direction, f"{arm} minus the target screen's raw B1", contrast(a, raw(target, "B1")))
            add(direction, f"{arm} minus the target screen's raw B2", contrast(a, raw(target, "B2")))
            add(direction, f"{arm} minus the B1 model trained on the source", contrast(a, arm_rho(source, fits.PRIMARY, "B1")))
            add(direction, f"{arm} minus TR", contrast(a, arm_rho(source, fits.PRIMARY, "TR")))
            add(direction, f"[TR; {arm}] minus TR", contrast(arm_rho(source, fits.PRIMARY, f"[TR; {arm}]"), arm_rho(source, fits.PRIMARY, "TR")))
            for basis in ("e1b", "shared genes"):
                if arm in ("x", "m", "[x; m]"):
                    add(direction, f"{arm}, basis {basis}", arm_rho(source, fits.PRIMARY, arm, basis=basis))
            add(direction, f"T of {arm}, mean-across-draws label", contrast(arm_rho(source, fits.PRIMARY, arm, column="y_mean", label="mean"),
                                                                          arm_rho(source, fits.PRIMARY, "C1", column="y_mean", label="mean")))
            for variant in ("draw 1", "draw 2", "draw 3"):
                m = rho(target, [assembled(source, fits.PRIMARY, arm, "forest", 0, matched=variant)], label)
                c = rho(target, [assembled(source, fits.PRIMARY, "C1", "forest", 0, matched=variant)], label)
                add(direction, f"T of {arm}, E4 matched features {variant}", contrast(m, c), note=CACHE_NOTE if arm == "z" else "")
            if "disjoint_label" in DATA and target in DATA["disjoint_label"]:
                half_label, half_cov, kept = DATA["disjoint_label"][target]
                v = [assembled(source, fits.PRIMARY, arm, "forest", s, side_suffix=" cell-disjoint") for s in SEEDS]
                add(direction, f"{arm}, cell-disjoint (half B features, half A label)", rho(target, v, half_label, kept),
                    note=CACHE_NOTE if arm == "z" else "")
                add(direction, f"R of {arm}, cell-disjoint", r_stat(target, v, half_label, half_cov, DATA["genes"][target], mask=kept))
        for masked, arm in (("x-", "x"), ("m-", "m"), ("[x; m]-", "[x; m]")):
            feature = DATA["target_is_feature"][target]
            for subset, mask in (("all", None), ("target is a feature", feature), ("target is not a feature", ~feature)):
                add(direction, f"{arm} minus {masked}, {subset}", contrast(arm_rho(source, fits.PRIMARY, arm, mask=mask), arm_rho(source, fits.PRIMARY, masked, mask=mask)))
        for other_arm in ("x", "C2", "[x; m]"):
            add(direction, f"z minus {other_arm}", contrast(arm_rho(source, fits.PRIMARY, "z"), arm_rho(source, fits.PRIMARY, other_arm)))
        for seed in (20270410, 20270411, 20270412, 20270413):
            for arm in FAMILY + MASKED:
                add(direction, f"R of {arm}, nuisance fold seed {seed}", arm_r(source, fits.PRIMARY, arm, seed=seed))
        add(direction, "C2 of the shared-gene basis", rho(target, [assembled(source, fits.PRIMARY, "z", "forest", s, basis="shared genes", c2_as_z=True) for s in SEEDS], label))
    return pd.DataFrame(rows)


def raw(screen: str, column: str):
    table = DATA["tables"][fits.PRIMARY][screen]
    return rho(screen, [table[column].to_numpy(dtype=np.float64)], table["y"].to_numpy(dtype=np.float64))


def size_matched_rows(family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SCREENS:
        target = fits.other(source)
        for endpoint in fits.MATCHED_GENES:
            label = label_of(target, endpoint)
            for arm in FAMILY:
                ts, rs, points = [], [], []
                for draw in range(fits.DRAWS):
                    a = rho(target, [assembled(source, endpoint, arm, "forest", s, draw=draw) for s in SEEDS], label)
                    c = rho(target, [assembled(source, endpoint, "C1", "forest", s, draw=draw) for s in SEEDS], label)
                    t = contrast(a, c)
                    if t is None:
                        continue
                    ts.append(t)
                    r = r_stat(target, [assembled(source, endpoint, arm, "forest", s, draw=draw) for s in SEEDS], label,
                               covariates_of(target, endpoint), DATA["genes"][target])
                    rs.append(r)
                if not ts:
                    continue
                t_mean = (float(np.mean([t[0] for t in ts])), np.mean([t[1] for t in ts], axis=0))
                r_mean = (float(np.mean([r[0] for r in rs])), np.mean([r[1] for r in rs], axis=0))
                full = contrast(arm_rho(source, endpoint, arm), arm_rho(source, endpoint, "C1"))
                share = float(np.mean([np.sign(t[0]) == np.sign(full[0]) for t in ts])) if full is not None else float("nan")
                rows.append({"direction": f"{source} to {target}", "endpoint": endpoint, "arm": arm, "training_genes": fits.MATCHED_GENES[endpoint],
                             "draws": len(ts), **{f"T_{k}": v for k, v in summary(t_mean).items()}, **{f"R_{k}": v for k, v in summary(r_mean).items()},
                             "share_of_draws_with_the_sign_of_the_full_T": share})
    return pd.DataFrame(rows)


def e1b_manifest() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E1b_fits_smoke" / "predictions_manifest.json"
    return REPO / "bioinf2027" / "results" / "E1" / "predictions_manifest.json"


def paired_with_e1() -> pd.DataFrame:
    """E6's predictions against E1b's external predictions of the matching cell (Section 6, item 6; Section 13.1, items 3 and 4)."""
    path = e1b_manifest()
    if not path.exists():
        return pd.DataFrame()
    manifest = json.loads(path.read_text())["files"]
    named = "training genes, rows per gene, cell line and laboratory, depth, row unit, guide design, feature namespace, gene universe"
    rows = []
    for endpoint, cell in fits.ENDPOINTS.items():
        seeds = seeds_of("forest", endpoint)
        pairing = ("E6's head seed 0 with E1b's head seed 0 (E6's seed rule at the 700-control endpoints; E6 plan, Section 16, item 3)"
                   if endpoint in ("27/700", "10/700") and seeds == (0,) and not SMOKE else
                   "head seed 0 on both sides (smoke)" if SMOKE else "head seeds 0 to 2 on both sides, averaged")
        for source in SCREENS:
            target = fits.other(source)
            label = label_of(target, endpoint)

            def e1(model):
                vectors = []
                for seed in seeds:
                    record = manifest.get(FitSpec(cell, "rebuilt", model, "forest", seed).key())
                    if record is None:
                        return None
                    file = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
                    if sha256(file) != record["sha256"]:
                        raise RuntimeError("E1b's prediction file of %s does not match its manifest" % cell)
                    _, _, values = transfer.load_predictions(file)
                    vectors.append(values[target])
                return rho(target, vectors, label)

            def entry(arm, e6_predictions, spearman, t_diff, **extra):
                return {"endpoint": endpoint, "e1_cell": cell, "target": target, "arm": arm, "e6_predictions": e6_predictions,
                        **{f"spearman_E6_minus_E1_{k}": v for k, v in summary(spearman).items()},
                        **{f"T_E6_minus_E1_{k}": v for k, v in summary(t_diff).items()},
                        "head_seeds": pairing, "differences_named": named, **extra}

            e1_values = {arm: e1(arm) for arm in FAMILY + ("C1", "C2")}
            c1_e1, c1_e6 = e1_values["C1"], arm_rho(source, endpoint, "C1")
            for arm in FAMILY + ("C2",):
                a6, a1 = arm_rho(source, endpoint, arm), e1_values[arm]
                rows.append(entry(arm, "gene-disjoint, primary x", contrast(a6, a1), contrast(contrast(a6, c1_e6), contrast(a1, c1_e1))))
            if endpoint == fits.PRIMARY:
                # Section 13.1, item 4: the pairing for x, m, [x; m] and C2 with E1b's x as E6's features
                for arm in ("x", "m", "[x; m]", "C2"):
                    a6, a1 = arm_rho(source, endpoint, arm, basis="e1b"), e1_values[arm]
                    rows.append(entry(arm, "gene-disjoint, E1b's x (each screen's own mapped features)", contrast(a6, a1),
                                      contrast(contrast(a6, c1_e6), contrast(a1, c1_e1)),
                                      note="the namespace difference to the primary-x rows: E6's primary x keeps the features mapped in both "
                                           "screens; here each screen's own mapping, as E1b; C1 does not depend on x"))
            if endpoint in fits.MATCHED_GENES:
                # Section 13.1, item 3: the training-size-matched arm, paired with G1 and G2 only
                for arm in FAMILY:
                    a1 = e1_values[arm]
                    spearman, t_diff = [], []
                    for draw in range(fits.DRAWS):
                        a_d = rho(target, [assembled(source, endpoint, arm, "forest", s, draw=draw) for s in seeds], label)
                        c_d = rho(target, [assembled(source, endpoint, "C1", "forest", s, draw=draw) for s in seeds], label)
                        if a_d is None or c_d is None or a1 is None or c1_e1 is None:
                            continue
                        spearman.append(contrast(a_d, a1))
                        t_diff.append(contrast(contrast(a_d, c_d), contrast(a1, c1_e1)))
                    mean = (lambda vs: (float(np.mean([v[0] for v in vs])), np.mean([v[1] for v in vs], axis=0)) if vs else None)
                    rows.append(entry(arm, "training-size-matched arm (%d training genes), mean over its draws" % fits.MATCHED_GENES[endpoint],
                                      mean(spearman), mean(t_diff), draws=len(spearman)))
    return pd.DataFrame(rows)


def oracle_rows() -> pd.DataFrame:
    rows = []
    for source in SCREENS:
        target = fits.other(source)
        s_table, t_table = DATA["tables"][fits.PRIMARY][source], DATA["tables"][fits.PRIMARY][target]
        s_genes = s_table.groupby("target_ensg")
        s_label = s_genes["y"].median()
        s_b = s_genes[list(e1cells.TRIVIAL)].median()
        genes = t_table["target_ensg"].astype(str)
        held = genes.isin(s_label.index).to_numpy()
        label = t_table["y"].to_numpy(dtype=np.float64)
        for name, predictor in (("source label of the same gene", genes.map(s_label).fillna(0).to_numpy()),
                                ("source control mean of the same gene", genes.map(s_b["B1"]).fillna(0).to_numpy())):
            rows.append({"direction": f"{source} to {target}", "reference": name, "constructs": int(held.sum()), "statistic": "Spearman",
                         **summary(rho(target, [predictor], label, held))})
            rows.append({"direction": f"{source} to {target}", "reference": name, "constructs": int(held.sum()), "statistic": "R",
                         **summary(r_stat(target, [predictor], label, covariates_of(target, fits.PRIMARY), DATA["genes"][target], mask=held))})
        s_cov = np.column_stack([genes.map(s_b[c]).fillna(0).to_numpy() for c in e1cells.TRIVIAL])
        ten = np.column_stack([covariates_of(target, fits.PRIMARY), s_cov])
        rows.append({"direction": f"{source} to {target}", "reference": "partial of the two labels given both screens' B0 to B4", "constructs": int(held.sum()),
                     "statistic": "R", **summary(r_stat(target, [genes.map(s_label).fillna(0).to_numpy()], label, ten, DATA["genes"][target], mask=held))})
        rows[-1]["note"] = "the source label shares the guide design with the target construct; not a ceiling for a representation (Section 13.1, item 10)"
    return pd.DataFrame(rows)


def refit_t_values(source: str, arm: str) -> np.ndarray:
    """T per refit (forest, seed 0, Section 15, item 6), for the refits found."""
    target = fits.other(source)
    label = label_of(target, fits.PRIMARY)
    values = []
    for r in range(DATA["refits"]):
        a = rho(target, [assembled(source, fits.PRIMARY, arm, "forest", 0, variant=f"refit {r}")], label)
        c = rho(target, [assembled(source, fits.PRIMARY, "C1", "forest", 0, variant=f"refit {r}")], label)
        if a is not None and c is not None:
            values.append(a[0] - c[0])
    return np.array(values)


def refit_rows(family: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SCREENS:
        target = fits.other(source)
        for arm in FAMILY:
            values = refit_t_values(source, arm)
            if not values.size:
                continue
            estimate = family[(family.direction == f"{source} to {target}") & (family.statistic == "T") & (family.arm == arm)]["estimate"].iloc[0]
            rows.append({"direction": f"{source} to {target}", "arm": arm, "refits": int(values.size), "T": estimate,
                         "p2.5": float(np.percentile(values, 2.5)), "p97.5": float(np.percentile(values, 97.5)),
                         "share_with_the_sign_of_T": float((np.sign(values) == np.sign(estimate)).mean())})
    return pd.DataFrame(rows)


def fold_rows() -> pd.DataFrame:
    """Within-fold Spearman for C1 and each arm, and the pooled Spearman with predictions centred within fold (Section 13.1, item 8)."""
    rows = []
    for source in SCREENS:
        target = fits.other(source)
        label = label_of(target, fits.PRIMARY)
        folds = folds_of(DATA["genes"][target], DATA["folds"][fits.FOLD_SEED])
        for arm in FAMILY + ("C1",):
            vectors = [assembled(source, fits.PRIMARY, arm, "forest", s) for s in SEEDS]
            if any(v is None for v in vectors):
                continue
            mean = np.mean(vectors, axis=0)
            centred = mean.copy()
            for k in range(5):
                centred[folds == k] -= mean[folds == k].mean()
            entry = {"direction": f"{source} to {target}", "arm": arm, "pooled_centred_within_fold": float(sps.spearmanr(centred, label).statistic)}
            for k in range(5):
                entry[f"fold_{k}"] = float(sps.spearmanr(mean[folds == k], label[folds == k]).statistic)
            rows.append(entry)
    return pd.DataFrame(rows)


def non_shared_rows() -> pd.DataFrame:
    rows = []
    for target in SCREENS:
        source = fits.other(target)
        causes = DATA["non_shared_causes"][target]
        for cause, count in causes.items():
            rows.append({"target": target, "source": source, "cause": cause, "constructs": count})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- inputs


def load() -> dict:
    inputs = fits.load_inputs()
    DATA["tables"] = fits.DATA["tables"]
    DATA["folds"] = fits.DATA["folds"]
    DATA["n"] = {s: len(DATA["tables"][fits.PRIMARY][s]) for s in SCREENS}
    DATA["genes"] = {s: DATA["tables"][fits.PRIMARY][s]["target_ensg"].astype(str).to_numpy() for s in SCREENS}
    DATA["target_is_feature"] = {s: fits.DATA[("x", s)]["primary"][1] >= 0 for s in SCREENS}
    held = {s: set(DATA["genes"][s].tolist()) for s in SCREENS}
    DATA["non_shared"] = {t: ~np.isin(DATA["genes"][t], list(held[fits.other(t)])) for t in SCREENS}
    DATA["non_shared_causes"] = non_shared_causes()
    DATA["draws"] = {}
    for screen in SCREENS:
        draw = resample.cluster_draw(DATA["genes"][screen], B, SEED[screen])
        DATA["draws"][screen] = {"draw": draw, "point": resample.ClusterDraw(weights=np.ones((1, DATA["n"][screen])), clusters=np.array([]), seed=-1, index_sha256="")}
    DATA["r5"] = json.loads((REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json").read_text())["r5_gate"]
    checks = {} if SMOKE else json.loads((OUT / "checks.json").read_text())
    DATA["refits"] = 1 if SMOKE else int(checks["refits"])
    DATA["seeds_700"] = (0,) if SMOKE else tuple(checks["seeds_of_the_700_control_endpoints"])
    if fits.DATA.get("e4"):
        DATA["disjoint_kept"] = {s: fits.DATA["e4"][(s, "cell-disjoint")][2] for s in SCREENS}
        DATA["disjoint_label"] = disjoint_labels()
    else:
        DATA["disjoint_kept"] = {s: np.zeros(DATA["n"][s], bool) for s in SCREENS}
    return inputs


def disjoint_labels() -> dict:
    """Half A's label and B0 to B4 at the G1 endpoint (27 cells, pooled control halves), in construct order."""
    out = {}
    manifest = e1cells.manifest("G1")
    for screen in SCREENS:
        name = f"external_{screen}_halves.csv"
        path = e1cells.endpoint_root() / "G1" / name
        if sha256(path) != manifest["tables"][name]["sha256"]:
            raise RuntimeError("%s does not match its manifest" % name)
        halves = pd.read_csv(path, float_precision="round_trip")
        a = halves[halves["half"] == "A"].set_index("construct")
        constructs = DATA["tables"][fits.PRIMARY][screen]["construct"].astype(str)
        label = constructs.map(a["y"]).to_numpy(dtype=np.float64)
        cov = np.column_stack([constructs.map(a[c]).to_numpy(dtype=np.float64) for c in e1cells.TRIVIAL])
        kept = DATA["disjoint_kept"][screen] & np.isfinite(label) & np.isfinite(cov).all(axis=1)
        out[screen] = (np.nan_to_num(label), np.nan_to_num(cov), kept)
    return out


def non_shared_causes() -> dict:
    """Why a target gene is not held by the source screen (Section 3.1)."""
    import h5py

    from bioinf2027 import e1inputs, singlecell

    out = {}
    for target in SCREENS:
        source = fits.other(target)
        eligibility = pd.read_csv(config.screen_paths()[source]["eligibility"])
        with h5py.File(e1inputs.paths()[f"{source}_single_cell"], "r") as handle:
            var = handle["var"]
            index = var.attrs["_index"]
            in_matrix = set(singlecell.read_annotation(var, index.decode() if isinstance(index, bytes) else str(index)).tolist())
        construct_gene = eligibility["construct"].astype(str).str.extract(r"(ENSG\d+)$")[0]
        cells = eligibility.groupby(construct_gene)["n_perturbed_cells"].max()
        counts: dict = {}
        for gene in DATA["genes"][target][DATA["non_shared"][target]]:
            if gene not in cells.index:
                cause = "no construct of the source library targets the gene"
            else:
                absent = gene not in in_matrix
                few = bool(cells[gene] < 27)
                cause = "both" if absent and few else ("absent from the source gene matrix" if absent else ("below the 27-cell floor" if few else "other"))
            counts[cause] = counts.get(cause, 0) + 1
        out[target] = counts
    return out


def family_pass_check() -> dict:
    """The full pass's values against the family pass's (E6 plan, Section 16, item 4)."""
    folder = OUT_DIR / "family_pass"
    if not (folder / "stats.json").exists():
        return {"family_pass": config.portable(folder), "passed": False, "note": "no family pass on file"}
    # The marker's rule changed between the passes (departure 7), so its column is not compared; both values are
    # listed instead, so that any change shows. Every other column of family.csv is compared.
    pairs = {"family.csv": ("family.csv", ["direction", "statistic", "arm"], (MARKER_CHANGED_BY_THE_FIX,)),
             "positive_controls.csv": ("positive_controls.csv", ["direction", "endpoint", "control"], ()),
             "readings.csv": ("readings.csv", ["direction", "arm"], ()),
             "refits.csv": ("refits.csv", ["direction", "arm"], ()),
             "paired_with_E1.csv": ("paired_with_E1.csv", ["endpoint", "target", "arm", "e6_predictions"], ()),
             "secondary.csv": ("secondary.csv", ["direction", "contrast"], ())}
    keys = ["direction", "statistic", "arm"]
    before = pd.read_csv(folder / "family.csv").set_index(keys)[MARKER_CHANGED_BY_THE_FIX]
    after = pd.read_csv(OUT_DIR / "family.csv").set_index(keys)[MARKER_CHANGED_BY_THE_FIX].reindex(before.index)
    marker = [{"direction": d, "statistic": s, "arm": a, "family_pass": bool(x), "full_pass": None if pd.isna(y) else bool(y)}
              for (d, s, a), x, y in zip(before.index, before.to_numpy(), after.to_numpy()) if pd.isna(y) or bool(x) != bool(y)]
    return {"family_pass": config.portable(folder), **fit_tiers.compare_passes(folder, OUT_DIR, pairs),
            "marker_rule_changed": {"column": MARKER_CHANGED_BY_THE_FIX, "declared": "departure 7 of bioinf2027/reports/E6.md",
                                    "rows_that_differ": marker}}


def verify_inputs() -> dict:
    """Stop rule 1 (plan Section 3.4): hash every data-root input of the fits and the statistics and stop if one
    differs from its record (bioinf2027/scripts/make_e6_input_records.py lists them; audit round 1, check 5)."""
    out = {}
    for name, path, record_file, entry, _ in input_records.listed():
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise SystemExit("%s differs from %s (stop rule 1); no statistic is computed" % (name, record_file))
        out[name] = {"path": config.portable(path), "bytes": path.stat().st_size, "sha256": digest, "record": record_file}
    return out


def main() -> None:
    started = time.time()
    checked_inputs = None if SMOKE else verify_inputs()
    load_predictions()
    inputs = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    family_pass = PASS == "family"
    family = family_table()
    outputs = {"family.csv": write(family, "family.csv")}
    positive = positive_controls((fits.PRIMARY,) if family_pass else None)
    outputs["positive_controls.csv"] = write(positive, "positive_controls.csv")
    secondary = secondary_rows(family_pass)
    outputs["secondary.csv"] = write(secondary, "secondary.csv")
    outputs["readings.csv"] = write(readings(family, positive, secondary), "readings.csv")
    outputs["refits.csv"] = write(refit_rows(family), "refits.csv")
    outputs["paired_with_E1.csv"] = write(paired_with_e1(), "paired_with_E1.csv")
    if not family_pass:
        outputs["transfer.csv"] = write(transfer_rows(), "transfer.csv")
        outputs["size_matched.csv"] = write(size_matched_rows(family), "size_matched.csv")
        outputs["oracle.csv"] = write(oracle_rows(), "oracle.csv")
        outputs["folds.csv"] = write(fold_rows(), "folds.csv")
        outputs["one_sided.csv"] = write(one_sided_rows(family), "one_sided.csv")
        outputs["non_shared.csv"] = write(non_shared_rows(), "non_shared.csv")
    e1_stats = REPO / "bioinf2027" / "results" / "E1" / "stats.json"
    digest_check = {}
    for screen in SCREENS:
        ours = DATA["draws"][screen]["draw"].index_sha256
        theirs = json.loads(e1_stats.read_text())["replicate_index_sha256"].get(screen) if e1_stats.exists() else None
        digest_check[screen] = {"E6": ours, "E1b": theirs, "equal": ours == theirs}
    stats = {"plan": PLAN, "pass": PASS, "replicates": B, "seeds": SEED,
             "nuisance_fold_seeds": {"R": FOLD_SEED_R, "robustness": list(NUISANCE_SEEDS)},
             "replicate_index_digests": digest_check,
             "inputs": {**inputs, "predictions_manifest": DATA["manifest"],
                        "e1b_predictions_manifest": file_record(e1b_manifest()) if e1b_manifest().exists() else None,
                        "data_root_inputs": checked_inputs,
                        "readouts": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json"),
                        "e1b_stats": file_record(e1_stats) if e1_stats.exists() else None},
             "outputs": outputs}
    if not family_pass:
        stats["family_pass_check"] = family_pass_check()
    stats["seconds"] = round(time.time() - started, 1)
    (OUT_DIR / "stats.json").write_text(json.dumps(strict_json(stats), indent=1, allow_nan=False) + "\n")
    print("[e6 stats] %s pass done in %.0f s%s" % (PASS, time.time() - started,
          "" if family_pass else "; family pass equal %s" % stats["family_pass_check"]["passed"]), flush=True)
    if not family_pass and not stats["family_pass_check"]["passed"]:
        print("[e6 stats] the full pass differs from the family pass: gate G3 (E6 plan, Section 16, item 4)", flush=True)
        raise SystemExit(3)


def write(frame: pd.DataFrame, name: str) -> dict:
    frame.to_csv(OUT_DIR / name, index=False)
    return file_record(OUT_DIR / name)


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
        raise SystemExit("a statistics run takes --pass family or --pass full (bioinf2027/plans/E6.md, Section 16)")
    if arguments == ["--smoke"]:
        SMOKE = True
        fits.SMOKE = True
        SEEDS = (0,)
        B = 20
        resample.BLOCK = 20
        OUT_DIR = config.storage_root() / "bioinf2027" / "scratch" / "E6_stats_smoke"
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    if PASS == "family":
        OUT_DIR = OUT_DIR / "family_pass"
    main()
