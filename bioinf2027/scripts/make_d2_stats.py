"""D2, the statistics of the token-drop arm, from E1b's and D2's saved predictions.

Design: bioinf2027/plans/D2.md, Sections 6, 10 and 11 (the family of 18 and the readings of 11.1).

Per cell of the primary pair (G5, G6), on the cell's rebuilt label under the forest (seeds averaged inside
each replicate):
- VCC test (target-gene cluster bootstrap, 10,000 replicates, seed 20270301): dR2_token = R2(z) - R2(z-),
  dR2_rest = R2(z-) - R2(C1), and the noise check R2(z~) - R2(z-);
- per screen (E1b's joint draws, seeds 20260901 and 20260902): drho_token = rho(z) - rho(z-), R(z-), and the
  external noise check rho(z~) - rho(z-).
Nine per cell, 18 in all, Holm at 0.05. The share lost, the preconditions, T(z-), the increment of [TR; z-]
over TR, the positive control of the drop, C2-, G1 and G2 and the version without targets lacking a token
are secondary and unadjusted.

Outputs under bioinf2027/results/D2/: family.csv, readings.csv, secondary.csv, stats.json. With --smoke: 20
replicates on the saved predictions, outputs to $IGA_STORAGE_ROOT/bioinf2027/scratch/D2_stats_smoke/, never
reported. The cell lists of the perturbed cells are checked against bioinf2027/results/D2/units.json before
they are read.
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

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "D2"
OUT_DIR = OUT  # where the outputs go; the scratch folder with --smoke
PLAN = "bioinf2027/plans/D2.md, Sections 6, 10 and 11"
B = 10000
EXT_SEED = {"RPE1": 20260901, "K562-essential": 20260902}
VCC_SEED = 20270301
FOLD_SEED = 20270409
SEEDS = (0, 1, 2)
ALPHA = 0.05
SHARE_MARGIN = 0.75
SCREENS = ("RPE1", "K562-essential")
CELLS = ("G5", "G6", "G1", "G2")

SMOKE = False
DATA: dict = {}


def manifest(path: Path) -> dict:
    return json.loads(path.read_text())["files"]


def predictions(files: dict, spec: FitSpec) -> dict | None:
    record = files.get(spec.key())
    if record is None:
        return None
    path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["sha256"]:
        raise RuntimeError("prediction file of %s does not match its manifest" % spec.key())
    return transfer.load_predictions(path)[2]


def seeds_of(cell, arm, label="rebuilt", representation="z", side="test", seeds=SEEDS):
    """Per-seed predictions of an arm: 'z' and 'C1' from E1b, 'z-' and 'z~' from D2."""
    out = []
    for seed in seeds:
        if arm in ("z", "C1", "TR", "[TR; z]", "C2", "x-"):
            values = predictions(DATA["e1b"], FitSpec(cell, label, arm if arm != "z" else "z", "forest", seed))
        elif arm == "[TR; z-]":
            values = predictions(DATA["d2"], FitSpec(f"D2 | {cell} | z-", label, "[TR; z]", "forest", seed))
        elif arm == "C2-":
            values = predictions(DATA["d2"], FitSpec(f"D2 | {cell} | x-", label, "C2", "forest", seed))
        else:
            values = predictions(DATA["d2"], FitSpec(f"D2 | {cell} | {arm}", label, representation, "forest", seed))
        if values is None:
            return None
        out.append(values[side])
    return out


def vcc_draw(cell: str) -> dict:
    if cell not in DATA["vcc_draws"]:
        groups = DATA["test_order"][cell]["target_gene"].astype(str).to_numpy()
        DATA["vcc_draws"][cell] = {"draw": resample.cluster_draw(groups, B, VCC_SEED), "point": ones(groups.size)}
    return DATA["vcc_draws"][cell]


def ones(n: int) -> resample.ClusterDraw:
    return resample.ClusterDraw(weights=np.ones((1, n)), clusters=np.array([]), seed=-1, index_sha256="")


def r2(cell: str, vectors: list, label: np.ndarray):
    if vectors is None:
        return None
    d = vcc_draw(cell)
    columns = np.column_stack(vectors)
    return float(resample.weighted_r2(label, columns, d["point"])[0].mean()), resample.weighted_r2(label, columns, d["draw"]).mean(axis=1)


def rho(screen: str, vectors: list, label: np.ndarray, mask=None):
    if vectors is None:
        return None
    columns = np.column_stack(vectors)
    d = DATA["draws"][screen]
    return (float(resample.weighted_spearman(columns, label, d["point"], mask=mask)[0].mean()),
            resample.weighted_spearman(columns, label, d["draw"], mask=mask).mean(axis=1))


def residual(values, covariates, genes):
    y = sps.rankdata(values) / (values.size + 1.0)
    unique = np.array(sorted(set(genes.tolist())))
    fold_of = dict(zip(unique[np.random.default_rng(FOLD_SEED).permutation(unique.size)], np.arange(unique.size) % 5))
    folds = np.array([fold_of[g] for g in genes])
    out = np.full(y.size, np.nan)
    with heads.single_thread():
        for f in range(5):
            test = folds == f
            out[test] = y[test] - heads.nuisance().fit(covariates[~test], y[~test]).predict(covariates[test])
    return out


def r_stat(screen, vectors, label, covariates, genes):
    if vectors is None:
        return None
    e_y = residual(label, covariates, genes)
    d = DATA["draws"][screen]
    points, draws = [], []
    for v in vectors:
        e_p = residual(v, covariates, genes)
        points.append(resample.weighted_pearson(e_p, e_y, d["point"])[0, 0])
        draws.append(resample.weighted_pearson(e_p, e_y, d["draw"])[:, 0])
    return float(np.mean(points)), np.mean(draws, axis=0)


def diff(a, b):
    if a is None or b is None:
        return None
    return a[0] - b[0], a[1] - b[1]


def summary(value) -> dict:
    if value is None:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "se": float("nan"), "detectable_2.80se": float("nan")}
    lo, hi, _ = percentile_interval(value[1])
    se = float(np.nanstd(value[1], ddof=1))
    return {"estimate": value[0], "lo": lo, "hi": hi, "p": resample.p_value(value[1]), "se": se, "detectable_2.80se": 2.80 * se}


def test_label(cell: str, column: str) -> np.ndarray:
    return DATA["test_order"][cell][column].to_numpy(dtype=np.float64)


def external(cell: str, screen: str):
    table = DATA["tables"][cell][screen]
    return (table["y"].to_numpy(dtype=np.float64), table[list(e1cells.TRIVIAL)].to_numpy(dtype=np.float64),
            table["target_ensg"].astype(str).to_numpy())


def share_lost(z, zm, c1):
    """(R2(z) - R2(z-)) / (R2(z) - R2(C1)); +inf or -inf where the denominator is at most 0 (Section 11.2, item 7)."""
    num_p, den_p = z[0] - zm[0], z[0] - c1[0]
    num, den = z[1] - zm[1], z[1] - c1[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = np.where(den > 0, num / den, np.where(num > 0, np.inf, -np.inf))
    point = num_p / den_p if den_p > 0 else (np.inf if num_p > 0 else -np.inf)
    undefined = int(np.isnan(draws).sum())
    draws = draws[~np.isnan(draws)]
    finite = draws[np.isfinite(draws)]
    # outer order statistics: interpolating between an infinite and a finite replicate gives NaN, and the
    # infinite replicates are meant to widen the interval
    if draws.size:
        lo, hi = np.percentile(draws, 2.5, method="lower"), np.percentile(draws, 97.5, method="higher")
    else:
        lo, hi = np.nan, np.nan
    return {"estimate": float(point), "lo": float(lo), "hi": float(hi), "replicates_with_a_non_positive_denominator": int((den <= 0).sum()),
            "finite_replicates": int(finite.size), "replicates_undefined": undefined,
            "interval_rule": "2.5th and 97.5th percentiles as the lower and higher order statistics, infinite replicates kept"}


def family_rows() -> pd.DataFrame:
    rows = []
    for cell in e1cells.PRIMARY:
        y = test_label(cell, "y")
        z, zm, zt, c1 = (r2(cell, seeds_of(cell, a), y) for a in ("z", "z-", "z~", "C1"))
        for name, value in (("dR2_token", diff(z, zm)), ("dR2_rest", diff(zm, c1)), ("VCC noise check R2(z~) - R2(z-)", diff(zt, zm))):
            rows.append({"cell": cell, "scope": "VCC test", "statistic": name, **summary(value)})
        for screen in SCREENS:
            label, cov, genes = external(cell, screen)
            rz, rzm, rzt = (rho(screen, seeds_of(cell, a, side=screen), label) for a in ("z", "z-", "z~"))
            rows.append({"cell": cell, "scope": screen, "statistic": "drho_token", **summary(diff(rz, rzm))})
            rows.append({"cell": cell, "scope": screen, "statistic": "R(z-)", **summary(r_stat(screen, seeds_of(cell, "z-", side=screen), label, cov, genes))})
            rows.append({"cell": cell, "scope": screen, "statistic": "external noise check rho(z~) - rho(z-)", **summary(diff(rzt, rzm))})
    frame = pd.DataFrame(rows)
    rejected, thresholds = resample.holm(frame["p"].to_numpy(), ALPHA)
    frame["holm_threshold"] = thresholds
    frame["rejected"] = rejected
    frame["state"] = [resample.holm_state(e, r) for e, r in zip(frame["estimate"], rejected)]
    gate = DATA["r5"]
    frame["r5_gate_passed"] = [bool(gate[f"{c} | VCC"]["passed"] and gate[f"{c} | {s}"]["passed"]) if s in SCREENS else True
                               for c, s in zip(frame["cell"], frame["scope"])]
    uninformative = DATA.get("uninformative", {})
    frame["uninformative_cell"] = [bool(uninformative.get(c, False)) for c in frame["cell"]]
    # an uninformative cell's statistics stay in the family, reported and not read (Section 11.2, item 9)
    frame["read"] = [not ((st == "R(z-)" and not g) or u) for st, g, u in zip(frame["statistic"], frame["r5_gate_passed"], frame["uninformative_cell"])]
    frame["family"] = "D2, 18 statistics, Holm at family-wise 0.05; at most 0.25 across E1, E4, D2 and E6"
    return frame


READING_TEXT = {
    "(a)": "a resolved part of z's VCC advantage over C1 is lost when the token is removed (share lost [interval]); no remainder is resolved "
           "(dR2_rest [interval], 2.80 x SE)",
    "(a')": "most of the embedding's VCC advantage over the Gaussian control is lost when the target's own token is removed",
    "(b)": "the token carries part of the signal (share lost [interval]); the remainder is described on VCC through the increment of "
           "[TR; z-] over TR (secondary) and externally through R(z-)",
    "(c)": "not attributable to the token",
    "(d)": "no loss resolved (interval and 2.80 x SE stated)",
    "(e)": "removing the token improves the arm, with no mechanism read into it",
    "(e, noise not resolved)": "not attributable to the token",
    "(f)": "no D2 reading for this cell: z fails its precondition",
    "uninformative": "undetermined: the cell is uninformative (Section 11.2, item 9)",
}
DROP_NOT_REMOVED = "; the drop did not remove the own-coordinate information under this head"


def secondary_value(secondary: pd.DataFrame, cell: str, quantity: str) -> dict:
    row = secondary[(secondary["cell"] == cell) & (secondary["quantity"] == quantity)]
    if row.empty:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan"), "detectable_2.80se": float("nan")}
    return {k: float(row[k].iloc[0]) if k in row and pd.notna(row[k].iloc[0]) else float("nan") for k in ("estimate", "lo", "hi", "detectable_2.80se")}


def vcc_code(cell: str, state: dict, secondary: pd.DataFrame) -> str:
    """The reading of Section 11.1 that one cell satisfies on VCC."""
    if DATA.get("uninformative", {}).get(cell, False):
        return "uninformative"
    if not secondary_value(secondary, cell, "precondition of z: R2(z) - R2(C1)")["lo"] > 0:
        return "(f)"
    t, rest, noise = (state[(cell, "VCC test", s)] for s in ("dR2_token", "dR2_rest", "VCC noise check R2(z~) - R2(z-)"))
    if t == "Holm-positive" and noise == "Holm-positive" and rest != "Holm-positive":
        return "(a')" if secondary_value(secondary, cell, "share lost")["lo"] >= SHARE_MARGIN else "(a)"
    if t == "Holm-positive" and rest == "Holm-positive" and noise == "Holm-positive":
        return "(b)"
    if t == "Holm-positive":
        return "(c)"
    if t == "inconclusive":
        return "(d)"
    if t == "Holm-negative":
        return "(e)" if noise == "Holm-negative" else "(e, noise not resolved)"
    return "undetermined"


def readings(family: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Section 11.1: a statement needs the same reading in G5 and G6; (a') needs the share-lost margin in both cells."""
    state = {(r.cell, r.scope, r.statistic): r.state for r in family.itertuples()}
    estimate = {(r.cell, r.scope, r.statistic): (r.estimate, r.lo, r.hi) for r in family.itertuples()}
    detectable = {(r.cell, r.scope, r.statistic): d for r, d in zip(family.itertuples(), family["detectable_2.80se"])}
    rows = []
    codes = {cell: vcc_code(cell, state, secondary) for cell in e1cells.PRIMARY}
    pair = [codes[c] for c in e1cells.PRIMARY]
    if "uninformative" in pair:
        code = "uninformative"  # Section 11.2, item 9: the pair's statements are undetermined
    elif set(pair) == {"(a)", "(a')"}:
        code = "(a)"  # (a') needs the margin in both cells; both satisfy (a)
    elif pair[0] == pair[1]:
        code = pair[0]
    else:
        code = "undetermined"
    if code == "uninformative" or code == "undetermined":
        statement = READING_TEXT["uninformative"] if code == "uninformative" else "undetermined: the cells of the pair give different readings"
    else:
        statement = READING_TEXT[code]
    if code == "(d)":
        drop = [secondary_value(secondary, c, "positive control of the drop: R2(z) - R2(z-) predicting B2 native")["lo"] for c in e1cells.PRIMARY]
        if not all(v > 0 for v in drop):
            statement += DROP_NOT_REMOVED
    entry = {"scope": "VCC", "reading": code, "statement": statement, **{c: codes[c] for c in e1cells.PRIMARY}}
    for cell in e1cells.PRIMARY:
        for name in ("dR2_token", "dR2_rest", "VCC noise check R2(z~) - R2(z-)"):
            key = (cell, "VCC test", name)
            entry.update({f"{cell} {name}": estimate[key][0], f"{cell} {name} lo": estimate[key][1], f"{cell} {name} hi": estimate[key][2],
                          f"{cell} {name} 2.80se": detectable[key]})
        share = secondary_value(secondary, cell, "share lost")
        entry.update({f"{cell} share lost": share["estimate"], f"{cell} share lost lo": share["lo"], f"{cell} share lost hi": share["hi"]})
    rows.append(entry)
    for screen in SCREENS:
        states = [(state[(c, screen, "drho_token")], state[(c, screen, "external noise check rho(z~) - rho(z-)")]) for c in e1cells.PRIMARY]
        uninformative = [c for c in e1cells.PRIMARY if DATA.get("uninformative", {}).get(c, False)]
        if uninformative:
            statement = READING_TEXT["uninformative"]
        elif all(s[0] == s[1] == "Holm-positive" for s in states):
            statement = "removing the token lowers z's external Spearman by drho_token [interval], beyond the noise-matched change"
        elif all(s[0] == s[1] == "Holm-negative" for s in states):
            statement = "removing the token raises z's external Spearman by drho_token [interval], beyond the noise-matched change"
        elif all(s[0] == "inconclusive" for s in states):
            statement = "no change resolved (interval and 2.80 x SE stated)"
        else:
            statement = "undetermined"
        # R(z-) counts toward E1's statement only where z- passes its own precondition in both cells (Section 11.2, item 8)
        precondition = all(secondary_value(secondary, c, "precondition of z-: R2(z-) - R2(C1)")["lo"] > 0 for c in e1cells.PRIMARY)
        gate = all(bool(family[(family["cell"] == c) & (family["scope"] == screen) & (family["statistic"] == "R(z-)")]["r5_gate_passed"].iloc[0])
                   for c in e1cells.PRIMARY)
        r_states = [state[(c, screen, "R(z-)")] for c in e1cells.PRIMARY]
        if uninformative or not gate:
            r_reading = "not read: " + ("an uninformative cell" if uninformative else "E1's R5 gate fails on this screen")
        elif not precondition:
            r_reading = "not read: z- fails its own precondition in a cell of the pair"
        elif r_states[0] != r_states[1]:
            r_reading = "undetermined: the cells differ"
        else:
            r_reading = {"Holm-positive": "z- carries information beyond the specified trivial predictors", "Holm-negative":
                         "z-'s prediction runs against the label once the trivial predictors are removed",
                         "inconclusive": "not shown to exceed the specified trivial predictors"}[r_states[0]]
        entry = {"scope": screen, "statement": statement, **{c: "; ".join(s) for c, s in zip(e1cells.PRIMARY, states)},
                 "R(z-) states": "; ".join(r_states), "R(z-) reading": r_reading, "z- precondition passed in both cells": precondition}
        for cell in e1cells.PRIMARY:
            key = (cell, screen, "drho_token")
            entry.update({f"{cell} drho_token": estimate[key][0], f"{cell} drho_token lo": estimate[key][1], f"{cell} drho_token hi": estimate[key][2],
                          f"{cell} drho_token 2.80se": detectable[key]})
        rows.append(entry)
    return pd.DataFrame(rows)


def secondary_rows() -> pd.DataFrame:
    rows = []

    def add(cell, quantity, value, **extra):
        rows.append({"cell": cell, "quantity": quantity, **summary(value), "adjustment": "secondary, unadjusted", **extra})

    for cell in CELLS:
        y = test_label(cell, "y")
        z, zm, c1 = (r2(cell, seeds_of(cell, a), y) for a in ("z", "z-", "C1"))
        add(cell, "precondition of z: R2(z) - R2(C1)", diff(z, c1))
        add(cell, "precondition of z-: R2(z-) - R2(C1)", diff(zm, c1))
        if cell not in e1cells.PRIMARY:
            add(cell, "dR2_token", diff(z, zm))
            add(cell, "dR2_rest", diff(zm, c1))
        if z is not None and zm is not None and c1 is not None:
            rows.append({"cell": cell, "quantity": "share lost", **share_lost(z, zm, c1), "adjustment": "secondary, unadjusted"})
        tr, trz, trzm = (r2(cell, seeds_of(cell, a), y) for a in ("TR", "[TR; z]", "[TR; z-]"))
        add(cell, "VCC test increment of [TR; z] over TR", diff(trz, tr))
        add(cell, "VCC test increment of [TR; z-] over TR", diff(trzm, tr))
        for name in ("B2", "B3"):
            label = test_label(cell, name)
            own_z = r2(cell, seeds_of_own(cell, "z", name), label)
            own_zm = r2(cell, seeds_of_own(cell, "z-", name), label)
            add(cell, f"positive control of the drop: R2(z) - R2(z-) predicting {name} native" if name == "B2" else f"positive control of the drop: R2(z) - R2(z-) predicting {name}",
                diff(own_z, own_zm))
        for screen in SCREENS:
            label, cov, genes = external(cell, screen)
            rz, rzm, rc1 = (rho(screen, seeds_of(cell, a, side=screen), label) for a in ("z", "z-", "C1"))
            add(cell, f"T(z-) on {screen}", diff(rzm, rc1), screen=screen)
            add(cell, f"T(z) on {screen}", diff(rz, rc1), screen=screen)
            if cell not in e1cells.PRIMARY:
                add(cell, f"drho_token on {screen}", diff(rz, rzm), screen=screen)
                add(cell, f"R(z-) on {screen}", r_stat(screen, seeds_of(cell, "z-", side=screen), label, cov, genes), screen=screen)
            has_token = DATA["has_token"][screen]
            add(cell, f"drho_token on {screen}, targets with a token", diff(rho(screen, seeds_of(cell, "z", side=screen), label, has_token),
                                                                          rho(screen, seeds_of(cell, "z-", side=screen), label, has_token)), screen=screen)
            add(cell, f"C2 minus C2- on {screen}", diff(rho(screen, seeds_of(cell, "C2", side=screen), label), rho(screen, seeds_of(cell, "C2-", side=screen), label)), screen=screen)
            add(cell, f"z- minus x- on {screen}", diff(rzm, rho(screen, seeds_of(cell, "x-", side=screen), label)), screen=screen)
            if cell in e1cells.PRIMARY:
                for lab, col in (("source", "y"), ("mean", "y_mean")):
                    lbl = DATA["tables"][cell][screen][col].to_numpy(dtype=np.float64)
                    add(cell, f"drho_token on {screen}, {lab} label", diff(rho(screen, seeds_of(cell, "z", label=lab, side=screen), lbl),
                                                                          rho(screen, seeds_of(cell, "z-", label=lab, side=screen), lbl)), screen=screen)
    return pd.DataFrame(rows)


def seeds_of_own(cell: str, arm: str, name: str):
    out = []
    for seed in SEEDS:
        values = predictions(DATA["d2"], FitSpec(f"D2 | {cell} | {arm}", f"own {name}", "z", "forest", seed))
        if values is None:
            return None
        out.append(values["test"])
    return out


def main() -> None:
    started = time.time()
    DATA["e1b"] = manifest(REPO / "bioinf2027" / "results" / "E1" / "predictions_manifest.json")
    DATA["d2"] = manifest(OUT / "predictions_manifest.json")
    DATA["tables"] = {cell: e1cells.load_tables(cell) for cell in CELLS}
    DATA["vcc_draws"] = {}
    # VCC test rows in the order E1b's predictions hold them: the cell's rows in the loader's order
    import make_e1b_fits as e1b

    harness_inputs = e1b.load_inputs(list(CELLS))
    DATA["test_order"] = {cell: e1b.DATA["cells"][cell].vcc[e1b.DATA["cells"][cell].split("test")] for cell in CELLS}
    DATA["draws"] = {}
    for screen in SCREENS:
        genes = DATA["tables"]["G1"][screen]["target_ensg"].astype(str).to_numpy()
        DATA["draws"][screen] = {"draw": resample.cluster_draw(genes, B, EXT_SEED[screen]), "point": ones(genes.size)}
    DATA["has_token"] = {}
    units = json.loads((OUT / "units.json").read_text())["files"]
    for screen, part in (("RPE1", "rpe1_perturbed"), ("K562-essential", "k562_perturbed")):
        path = config.storage_root() / "bioinf2027" / "D2" / "units" / f"{part}.parquet"
        if sha256(path) != units[part]["sha256"]:
            raise RuntimeError("the %s cell list does not match bioinf2027/results/D2/units.json" % part)
        frame = pd.read_parquet(path)
        no_token = set(frame.loc[frame["drop_token"] < 0, "construct"])
        DATA["has_token"][screen] = ~DATA["tables"]["G1"][screen]["construct"].astype(str).isin(no_token).to_numpy()
    DATA["r5"] = json.loads((REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json").read_text())["r5_gate"]
    aggregate = json.loads((OUT / "aggregate.json").read_text())
    DATA["uninformative"] = {cell: bool(v["uninformative"]) for cell, v in aggregate["uninformative_cells"].items()}
    family = family_rows()
    secondary = secondary_rows()
    outputs = {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("family.csv", family), ("secondary.csv", secondary), ("readings.csv", readings(family, secondary))):
        frame.to_csv(OUT_DIR / name, index=False)
        outputs[name] = file_record(OUT_DIR / name)
    stats = {"plan": PLAN, "replicates": B, "seeds": {"external": EXT_SEED, "vcc": VCC_SEED, "nuisance folds": FOLD_SEED},
             "replicate_index_sha256": {s: DATA["draws"][s]["draw"].index_sha256 for s in SCREENS},
             "inputs": {"e1b_predictions": file_record(REPO / "bioinf2027" / "results" / "E1" / "predictions_manifest.json"),
                        "d2_predictions": file_record(OUT / "predictions_manifest.json"), "aggregate": file_record(OUT / "aggregate.json"),
                        "units": file_record(OUT / "units.json"), "e1b_harness": harness_inputs,
                        "readouts": file_record(REPO / "bioinf2027" / "endpoints" / "readouts" / "readouts.json")},
             "smoke": SMOKE,
             "outputs": outputs, "seconds": round(time.time() - started, 1)}
    (OUT_DIR / "stats.json").write_text(json.dumps(strict_json(stats), indent=1, allow_nan=False) + "\n")
    print("[d2 stats] done in %.0f s" % (time.time() - started), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--smoke"]:
        SMOKE = True
        B = 20
        OUT_DIR = config.storage_root() / "bioinf2027" / "scratch" / "D2_stats_smoke"
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    main()
