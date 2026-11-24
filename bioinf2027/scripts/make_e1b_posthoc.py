"""E1b post hoc analyses, decided at gate G3 on 2026-11-12: post hoc, secondary and unadjusted.

Plan: bioinf2027/plans/E1b_posthoc.md, as revised after its review
(bioinf2027/reviews/E1b_posthoc_plan_r1.md). Nothing is refitted. Every value comes from the saved
E1b predictions (bioinf2027/results/E1/predictions_manifest.json, each file checked against its
digest), through the functions of bioinf2027/scripts/make_e1b_stats.py, used unchanged.

- VCC test, cells G5 and G6 (G6 the cell of record; G5 beside it, "not the submitted statistic"). The
  precondition's target-gene cluster bootstrap: 10,000 replicates, seed 20270301, with per-seed R² and
  Spearman averaged over head seeds 0 to 2 inside each replicate.
  - (a0) z (forest) minus C1 (boosting), with its batch-resampled interval (seed 20270302);
  - (a1) z minus the B0 head, (a1b) minus the B1 head, (a1c) minus the B2 head;
  - (a2) [TR; z] minus TR, (a3) [z; B2] minus [C1; B2], (a4) z minus C2, all on the existing label;
  - (b) the z minus C1 margin on the existing label minus the same on the rebuilt label, and the same
    for B0, B1, B2 and TR, all from one joint draw; beside it the labels' variances and the paired MSE
    difference.
- External, descriptive, cells G5 and G6 on both screens, with E1b's joint draws (seeds 20260901 and
  20260902): the Spearman of the existing-label with the rebuilt-label z prediction, and of each
  with B0, B1, B2, the norm of z and the external label; C1 is given for reference. A VCC reference
  is given on the test rows.
- L of C2 and C3 (T on the rebuilt label minus T on the existing label), on E1b's joint draws.

Before anything is written, E1b values are recomputed from the same inputs (plan, Section 5). A
difference above 1e-12 stops the run, and nothing is written.

Outputs under bioinf2027/results/E1/posthoc/: posthoc_vcc.csv, posthoc_label_scale.csv,
posthoc_external_ranking.csv, posthoc_vcc_ranking.csv, posthoc_l_controls.csv, posthoc.json. E1b's
own outputs are read, never written. With --smoke: 20 replicates, point values checked only, outputs
to the scratch folder, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_e1b_stats as e1s  # noqa: E402
from bioinf2027 import config, e1cells, e1inputs, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "bioinf2027" / "results" / "E1"
OUT = E1 / "posthoc"
PLAN = "bioinf2027/plans/E1b_posthoc.md"
STATUS = "post hoc, secondary, unadjusted (bioinf2027/plans/E1b_posthoc.md)"
CELLS = e1cells.PRIMARY
RECORD_CELL = "G6"
SCREENS = e1s.SCREENS
TOLERANCE = 1e-12
SUBMITTED_A0 = 0.1645
SMOKE = False
Z_RECORDS = {"RPE1": ("rpe1_z", "rpe1_z_index"), "K562-essential": ("k562_z", "k562_z_index")}
NOT_RECORD = "not the submitted statistic (G5, 2,198 rows with at least 27 cells); reported beside G6, no reading"

# (name, description, (representation, head) of the first, (representation, head) of the second), existing label
EXISTING_QUANTITIES = (
    ("a0", "z (forest) minus C1 (boosting); reference, the comparator of the submitted margin", ("z", "forest"), ("C1", "boosting")),
    ("a1", "z minus the B0 head (cell count)", ("z", "forest"), ("B0", "forest")),
    ("a1b", "z minus the B1 head (control-cell mean of the target gene)", ("z", "forest"), ("B1", "forest")),
    ("a1c", "z minus the B2 head (own shift)", ("z", "forest"), ("B2", "forest")),
    ("a2", "[TR; z] minus TR", ("[TR; z]", "forest"), ("TR", "forest")),
    ("a3", "[z; B2] minus [C1; B2]", ("[z; B2]", "forest"), ("[C1; B2]", "forest")),
    ("a4", "z minus C2; reference, the comparison behind the submitted z minus C2", ("z", "forest"), ("C2", "forest")),
)
B_ARMS = (("b", "z"), ("b-B0", "B0"), ("b-B1", "B1"), ("b-B2", "B2"), ("b-TR", "TR"))
READ_IN_G6 = ("a1", "a1b", "a1c", "a2", "a3", "b")
READINGS = {
    "a1": ("z exceeds cell count alone on the all-cells label", "z is not shown to differ from cell count alone", "cell count alone exceeds z"),
    "a1b": ("z exceeds the control-cell mean alone", "z is not shown to differ from the control-cell mean alone", "the control-cell mean alone exceeds z"),
    "a1c": ("z exceeds the own shift alone", "z is not shown to differ from the own shift alone", "the own shift alone exceeds z"),
    "a2": ("z adds to cell count, control mean and own shift", "nothing added by z is resolved",
           "adding z lowers R-squared under the forest; not read as evidence about z (dilution neither shown nor excluded)"),
    "a3": ("z beats a Gaussian block of the same width beside B2; consistent with z carrying B0 or B1", "no difference resolved",
           "the Gaussian block does better"),
    "b": ("the margin over C1 is larger on the all-cells label than on the rebuilt label", "the change in the margin is not resolved as a paired difference",
          "the margin is larger on the rebuilt label"),
}


class Stop(RuntimeError):
    """A reproduction check failed or an input is missing: nothing is written (hard limit 9)."""


# --------------------------------------------------------------------------- inputs


def external_table(cell: str, screen: str) -> pd.DataFrame:
    data = e1s.DATA["cells"][cell]
    return e1s.DATA["external"][(screen, data.external_source[screen])]


def load() -> dict:
    e1s.DATA["predictions"] = e1s.Predictions(e1s.OUT / "predictions_manifest.json")
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    e1s.DATA["cells"], e1s.DATA["external"], e1s.DATA["external_source"] = {}, {}, {}
    for cell in CELLS:
        # G5 and G6 are row-level cells; the argument `modelled` is used only for gene-level cells
        if e1cells.CELLS[cell]["unit"] != "row":
            raise Stop("%s is not a row-level cell" % cell)
        data = e1cells.build_cell(cell, rows, means)
        e1s.DATA["cells"][cell] = data
        for screen in data.spec["screens"]:
            e1s.DATA["external"][(screen, data.external_source[screen])] = data.external[screen]
        e1s.DATA["external_source"][cell] = data.external_source[data.spec["screens"][0]]
    e1s.STATE["draws"] = e1s.build_draws()
    e1s.STATE["vcc_draws"] = {}
    records = json.loads((E1 / "input_records.json").read_text())["inputs"]
    paths = e1inputs.paths()
    norms = {}
    for screen in SCREENS:
        for key, name in zip(Z_RECORDS[screen], (f"{screen}_z", f"{screen}_z_index")):
            if sha256(paths[name]) != records[key]["sha256"]:
                raise Stop("%s does not match bioinf2027/results/E1/input_records.json, %s" % (name, key))
        constructs = None
        for cell in CELLS:
            table = external_table(cell, screen)
            if constructs is None:
                constructs = table["construct"].astype(str).to_numpy()
            elif not np.array_equal(constructs, table["construct"].astype(str).to_numpy()):
                raise Stop("the external constructs of %s differ between the cells" % screen)
            for column in ("B0", "B1", "B2", "y"):
                if not np.isfinite(table[column].to_numpy(dtype=np.float64)).all():
                    raise Stop("%s, %s: column %s is not finite" % (cell, screen, column))
            # the joint draw resamples the target genes of the constructs in this order
            if not np.array_equal(table["target_ensg"].astype(str).to_numpy(), e1s.STATE["draws"][screen]["groups"]):
                raise Stop("%s, %s: the constructs are not in the order of the joint draw" % (cell, screen))
        z = e1inputs._ordered_z(screen, constructs)
        norms[screen] = np.linalg.norm(z.astype(np.float64), axis=1)
    return {"z_norm": norms}


CACHE: dict = {}


def vcc(cell: str, label: str, representation: str, head: str, kind: str = "genes") -> dict:
    key = ("vcc", cell, label, representation, head, kind)
    if key not in CACHE:
        if len(e1s.DATA["predictions"].seeds(cell, label, representation, head)) != len(e1s.SEEDS):
            raise Stop("%s | %s | %s | %s does not have every head seed" % (cell, label, representation, head))
        CACHE[key] = e1s.vcc_r2(cell, label, representation, head, kind=kind)
    return CACHE[key]


def test_rows(cell: str) -> pd.DataFrame:
    data = e1s.DATA["cells"][cell]
    return data.vcc[data.split("test")]


def external_predictions(cell: str, label: str, representation: str, screen: str) -> list[np.ndarray]:
    fits = e1s.DATA["predictions"].seeds(cell, label, representation, "forest")
    if len(fits) != len(e1s.SEEDS) or any(screen not in f for f in fits):
        raise Stop("missing external predictions: %s | %s | %s | %s" % (cell, label, representation, screen))
    return [np.asarray(f[screen], dtype=np.float64) for f in fits]


def spearman_pairs(columns: list[np.ndarray], others: list[np.ndarray], draws: dict) -> tuple[float, np.ndarray]:
    """Spearman of columns[k] with others[k] per head seed k, averaged over seeds; point and replicates."""
    points, replicates = [], []
    for column, other in zip(columns, others):
        points.append(resample.weighted_spearman(column, other, draws["point"])[0, 0])
        replicates.append(resample.weighted_spearman(column, other, draws["draw"])[:, 0])
    return float(np.mean(points)), np.mean(replicates, axis=0)


def external_rho(cell: str, label: str, representation: str, screen: str) -> tuple[float, np.ndarray]:
    """As make_e1b_stats.get_rho on all constructs: the mean over seeds of the external Spearman with y."""
    key = ("external", cell, label, representation, screen)
    if key not in CACHE:
        y = external_table(cell, screen)["y"].to_numpy(dtype=np.float64)
        predictions = external_predictions(cell, label, representation, screen)
        CACHE[key] = spearman_pairs(predictions, [y] * len(predictions), e1s.STATE["draws"][screen])
    return CACHE[key]


def t_value(cell, label, representation, screen):
    return e1s.contrast(external_rho(cell, label, representation, screen), external_rho(cell, label, "C1", screen))


def l_value(cell, representation, screen):
    return e1s.contrast(t_value(cell, "rebuilt", representation, screen), t_value(cell, "existing", representation, screen))


def tied_share(predictions: list[np.ndarray]) -> float:
    """1 - distinct predicted values / number of predictions, averaged over the head seeds."""
    return float(np.mean([1.0 - np.unique(p).size / p.size for p in predictions]))


# --------------------------------------------------------------------------- reproduction checks


def compare(checks: list, name: str, mine: float, theirs: float, where: str) -> None:
    checks.append({"check": name, "where": where, "recomputed": float(mine), "file": float(theirs), "difference": abs(float(mine) - float(theirs))})


def one_row(frame: pd.DataFrame, where: str) -> pd.Series:
    if len(frame) != 1:
        raise Stop("%s: %d rows where one was expected" % (where, len(frame)))
    return frame.iloc[0]


def reproduction() -> dict:
    checks: list = []
    pre = pd.read_csv(E1 / "preconditions.csv", float_precision="round_trip")
    test = pd.read_csv(E1 / "vcc_test.csv", float_precision="round_trip")
    ext = pd.read_csv(E1 / "transfer.csv", float_precision="round_trip")
    family = pd.read_csv(E1 / "primary_family.csv", float_precision="round_trip")
    intervals = not SMOKE
    for cell in CELLS:
        for label in ("existing", "rebuilt"):
            where = f"preconditions.csv, {cell}, {label}, z"
            row = one_row(pre[(pre.cell == cell) & (pre.label == label) & (pre.arm == "z")], where)
            a, c1 = vcc(cell, label, "z", "forest"), vcc(cell, label, "C1", "forest")
            d = e1s.summary(*e1s.contrast(a["r2"], c1["r2"]))
            ds = e1s.contrast(a["spearman"], c1["spearman"])
            values = {"r2_arm": a["r2"][0], "r2_C1": c1["r2"][0], "r2_difference_estimate": d["estimate"], "spearman_difference": ds[0]}
            if intervals:
                lo, hi = e1s.resample_interval(ds[1])[:2]
                values.update({"r2_difference_lo": d["lo"], "r2_difference_hi": d["hi"], "spearman_difference_lo": lo, "spearman_difference_hi": hi})
            for column, mine in values.items():
                compare(checks, "1 precondition", mine, row[column], f"{where}, {column}")
        rows2 = [("existing", r, "forest") for r in ("B0", "B1", "B2", "C2", "TR", "[TR; z]", "[z; B2]", "[C1; B2]")]
        rows2 += [("rebuilt", r, "forest") for r in ("B0", "B1", "B2", "TR")] + [("existing", "C1", "boosting")]
        for label, representation, head in rows2:
            where = f"vcc_test.csv, {cell}, {label}, {representation}, {head}"
            row = one_row(test[(test.cell == cell) & (test.label == label) & (test.representation == representation) & (test["head"] == head)], where)
            s = e1s.summary(*vcc(cell, label, representation, head)["r2"])
            values = {"row_r2": s["estimate"], **({"row_r2_lo": s["lo"], "row_r2_hi": s["hi"]} if intervals else {})}
            for column, mine in values.items():
                compare(checks, "2 VCC test", mine, row[column], f"{where}, {column}")
        for screen in SCREENS:
            for label in ("existing", "rebuilt"):
                base = ext[(ext.cell == cell) & (ext.screen == screen) & (ext.label == label) & (ext["head"] == "forest") & (ext.subset == "all")]
                where = f"transfer.csv, {cell}, {screen}, {label}, z"
                s = e1s.summary(*external_rho(cell, label, "z", screen))
                row = one_row(base[base.representation == "z"], where)
                for column in ("estimate", "lo", "hi") if intervals else ("estimate",):
                    compare(checks, "3 external Spearman", s[column], row[f"spearman_{column}"], f"{where}, spearman_{column}")
                for representation in ("C2", "C3"):
                    where = f"transfer.csv, {cell}, {screen}, {label}, {representation}"
                    s = e1s.summary(*t_value(cell, label, representation, screen))
                    row = one_row(base[base.representation == representation], where)
                    for column in ("estimate", "lo", "hi") if intervals else ("estimate",):
                        compare(checks, "3 external T", s[column], row[f"minus_C1_{column}"], f"{where}, minus_C1_{column}")
            where = f"primary_family.csv, {cell}, {screen}, L, z"
            row = one_row(family[(family.cell == cell) & (family.screen == screen) & (family.statistic == "L") & (family.arm == "z")], where)
            s = e1s.summary(*l_value(cell, "z", screen))
            for column in ("estimate", "lo", "hi") if intervals else ("estimate",):
                compare(checks, "4 L(z)", s[column], row[column], f"{where}, {column}")
    stats = json.loads((E1 / "stats.json").read_text())
    digests = {}
    for screen in SCREENS:
        mine = e1s.STATE["draws"][screen]["draw"].index_sha256
        digests[screen] = {"recomputed": mine, "stats.json": stats["replicate_index_sha256"][screen], "equal": mine == stats["replicate_index_sha256"][screen]}
    if intervals and not all(d["equal"] for d in digests.values()):
        raise Stop("the external replicate index differs from bioinf2027/results/E1/stats.json: %s" % digests)
    failed = [c for c in checks if not c["difference"] <= TOLERANCE]
    if failed:
        raise Stop("%d reproduction checks differ by more than %g, the first: %s" % (len(failed), TOLERANCE, failed[0]))
    a0 = e1s.contrast(vcc(RECORD_CELL, "existing", "z", "forest")["r2"], vcc(RECORD_CELL, "existing", "C1", "boosting")["r2"])[0]
    if abs(a0 - SUBMITTED_A0) > 5e-5:
        raise Stop("the G6 (a0) point estimate %.6f does not equal the submitted +0.1645 to four decimals" % a0)
    return {"tolerance": TOLERANCE, "values_compared": len(checks), "largest_difference": max(c["difference"] for c in checks),
            "external_replicate_index": digests, "intervals_compared": intervals,
            "g6_a0_against_submitted": {"recomputed": a0, "submitted": SUBMITTED_A0, "main_tex": "earlier_revision/v7/main.tex:512"},
            "checks": checks}


def support_check(norms: dict) -> dict:
    support = pd.read_csv(E1 / "support.csv", float_precision="round_trip")
    out = {}
    for screen in SCREENS:
        theirs = float(one_row(support[(support.cell == "G5") & (support.screen == screen) & (support.quantity == "norm of z")], "support.csv")["screen_median"])
        mine = float(np.median(norms[screen]))
        if abs(mine - theirs) > TOLERANCE:
            raise Stop("the median norm of z on %s (%r) differs from support.csv (%r)" % (screen, mine, theirs))
        out[screen] = {"median_norm_of_z": mine, "support.csv": theirs}
    return out


# --------------------------------------------------------------------------- VCC


def reading(name: str, cell: str, lo: float, hi: float) -> str:
    if cell != RECORD_CELL:
        return NOT_RECORD
    if name not in READINGS:
        return "reference only; no reading"
    above, inside, below = READINGS[name]
    if lo > 0:
        return above
    if hi < 0:
        return below
    return inside


def vcc_entry(cell, name, description, r2, rho, first, second, n_rows, n_genes) -> dict:
    s, sr = e1s.summary(*r2), e1s.summary(*rho)
    return {"cell": cell, "quantity": name, "description": description, "n_test_rows": n_rows, "n_test_genes": n_genes,
            "r2_difference": s["estimate"], "r2_lo": s["lo"], "r2_hi": s["hi"], "r2_p": s["p"], "r2_se": s["se"], "r2_detectable_2.80se": s["detectable_2.80se"],
            "spearman_difference": sr["estimate"], "spearman_lo": sr["lo"], "spearman_hi": sr["hi"], "spearman_p": sr["p"],
            "first": first[0], "r2_first": first[1], "spearman_first": first[2], "second": second[0], "r2_second": second[1], "spearman_second": second[2],
            "reading_on_r2": reading(name, cell, s["lo"], s["hi"])}


def vcc_rows() -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        for name, description, (arm, arm_head), (other, other_head) in EXISTING_QUANTITIES:
            a, c = vcc(cell, "existing", arm, arm_head), vcc(cell, "existing", other, other_head)
            entry = vcc_entry(cell, name, description + ", existing label", e1s.contrast(a["r2"], c["r2"]), e1s.contrast(a["spearman"], c["spearman"]),
                              (f"{arm} ({arm_head})", a["r2"][0], a["spearman"][0]), (f"{other} ({other_head})", c["r2"][0], c["spearman"][0]), a["n_rows"], a["n_genes"])
            if name == "a0":
                ab, cb = vcc(cell, "existing", arm, arm_head, kind="batches"), vcc(cell, "existing", other, other_head, kind="batches")
                lo, hi, _ = e1s.resample_interval(e1s.contrast(ab["r2"], cb["r2"])[1])
                entry.update({"batch_resampled_lo": lo, "batch_resampled_hi": hi})
            if name == "a2":
                entry["ceiling"] = 1.0 - c["r2"][0]
            rows.append(entry)
        for name, arm in B_ARMS:
            margins = {}
            for label in ("existing", "rebuilt"):
                a, c1 = vcc(cell, label, arm, "forest"), vcc(cell, label, "C1", "forest")
                margins[label] = {"r2": e1s.contrast(a["r2"], c1["r2"]), "spearman": e1s.contrast(a["spearman"], c1["spearman"]), "n": (a["n_rows"], a["n_genes"])}
            if margins["existing"]["n"] != margins["rebuilt"]["n"]:
                raise Stop("%s: the two labels do not share the test rows" % cell)
            description = f"{arm} minus C1 (forest) on the existing label minus the same on the rebuilt label, one joint draw"
            if name != "b":
                description += "; reference for b"
            entry = vcc_entry(cell, name, description, e1s.contrast(margins["existing"]["r2"], margins["rebuilt"]["r2"]),
                              e1s.contrast(margins["existing"]["spearman"], margins["rebuilt"]["spearman"]),
                              ("existing-label margin", margins["existing"]["r2"][0], margins["existing"]["spearman"][0]),
                              ("rebuilt-label margin", margins["rebuilt"]["r2"][0], margins["rebuilt"]["spearman"][0]), *margins["existing"]["n"])
            lo, hi = entry["r2_lo"], entry["r2_hi"]
            entry["margin_falls_on_the_rebuilt_label"] = bool(lo > 0)
            if name == "b" and cell == RECORD_CELL:
                pre = vcc(cell, "existing", "z", "forest"), vcc(cell, "existing", "C1", "forest")
                existing_lo = e1s.summary(*e1s.contrast(pre[0]["r2"], pre[1]["r2"]))["lo"]
                entry["margin_lost_stated"] = bool(existing_lo > 0 and lo > 0)
            rows.append(entry)
    frame = pd.DataFrame(rows)
    record = frame[(frame.cell == RECORD_CELL) & frame.quantity.isin(READ_IN_G6)]
    rejected, thresholds = resample.holm(record["r2_p"].to_numpy(), e1s.ALPHA)
    frame["descriptive_holm_g6"] = ""
    for index, r, t in zip(record.index, rejected, thresholds):
        frame.loc[index, "descriptive_holm_g6"] = "%s at threshold %.4f (descriptive, not a decision rule)" % ("rejected" if r else "not rejected", t)
    frame["status"] = STATUS
    return frame


def weighted_mse(y: np.ndarray, predictions: np.ndarray, draw) -> np.ndarray:
    """Mean squared error per replicate, averaged over the head-seed columns."""
    total = draw.weights.sum(axis=1)
    return np.mean([draw.weights @ ((y - predictions[:, j]) ** 2) / total for j in range(predictions.shape[1])], axis=0)


def label_scale_rows() -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        draws = e1s.vcc_draw(cell, "genes")
        out = {}
        for label in ("existing", "rebuilt"):
            y = e1s.vcc_label(cell, label)
            z = np.column_stack([f["test"] for f in e1s.DATA["predictions"].seeds(cell, label, "z", "forest")])
            c1 = np.column_stack([f["test"] for f in e1s.DATA["predictions"].seeds(cell, label, "C1", "forest")])
            point = weighted_mse(y, z, draws["point"])[0] - weighted_mse(y, c1, draws["point"])[0]
            replicates = weighted_mse(y, z, draws["draw"]) - weighted_mse(y, c1, draws["draw"])
            out[label] = (point, replicates)
            rows.append({"cell": cell, "quantity": f"variance of the {label} label over the test rows", "estimate": float(np.var(y)),
                         "lo": float("nan"), "hi": float("nan"), "n_test_rows": int(y.size)})
            s = e1s.summary(point, replicates)
            rows.append({"cell": cell, "quantity": f"MSE of z minus MSE of C1 (forest), {label} label", "estimate": s["estimate"], "lo": s["lo"], "hi": s["hi"], "n_test_rows": int(y.size)})
        s = e1s.summary(*e1s.contrast(out["existing"], out["rebuilt"]))
        rows.append({"cell": cell, "quantity": "MSE of z minus MSE of C1, existing label minus rebuilt label, one joint draw",
                     "estimate": s["estimate"], "lo": s["lo"], "hi": s["hi"], "n_test_rows": int(y.size)})
    frame = pd.DataFrame(rows)
    frame["role"] = ["beside b" if c == RECORD_CELL else NOT_RECORD for c in frame.cell]
    frame["status"] = STATUS
    return frame


# --------------------------------------------------------------------------- rankings (c)


def external_rows(norms: dict) -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        for screen in SCREENS:
            table = external_table(cell, screen)
            draws = e1s.STATE["draws"][screen]
            covariates = {"B0 (cell count)": table["B0"].to_numpy(dtype=np.float64),
                          "B1 (control-cell mean of the target gene)": table["B1"].to_numpy(dtype=np.float64),
                          "B2 native (own shift)": table["B2"].to_numpy(dtype=np.float64),
                          "norm of z": norms[screen], "external label y": table["y"].to_numpy(dtype=np.float64)}
            for arm, role in (("z", "z"), ("C1", "reference")):
                predictions = {label: external_predictions(cell, label, arm, screen) for label in ("existing", "rebuilt")}
                pairs = [("existing-label prediction", "rebuilt-label prediction", predictions["existing"], predictions["rebuilt"])]
                for label in ("existing", "rebuilt"):
                    for name, values in covariates.items():
                        pairs.append((f"{label}-label prediction", name, predictions[label], [values] * len(predictions[label])))
                for first, second, a, b in pairs:
                    s = e1s.summary(*spearman_pairs(a, b, draws))
                    rows.append({"cell": cell, "screen": screen, "arm": arm, "role": role, "first": first, "second": second,
                                 "n_constructs": int(table.shape[0]), "spearman": s["estimate"], "lo": s["lo"], "hi": s["hi"],
                                 "tied_share_first": tied_share(a),
                                 "tied_share_second": tied_share(b) if second.endswith("prediction") else float("nan")})
    frame = pd.DataFrame(rows)
    frame["kind"] = "descriptive; no statement is read from the interval"
    frame["status"] = STATUS
    return frame


def vcc_ranking_rows() -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        test = test_rows(cell)
        data = e1s.DATA["cells"][cell]
        draws = e1s.vcc_draw(cell, "genes")
        covariates = {name: test[name].to_numpy(dtype=np.float64) for name in ("B0", "B1", "B2")}
        for name, values in covariates.items():
            if not np.isfinite(values).all():
                raise Stop("%s: VCC test column %s is not finite" % (cell, name))
        z_norm = np.linalg.norm(data.z[data.split("test")].astype(np.float64), axis=1)
        for label in ("existing", "rebuilt"):
            predictions = [np.asarray(f["test"], dtype=np.float64) for f in e1s.DATA["predictions"].seeds(cell, label, "z", "forest")]
            for name, values in covariates.items():
                s = e1s.summary(*spearman_pairs(predictions, [values] * len(predictions), draws))
                rows.append({"cell": cell, "label": label, "first": f"{label}-label z prediction", "second": name, "n_test_rows": int(values.size),
                             "spearman": s["estimate"], "lo": s["lo"], "hi": s["hi"], "tied_share_first": tied_share(predictions)})
            s = e1s.summary(*spearman_pairs(predictions, [z_norm] * len(predictions), draws))
            rows.append({"cell": cell, "label": label, "first": f"{label}-label z prediction", "second": "norm of z", "n_test_rows": int(z_norm.size),
                         "spearman": s["estimate"], "lo": s["lo"], "hi": s["hi"], "tied_share_first": tied_share(predictions)})
        for name, values in covariates.items():
            s = e1s.summary(*spearman_pairs([z_norm], [values], draws))
            rows.append({"cell": cell, "label": "either", "first": "norm of z", "second": name, "n_test_rows": int(values.size),
                         "spearman": s["estimate"], "lo": s["lo"], "hi": s["hi"], "tied_share_first": float("nan")})
    frame = pd.DataFrame(rows)
    frame["kind"] = "descriptive; VCC reference for the external ranking check; no statement is read from the interval"
    frame["status"] = STATUS
    return frame


# --------------------------------------------------------------------------- L of C2 and C3


def l_control_rows() -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        for screen in SCREENS:
            for representation in ("C2", "C3"):
                entry = {"cell": cell, "screen": screen, "arm": representation, "n_constructs": int(external_table(cell, screen).shape[0])}
                for name, value in (("T_rebuilt", t_value(cell, "rebuilt", representation, screen)), ("T_existing", t_value(cell, "existing", representation, screen)),
                                    ("L", l_value(cell, representation, screen))):
                    s = e1s.summary(*value)
                    entry.update({f"{name}_{k}": s[k] for k in ("estimate", "lo", "hi", "p", "detectable_2.80se")})
                rows.append(entry)
    frame = pd.DataFrame(rows)
    frame["note"] = "L = T on the rebuilt label minus T on the existing label, as for the family arms; results review blocking issue 2"
    frame["status"] = STATUS
    return frame


# --------------------------------------------------------------------------- main


def main() -> None:
    started = time.time()
    loaded = load()
    checks = reproduction()
    support = support_check(loaded["z_norm"])
    frames = {"posthoc_vcc.csv": vcc_rows(), "posthoc_label_scale.csv": label_scale_rows(),
              "posthoc_external_ranking.csv": external_rows(loaded["z_norm"]), "posthoc_vcc_ranking.csv": vcc_ranking_rows(),
              "posthoc_l_controls.csv": l_control_rows()}
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, frame in frames.items():
        frame.to_csv(OUT / name, index=False)
        outputs[name] = file_record(OUT / name)
    record = {
        "plan": PLAN, "status": STATUS, "decided": "bioinf2027/GATE.md, top section (gate G3, 2026-11-12), item 2",
        "review": "bioinf2027/reviews/E1b_posthoc_plan_r1.md", "cell_of_record": RECORD_CELL,
        "settings": {"replicates": e1s.B, "vcc_seed": e1s.VCC_SEED, "batch_seed": e1s.BATCH_SEED, "external_seeds": e1s.EXT_SEED,
                     "head_seeds": list(e1s.SEEDS), "tolerance": TOLERANCE, "alpha_of_the_descriptive_holm": e1s.ALPHA},
        "vcc_replicate_index_sha256": {f"{c} | {k}": v["draw"].index_sha256 for (c, k), v in e1s.STATE["vcc_draws"].items()},
        "external_replicate_index_sha256": {s: e1s.STATE["draws"][s]["draw"].index_sha256 for s in SCREENS},
        "inputs": {"predictions_manifest": file_record(e1s.OUT / "predictions_manifest.json"),
                   **{name: file_record(E1 / name) for name in ("preconditions.csv", "vcc_test.csv", "transfer.csv", "primary_family.csv",
                                                                 "support.csv", "stats.json", "input_records.json")},
                   "cell_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in CELLS}},
        "reproduction": checks, "norm_of_z_against_support.csv": support,
        "outputs": outputs, "smoke": SMOKE, "seconds": round(time.time() - started, 1),
    }
    (OUT / "posthoc.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[posthoc] %d reproduction values within %g (largest %.2e); G6 a0 %.4f; %s; %.0f s"
          % (checks["values_compared"], TOLERANCE, checks["largest_difference"], checks["g6_a0_against_submitted"]["recomputed"],
             ", ".join(f"{k} {len(v)} rows" for k, v in frames.items()), time.time() - started), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--smoke"]:
        SMOKE = True
        e1s.B = 20
        resample.BLOCK = 20
        OUT = config.storage_root() / "bioinf2027" / "scratch" / "E1b_posthoc_smoke"
        PLAN = PLAN + "; smoke run, 20 replicates, never reported"
    elif sys.argv[1:]:
        raise SystemExit("unknown arguments: %s" % sys.argv[1:])
    main()
