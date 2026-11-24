"""E2, the supplement added after the results review: items (a), (b) and (e) of gate G3, item 5 (the author's answers
of 2026-11-16, bioinf2027/GATE.md; the review: bioinf2027/reviews/E2_results_r1.md and _r2.md, "Missing controls or
analyses"). From E2's and E1b's saved predictions, E1b's draws and E6's draws through the code of
make_e2_stats.py; nothing is refitted. Item (c) is make_e2_scgpt_target_input.py.

- (a) In (iii), e - C2 for scGPT, Stack and PCA-50, both directions: e minus E6's C2 (the random projection of x of
  width 768 that E6's z - C2 uses), paired from E6's draws (seeds 20260901 and 20260902, the same replicates as
  E1b's). Geneformer's z - C2 beside them, as the reference. Descriptive, between representations of different
  widths; added after the review.
- (b) Post hoc splits, within each subset of each replicate (as E6 split by "target is a feature"): Stack by
  whether the target gene is in Stack's gene list (basecount_1000per_15000max.pkl, exact symbol, as
  make_e2_coverage.py matches), PCA-50 by whether the target gene is among the 2,000 genes PCA-50 is computed from
  (E1b's target_is_feature). VCC: G6 and G5, forest, R-squared of e and of C1 on both labels, e - C1 (P-a on the
  existing label, P-r on the rebuilt) and P-d, the target-gene cluster bootstrap (seed 20270301) restricted to the
  subset. (iii): T = e - C1 and the Spearman of e and of C1, both directions, E6's draws restricted to the subset.
- (e) A crossed gene by batch interval for P-a, P-d and N0 - C1 in G6 (forest), for Geneformer, scGPT, Stack and
  PCA-50: the pigeonhole bootstrap (Owen, 2007), in which a row's weight in a replicate is the product of its target
  gene's weight in the target-gene draw (seed 20270301) and its batch's weight in the batch draw (seed 20270302),
  so that no new draw is made and every contrast stays inside each replicate. It is somewhat conservative for a
  crossed design (its variance counts the residual term more than once). A sensitivity analysis.

Checks before anything is written (hard limit 9; a failed check raises Stop): every point value and interval that
make_e2_stats.py wrote for the same quantity (vcc.csv: r2, r2_lo, r2_hi and the batch-resampled interval; e6.csv: the
Spearman rows) is reproduced to 1e-12 by this script's draws; Geneformer's z - C2 equals E6's saved value
(bioinf2027/results/E6/secondary.csv) to 1e-12; each split's computation with every row in the subset equals the
unsplit value exactly; every planned fit is present. Outputs under bioinf2027/results/E2/supplement/:
e_minus_c2.csv, splits_vcc.csv, splits_e6.csv, crossed.csv and supplement.json.

With --smoke: 20 replicates on the smoke predictions (head seed 0), outputs to
$IGA_STORAGE_ROOT/bioinf2027/scratch/E2_supplement_smoke/, never reported; the comparison with the saved tables is
skipped (they come from the real predictions), the other checks run.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, resample  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402

import make_e2_coverage as cov  # noqa: E402
import make_e2_stats as st  # noqa: E402

e1s, e6s, e6f, att = st.e1s, st.e6s, st.e6f, st.att
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "supplement"
STATS = REPO / "bioinf2027" / "results" / "E2" / "stats"
E6_SECONDARY = REPO / "bioinf2027" / "results" / "E6" / "secondary.csv"
ADDED = "added after the results review (gate G3, item 5; bioinf2027/GATE.md)"
CROSSED_NOTE = ("pigeonhole bootstrap (Owen, 2007): a row's weight is its target gene's weight (seed 20270301) times its batch's "
                "weight (seed 20270302); somewhat conservative for a crossed design")
TOLERANCE = 1e-12
CROSSED_CELL = "G6"
CROSSED_ENCODERS = ("geneformer", "scgpt", "stack", "pca50")
CROSSED_QUANTITIES = ("e - C1", "P-d", "N0 - C1")
C2_ENCODERS = ("scgpt", "stack", "pca50")
SPLIT_ENCODERS = ("stack", "pca50")
SPLIT_NAMES = {"stack": ("target gene in Stack's gene list", "target gene not in Stack's gene list"),
               "pca50": ("target gene among PCA-50's 2,000 genes", "target gene not among PCA-50's 2,000 genes")}
SMOKE = False
Stop = st.Stop
QUANTITY = {q[0]: q for q in st.QUANTITIES}


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E2_supplement_smoke" if SMOKE else OUT


def close(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= TOLERANCE


class Checks:
    def __init__(self) -> None:
        self.compared, self.largest, self.failed = 0, 0.0, []

    def value(self, where: str, mine: float, saved: float) -> None:
        mine, saved = float(mine), float(saved)
        self.compared += 1
        if not (np.isnan(mine) and np.isnan(saved)):
            self.largest = max(self.largest, abs(mine - saved))
        if not close(mine, saved):
            self.failed.append({"where": where, "mine": mine, "saved": saved})

    def record(self) -> dict:
        return {"values_compared": self.compared, "largest_difference": self.largest, "tolerance": TOLERANCE, "failed": self.failed}


def saved(frame: pd.DataFrame, **where) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for k, v in where.items():
        mask &= (frame[k] == v).to_numpy()
    if mask.sum() != 1:
        raise Stop("%d saved rows for %s" % (mask.sum(), where))
    return frame[mask].iloc[0]


# --------------------------------------------------------------------------- (e) the crossed interval


def crossed_draw(cell: str) -> dict:
    genes, batches = e1s.vcc_draw(cell, "genes"), e1s.vcc_draw(cell, "batches")
    if genes["draw"].weights.shape != batches["draw"].weights.shape:
        raise Stop("the gene and batch draws of %s differ in shape" % cell)
    weights = genes["draw"].weights * batches["draw"].weights
    digest = hashlib.sha256((genes["draw"].index_sha256 + "|" + batches["draw"].index_sha256).encode()).hexdigest()
    draw = resample.ClusterDraw(weights=weights, clusters=np.array([]), seed=-1, index_sha256=digest)
    e1s.STATE["vcc_draws"][(cell, "crossed")] = {"draw": draw, "point": genes["point"], "groups": genes["groups"]}
    return {"replicates": int(weights.shape[0]), "rows": int(weights.shape[1]), "target_genes": int(np.unique(genes["groups"]).size),
            "batches": int(np.unique(batches["groups"]).size), "replicates_with_no_row": int((weights.sum(axis=1) == 0).sum()),
            "gene_draw_index_sha256": genes["draw"].index_sha256, "batch_draw_index_sha256": batches["draw"].index_sha256}


def crossed_rows(vcc: pd.DataFrame | None, checks: Checks) -> list[dict]:
    rows = []
    cell, head = CROSSED_CELL, "forest"
    for encoder in CROSSED_ENCODERS:
        for quantity in CROSSED_QUANTITIES:
            _, used, kind, role, formula, description = QUANTITY[quantity]
            if not st.wanted_quantity(encoder, quantity, used, cell):
                raise Stop("%s, %s: a representation is not fitted" % (encoder, quantity))

            def v(r, label, k, draw):
                s = st.score(cell, label, st.role_name(encoder, r), head, draw)
                return None if s is None else s[k]

            for label in (st.LABELS if kind == "label" else (st.PAIRED,)):
                value = {d: formula(lambda r, lab, k, d=d: v(r, lab, k, d), "r2", label) for d in ("genes", "batches", "crossed")}
                rho = {d: formula(lambda r, lab, k, d=d: v(r, lab, k, d), "spearman", label) for d in ("genes", "crossed")}
                if any(x is None for x in list(value.values()) + list(rho.values())):
                    raise Stop("%s, %s, %s: a planned fit is missing" % (encoder, quantity, label))
                s = {d: e1s.summarise(x) for d, x in value.items()}
                sp = {d: e1s.summarise(x) for d, x in rho.items()}
                if vcc is not None:
                    row = saved(vcc, encoder=encoder, cell=cell, head=head, quantity=quantity, label=label)
                    where = "vcc.csv %s | %s | %s" % (encoder, quantity, label)
                    for mine, theirs in (("estimate", "r2"), ("lo", "r2_lo"), ("hi", "r2_hi")):
                        checks.value(where + " | " + theirs, s["genes"][mine], row[theirs])
                    checks.value(where + " | spearman", sp["genes"]["estimate"], row["spearman"])
                    if np.isfinite(row["batch_resampled_lo"]):
                        checks.value(where + " | batch lo", s["batches"]["lo"], row["batch_resampled_lo"])
                        checks.value(where + " | batch hi", s["batches"]["hi"], row["batch_resampled_hi"])
                rows.append({"encoder": encoder, "width": st.width(encoder), "cell": cell, "head": head, "quantity": quantity,
                             "fact": st.fact(quantity, label), "label": label, "r2": s["genes"]["estimate"],
                             "gene_lo": s["genes"]["lo"], "gene_hi": s["genes"]["hi"], "batch_lo": s["batches"]["lo"], "batch_hi": s["batches"]["hi"],
                             "crossed_lo": s["crossed"]["lo"], "crossed_hi": s["crossed"]["hi"], "crossed_p": s["crossed"]["p"],
                             "crossed_se": s["crossed"]["se"], "crossed_replicates_undefined": s["crossed"]["replicates_undefined"],
                             "state_gene_draw": att.state(s["genes"]["lo"], s["genes"]["hi"]),
                             "state_batch_draw": att.state(s["batches"]["lo"], s["batches"]["hi"]),
                             "state_crossed": att.state(s["crossed"]["lo"], s["crossed"]["hi"]),
                             "spearman": sp["genes"]["estimate"], "spearman_crossed_lo": sp["crossed"]["lo"], "spearman_crossed_hi": sp["crossed"]["hi"],
                             "description": description, "role": "sensitivity analysis, " + ADDED, "method": CROSSED_NOTE,
                             "markers": st.markers(encoder, "VCC")})
    return rows


# --------------------------------------------------------------------------- (a) e - C2 in (iii)


def c2_rows(e6: pd.DataFrame | None, secondary: pd.DataFrame, checks: Checks) -> list[dict]:
    rows = []
    for encoder in (st.GENEFORMER,) + C2_ENCODERS:
        for source in st.SOURCES:
            direction = f"{source} to {e6f.other(source)}"
            e, c2 = st.e6_rho(encoder, source, "e"), st.e6_rho("C2", source, "e")
            if e is None or c2 is None:
                raise Stop("(iii) %s, %s: a planned fit is missing" % (encoder, direction))
            value = st.summary(st.minus(e, c2))
            if not SMOKE:
                if e6 is not None:
                    for name, rv, enc in ((st.roles(encoder)["e"], e, encoder), ("C2", c2, "C2")):
                        row = saved(e6, encoder=enc, direction=direction, quantity=f"rho({name})")
                        checks.value("e6.csv %s | %s | rho(%s)" % (enc, direction, name), rv[0], row["estimate"])
                        lo, hi, _ = e1s.resample_interval(rv[1])
                        checks.value("e6.csv %s | %s | rho(%s) lo" % (enc, direction, name), lo, row["lo"])
                        checks.value("e6.csv %s | %s | rho(%s) hi" % (enc, direction, name), hi, row["hi"])
                if encoder == st.GENEFORMER:
                    row = saved(secondary, direction=direction, contrast="z minus C2")
                    for k in ("estimate", "lo", "hi"):
                        checks.value("E6 secondary.csv %s | z minus C2 | %s" % (direction, k), value[k], row[k])
            rows.append({"encoder": encoder, "width": st.width(encoder), "direction": direction, "endpoint": e6f.PRIMARY, "head": "forest",
                         "quantity": "e - C2" if encoder != st.GENEFORMER else "z - C2", "rho_e": e[0], "rho_C2": c2[0], **value,
                         "state": att.state(value["lo"], value["hi"]),
                         "role": ("between representations of different widths (C2 is E6's random projection of x, width 768), descriptive, "
                                  + ADDED) if encoder != st.GENEFORMER else "reference: E6's z - C2, recomputed",
                         "markers": st.markers(encoder, "Replogle")})
    return rows


# --------------------------------------------------------------------------- (b) the post hoc splits


def flags(column: pd.Series) -> np.ndarray:
    """A saved true/false column as booleans; any other type stops (a text column would read as all true)."""
    if column.dtype != bool:
        raise Stop("the column %s is %s, not boolean" % (column.name, column.dtype))
    return column.to_numpy()


def stack_genes() -> set:
    return cov.vocabularies()["stack"]


def vcc_masked(cell: str, label: str, representation: str, head: str, mask: np.ndarray) -> dict | None:
    """make_e1b_stats.vcc_r2 restricted to the rows of the mask in every replicate (the R-squared within the subset)."""
    wanted = st.wanted_seeds(representation, head)
    fits = st.DATA["lookup"].seeds(cell, label, representation, head, seeds=wanted)
    if len(fits) != len(wanted):
        raise Stop("VCC %s | %s | %s | %s: %d of %d head seeds" % (cell, label, representation, head, len(fits), len(wanted)))
    y = e1s.vcc_label(cell, label)
    draws = e1s.vcc_draw(cell, "genes")
    predictions = np.column_stack([f["test"] for f in fits])
    m = np.asarray(mask, dtype=bool)
    sub = resample.ClusterDraw(weights=np.ascontiguousarray(draws["draw"].weights[:, m]), clusters=draws["draw"].clusters,
                               seed=draws["draw"].seed, index_sha256=draws["draw"].index_sha256)
    point = e1s._ones(int(m.sum()))
    r2 = (float(resample.weighted_r2(y[m], predictions[m], point)[0].mean()), resample.weighted_r2(y[m], predictions[m], sub).mean(axis=1))
    rho = (float(resample.weighted_spearman(predictions[m], y[m], point)[0].mean()), resample.weighted_spearman(predictions[m], y[m], sub).mean(axis=1))
    return {"r2": r2, "spearman": rho, "n_rows": int(m.sum()), "n_genes": int(np.unique(draws["groups"][m]).size)}


def same(a: tuple, b: tuple) -> bool:
    return a[0] == b[0] and np.array_equal(a[1], b[1], equal_nan=True)


def vcc_split_rows(genes: set, unsplit: list) -> list[dict]:
    rows = []
    head = "forest"
    for encoder in SPLIT_ENCODERS:
        for cell in st.CELLS:
            data = e1s.DATA["cells"][cell]
            test = data.vcc[data.split("test")]
            inside = test["target_gene"].astype(str).isin(genes).to_numpy() if encoder == "stack" else flags(test["target_is_feature"])
            names = st.role_name(encoder, "e"), st.role_name(encoder, "C1")
            for subset, mask in (("all", np.ones(inside.size, dtype=bool)), (SPLIT_NAMES[encoder][0], inside), (SPLIT_NAMES[encoder][1], ~inside)):
                if not mask.any():
                    rows.append({"encoder": encoder, "cell": cell, "subset": subset, "n_test_rows": 0, "n_test_genes": 0, "note": "no row in the subset"})
                    continue
                s = {(r, label): vcc_masked(cell, label, n, head, mask) for r, n in zip(("e", "C1"), names) for label in st.LABELS}
                if subset == "all":  # every row: the unsplit value of make_e2_stats.score, exactly
                    for (r, label), x in s.items():
                        full = st.score(cell, label, st.role_name(encoder, r), head)
                        if not (same(x["r2"], full["r2"]) and same(x["spearman"], full["spearman"])):
                            unsplit.append("VCC %s | %s | %s | %s" % (encoder, cell, r, label))
                n = s[("e", "existing")]
                values = {f"R2({names[0]}) | {label}": s[("e", label)]["r2"] for label in st.LABELS}
                values.update({f"R2({names[1]}) | {label}": s[("C1", label)]["r2"] for label in st.LABELS})
                margin = {label: st.minus(s[("e", label)]["r2"], s[("C1", label)]["r2"]) for label in st.LABELS}
                values.update({"e - C1 | existing (P-a)": margin["existing"], "e - C1 | rebuilt (P-r)": margin["rebuilt"],
                               "P-d": st.minus(margin["existing"], margin["rebuilt"])})
                for quantity, value in values.items():
                    q = st.summary(value)
                    rows.append({"encoder": encoder, "width": st.width(encoder), "cell": cell, "head": head, "subset": subset,
                                 "n_test_rows": n["n_rows"], "n_test_genes": n["n_genes"], "quantity": quantity, **q,
                                 "state": att.state(q["lo"], q["hi"]) if not quantity.startswith("R2(") else "",
                                 "role": "post hoc, " + ADDED, "markers": st.markers(encoder, "VCC")})
    return rows


def e6_split_rows(genes: set, unsplit: list) -> list[dict]:
    rows = []
    for encoder in SPLIT_ENCODERS:
        for source in st.SOURCES:
            target = e6f.other(source)
            direction = f"{source} to {target}"
            table = e6s.DATA["tables"][e6f.PRIMARY][target]
            inside = table["target_symbol"].astype(str).isin(genes).to_numpy() if encoder == "stack" else flags(table["target_is_feature"])
            label = e6s.label_of(target, e6f.PRIMARY)
            vectors = {r: st.e6_vectors(encoder, source, r) for r in ("e", "C1")}
            if any(v is None or any(x is None for x in v) for v in vectors.values()):
                raise Stop("(iii) %s, %s: a planned fit is missing" % (encoder, direction))
            for r in ("e", "C1"):  # every construct: the unsplit value of make_e2_stats.e6_rho, exactly
                if not same(e6s.rho(target, vectors[r], label, None), st.e6_rho(encoder, source, r)):
                    unsplit.append("(iii) %s | %s | %s" % (encoder, direction, r))
            for subset, mask in (("all", None), (SPLIT_NAMES[encoder][0], inside), (SPLIT_NAMES[encoder][1], ~inside)):
                m = np.ones(inside.size, dtype=bool) if mask is None else mask
                rho = {r: e6s.rho(target, vectors[r], label, mask) for r in ("e", "C1")}
                genes_in = int(np.unique(table["target_ensg"].astype(str).to_numpy()[m]).size)
                for quantity, value in ((f"rho({st.roles(encoder)['e']})", rho["e"]), ("rho(C1)", rho["C1"]), ("T", st.minus(rho["e"], rho["C1"]))):
                    q = st.summary(value)
                    rows.append({"encoder": encoder, "width": st.width(encoder), "direction": direction, "endpoint": e6f.PRIMARY, "head": "forest",
                                 "subset": subset, "n_constructs": int(m.sum()), "n_target_genes": genes_in, "quantity": quantity, **q,
                                 "state": att.state(q["lo"], q["hi"]) if quantity == "T" else "", "role": "post hoc, " + ADDED,
                                 "markers": st.markers(encoder, "Replogle")})
    return rows


# --------------------------------------------------------------------------- main


def main() -> None:
    started = time.time()
    st.DATA["missing"] = []
    st.DATA["order_marker"] = st.order_marker()
    predictions = st.load_predictions()
    inputs = {**st.load_cells(), **st.load_e6()}
    checks = Checks()
    vcc = None if SMOKE else pd.read_csv(STATS / "vcc.csv", float_precision="round_trip")
    e6 = None if SMOKE else pd.read_csv(STATS / "e6.csv", float_precision="round_trip")
    secondary = pd.read_csv(E6_SECONDARY, float_precision="round_trip")
    draw = crossed_draw(CROSSED_CELL)
    crossed = pd.DataFrame(crossed_rows(vcc, checks))
    print("[e2 supplement] (e) crossed intervals: %d rows, %.0f s" % (len(crossed), time.time() - started), flush=True)
    c2 = pd.DataFrame(c2_rows(e6, secondary, checks))
    print("[e2 supplement] (a) e - C2: %d rows, %.0f s" % (len(c2), time.time() - started), flush=True)
    genes, unsplit = stack_genes(), []
    splits_vcc = pd.DataFrame(vcc_split_rows(genes, unsplit))
    splits_e6 = pd.DataFrame(e6_split_rows(genes, unsplit))
    print("[e2 supplement] (b) splits: %d VCC rows, %d (iii) rows, %.0f s" % (len(splits_vcc), len(splits_e6), time.time() - started), flush=True)
    if checks.failed or unsplit or st.DATA["missing"]:
        raise Stop("a check failed: %s" % json.dumps({"reproduction": checks.failed[:10], "unsplit_not_equal": unsplit, "missing": st.DATA["missing"]}))
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    frames = {"crossed.csv": crossed, "e_minus_c2.csv": c2, "splits_vcc.csv": splits_vcc, "splits_e6.csv": splits_e6}
    for name, frame in frames.items():
        frame.to_csv(folder / name, index=False)
    record = {"item": "E2, supplement (gate G3, item 5 (a), (b) and (e); bioinf2027/GATE.md)", "status": ADDED, "smoke": SMOKE,
              "settings": {"replicates": e1s.B, "vcc_seed": e1s.VCC_SEED, "batch_seed": e1s.BATCH_SEED, "e6_seeds": e6s.SEED,
                           "e6_endpoint": e6f.PRIMARY, "tolerance": TOLERANCE, "crossed_cell": CROSSED_CELL},
              "crossed_draw": draw, "crossed_method": CROSSED_NOTE,
              "checks": {"reproduction_of_saved_values": checks.record(), "unsplit_equal_to_the_unsplit_value": not unsplit,
                         "planned_fits_present": not st.DATA["missing"]},
              "stack_gene_list": {"genes": len(genes), "record": file_record(REPO / "bioinf2027" / "results" / "E2" / "downloads.json")},
              "inputs": {**inputs, "predictions": predictions, "stats": {n: file_record(STATS / n) for n in ("vcc.csv", "e6.csv", "stats.json")},
                         "e6_secondary": file_record(E6_SECONDARY)},
              "code": {config.portable(Path(m.__file__).resolve()): sha256(Path(m.__file__)) for m in (sys.modules[__name__], st, cov, e1s, e6s, resample)},
              "outputs": {n: file_record(folder / n) for n in frames}, "seconds": round(time.time() - started, 1)}
    (folder / "supplement.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 supplement] done in %.0f s; %d saved values reproduced (largest difference %.2e)%s"
          % (time.time() - started, checks.compared, checks.largest, " (smoke run)" if SMOKE else ""), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--smoke"]:
        SMOKE = True
        st.SMOKE = True
        e1s.B = 20
        resample.BLOCK = 20
    elif arguments:
        raise SystemExit("usage: make_e2_supplement.py [--smoke]")
    main()
