"""E1b attribution: does z add to cell count on the submitted label? With D2's z- against the one-variable heads.

Note: bioinf2027/plans/E1b_attribution.md (decided by the author at gate G2, items 7 and 12; reviewed once,
bioinf2027/reviews/E1b_attribution_plan_r1.md). Every value is post hoc, secondary and unadjusted. E1b's and
D2's audited outputs are not changed.

Two steps, each run under bioinf2027/scripts/run_item.py --item E1:

--fits   84 new fits: [B0; z], [B0; C1] and [B0; C3] (B0 followed by E1b's z, C1 or C3 block, built by
         transfer.Blocks on the same training rows), labels existing and rebuilt, cells G5 and G6, forest and
         boosting (head seeds 0 to 2) and ridge (seed 0), as E1b's main fits. Before any new fit, four E1b fits
         are refitted through this script's fitting code without the B0 column and compared exactly with E1b's
         saved files (check 1), and the blocks are compared with transfer.Blocks (check 2). Predictions under
         $IGA_STORAGE_ROOT/bioinf2027/E1b/attribution/predictions/; manifest and protocol under
         bioinf2027/results/E1/attribution/.
--stats  The VCC-test contrasts of the note, Section 2, from the target-gene cluster bootstrap of E1b's
         precondition (10,000 replicates, seed 20270301) through make_e1b_stats.vcc_r2, unchanged; D2's z- from
         make_d2_stats.predictions. Reproduction checks first (note, Section 5); outputs attribution_vcc.csv,
         attribution_s2.csv and attribution.json.

With --smoke (after --fits or --stats): forest 5 trees, boosting 20 iterations, 20 replicates, outputs to
$IGA_STORAGE_ROOT/bioinf2027/scratch/E1b_attribution_smoke/, never reported. Check 1 runs with the real head
settings in the smoke fit step.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import copy  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, e1inputs, heads, resample, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import make_d2_stats as d2s  # noqa: E402
import make_e1b_fits as e1f  # noqa: E402
import make_e1b_stats as e1s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E1 = REPO / "bioinf2027" / "results" / "E1"
OUT = E1 / "attribution"
PLAN = "bioinf2027/plans/E1b_attribution.md"
STATUS = "post hoc, secondary, unadjusted (bioinf2027/plans/E1b_attribution.md)"
NEW = {"[B0; z]": "z", "[B0; C1]": "C1", "[B0; C3]": "C3"}
CELLS = tuple(e1cells.PRIMARY)
LABELS = ("existing", "rebuilt")
HEAD_SEEDS = (("forest", (0, 1, 2)), ("boosting", (0, 1, 2)), ("ridge", (0,)))
RECORD_CELL = "G6"
TOLERANCE = 1e-12
WORKERS = 4
SMOKE = False
DATA: dict = {}


class Stop(RuntimeError):
    """A check failed or an input is missing: nothing is written (hard limit 9)."""


def scratch() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E1b_attribution_smoke"


def predictions_root() -> Path:
    if SMOKE:
        return scratch() / "predictions"
    return config.storage_root() / "bioinf2027" / "E1b" / "attribution" / "predictions"


def out_dir() -> Path:
    return scratch() if SMOKE else OUT


# --------------------------------------------------------------------------- fitting


class Matrix:
    """B0 followed by the block of E1b's fit of the base representation; with_b0=False gives E1b's own matrix."""

    def __init__(self, spec: FitSpec, train: transfer.Side, rows: np.ndarray, with_b0: bool = True) -> None:
        self.with_b0 = with_b0 and spec.representation in NEW
        base = NEW.get(spec.representation, spec.representation)
        self.blocks = transfer.Blocks(FitSpec(spec.context, spec.label, base, spec.head, spec.seed, spec.variant, spec.realisation), train, rows)

    def __call__(self, side: transfer.Side) -> np.ndarray:
        block = self.blocks.matrix(side)
        if not self.with_b0:
            return block
        return np.concatenate([side.b[:, [0]], block], axis=1).astype(np.float32)


def fit(context: transfer.Context, spec: FitSpec, with_b0: bool = True) -> tuple[dict, dict]:
    """transfer.run_fit, step by step, with the matrix of Matrix: the predictions on every side and the record."""
    rows = transfer.training_rows(context, spec, e1f.REFIT_SEED)
    y = context.labels[spec.label][rows]
    if not np.isfinite(y).all():
        raise RuntimeError("%s: a training label is missing" % spec.key())
    matrix = Matrix(spec, context.train, rows, with_b0)
    start = time.time()
    record, out = {}, {}
    with heads.single_thread():
        x_train = matrix(context.train)[rows]
        if spec.head == "ridge":
            val = context.sides["val"]
            val_label = context.labels[spec.label + "@val"]
            val_mask = np.isfinite(val_label)
            model, scaler, choice = heads.fit_ridge_by_validation(x_train, y, matrix(val)[val_mask], val_label[val_mask])
            record["ridge"] = choice

            def predict(m):
                return model.predict(scaler.transform(np.asarray(m, dtype=np.float64)))
        else:
            model = transfer._head(spec).fit(x_train, y)

            def predict(m):
                return model.predict(m)

        for name, side in context.sides.items():
            m = matrix(side)
            values = np.full(side.n, np.nan)
            usable = np.isfinite(m).all(axis=1)
            if usable.any():
                values[usable] = predict(m[usable])
            out[name] = values
    record["seconds"] = time.time() - start
    record["training_rows"] = int(rows.size)
    return out, record


def save(spec: FitSpec, out: dict, record: dict) -> dict:
    folder = predictions_root() / spec.context
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / spec.file_name()
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, spec=np.array(json.dumps(spec.as_dict(), sort_keys=True)), record=np.array(json.dumps(record, sort_keys=True)),
             **{name.replace("|", "__"): values for name, values in out.items()})
    os.replace(temporary, path)
    return {"spec": spec.as_dict(), "file": file_record(path), "seconds": record["seconds"]}


def worker(spec: FitSpec) -> dict:
    path = predictions_root() / spec.context / spec.file_name()
    if path.exists():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == json.dumps(spec.as_dict(), sort_keys=True):
                return {"spec": spec.as_dict(), "file": file_record(path), "reused": True}
    out, record = fit(DATA["contexts"][spec.context], spec)
    return {**save(spec, out, record), "reused": False}


def check_worker(spec: FitSpec) -> dict:
    """Check 1: an E1b fit refitted through this script's code (no B0 column), compared exactly with E1b's file."""
    out, record = fit(DATA["contexts"][spec.context], spec, with_b0=False)
    saved = DATA["e1b"].get(spec.context, spec.label, spec.representation, spec.head, spec.seed)
    if saved is None:
        raise Stop("check 1: E1b holds no %s" % spec.key())
    sides = {}
    for name, values in out.items():
        if name not in saved:
            raise Stop("check 1: E1b's %s has no side %s" % (spec.key(), name))
        sides[name] = {"equal": bool(np.array_equal(values, saved[name], equal_nan=True)),
                       "largest_difference": float(np.nanmax(np.abs(values - saved[name]))) if np.isfinite(values).any() else 0.0}
    extra = sorted(k for k in saved if not k.startswith("_") and "|" not in k and k not in out)
    return {"fit": spec.key(), "seconds": round(record["seconds"], 1), "sides": sides, "sides_missing_here": extra,
            "passed": all(v["equal"] for v in sides.values()) and not extra}


def tasks() -> list[FitSpec]:
    out = [FitSpec(cell, label, r, head, seed) for cell in CELLS for label in LABELS for r in NEW for head, seeds in HEAD_SEEDS for seed in seeds]
    if SMOKE:
        out = [t for t in out if t.seed == 0]

    def cost(t: FitSpec):
        weight = {"forest": 1.0, "boosting": 0.05, "ridge": 0.02}[t.head]
        return -DATA["train_rows"][t.context] * weight * (1.4 if t.representation == "[B0; C1]" else 1.0)

    return sorted(out, key=cost)


CHECK_FITS = (FitSpec("G6", "existing", "C1", "boosting", 0), FitSpec("G6", "existing", "z", "ridge", 0),
              FitSpec("G5", "existing", "z", "forest", 0), FitSpec("G6", "existing", "B0", "forest", 0))


def blocks_check() -> dict:
    """Check 2: the matrix is the side's B0 followed by transfer.Blocks of the base representation, exactly."""
    out = {}
    for cell in CELLS:
        context = DATA["contexts"][cell]
        rows = transfer.training_rows(context, FitSpec(cell, "existing", "z", "forest", 0), e1f.REFIT_SEED)
        for representation, base in NEW.items():
            spec = FitSpec(cell, "existing", representation, "forest", 0)
            matrix = Matrix(spec, context.train, rows)
            reference = transfer.Blocks(FitSpec(cell, "existing", base, "forest", 0), context.train, rows)
            for name, side in [("train", context.train)] + list(context.sides.items()):
                m = matrix(side)
                ok = (m.shape[1] == reference.matrix(side).shape[1] + 1
                      and np.array_equal(m[:, 0], side.b[:, 0].astype(np.float32), equal_nan=True)
                      and np.array_equal(m[:, 1:], reference.matrix(side).astype(np.float32), equal_nan=True))
                out[f"{cell} | {representation} | {name}"] = bool(ok)
                if not ok:
                    raise Stop("check 2: the %s matrix of %s on %s is not B0 followed by the %s block" % (representation, cell, name, base))
    return {"compared": len(out), "passed": True}


def run_fits() -> None:
    started = time.time()
    inputs = e1f.load_inputs(list(CELLS))
    order = list(e1cells.CELLS)
    DATA["contexts"] = {cell: e1f.build_context(cell, order.index(cell), {}) for cell in CELLS}
    DATA["train_rows"] = {cell: int(e1f.DATA["cells"][cell].split("train").sum()) for cell in CELLS}
    DATA["e1b"] = e1s.Predictions(E1 / "predictions_manifest.json")
    checks = {"blocks": blocks_check()}
    context = multiprocessing.get_context("fork")
    real_heads = (copy.deepcopy(heads.FOREST), copy.deepcopy(heads.BOOSTING))  # check 1 always runs with E1b's settings
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        results = list(pool.map(check_worker, CHECK_FITS))
    checks["pipeline_equality"] = results
    print("[attribution] check 1: %s" % "; ".join("%s %s (%.0f s)" % (r["fit"], "equal" if r["passed"] else "DIFFERS", r["seconds"]) for r in results), flush=True)
    if not all(r["passed"] for r in results):
        raise Stop("check 1 failed: %s" % json.dumps(results))
    if SMOKE:
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
    listed = tasks()
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"plan": PLAN, "fits": len(listed), "representations": NEW, "labels": list(LABELS), "cells": list(CELLS),
                "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING, "ridge grid": list(heads.RIDGE_GRID),
                          "threads": "one: forest n_jobs=1, OpenMP and BLAS limited to one thread"},
                "check_1_heads": {"forest": real_heads[0], "boosting": real_heads[1]},
                "seeds": {"heads": {h: list(s) for h, s in HEAD_SEEDS}, "C1 and C3 realisation": 0, "refit seed (unused, all training rows)": e1f.REFIT_SEED},
                "inputs": {**inputs, "e1b_predictions": file_record(E1 / "predictions_manifest.json"), "e1b_protocol": file_record(E1 / "protocol.json")},
                "fitting_code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (transfer, heads, sys.modules["bioinf2027.representations"])},
                "script": {config.portable(Path(__file__)): sha256(Path(__file__))},
                "predictions": config.portable(predictions_root()), "smoke": SMOKE, "checks": checks}
    protocol_path = folder / "protocol.json"
    protocol_path.write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        pending = {pool.submit(worker, t): t for t in listed}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as error:  # recorded and reported; the run continues
                    failed.append({"fit": spec.key(), "error": repr(error)})
                    print("[attribution] FAILED %s: %r" % (spec.key(), error), flush=True)
                    continue
                if len(done) % 10 == 0 or not pending:
                    print("[attribution] %d of %d done, %d failed, %.0f s" % (len(done), len(listed), len(failed), time.time() - started), flush=True)
    manifest = {"protocol": file_record(protocol_path), "fits": len(listed), "completed": len(done), "failed": failed,
                "reused": sum(bool(r.get("reused")) for r in done), "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done},
                "seconds": round(time.time() - started, 1)}
    (folder / "predictions_manifest.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[attribution] %d completed, %d failed, %.0f s" % (len(done), len(failed), time.time() - started), flush=True)
    if failed:
        raise SystemExit(1)


# --------------------------------------------------------------------------- statistics


class Lookup:
    """One lookup over this note's manifest and E1b's: make_e1b_stats.vcc_r2 reads DATA["predictions"]."""

    def __init__(self, first: e1s.Predictions, second: e1s.Predictions) -> None:
        shared = set(first.files) & set(second.files)
        if shared:
            raise Stop("check 5: the two manifests share %d keys" % len(shared))
        self.sources = (first, second)

    def get(self, *arguments, **keywords):
        for source in self.sources:
            values = source.get(*arguments, **keywords)
            if values is not None:
                return values
        return None

    def seeds(self, context, label, representation, head="forest", variant=e1s.ALL_ROWS, realisation=0, seeds=e1s.SEEDS):
        out = []
        for seed in seeds:
            values = self.get(context, label, representation, head, seed, variant, realisation)
            if values is not None:
                out.append(values)
        return out


CACHE: dict = {}


def score(cell: str, label: str, representation: str, head: str, kind: str = "genes") -> dict:
    key = (cell, label, representation, head, kind)
    if key not in CACHE:
        wanted = dict(HEAD_SEEDS)[head]
        if SMOKE and representation in NEW:
            wanted = (0,)  # the smoke fits hold head seed 0 only
        if representation == "z-":
            fits = [d2s.predictions(DATA["d2"], FitSpec(f"D2 | {cell} | z-", label, "z", head, s)) for s in wanted]
            if any(f is None or "test" not in f for f in fits):
                raise Stop("%s | %s | z- | %s: a D2 fit is missing" % (cell, label, head))
            CACHE[key] = scored(cell, label, [f["test"] for f in fits], kind)
        else:
            if len(e1s.DATA["predictions"].seeds(cell, label, representation, head, seeds=wanted)) != len(wanted):
                raise Stop("%s | %s | %s | %s does not have every head seed" % (cell, label, representation, head))
            CACHE[key] = e1s.vcc_r2(cell, label, representation, head, kind=kind)
    return CACHE[key]


def scored(cell: str, label: str, vectors: list, kind: str) -> dict:
    """make_e1b_stats.vcc_r2 for predictions held outside its lookup (D2's z-), the same computation."""
    y = e1s.vcc_label(cell, label)
    draws = e1s.vcc_draw(cell, kind)
    predictions = np.column_stack(vectors)
    r2_point = resample.weighted_r2(y, predictions, draws["point"])[0].mean()
    r2 = resample.weighted_r2(y, predictions, draws["draw"]).mean(axis=1)
    rho_point = resample.weighted_spearman(predictions, y, draws["point"])[0].mean()
    rho = resample.weighted_spearman(predictions, y, draws["draw"]).mean(axis=1)
    return {"r2": (float(r2_point), r2), "spearman": (float(rho_point), rho), "n_rows": int(y.size), "n_genes": int(np.unique(draws["groups"]).size)}


def minus(a, b):
    return e1s.contrast(a, b)


def half(a):
    return a[0] / 2.0, a[1] / 2.0


def state(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "undefined"
    return "above zero" if lo > 0 else ("below zero" if hi < 0 else "contains zero")


READINGS = {
    "d1": ("adding z to cell count raises R-squared under this head", "nothing added by z to cell count is resolved",
           "adding z to cell count lowers R-squared under this head; read only with (d2) and (d3)"),
    "d2": ("a Gaussian block of 768 columns raises R-squared beside cell count: an effect of the head", "no effect of a block of this width is resolved",
           "a Gaussian block of 768 columns lowers R-squared beside cell count (dilution)"),
    "d2'": ("z's own columns with the rows shuffled raise R-squared beside cell count: an effect of the head",
            "no effect of z's shuffled columns is resolved", "z's own columns with the rows shuffled lower R-squared beside cell count (dilution)"),
    "d3": ("beside cell count, z gives this forest more than an independent Gaussian block of the same width; not separated from a difference in block "
           "structure (see d3'); consistent with z carrying the control-cell mean or the own shift, which neither model holds",
           "beside cell count, z is not shown to differ from a Gaussian block of the same width", "beside cell count, a Gaussian block does better than z"),
    "d3'": ("beside cell count, z gives this forest more than its own columns with the rows shuffled; consistent with z carrying the control-cell mean or "
            "the own shift", "beside cell count, z is not shown to differ from its row-shuffled copy", "beside cell count, the row-shuffled copy does better than z"),
    "Delta": ("beside cell count, z's advantage over a Gaussian block of the same width is smaller than without it", "the change of the advantage is not resolved",
              "beside cell count, z's advantage over the Gaussian block is larger than without it"),
    "D1/2": ("not read", "not read",
             "beside cell count, z's advantage over the Gaussian block is less than half of its forest margin over C1 without cell count"),
    "e1": ("z- exceeds cell count alone", "z- is not shown to differ from cell count alone", "cell count alone exceeds z-"),
    "e2": ("z- exceeds the control-cell mean alone", "z- is not shown to differ from the control-cell mean alone", "the control-cell mean alone exceeds z-"),
    "e3": ("z- exceeds the own shift alone", "z- is not shown to differ from the own shift alone", "the own shift alone exceeds z-"),
}
READ = tuple(READINGS)


def reading(name: str, s: str, d2_state: str | None = None) -> str:
    above, contains, below = READINGS[name]
    if name == "d1" and s == "above zero" and d2_state == "above zero":
        return "not read: a Gaussian block of the same width also raises it (d2 above zero)"
    return {"above zero": above, "contains zero": contains, "below zero": below}.get(s, "undefined")


def quantities(cell: str, label: str, head: str) -> dict:
    """The note's quantities for one cell, label and head, each (R-squared pair, Spearman pair, description)."""
    s = {r: score(cell, label, r, head) for r in ("B0", "[B0; z]", "[B0; C1]", "[B0; C3]", "z", "C1", "C3")}

    def pair(f):
        return f("r2"), f("spearman")

    out = {
        "d1": (*pair(lambda k: minus(s["[B0; z]"][k], s["B0"][k])), "[B0; z] minus B0"),
        "d2": (*pair(lambda k: minus(s["[B0; C1]"][k], s["B0"][k])), "[B0; C1] minus B0 (control)"),
        "d2'": (*pair(lambda k: minus(s["[B0; C3]"][k], s["B0"][k])), "[B0; C3] minus B0 (control, z's columns with the rows shuffled)"),
        "d3": (*pair(lambda k: minus(s["[B0; z]"][k], s["[B0; C1]"][k])), "[B0; z] minus [B0; C1]"),
        "d3'": (*pair(lambda k: minus(s["[B0; z]"][k], s["[B0; C3]"][k])), "[B0; z] minus [B0; C3]"),
        "Delta": (*pair(lambda k: minus(minus(s["z"][k], s["C1"][k]), minus(s["[B0; z]"][k], s["[B0; C1]"][k]))),
                  "Delta = [z minus C1] minus [[B0; z] minus [B0; C1]], inside each replicate"),
        "D1/2": (*pair(lambda k: minus(minus(s["[B0; z]"][k], s["[B0; C1]"][k]), half(minus(s["z"][k], s["C1"][k])))),
                 "D1/2 = [[B0; z] minus [B0; C1]] minus half of [z minus C1], inside each replicate"),
        "M": (*pair(lambda k: minus(s["z"][k], s["C1"][k])), "M = z minus C1; reference"),
        "z - C3": (*pair(lambda k: minus(s["z"][k], s["C3"][k])), "z minus C3; reference"),
        "[C1; B2] - B2": (*pair(lambda k: minus(score(cell, label, "[C1; B2]", head)[k], score(cell, label, "B2", head)[k])),
                          "[C1; B2] minus the B2 head; reference for the effect of a 768-column block beside one column"),
    }
    if label == "existing":
        zm = score(cell, label, "z-", head)
        for name, other, text in (("e1", "B0", "cell count"), ("e2", "B1", "control-cell mean"), ("e3", "B2", "own shift")):
            out[name] = (*pair(lambda k, o=other: minus(zm[k], score(cell, label, o, head)[k])), f"z- minus the {other} head ({text})")
        if head == "forest":
            out["a2"] = (*pair(lambda k: minus(score(cell, label, "[TR; z]", head)[k], score(cell, label, "TR", head)[k])),
                         "[TR; z] minus TR (post hoc note's a2); beside d3 and d3'")
    return out, s


def vcc_rows() -> pd.DataFrame:
    rows = []
    for cell in CELLS:
        for label in LABELS:
            for head, _ in HEAD_SEEDS:
                values, scores = quantities(cell, label, head)
                for name, (r2, rho, description) in values.items():
                    r = e1s.summary(*r2)
                    sp = e1s.summary(*rho)
                    entry = {"cell": cell, "label": label, "head": head, "quantity": name, "description": description,
                             "n_test_rows": scores["z"]["n_rows"], "n_test_genes": scores["z"]["n_genes"],
                             "r2": r["estimate"], "r2_lo": r["lo"], "r2_hi": r["hi"], "r2_p": r["p"], "r2_se": r["se"], "r2_detectable_2.80se": r["detectable_2.80se"],
                             "spearman": sp["estimate"], "spearman_lo": sp["lo"], "spearman_hi": sp["hi"], "state": state(r["lo"], r["hi"])}
                    if name == "d3":
                        m = values["M"][0][0]
                        entry["half_of_M_description"] = m / 2.0
                        entry["ratio_d3_to_M_description"] = r["estimate"] / m if m != 0 else float("nan")
                    if cell == RECORD_CELL and head == "forest" and name in ("d3", "d3'", "Delta"):
                        batch = {k: score(cell, label, k, head, kind="batches") for k in ("B0", "[B0; z]", "[B0; C1]", "[B0; C3]", "z", "C1")}
                        if name == "d3":
                            v = minus(batch["[B0; z]"]["r2"], batch["[B0; C1]"]["r2"])
                        elif name == "d3'":
                            v = minus(batch["[B0; z]"]["r2"], batch["[B0; C3]"]["r2"])
                        else:
                            v = minus(minus(batch["z"]["r2"], batch["C1"]["r2"]), minus(batch["[B0; z]"]["r2"], batch["[B0; C1]"]["r2"]))
                        lo, hi, _ = e1s.resample_interval(v[1])
                        entry.update({"batch_resampled_lo": lo, "batch_resampled_hi": hi,
                                      "batch_resampled_note": "sensitivity analysis: whole batches, 10,000 replicates, seed 20270302"})
                    entry["note"] = "" if cell == RECORD_CELL else "not the submitted statistic (G5); reported beside G6, no reading"
                    rows.append(entry)
                for name in ("B0", "[B0; z]", "[B0; C1]", "[B0; C3]", "z", "C1", "C3"):
                    r = e1s.summary(*scores[name]["r2"])
                    rows.append({"cell": cell, "label": label, "head": head, "quantity": f"R2({name})", "description": "reference",
                                 "n_test_rows": scores[name]["n_rows"], "n_test_genes": scores[name]["n_genes"],
                                 "r2": r["estimate"], "r2_lo": r["lo"], "r2_hi": r["hi"], "r2_p": float("nan"), "state": "",
                                 "note": "" if cell == RECORD_CELL else "not the submitted statistic (G5)"})
    frame = pd.DataFrame(rows)
    # readings: G6, existing label, forest only; the (d1) reading conditional on (d2)
    frame["reading"] = ""
    record = (frame.cell == RECORD_CELL) & (frame.label == "existing") & (frame["head"] == "forest")
    d2_state = frame.loc[record & (frame.quantity == "d2"), "state"].iloc[0]
    for index in frame.index[record & frame.quantity.isin(READ)]:
        frame.loc[index, "reading"] = reading(frame.loc[index, "quantity"], frame.loc[index, "state"], d2_state)
    # boosting agreement with the forest (sign and resolution), G6
    frame["boosting_agrees_with_forest"] = ""
    for label in LABELS:
        for name in READ:
            f = frame[(frame.cell == RECORD_CELL) & (frame.label == label) & (frame["head"] == "forest") & (frame.quantity == name)]
            b = frame[(frame.cell == RECORD_CELL) & (frame.label == label) & (frame["head"] == "boosting") & (frame.quantity == name)]
            if len(f) and len(b):
                frame.loc[f.index, "boosting_agrees_with_forest"] = str(bool(f["state"].iloc[0] == b["state"].iloc[0]))
    rejected, thresholds = resample.holm(frame.loc[record & frame.quantity.isin(READ), "r2_p"].to_numpy(), e1s.ALPHA)
    frame["descriptive_holm_g6"] = ""
    for index, r, t in zip(frame.index[record & frame.quantity.isin(READ)], rejected, thresholds):
        frame.loc[index, "descriptive_holm_g6"] = "%s at threshold %.4f (descriptive, not a decision rule)" % ("rejected" if r else "not rejected", t)
    frame["status"] = STATUS
    return frame


def s2_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for head in ("forest", "boosting"):
        get = {r.quantity: r for r in frame[(frame.cell == RECORD_CELL) & (frame.label == "existing") & (frame["head"] == head)].itertuples()}
        a = get["d3"].state == "above zero" and get["d3'"].state == "above zero"
        a1 = get["d3"].state == "above zero" and get["d3'"].state != "above zero"
        b = get["Delta"].state == "above zero"
        c = bool(get["D1/2"].r2_hi < 0)
        for row, holds, condition, text in (
                ("A", a, "(d3) and (d3') both resolved above zero",
                 "beside the perturbed-cell count, z gives this forest more than an independent Gaussian block and more than its own row-shuffled columns of the "
                 "same width; consistent with z carrying the control-cell mean or the own shift"),
                ("A'", a1, "(d3) resolved above zero, (d3') not",
                 "beside cell count, z gives this forest more than an independent Gaussian block of the same width; not separated from a difference in block structure"),
                ("B", b, "Delta resolved above zero", "beside cell count, z's advantage over a Gaussian block of the same width is smaller than without it"),
                ("C", c, "D1/2 upper bound below zero",
                 "beside cell count, z's advantage over the Gaussian block is less than half of its forest margin over C1 without it"),
                ("D", not (a or b or c), "none of A, B and C", "not determined")):
            rows.append({"cell": RECORD_CELL, "label": "existing", "head": head, "row": row, "condition": condition, "holds": bool(holds), "reading": text,
                         "role": "the S2 reading (forest)" if head == "forest" else "beside the forest: the S2 reading is specific to the forest unless boosting agrees",
                         "status": STATUS})
    out = pd.DataFrame(rows)
    forest = out[out["head"] == "forest"].set_index("row")["holds"]
    boosting = out[out["head"] == "boosting"].set_index("row")["holds"]
    out["boosting_agrees"] = out["row"].map(lambda r: bool(forest[r] == boosting[r]))
    return out


def reproduction() -> dict:
    largest, compared = 0.0, 0

    def compare(what, ours, theirs):
        nonlocal largest, compared
        difference = abs(float(ours) - float(theirs))
        largest = max(largest, difference)
        compared += 1
        if not difference <= TOLERANCE:
            raise Stop("check 3: %s is %r here and %r on file; nothing is written" % (what, ours, theirs))

    names = ("estimate",) if SMOKE else ("estimate", "lo", "hi")
    pre = pd.read_csv(E1 / "preconditions.csv")
    for cell in CELLS:
        for label in LABELS:
            row = pre[(pre.cell == cell) & (pre.label == label) & (pre.arm == "z")].iloc[0]
            z, c1 = score(cell, label, "z", "forest"), score(cell, label, "C1", "forest")
            compare(f"{cell} {label} r2_arm", z["r2"][0], row.r2_arm)
            compare(f"{cell} {label} r2_C1", c1["r2"][0], row.r2_C1)
            s = e1s.summary(*minus(z["r2"], c1["r2"]))
            for n, column in zip(names, ("r2_difference_estimate", "r2_difference_lo", "r2_difference_hi")):
                compare(f"{cell} {label} z - C1 {n}", s[n], getattr(row, column))
    posthoc = pd.read_csv(E1 / "posthoc" / "posthoc_vcc.csv")
    for cell in CELLS:
        for name, (a, b) in {"a1": ("z", "B0"), "a1b": ("z", "B1"), "a1c": ("z", "B2"), "a2": ("[TR; z]", "TR")}.items():
            row = posthoc[(posthoc.cell == cell) & (posthoc.quantity == name)].iloc[0]
            s = e1s.summary(*minus(score(cell, "existing", a, "forest")["r2"], score(cell, "existing", b, "forest")["r2"]))
            for n, column in zip(names, ("r2_difference", "r2_lo", "r2_hi")):
                compare(f"{cell} {name} {n}", s[n], getattr(row, column))
    existing = pd.read_csv(REPO / "bioinf2027" / "results" / "D2" / "supplement" / "existing_label.csv")
    for cell in CELLS:
        row = existing[(existing.cell == cell) & (existing.quantity == "R2(z-)") & (existing.replicates == "target-gene clusters")].iloc[0]
        s = e1s.summary(*score(cell, "existing", "z-", "forest")["r2"])
        for n in names:
            compare(f"{cell} R2(z-) {n}", s[n], getattr(row, n))
    digests = {}
    recorded = json.loads((E1 / "posthoc" / "posthoc.json").read_text())["vcc_replicate_index_sha256"]
    for cell in CELLS:
        for kind, name in (("genes", "genes"), ("batches", "batches")):
            key = f"{cell} | {name}"
            ours = e1s.vcc_draw(cell, kind)["draw"].index_sha256
            digests[key] = {"here": ours, "posthoc.json": recorded.get(key), "equal": ours == recorded.get(key)}
            if not SMOKE and not digests[key]["equal"]:
                raise Stop("check 3: the replicate index of %s differs from the post hoc note's" % key)
    return {"values_compared": compared, "largest_difference": largest, "tolerance": TOLERANCE, "replicate_index_digests": digests, "passed": True}


def control_cells() -> dict:
    out = {}
    for cell in CELLS:
        batches = sorted(set(e1s.DATA["cells"][cell].vcc.loc[e1s.DATA["cells"][cell].split("test"), "batch"].astype(str)))
        counts = [DATA["control_counts"][b] for b in batches]
        out[cell] = {"batches": len(batches), "smallest": int(min(counts)), "median": float(np.median(counts)), "largest": int(max(counts))}
    return out


def run_stats() -> None:
    started = time.time()
    folder = out_dir()
    manifest_path = folder / "predictions_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["failed"] or manifest["completed"] != manifest["fits"]:
        raise Stop("check 4: the fit step did not complete every fit")
    e1s.DATA["predictions"] = Lookup(e1s.Predictions(manifest_path), e1s.Predictions(E1 / "predictions_manifest.json"))
    DATA["d2"] = d2s.manifest(REPO / "bioinf2027" / "results" / "D2" / "predictions_manifest.json")
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    DATA["control_counts"] = means.counts
    e1s.DATA["cells"] = {}
    for cell in CELLS:
        if e1cells.CELLS[cell]["unit"] != "row":
            raise Stop("%s is not a row-level cell" % cell)
        e1s.DATA["cells"][cell] = e1cells.build_cell(cell, rows, means)
    e1s.STATE["vcc_draws"] = {}
    checks = {"reproduction": reproduction(), "manifests_disjoint": True, "fits_complete": True}
    frame = vcc_rows()
    s2 = s2_rows(frame)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "attribution_vcc.csv", index=False)
    s2.to_csv(folder / "attribution_s2.csv", index=False)
    record = {"plan": PLAN, "replicates": e1s.B, "seeds": {"target-gene clusters": e1s.VCC_SEED, "batches": e1s.BATCH_SEED},
              "heads": {h: list(s) for h, s in HEAD_SEEDS}, "cell_of_record": RECORD_CELL,
              "inputs": {"attribution_predictions": file_record(manifest_path), "e1b_predictions": file_record(E1 / "predictions_manifest.json"),
                         "d2_predictions": file_record(REPO / "bioinf2027" / "results" / "D2" / "predictions_manifest.json"),
                         "preconditions": file_record(E1 / "preconditions.csv"), "posthoc_vcc": file_record(E1 / "posthoc" / "posthoc_vcc.csv"),
                         "d2_existing_label": file_record(REPO / "bioinf2027" / "results" / "D2" / "supplement" / "existing_label.csv"),
                         "cell_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in CELLS}},
              "control_cells_per_batch_over_the_test_rows": control_cells(), "checks": checks,
              "outputs": {"attribution_vcc.csv": file_record(folder / "attribution_vcc.csv"), "attribution_s2.csv": file_record(folder / "attribution_s2.csv")},
              "smoke": SMOKE, "seconds": round(time.time() - started, 1)}
    (folder / "attribution.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[attribution stats] done in %.0f s; %d rows; S2 rows holding (forest): %s" % (
        time.time() - started, len(frame), ", ".join(s2[(s2["head"] == "forest") & s2["holds"]]["row"])), flush=True)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
        e1f.SMOKE = False  # E1b's own predictions are read from E1b's folder
        e1s.B = 20
        resample.BLOCK = 20
    if arguments == ["--fits"]:
        run_fits()
    elif arguments == ["--stats"]:
        run_stats()
    else:
        raise SystemExit("a run takes --fits or --stats (and optionally --smoke)")
