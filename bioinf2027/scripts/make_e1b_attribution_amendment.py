"""E1b attribution, amendment 1: structure-matched Delta' and D1/2', and the dilution controls of [TR; z] - TR.

Note: bioinf2027/plans/E1b_attribution.md, Section 11 (decided by the author at gate G3, item 14, 2026-11-14;
no further review round; one audit). Every value is post hoc, secondary and unadjusted. The attribution
outputs of Sections 1 to 10 and E1b's audited outputs are not changed; make_e1b_attribution.py is imported
unchanged.

Two steps, each run under bioinf2027/scripts/run_item.py --item E1:

--fits   24 new fits in G6: [TR; C1] and [TR; C3] (the raw columns B0, B1 and B2 of the side followed by E1b's
         C1 or C3 block, built by transfer.Blocks on the same training rows), labels existing and rebuilt,
         forest and boosting, head seeds 0 to 2. Before any new fit, E1b's [TR; z] forest (existing) and
         boosting (rebuilt) at head seed 0 are refitted through this script's code and compared exactly with
         E1b's files (check A1), and the blocks are compared with transfer.Blocks (check A2). Predictions
         under $IGA_STORAGE_ROOT/bioinf2027/E1b/attribution/amendment/predictions/; manifest and protocol
         under bioinf2027/results/E1/attribution/amendment/.
--stats  Part (g): Delta' and D1/2' (C3 in place of C1) in G5 and G6, both labels, three heads, from saved
         predictions. Part (h): [TR; z] - TR, [TR; C1] - TR, [TR; C3] - TR, [TR; z] - [TR; C1] and
         [TR; z] - [TR; C3] in G6, both labels, forest and boosting. Target-gene cluster bootstrap of E1b's
         precondition (10,000 replicates, seed 20270301) through make_e1b_stats.vcc_r2, unchanged.
         Reproduction checks first (A3 to A5); outputs amendment_vcc.csv, amendment_rows.csv and
         amendment.json.

With --smoke (after --fits or --stats): forest 5 trees, boosting 20 iterations, head seed 0, 20 replicates,
outputs to $IGA_STORAGE_ROOT/bioinf2027/scratch/E1b_attribution_amendment_smoke/, never reported. Check A1
runs with the real head settings in the smoke fit step.
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

import make_e1b_attribution as att  # noqa: E402
import make_e1b_fits as e1f  # noqa: E402
import make_e1b_stats as e1s  # noqa: E402

REPO = att.REPO
E1 = att.E1
ATTRIBUTION = att.OUT
OUT = ATTRIBUTION / "amendment"
PLAN = "bioinf2027/plans/E1b_attribution.md, Section 11"
STATUS = "post hoc, secondary, unadjusted (bioinf2027/plans/E1b_attribution.md, Section 11)"
NEW = {"[TR; C1]": "C1", "[TR; C3]": "C3"}
RECORD_CELL = "G6"
G_CELLS = tuple(e1cells.PRIMARY)
LABELS = ("existing", "rebuilt")
H_HEAD_SEEDS = (("forest", (0, 1, 2)), ("boosting", (0, 1, 2)))
G_HEAD_SEEDS = (("forest", (0, 1, 2)), ("boosting", (0, 1, 2)), ("ridge", (0,)))
TOLERANCE = 1e-12
WORKERS = 4
SMOKE = False
DATA: dict = {}


class Stop(RuntimeError):
    """A check failed or an input is missing: nothing is written (hard limit 9)."""


def scratch() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E1b_attribution_amendment_smoke"


def predictions_root() -> Path:
    if SMOKE:
        return scratch() / "predictions"
    return config.storage_root() / "bioinf2027" / "E1b" / "attribution" / "amendment" / "predictions"


def out_dir() -> Path:
    return scratch() if SMOKE else OUT


# --------------------------------------------------------------------------- fitting


class Matrix:
    """TR (the side's raw B0, B1 and B2) followed by the block of E1b's fit of the base representation."""

    def __init__(self, spec: FitSpec, train: transfer.Side, rows: np.ndarray, base: str | None = None) -> None:
        base = base or NEW[spec.representation]
        self.blocks = transfer.Blocks(FitSpec(spec.context, spec.label, base, spec.head, spec.seed, spec.variant, spec.realisation), train, rows)

    def __call__(self, side: transfer.Side) -> np.ndarray:
        return np.concatenate([side.b[:, :3], self.blocks.matrix(side)], axis=1).astype(np.float32)


def fit(context: transfer.Context, spec: FitSpec, base: str | None = None) -> tuple[dict, dict]:
    """make_e1b_attribution.fit, step by step, with the TR-prefixed matrix (forest and boosting only)."""
    if spec.head not in ("forest", "boosting"):
        raise ValueError("the amendment fits forest and boosting only")
    rows = transfer.training_rows(context, spec, e1f.REFIT_SEED)
    y = context.labels[spec.label][rows]
    if not np.isfinite(y).all():
        raise RuntimeError("%s: a training label is missing" % spec.key())
    matrix = Matrix(spec, context.train, rows, base)
    start = time.time()
    out = {}
    with heads.single_thread():
        model = transfer._head(spec).fit(matrix(context.train)[rows], y)
        for name, side in context.sides.items():
            m = matrix(side)
            values = np.full(side.n, np.nan)
            usable = np.isfinite(m).all(axis=1)
            if usable.any():
                values[usable] = model.predict(m[usable])
            out[name] = values
    record = {"seconds": time.time() - start, "training_rows": int(rows.size)}
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


CHECK_FITS = (FitSpec("G6", "existing", "[TR; z]", "forest", 0), FitSpec("G6", "rebuilt", "[TR; z]", "boosting", 0))


def check_worker(spec: FitSpec) -> dict:
    """Check A1: E1b's [TR; z] refitted through this script's code (TR prefix, base z), compared exactly."""
    out, record = fit(DATA["contexts"][spec.context], spec, base="z")
    saved = DATA["e1b"].get(spec.context, spec.label, spec.representation, spec.head, spec.seed)
    if saved is None:
        raise Stop("check A1: E1b holds no %s" % spec.key())
    sides = {}
    for name, values in out.items():
        if name not in saved:
            raise Stop("check A1: E1b's %s has no side %s" % (spec.key(), name))
        sides[name] = {"equal": bool(np.array_equal(values, saved[name], equal_nan=True)),
                       "largest_difference": float(np.nanmax(np.abs(values - saved[name]))) if np.isfinite(values).any() else 0.0}
    extra = sorted(k for k in saved if not k.startswith("_") and "|" not in k and k not in out)
    return {"fit": spec.key(), "seconds": round(record["seconds"], 1), "sides": sides, "sides_missing_here": extra,
            "passed": all(v["equal"] for v in sides.values()) and not extra}


def blocks_check() -> dict:
    """Check A2: the matrix is the side's B0, B1 and B2 followed by transfer.Blocks of C1 or C3, exactly."""
    context = DATA["contexts"][RECORD_CELL]
    rows = transfer.training_rows(context, FitSpec(RECORD_CELL, "existing", "z", "forest", 0), e1f.REFIT_SEED)
    compared = 0
    for representation, base in NEW.items():
        matrix = Matrix(FitSpec(RECORD_CELL, "existing", representation, "forest", 0), context.train, rows)
        reference = transfer.Blocks(FitSpec(RECORD_CELL, "existing", base, "forest", 0), context.train, rows)
        for name, side in [("train", context.train)] + list(context.sides.items()):
            m = matrix(side)
            block = reference.matrix(side)
            ok = (m.shape[1] == block.shape[1] + 3
                  and np.array_equal(m[:, :3], side.b[:, :3].astype(np.float32), equal_nan=True)
                  and np.array_equal(m[:, 3:], block.astype(np.float32), equal_nan=True))
            compared += 1
            if not ok:
                raise Stop("check A2: the %s matrix on %s is not TR followed by the %s block" % (representation, name, base))
    return {"compared": compared, "passed": True}


def tasks() -> list[FitSpec]:
    out = [FitSpec(RECORD_CELL, label, r, head, seed) for label in LABELS for r in NEW for head, seeds in H_HEAD_SEEDS for seed in seeds]
    if SMOKE:
        out = [t for t in out if t.seed == 0]
    weight = {"forest": 1.0, "boosting": 0.05}
    return sorted(out, key=lambda t: -weight[t.head] * (1.15 if t.representation == "[TR; C1]" else 1.0))


def run_fits() -> None:
    started = time.time()
    inputs = e1f.load_inputs([RECORD_CELL])
    order = list(e1cells.CELLS)
    DATA["contexts"] = {RECORD_CELL: e1f.build_context(RECORD_CELL, order.index(RECORD_CELL), {})}
    DATA["e1b"] = e1s.Predictions(E1 / "predictions_manifest.json")
    checks = {"blocks": blocks_check()}
    context = multiprocessing.get_context("fork")
    real_heads = (copy.deepcopy(heads.FOREST), copy.deepcopy(heads.BOOSTING))  # check A1 always runs with E1b's settings
    with ProcessPoolExecutor(max_workers=len(CHECK_FITS), mp_context=context) as pool:
        results = list(pool.map(check_worker, CHECK_FITS))
    checks["pipeline_equality"] = results
    print("[amendment] check A1: %s" % "; ".join("%s %s (%.0f s)" % (r["fit"], "equal" if r["passed"] else "DIFFERS", r["seconds"]) for r in results), flush=True)
    if not all(r["passed"] for r in results):
        raise Stop("check A1 failed: %s" % json.dumps(results))
    if SMOKE:
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
    listed = tasks()
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    imported = (att, e1f, e1s, sys.modules["make_d2_stats"])
    protocol = {"plan": PLAN, "fits": len(listed), "representations": NEW, "labels": list(LABELS), "cells": [RECORD_CELL],
                "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING, "threads": "one: forest n_jobs=1, OpenMP and BLAS limited to one thread"},
                "check_A1_heads": {"forest": real_heads[0], "boosting": real_heads[1]},
                "seeds": {"heads": {h: list(s) for h, s in H_HEAD_SEEDS}, "C1 and C3 realisation": 0, "refit seed (unused, all training rows)": e1f.REFIT_SEED},
                "inputs": {**inputs, "e1b_predictions": file_record(E1 / "predictions_manifest.json"), "e1b_protocol": file_record(E1 / "protocol.json")},
                "fitting_code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (transfer, heads, sys.modules["bioinf2027.representations"])},
                "script": {config.portable(Path(__file__)): sha256(Path(__file__))},
                "imported_scripts": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in imported},
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
                    print("[amendment] FAILED %s: %r" % (spec.key(), error), flush=True)
                    continue
                if len(done) % 4 == 0 or not pending:
                    print("[amendment] %d of %d done, %d failed, %.0f s" % (len(done), len(listed), len(failed), time.time() - started), flush=True)
    manifest = {"protocol": file_record(protocol_path), "fits": len(listed), "completed": len(done), "failed": failed,
                "reused": sum(bool(r.get("reused")) for r in done), "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done},
                "seconds": round(time.time() - started, 1)}
    (folder / "predictions_manifest.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[amendment] %d completed, %d failed, %.0f s" % (len(done), len(failed), time.time() - started), flush=True)
    if failed:
        raise SystemExit(1)


# --------------------------------------------------------------------------- statistics


class Lookup:
    """One lookup over the amendment's, the attribution's and E1b's manifests: make_e1b_stats.vcc_r2 reads DATA["predictions"]."""

    def __init__(self, *sources: e1s.Predictions) -> None:
        for i, a in enumerate(sources):
            for b in sources[i + 1:]:
                shared = set(a.files) & set(b.files)
                if shared:
                    raise Stop("check A5: two manifests share %d keys" % len(shared))
        self.sources = sources

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
        wanted = dict(G_HEAD_SEEDS)[head]
        if SMOKE and representation in NEW:
            wanted = (0,)  # the smoke fits hold head seed 0 only
        if len(e1s.DATA["predictions"].seeds(cell, label, representation, head, seeds=wanted)) != len(wanted):
            raise Stop("check A4: %s | %s | %s | %s does not have every head seed" % (cell, label, representation, head))
        CACHE[key] = e1s.vcc_r2(cell, label, representation, head, kind=kind)
    return CACHE[key]


minus = att.minus
half = att.half
state = att.state

READINGS = {
    "Delta'": ("beside cell count, z's advantage over its row-shuffled copy is smaller than without it", "the change of the advantage is not resolved",
               "beside cell count, z's advantage over its row-shuffled copy is larger than without it"),
    "D1/2'": ("not read", "not read",
              "beside cell count, z's advantage over its row-shuffled copy is less than half of its forest margin over C3 without cell count"),
    "h1": ("adding z to the three trivial predictors raises R-squared under this head", "nothing added by z to the three trivial predictors is resolved",
           "adding z to the three trivial predictors lowers R-squared under this head; read only with (h2), (h2'), (h3) and (h3')"),
    "h2": ("a Gaussian block of 768 columns raises R-squared beside the three trivial predictors: an effect of the head",
           "no effect of a block of this width is resolved",
           "a Gaussian block of 768 columns lowers R-squared beside the three trivial predictors (dilution)"),
    "h2'": ("z's own columns with the rows shuffled raise R-squared beside the three trivial predictors: an effect of the head",
            "no effect of z's shuffled columns is resolved",
            "z's own columns with the rows shuffled lower R-squared beside the three trivial predictors (dilution)"),
    "h3": ("beside the three trivial predictors, z gives this forest more than an independent Gaussian block of the same width; not separated from a "
           "difference in block structure (see h3')", "beside the three trivial predictors, z is not shown to differ from a Gaussian block of the same width",
           "beside the three trivial predictors, a Gaussian block does better than z"),
    "h3'": ("beside the three trivial predictors, z gives this forest more than its own columns with the rows shuffled",
            "beside the three trivial predictors, z is not shown to differ from its row-shuffled copy",
            "beside the three trivial predictors, the row-shuffled copy does better than z"),
}
READ = tuple(READINGS)


def reading(name: str, s: str, h2_states: tuple[str, str]) -> str:
    above, contains, below = READINGS[name]
    if name == "h1" and s == "above zero" and "above zero" in h2_states:
        return "not read: a block of the same width also raises it ((h2) or (h2') above zero)"
    return {"above zero": above, "contains zero": contains, "below zero": below}.get(s, "undefined")


def g_quantities(cell: str, label: str, head: str) -> tuple[dict, dict]:
    s = {r: score(cell, label, r, head) for r in ("z", "C3", "[B0; z]", "[B0; C3]")}
    out = {}
    for k in ("r2", "spearman"):
        out.setdefault("Delta'", []).append(minus(minus(s["z"][k], s["C3"][k]), minus(s["[B0; z]"][k], s["[B0; C3]"][k])))
        out.setdefault("D1/2'", []).append(minus(minus(s["[B0; z]"][k], s["[B0; C3]"][k]), half(minus(s["z"][k], s["C3"][k]))))
        out.setdefault("M'", []).append(minus(s["z"][k], s["C3"][k]))
    text = {"Delta'": "Delta' = [z minus C3] minus [[B0; z] minus [B0; C3]], inside each replicate",
            "D1/2'": "D1/2' = [[B0; z] minus [B0; C3]] minus half of [z minus C3], inside each replicate",
            "M'": "M' = z minus C3; reference inside Delta' and D1/2' only, an unread row (author's decision, gate G3 item 13)"}
    return {n: (v[0], v[1], text[n]) for n, v in out.items()}, s


def h_quantities(label: str, head: str) -> tuple[dict, dict]:
    s = {r: score(RECORD_CELL, label, r, head) for r in ("TR", "[TR; z]", "[TR; C1]", "[TR; C3]")}
    pairs = {"h1": ("[TR; z]", "TR", "[TR; z] minus TR (the post hoc note's a2 on the existing label under the forest)"),
             "h2": ("[TR; C1]", "TR", "[TR; C1] minus TR (control)"),
             "h2'": ("[TR; C3]", "TR", "[TR; C3] minus TR (control, z's columns with the rows shuffled)"),
             "h3": ("[TR; z]", "[TR; C1]", "[TR; z] minus [TR; C1]"),
             "h3'": ("[TR; z]", "[TR; C3]", "[TR; z] minus [TR; C3]")}
    return {n: (minus(s[a]["r2"], s[b]["r2"]), minus(s[a]["spearman"], s[b]["spearman"]), t) for n, (a, b, t) in pairs.items()}, s


def entry(cell, label, head, part, name, r2, rho, description, n) -> dict:
    r = e1s.summary(*r2)
    sp = e1s.summary(*rho)
    return {"part": part, "cell": cell, "label": label, "head": head, "quantity": name, "description": description,
            "n_test_rows": n["n_rows"], "n_test_genes": n["n_genes"],
            "r2": r["estimate"], "r2_lo": r["lo"], "r2_hi": r["hi"], "r2_p": r["p"], "r2_se": r["se"], "r2_detectable_2.80se": r["detectable_2.80se"],
            "spearman": sp["estimate"], "spearman_lo": sp["lo"], "spearman_hi": sp["hi"],
            "state": "not read (author's decision, gate G3 item 13)" if name == "M'" else state(r["lo"], r["hi"]),
            "note": "" if cell == RECORD_CELL else "not the submitted statistic (G5); reported beside G6, no reading"}


def reference(cell, label, head, part, name, scores) -> dict:
    r = e1s.summary(*scores[name]["r2"])
    return {"part": part, "cell": cell, "label": label, "head": head, "quantity": f"R2({name})", "description": "reference",
            "n_test_rows": scores[name]["n_rows"], "n_test_genes": scores[name]["n_genes"],
            "r2": r["estimate"], "r2_lo": r["lo"], "r2_hi": r["hi"], "r2_p": float("nan"), "state": "",
            "note": "" if cell == RECORD_CELL else "not the submitted statistic (G5)"}


def vcc_rows() -> pd.DataFrame:
    rows = []
    for cell in G_CELLS:
        for label in LABELS:
            for head, _ in G_HEAD_SEEDS:
                values, scores = g_quantities(cell, label, head)
                for name, (r2, rho, text) in values.items():
                    e = entry(cell, label, head, "g", name, r2, rho, text, scores["z"])
                    if cell == RECORD_CELL and head == "forest" and name == "Delta'":
                        b = {k: score(cell, label, k, head, kind="batches") for k in ("z", "C3", "[B0; z]", "[B0; C3]")}
                        v = minus(minus(b["z"]["r2"], b["C3"]["r2"]), minus(b["[B0; z]"]["r2"], b["[B0; C3]"]["r2"]))
                        lo, hi, _ = e1s.resample_interval(v[1])
                        e.update({"batch_resampled_lo": lo, "batch_resampled_hi": hi,
                                  "batch_resampled_note": "sensitivity analysis: whole batches, 10,000 replicates, seed 20270302"})
                    rows.append(e)
    for label in LABELS:
        for head, _ in H_HEAD_SEEDS:
            values, scores = h_quantities(label, head)
            for name, (r2, rho, text) in values.items():
                rows.append(entry(RECORD_CELL, label, head, "h", name, r2, rho, text, scores["TR"]))
            for name in ("TR", "[TR; z]", "[TR; C1]", "[TR; C3]"):
                rows.append(reference(RECORD_CELL, label, head, "h", name, scores))
    frame = pd.DataFrame(rows)
    record = (frame.cell == RECORD_CELL) & (frame.label == "existing") & (frame["head"] == "forest")
    frame["reading"] = ""
    h2_states = tuple(frame.loc[record & frame.quantity.isin(["h2", "h2'"]), "state"])
    for index in frame.index[record & frame.quantity.isin(READ)]:
        frame.loc[index, "reading"] = reading(frame.loc[index, "quantity"], frame.loc[index, "state"], h2_states)
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


ROW_TEXT = {
    "B'": ("Delta' resolved above zero", "beside cell count, z's advantage over its row-shuffled copy of the same width is smaller than without it"),
    "C'": ("D1/2' upper bound below zero", "beside cell count, z's advantage over its row-shuffled copy is less than half of its margin over C3 without it"),
    "E": ("(h3) and (h3') both resolved above zero", "beside cell count, the control-cell expression of the target and the own shift, z gives the head more "
          "than an independent Gaussian block and more than its own row-shuffled columns of the same width"),
    "E'": ("(h3) resolved above zero, (h3') not", "beside the three trivial predictors, z gives the head more than a Gaussian block of the same width; not "
           "separated from a difference in block structure"),
    "F": ("neither (h3) nor (h3') resolved above zero", "beside the three trivial predictors, z is not shown to give the head more than a block of the same width"),
}


def s2_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for head in ("forest", "boosting"):
        get = {r.quantity: r for r in frame[(frame.cell == RECORD_CELL) & (frame.label == "existing") & (frame["head"] == head)].itertuples()}
        h3, h3p = get["h3"].state == "above zero", get["h3'"].state == "above zero"
        holds = {"B'": get["Delta'"].state == "above zero", "C'": bool(get["D1/2'"].r2_hi < 0), "E": h3 and h3p, "E'": h3 and not h3p,
                 "F": not h3 and not h3p}
        for row, (condition, text) in ROW_TEXT.items():
            rows.append({"cell": RECORD_CELL, "label": "existing", "head": head, "row": row, "condition": condition, "holds": bool(holds[row]),
                         "reading": text, "role": "the reading (forest)" if head == "forest" else "beside the forest: a row is specific to the forest unless boosting agrees",
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
            raise Stop("check A3: %s is %r here and %r on file; nothing is written" % (what, ours, theirs))

    names = ("estimate",) if SMOKE else ("estimate", "lo", "hi")
    posthoc = pd.read_csv(E1 / "posthoc" / "posthoc_vcc.csv")
    row = posthoc[(posthoc.cell == RECORD_CELL) & (posthoc.quantity == "a2")].iloc[0]
    s = e1s.summary(*minus(score(RECORD_CELL, "existing", "[TR; z]", "forest")["r2"], score(RECORD_CELL, "existing", "TR", "forest")["r2"]))
    for n, column in zip(names, ("r2_difference", "r2_lo", "r2_hi")):
        compare(f"{RECORD_CELL} a2 {n}", s[n], getattr(row, column))
    saved = pd.read_csv(ATTRIBUTION / "attribution_vcc.csv")
    for cell in G_CELLS:
        for label in LABELS:
            for head, _ in G_HEAD_SEEDS:
                for name, (a, b) in {"d3'": ("[B0; z]", "[B0; C3]"), "z - C3": ("z", "C3")}.items():
                    row = saved[(saved.cell == cell) & (saved.label == label) & (saved["head"] == head) & (saved.quantity == name)].iloc[0]
                    s = e1s.summary(*minus(score(cell, label, a, head)["r2"], score(cell, label, b, head)["r2"]))
                    for n, column in zip(names, ("r2", "r2_lo", "r2_hi")):
                        compare(f"{cell} {label} {head} {name} {n}", s[n], getattr(row, column))
    digests = {}
    recorded = json.loads((E1 / "posthoc" / "posthoc.json").read_text())["vcc_replicate_index_sha256"]
    for cell in G_CELLS:
        for kind in ("genes", "batches"):
            key = f"{cell} | {kind}"
            ours = e1s.vcc_draw(cell, kind)["draw"].index_sha256
            digests[key] = {"here": ours, "posthoc.json": recorded.get(key), "equal": ours == recorded.get(key)}
            if not SMOKE and not digests[key]["equal"]:
                raise Stop("check A3: the replicate index of %s differs from the post hoc note's" % key)
    return {"values_compared": compared, "largest_difference": largest, "tolerance": TOLERANCE, "replicate_index_digests": digests, "passed": True}


def run_stats() -> None:
    started = time.time()
    folder = out_dir()
    manifest_path = folder / "predictions_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["failed"] or manifest["completed"] != manifest["fits"]:
        raise Stop("check A4: the fit step did not complete every fit")
    e1s.DATA["predictions"] = Lookup(e1s.Predictions(manifest_path), e1s.Predictions(ATTRIBUTION / "predictions_manifest.json"),
                                     e1s.Predictions(E1 / "predictions_manifest.json"))
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    e1s.DATA["cells"] = {}
    for cell in G_CELLS:
        if e1cells.CELLS[cell]["unit"] != "row":
            raise Stop("%s is not a row-level cell" % cell)
        e1s.DATA["cells"][cell] = e1cells.build_cell(cell, rows, means)
    e1s.STATE["vcc_draws"] = {}
    checks = {"reproduction": reproduction(), "manifests_disjoint": True, "fits_complete": True}
    frame = vcc_rows()
    s2 = s2_rows(frame)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "amendment_vcc.csv", index=False)
    s2.to_csv(folder / "amendment_rows.csv", index=False)
    record = {"plan": PLAN, "replicates": e1s.B, "seeds": {"target-gene clusters": e1s.VCC_SEED, "batches": e1s.BATCH_SEED},
              "heads": {"part g": {h: list(s) for h, s in G_HEAD_SEEDS}, "part h": {h: list(s) for h, s in H_HEAD_SEEDS}}, "cell_of_record": RECORD_CELL,
              "inputs": {"amendment_predictions": file_record(manifest_path), "attribution_predictions": file_record(ATTRIBUTION / "predictions_manifest.json"),
                         "e1b_predictions": file_record(E1 / "predictions_manifest.json"), "attribution_vcc": file_record(ATTRIBUTION / "attribution_vcc.csv"),
                         "posthoc_vcc": file_record(E1 / "posthoc" / "posthoc_vcc.csv"), "posthoc_json": file_record(E1 / "posthoc" / "posthoc.json"),
                         "cell_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in G_CELLS}},
              "script": {config.portable(Path(__file__)): sha256(Path(__file__))},
              "imported_scripts": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (att, e1f, e1s, sys.modules["make_d2_stats"])},
              "checks": checks,
              "outputs": {"amendment_vcc.csv": file_record(folder / "amendment_vcc.csv"), "amendment_rows.csv": file_record(folder / "amendment_rows.csv")},
              "smoke": SMOKE, "seconds": round(time.time() - started, 1)}
    (folder / "amendment.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[amendment stats] done in %.0f s; %d rows; rows holding (forest): %s" % (
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
