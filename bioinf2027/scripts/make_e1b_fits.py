"""E1b, every head fit of the evaluation, with its predictions saved per fit.

Design: bioinf2027/plans/E1_design.md, Section 10, as amended by Sections 21 to 28 (Section 28 fixes
the details the fits need). Inputs: the E1a tables of every cell (refused unless they match their
manifests), the feature inputs of bioinf2027/src/bioinf2027/e1inputs.py, and the pre-run checks of
bioinf2027/scripts/make_e1b_checks.py, which must have passed (bioinf2027/results/E1/checks.json).
No statistic is computed here; bioinf2027/scripts/make_e1b_stats.py reads the saved predictions.

Each fit's predictions (VCC validation and test rows of its cell, every external screen of its
cell, and the release basis of the screens for representations of x) go to
$IGA_STORAGE_ROOT/bioinf2027/E1b/predictions/<context>/ with the fit's specification. A fit whose
file exists with the same specification is not repeated, so an interrupted run resumes. The
protocol (bioinf2027/results/E1/protocol.json) is written and read back before the first fit, and
the manifest of every prediction file (bioinf2027/results/E1/predictions_manifest.json) after the
last.

Options: --plan prints the task counts and exits; --smoke fits a small subset with 5 trees and 20
boosting iterations into the scratch folder, never reported.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioinf2027 import config, e1cells, e1inputs, features, heads, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E1"
PLAN = "bioinf2027/plans/E1_design.md, Section 10, as amended by Sections 21 to 28"
WORKERS = 12
REFIT_SEED = 20270414
NULL_REALISATIONS = {"C1": 100, "C3": 100, "C2": 20}
SEEDS = (0, 1, 2)
LABEL_COLUMNS = {"rebuilt": "y", "source": "src_y", "mean": "y_mean", "existing": "existing_label"}
EXTERNAL_BASES = {"RPE1": ("release",), "K562-essential": ("release",)}
SMOKE = False

DATA: dict = {}

R = transfer
NON_B = R.FAMILY + R.MASKED + R.CONTROLS
ALL_MODELS = R.FAMILY + R.MASKED + R.CONTROLS + R.PIPELINE + R.TRIVIAL_MODELS + R.ONE_VARIABLE
FAMILY_REPEAT = R.FAMILY + R.MASKED + ("C1",)
REFIT_MODELS = R.FAMILY + ("C1",)


def predictions_root() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E1b_fits_smoke" / "predictions"
    return config.storage_root() / "bioinf2027" / "E1b" / "predictions"


# --------------------------------------------------------------------------- contexts


def _side(data: e1cells.CellData, split: str) -> transfer.Side:
    mask = data.split(split)
    return transfer.Side(
        name=split,
        control_split=split,
        z=data.z[mask],
        x={"main": data.x[mask]},
        b=data.vcc[list(e1cells.TRIVIAL)].to_numpy(dtype=np.float64)[mask],
        target_column=data.target_column[mask],
    )


def _external_side(screen: str, table: pd.DataFrame) -> transfer.Side:
    inputs = DATA["screens"][screen]
    if not np.array_equal(table["construct"].astype(str).to_numpy(), inputs.constructs):
        raise RuntimeError("the %s table and the feature inputs list different constructs" % screen)
    return transfer.Side(
        name=screen,
        control_split=screen,
        z=inputs.z,
        x={"main": DATA["external_x"][screen], "release": DATA["release_x"][screen]},
        b=table[list(e1cells.TRIVIAL)].to_numpy(dtype=np.float64),
        target_column=inputs.target_column,
    )


def build_context(cell: str, index: int, foreign: dict) -> transfer.Context:
    data = DATA["cells"][cell]
    sides = {split: _side(data, split) for split in ("val", "test")}
    for screen in data.spec["screens"]:
        sides[screen] = _external_side(screen, data.external[screen])
    train_mask = data.split("train")
    val_mask = data.split("val")
    labels = {}
    for name, column in LABEL_COLUMNS.items():
        values = data.vcc[column].to_numpy(dtype=np.float64)
        labels[name] = values[train_mask]
        labels[name + "@val"] = values[val_mask]
    for name, other in foreign.items():
        source = DATA["cells"][other].vcc.set_index(["split", "batch", "target_gene"])["y"]
        values = source.loc[list(zip(data.vcc["split"], data.vcc["batch"], data.vcc["target_gene"]))].to_numpy(dtype=np.float64)
        labels[name] = values[train_mask]
        labels[name + "@val"] = values[val_mask]
    return transfer.Context(
        name=cell,
        train=_side(data, "train"),
        labels=labels,
        sides=sides,
        meta={"n_guides": data.vcc["n_guides"].to_numpy()[train_mask] if "n_guides" in data.vcc else np.ones(int(train_mask.sum())),
              "genes": data.vcc["target_gene"].astype(str).to_numpy()[train_mask], "context_index": index},
    )


# --------------------------------------------------------------------------- tasks


def tasks_for(cell: str, refits: int) -> list[FitSpec]:
    out: list[FitSpec] = []

    def add(label, representations, head_seeds, variant="all training rows", realisation=0):
        for r in representations:
            for head, seeds in head_seeds:
                for seed in seeds:
                    out.append(FitSpec(cell, label, r, head, seed, variant, realisation))

    every_head = (("forest", SEEDS), ("boosting", SEEDS), ("ridge", (0,)))
    positive = (("monotone", (0,)),)
    spec = e1cells.CELLS[cell]
    derived = "base" in spec
    add("rebuilt", ALL_MODELS, every_head)
    add("rebuilt", R.POSITIVE, positive)
    if cell in e1cells.PRIMARY:
        add("existing", ALL_MODELS, every_head)
        add("existing", R.POSITIVE, positive)
        add("source", FAMILY_REPEAT, (("forest", SEEDS),))
        add("mean", FAMILY_REPEAT, (("forest", SEEDS),))
        for control, count in NULL_REALISATIONS.items():
            for k in range(1, count):
                add("rebuilt", (control,), (("forest", (0,)),), realisation=k)
        for r in range(refits):
            add("rebuilt", REFIT_MODELS, (("forest", (0,)),), variant=f"refit {r}")
        for variant in ("target is a feature", "target is not a feature"):
            add("rebuilt", REFIT_MODELS, (("forest", SEEDS),), variant=variant)
        for label in ("rebuilt", "existing"):
            add(label, REFIT_MODELS, (("forest", SEEDS),), variant="single-guide rows")
        other = {"G5": "G1", "G6": "G2"}[cell]
        add(f"rebuilt:{other}", REFIT_MODELS, (("forest", SEEDS),))
    elif not derived:
        add("existing", NON_B, every_head)
    return out


def cost(spec: FitSpec) -> float:
    """Relative cost of a fit: training rows times columns for the forest, smaller for the rest."""
    rows = DATA["train_rows"][spec.context]
    columns = {"z": 768, "x": 2000, "[x; m]": 2004, "x-": 2000, "[x; m]-": 2004, "C1": 768, "C2": 768, "C3": 768,
               "[C1; B2]": 769, "[z; B2]": 769, "[TR; z]": 771, "[TR; x]": 2003, "[TR; [x; m]]": 2007}.get(spec.representation, 5)
    weight = {"forest": 1.0, "boosting": 0.05, "ridge": 0.02, "monotone": 0.001}[spec.head]
    tier = 0 if (spec.context in e1cells.PRIMARY and spec.label in ("rebuilt", "existing") and spec.variant == "all training rows"
                 and spec.realisation == 0 and spec.head == "forest") else 1
    return (tier, -rows * columns * weight)


def all_tasks(refits: int) -> list[FitSpec]:
    tasks = []
    for cell in e1cells.CELLS:
        tasks.extend(tasks_for(cell, refits))
    keys = [t.key() for t in tasks]
    if len(keys) != len(set(keys)):
        raise RuntimeError("a fit is listed twice")
    return sorted(tasks, key=cost)


def worker(spec: FitSpec) -> dict:
    context = DATA["contexts"][spec.context]
    folder = predictions_root() / spec.context.replace(" ", "_")
    return transfer.run_fit(context, spec, folder, REFIT_SEED, EXTERNAL_BASES)


# --------------------------------------------------------------------------- main


def checks_accepted(checks: dict) -> dict:
    """Whether the pre-run checks allow the fits: all passed, or the only failures are those the author accepted.

    At gate G3 the author accepted the harness failures on the external values of C1 and C3
    (bioinf2027/plans/E1_design.md, Section 29; bioinf2027/results/E1/harness_decision.json). The
    decision holds only for the checks file and harness table it names, and only if every other check
    passed and every failing harness row is one it lists.
    """
    if checks["all_passed"]:
        return {"accepted": True, "by": "every check passed"}
    decision_path = OUT / "harness_decision.json"
    if not decision_path.exists():
        return {"accepted": False, "by": "a check failed and no decision is recorded"}
    decision = json.loads(decision_path.read_text())
    if decision["checks_file"]["sha256"] != sha256(OUT / "checks.json") or decision["harness_file"]["sha256"] != sha256(OUT / "harness.csv"):
        return {"accepted": False, "by": "the decision names other check files"}
    others = checks["external_x"]["RPE1"]["passed"] and checks["external_x"]["K562-essential"]["passed"] and checks["check_f"]["passed"]
    failing = {(r["arm"], r["scope"]) for r in pd.read_csv(OUT / "harness.csv").to_dict("records") if not bool(r["passed"])}
    allowed = {(r["arm"], r["scope"]) for r in decision["accepted_failures"]}
    accepted = bool(others and failing <= allowed and all(arm in ("C1", "C3") and scope != "VCC test" for arm, scope in allowed))
    return {"accepted": accepted, "by": "the author's decision at gate G3" if accepted else "failures beyond the decision", "decision": file_record(decision_path)}


def load_inputs(cells) -> dict:
    checks = json.loads((OUT / "checks.json").read_text())
    acceptance = checks_accepted(checks)
    if not acceptance["accepted"]:
        raise SystemExit("the E1b pre-run checks do not allow the fits (%s); no fit is run" % acceptance["by"])
    DATA["screens"], DATA["external_x"], DATA["release_x"] = {}, {}, {}
    records = {}
    for screen in e1inputs.SCREENS:
        endpoint = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
        inputs = e1inputs.load_screen_inputs(screen, endpoint)
        DATA["screens"][screen] = inputs
        DATA["external_x"][screen] = inputs.x["full_pass"] if screen == "RPE1" else inputs.x["submitted"]
        DATA["release_x"][screen] = e1inputs.release_mapped(screen, inputs.constructs)
        records[screen] = {"e1b_x": "full-pass mapping" if screen == "RPE1" else "submitted array (equal to the full-pass mapping)",
                           "full_pass_mapping": checks["external_x"][screen]["full_pass_mapping"],
                           "release": file_record(e1inputs.paths()[f"{screen}_release"])}
    rows = e1inputs.load_vcc_rows()
    means = e1inputs.load_control_means()
    modelled = e1cells.load_tables("G2")["vcc"]
    DATA["cells"] = {cell: e1cells.build_cell(cell, rows, means, modelled=modelled) for cell in cells}
    return {"checks": file_record(OUT / "checks.json"), "checks_acceptance": acceptance, "refits": int(checks["timing_probe"]["refits"]), "screens": records,
            "cell_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in cells},
            "input_records": file_record(OUT / "input_records.json")}


def write_protocol(tasks: list[FitSpec], inputs: dict) -> Path:
    counts = {}
    for t in tasks:
        key = f"{t.context} | {t.label} | {t.head}"
        counts[key] = counts.get(key, 0) + 1
    protocol = {
        "plan": PLAN,
        "fits": len(tasks),
        "fits_by_context_label_head": counts,
        "representations": list(transfer.REPRESENTATIONS),
        "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING, "ridge grid": list(heads.RIDGE_GRID), "monotone": heads.MONOTONE,
                  "threads": "one: forest n_jobs=1, OpenMP and BLAS limited to one thread"},
        "seeds": {"heads": list(SEEDS), "refits": REFIT_SEED, "null band realisations": NULL_REALISATIONS},
        "refits": inputs["refits"],
        "labels": LABEL_COLUMNS,
        "external_bases": {"main": "RPE1: full-pass mapping of the native delta; K562-essential: the submitted array, equal to it",
                           "release": "the release pseudobulk mapped to the 2,000 features, secondary basis, representations of x only"},
        "inputs": inputs,
        "predictions": config.portable(predictions_root()),
    }
    path = (OUT if not SMOKE else predictions_root().parent) / "protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        handle.write(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if json.loads(path.read_text())["fits"] != len(tasks):
        raise RuntimeError("the protocol did not read back")
    return path


def main(plan_only: bool = False) -> None:
    started = time.time()
    cells = list(e1cells.CELLS)
    inputs = load_inputs(cells)
    refits = 2 if SMOKE else inputs["refits"]
    DATA["train_rows"] = {cell: int(DATA["cells"][cell].split("train").sum()) for cell in cells}
    tasks = all_tasks(refits)
    if SMOKE:
        keep = {"G5", "G3"}
        tasks = [t for t in tasks if t.context in keep and (t.realisation < 2) and (t.seed == 0)]
    print("[fits] %d fits over %d contexts; refits per primary cell %d" % (len(tasks), len(cells), refits), flush=True)
    by_head = pd.Series([t.head for t in tasks]).value_counts().to_dict()
    print("[fits] by head %s" % by_head, flush=True)
    if plan_only:
        return
    protocol_path = write_protocol(tasks, inputs)
    foreign = {"G5": {"rebuilt:G1": "G1"}, "G6": {"rebuilt:G2": "G2"}}
    DATA["contexts"] = {cell: build_context(cell, i, foreign.get(cell, {})) for i, cell in enumerate(cells)}
    context = multiprocessing.get_context("fork")
    done, failed, reused = [], [], 0
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        pending = {pool.submit(worker, t): t for t in tasks}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:  # recorded and reported; the run continues
                    failed.append({"fit": spec.key(), "error": repr(error)})
                    print("[fits] FAILED %s: %r" % (spec.key(), error), flush=True)
                    continue
                done.append(result)
                reused += int(result["reused"])
                if len(done) % 25 == 0 or not pending:
                    print("[fits] %d of %d done (%d reused, %d failed), %.0f s" % (len(done), len(tasks), reused, len(failed), time.time() - started), flush=True)
    manifest = {
        "protocol": file_record(protocol_path),
        "fits": len(tasks),
        "completed": len(done),
        "failed": failed,
        "files": {transfer.FitSpec(**r["spec"]).key(): r["file"] for r in done},
        "seconds": round(time.time() - started, 1),
    }
    path = (OUT if not SMOKE else predictions_root().parent) / "predictions_manifest.json"
    path.write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[fits] %d completed, %d failed; manifest %s" % (len(done), len(failed), config.portable(path)), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments == ["--smoke"]:
        SMOKE = True
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
        PLAN = PLAN + "; smoke run, 5 trees, never reported"
        main()
    elif arguments == ["--plan"]:
        main(plan_only=True)
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    else:
        main()
