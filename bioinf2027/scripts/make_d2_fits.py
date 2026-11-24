"""D2, the head fits of the token-drop arm: z- and the noise-matched z~ in E1b's harness.

Design: bioinf2027/plans/D2.md, Sections 5 (item 5), 10, 11 and 13. The arms of
bioinf2027/scripts/make_d2_aggregate.py (bioinf2027/results/D2/aggregate.json) enter E1b's harness in
place of z, with E1b's heads, seeds, rows, cells and preprocessing (bioinf2027/plans/E1_design.md,
Sections 10 and 28): the reference of z in each cell (the row's batch in G5 and G6, the pooled controls
in G1 and G2; Section 10.1, item 6).

Fits per cell of the two pairs: z- on the rebuilt, existing, source-mode and mean-across-draws labels under
the forest (seeds 0 to 2) and the secondary heads; z~ on the rebuilt and existing labels; [TR; z-]; the
refits of E1b on z- in G5 and G6; the positive control of the drop (B2 native and B3 of the VCC rows as the
label, predicted from z and from z-; Section 10.3, item 3); and C2- (the fixed projection of the masked x;
Section 11.3, item 1).

Options: --plan prints the counts; --smoke fits a subset with 5 trees into the scratch folder; --workers N
(default 12; 1 to 16) sets the number of worker processes and changes no prediction, since each fit runs
single-threaded with its own seed.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, heads, representations as rep, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import fit_tiers  # noqa: E402
import make_e1b_fits as e1b  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "D2"
PLAN = "bioinf2027/plans/D2.md, Sections 5, 10, 11 and 13"
CELLS = ("G5", "G6", "G1", "G2")
SEEDS = (0, 1, 2)
WORKERS = 12
SMOKE = False

DATA: dict = {}


def predictions_root() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "D2_fits_smoke" / "predictions"
    return config.storage_root() / "bioinf2027" / "D2" / "predictions"


def load_arms() -> dict:
    record = json.loads((OUT / "aggregate.json").read_text())
    path = config.storage_root() / record["arms"]["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["arms"]["sha256"]:
        raise RuntimeError("the D2 arms differ from their record")
    with np.load(path, allow_pickle=False) as store:
        DATA["arms"] = {k: store[k] for k in store.files}
    return file_record(OUT / "aggregate.json")


def arm_context(cell: str, arm: str, index: int) -> transfer.Context:
    """E1b's context of the cell with z replaced by z- or z~ (arm 'z-' or 'z~'), unchanged (arm 'z'), or with x masked for C2- (arm 'x-')."""
    base = e1b.build_context(cell, index, {})
    context = copy.copy(base)
    context.sides = dict(base.sides)
    data = e1b.DATA["cells"][cell]
    if arm in ("z-", "z~"):
        reference = "batch" if e1cells.CELLS[cell]["features"] == "batch" else "pooled"
        name = {"z-": "z_minus", "z~": "z_tilde"}[arm]
        keys = DATA["arms"]["VCC_keys"]
        position = {k: i for i, k in enumerate(keys.tolist())}
        order = np.array([position["|".join(k)] for k in data.vcc[["split", "batch", "target_gene"]].astype(str).to_numpy()])
        values = DATA["arms"][f"VCC_{name}_{reference}"]
        for split, side_name in (("train", None), ("val", "val"), ("test", "test")):
            rows = order[data.split(split)]
            old = context.train if side_name is None else context.sides[side_name]
            new = transfer.Side(name=old.name, control_split=old.control_split, z=values[rows], x=old.x, b=old.b, target_column=old.target_column)
            if side_name is None:
                context.train = new
            else:
                context.sides[side_name] = new
        for screen in data.spec["screens"]:
            old = context.sides[screen]
            if not np.array_equal(DATA["arms"][f"{screen}_constructs"], e1b.DATA["screens"][screen].constructs):
                raise RuntimeError("the D2 arms list the %s constructs in another order" % screen)
            context.sides[screen] = transfer.Side(name=screen, control_split=screen, z=DATA["arms"][f"{screen}_{name}"], x=old.x, b=old.b,
                                                  target_column=old.target_column)
    elif arm == "z":
        pass  # the cached z: the counterpart of z- for the positive control of the drop
    elif arm == "x-":
        def masked(side):
            return transfer.Side(name=side.name, control_split=side.control_split, z=side.z,
                                 x={k: rep.mask_target(v, side.target_column) for k, v in side.x.items()}, b=side.b, target_column=side.target_column)
        context.train = masked(context.train)
        context.sides = {k: masked(v) for k, v in context.sides.items()}
    else:
        raise ValueError(arm)
    for name in ("B2", "B3"):
        values = data.vcc[name].to_numpy(dtype=np.float64)
        context.labels[f"own {name}"] = values[data.split("train")]
        context.labels[f"own {name}@val"] = values[data.split("val")]
    context.name = f"D2 | {cell} | {arm}"
    return context


def task_list(refits: int) -> list[tuple[dict, FitSpec]]:
    out = []

    def add(cell, arm, label, models, heads_seeds, variant="all training rows"):
        name = f"D2 | {cell} | {arm}" if arm else cell
        for r in models:
            for head, seeds in heads_seeds:
                for seed in seeds:
                    out.append(({"cell": cell, "arm": arm}, FitSpec(name, label, r, head, seed, variant)))

    every_head = (("forest", SEEDS), ("boosting", SEEDS), ("ridge", (0,)))
    for cell in CELLS:
        primary = cell in e1cells.PRIMARY
        add(cell, "z-", "rebuilt", ("z",), every_head)
        add(cell, "z-", "rebuilt", ("[TR; z]",), (("forest", SEEDS),))
        add(cell, "z-", "existing", ("z",), every_head if primary else (("forest", SEEDS),))
        add(cell, "z~", "rebuilt", ("z",), (("forest", SEEDS),))
        add(cell, "z~", "existing", ("z",), (("forest", SEEDS),))
        add(cell, "x-", "rebuilt", ("C2",), (("forest", SEEDS),))
        for name in ("B2", "B3"):
            add(cell, "z-", f"own {name}", ("z",), (("forest", SEEDS),))
            add(cell, "z", f"own {name}", ("z",), (("forest", SEEDS),))
        if primary:
            add(cell, "z-", "source", ("z",), (("forest", SEEDS),))
            add(cell, "z-", "mean", ("z",), (("forest", SEEDS),))
            for r in range(refits):
                add(cell, "z-", "rebuilt", ("z",), (("forest", (0,)),), variant=f"refit {r}")
    keys = [s.key() for _, s in out]
    if len(keys) != len(set(keys)):
        raise RuntimeError("a fit is listed twice")
    return out


def worker(description: dict, spec: FitSpec, index: int) -> dict:
    key = (description["cell"], description["arm"])
    if key not in DATA["contexts"]:
        DATA["contexts"][key] = arm_context(description["cell"], description["arm"], index)
    folder = predictions_root() / spec.file_name()[:2]
    return transfer.run_fit(DATA["contexts"][key], spec, folder, e1b.REFIT_SEED, {})


def main(mode: str) -> None:
    started = time.time()
    arms_record = load_arms()
    inputs = e1b.load_inputs(list(CELLS))
    refits = 1 if SMOKE else inputs["refits"]
    tasks = task_list(refits)
    if SMOKE:
        tasks = [t for t in tasks if t[1].seed == 0 and t[0]["cell"] == "G5"]
    print("[d2 fits] %d fits" % len(tasks), flush=True)
    if mode == "plan":
        return
    DATA["contexts"] = {}
    folder = predictions_root().parent if SMOKE else OUT
    folder.mkdir(parents=True, exist_ok=True)
    # the context index of a cell is E1b's, so that the refits draw the same training genes as E1b's refits
    index = {cell: list(e1cells.CELLS).index(cell) for cell in CELLS}
    protocol = {"plan": PLAN, "fits": len(tasks), "refits": refits, "inputs": {**inputs, "arms": arms_record},
                "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING, "ridge grid": list(heads.RIDGE_GRID)},
                "refit_draws": "E1b's (seed 20270414, the cell's index in the grid), so that z- and z are refitted on the same genes",
                "predictions": config.portable(predictions_root())}
    (folder / "fit_protocol.json").write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    context = multiprocessing.get_context("fork")
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        pending = {pool.submit(worker, d, s, index[d["cell"]]): s for d, s in tasks}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as error:  # recorded and reported; the run continues
                    failed.append({"fit": spec.key(), "error": repr(error)})
                    print("[d2 fits] FAILED %s: %r" % (spec.key(), error), flush=True)
                    continue
                if len(done) % 25 == 0 or not pending:
                    print("[d2 fits] %d of %d done, %d failed, %.0f s" % (len(done), len(tasks), len(failed), time.time() - started), flush=True)
    manifest = {"protocol": file_record(folder / "fit_protocol.json"), "fits": len(tasks), "completed": len(done), "failed": failed,
                "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done}, "seconds": round(time.time() - started, 1)}
    (folder / "predictions_manifest.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[d2 fits] %d completed, %d failed" % (len(done), len(failed)), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    WORKERS, arguments = fit_tiers.parse_workers(sys.argv[1:], WORKERS)
    if arguments == ["--smoke"]:
        SMOKE = True
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
        main("fit")
    elif arguments == ["--plan"]:
        main("plan")
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    else:
        main("fit")
