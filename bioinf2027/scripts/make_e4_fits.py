"""E4, every head fit on the matched features.

Design: bioinf2027/plans/E4.md, Sections 4 and 5 as amended by Sections 10 to 13 (Section 14 fixes the
details). Every fit uses E1b's harness and specification (bioinf2027/src/bioinf2027/transfer.py; the
heads, seeds and preprocessing of bioinf2027/plans/E1_design.md, Sections 10 and 28) with the features
of bioinf2027/scripts/make_e4_features.py in place of the all-cell features. The all-cell pipeline is
E1b's own fits, which are not repeated, except the corner that applies all-cell heads to matched
external features: those fits are repeated with the matched sides added, and their predictions on the
all-cell sides are checked against E1b's saved predictions. They also score the cache-built all-cell
external z (E4 plan, Section 15), the all-cell side of F of z and the heads of the E1b z sensitivity.

Fit sets per cell of the two pairs (G5, G6, G1, G2) and feature replicate k (draws 1 to 3): the matched
pipeline on the rebuilt label; the fourth corner (existing label); the source-mode label; the shared-gene
size-factor variant; the cell-disjoint variant (R5 half A's label, half B's features); refits on replicate
1. The depth-thinned fits (G5-DT, G6-DT) are added when the thinned features exist.

Tiers (bioinf2027/plans/E4.md, Section 16; bioinf2027/scripts/fit_tiers.py): --tier N runs every listed
fit up to tier N, without the fits the amendment drops (Section 16, item 3). Each run writes
protocol_tier<N>.json and predictions_manifest_tier<N>.json; before fitting, it compares its inputs,
heads, seeds and fitting code with every earlier tier's protocol and refuses to run (and so to reuse
an earlier tier's predictions) if any differs.

Options: --tier N (required for a fit run); --workers N (default 12; changes no prediction); --plan prints the task counts per tier; --smoke fits a
small subset with 5 trees into the scratch folder (head seed 0, G5 and its depth-thinned cells, feature
draw 1), with --tier N as well.

--slots runs each fit in a process of its own (fit_slots.py), so that the memory rules can shed one worker at a time,
and --workers-file (a path under the data root) a file whose integer sets the number of fits in progress as fits
finish; --smallest-first takes the fits in ascending order of their recorded price (E1b's recorded seconds per
training row and column of each head, times the fit's rows and columns, as progress_status.py prices E4) instead of
the largest first (the author's instructions of 2026-11-15 and 2026-11-16, bioinf2027/GATE.md). Each fit is a function
of its specification and of the inputs, so none of these changes a value.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
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

from bioinf2027 import config, e1cells, heads, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import fit_slots  # noqa: E402
import fit_tiers  # noqa: E402
import make_e1b_fits as e1b  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E4"
PLAN = "bioinf2027/plans/E4.md, Sections 4 and 5 as amended by Sections 10 to 14"
WORKERS = 12
WORKERS_FILE: Path | None = None  # --workers-file <path under the data root>: the worker count, read as fits finish (changes no value)
SLOTS = False  # --slots: one process per fit, so that one worker can be shed at a time (fit_slots.py; changes no value)
SMALLEST_FIRST = False  # --smallest-first: ascending recorded price (changes no value)
SEEDS = (0, 1, 2)
CELLS = ("G5", "G6", "G1", "G2")
REPLICATES = (1, 2, 3)
REFIT_SEED = 20270414
R = transfer
MATCHED_MODELS = R.FAMILY + R.MASKED + R.CONTROLS + ("TR", "[C1; B2]", "[z; B2]")
CORNER_MODELS = R.FAMILY + ("C1",)
SOURCE_MODELS = R.FAMILY + R.MASKED + ("C1",)
SIZE_FACTOR_MODELS = ("x", "m", "[x; m]")
SMOKE = False

DATA: dict = {}


def predictions_root() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E4_fits_smoke" / "predictions"
    return config.storage_root() / "bioinf2027" / "E4" / "predictions"


# --------------------------------------------------------------------------- inputs


def load_features() -> dict:
    record = json.loads((OUT / "features.json").read_text())
    if not record.get("checks_passed"):
        raise SystemExit("the E4 builder checks have not passed (bioinf2027/results/E4/features.json); no fit is run")
    DATA["features"] = {}
    for key, entry in record["features"].items():
        path = config.storage_root() / entry["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError("the E4 feature file %s differs from its record" % key)
        DATA["features"][key] = path
    thinned = OUT / "features_thinned.json"
    DATA["thinned"] = {}
    if thinned.exists():
        record_t = json.loads(thinned.read_text())
        for key, entry in record_t["features"].items():
            path = config.storage_root() / entry["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
            if sha256(path) != entry["sha256"]:
                raise RuntimeError("the thinned feature file %s differs from its record" % key)
            DATA["thinned"][key] = path
    return {"features": file_record(OUT / "features.json"), "thinned": file_record(thinned) if thinned.exists() else None}


def feature_arrays(cell: str, variant: str) -> dict:
    with np.load(DATA["features"][f"{cell} | {variant}"], allow_pickle=False) as store:
        return {k: store[k] for k in store.files}


def halves_labels(cell: str) -> dict:
    """R5 half A labels: VCC at 10 cells (the deciding version in G5 and G6), externally at the cell's n_pert."""
    manifest = e1cells.manifest(cell)
    out = {}
    for name, record in manifest["tables"].items():
        if not name.endswith("halves.csv"):
            continue
        path = e1cells.endpoint_root() / cell / name
        if sha256(path) != record["sha256"]:
            raise RuntimeError("%s/%s does not match its manifest" % (cell, name))
        out[name] = pd.read_csv(path, float_precision="round_trip")
    return out


# --------------------------------------------------------------------------- contexts


def matched_context(cell: str, variant: str, index: int, foreign: dict | None = None, all_cell_extra: bool = False,
                    draw_level_b: bool = False) -> transfer.Context:
    """E1b's context of the cell with the matched features of one variant in place of the all-cell ones.

    With ``all_cell_extra`` the context keeps E1b's all-cell features for training and every side, and
    adds the matched external features of the variant as extra sides (the corner that scores all-cell
    heads on matched external features).
    """
    base = e1b.build_context(cell, index, foreign or {})
    arrays = feature_arrays(cell, variant)
    keys = np.array(arrays["VCC_keys"])
    data = e1b.DATA["cells"][cell]
    position = {k: i for i, k in enumerate(keys.tolist())}
    order_all = np.array([position["|".join(k)] for k in data.vcc[["split", "batch", "target_gene"]].astype(str).to_numpy()])
    context = copy.copy(base)
    context.sides = dict(base.sides)
    if all_cell_extra:
        for screen in data.spec["screens"]:
            ext = base.sides[screen]
            context.sides[f"{screen} matched"] = transfer.Side(name=f"{screen} matched", control_split=screen, z=arrays[f"{screen}_z"],
                                                             x={"main": arrays[f"{screen}_x"]}, b=ext.b, target_column=ext.target_column)
        return context
    for split, name in (("train", None), ("val", "val"), ("test", "test")):
        mask = data.split(split)
        rows = order_all[mask]
        old = base.train if name is None else base.sides[name]
        new = transfer.Side(name=old.name, control_split=old.control_split, z=arrays["VCC_z"][rows], x={"main": arrays["VCC_x"][rows]},
                            b=old.b, target_column=old.target_column)
        if name is None:
            context.train = new
        else:
            context.sides[name] = new
    for screen in data.spec["screens"]:
        if f"{screen}_x" not in arrays:
            context.sides.pop(screen, None)
            continue
        old = base.sides[screen]
        context.sides[screen] = transfer.Side(name=screen, control_split=screen, z=arrays[f"{screen}_z"], x={"main": arrays[f"{screen}_x"]},
                                              b=old.b, target_column=old.target_column)
        if variant.startswith("draw "):
            # matched heads scored on the all-cell external features: the other corner of the 2 x 2 (Section 10.3, item 2),
            # with the cache-built all-cell z (Section 15)
            cache = feature_arrays("external", "all-cell")
            context.sides[f"{screen} all-cell"] = transfer.Side(name=f"{screen} all-cell", control_split=screen, z=cache[f"{screen}_z"],
                                                                x={"main": old.x["main"]}, b=old.b, target_column=old.target_column)
    if draw_level_b and variant.startswith("draw "):
        # the positive control for F: B2 native of the draw's own cells in place of the pool's (Section 10.3, item 3)
        def with_draw_b(side, b_draw):
            b = side.b.copy()
            b[:, 2] = b_draw[:, 2]
            return transfer.Side(name=side.name, control_split=side.control_split, z=side.z, x=side.x, b=b, target_column=side.target_column)

        for split, name in (("train", None), ("val", "val"), ("test", "test")):
            rows = order_all[data.split(split)]
            if name is None:
                context.train = with_draw_b(context.train, arrays["VCC_b"][rows])
            else:
                context.sides[name] = with_draw_b(context.sides[name], arrays["VCC_b"][rows])
        for screen in data.spec["screens"]:
            context.sides[screen] = with_draw_b(context.sides[screen], arrays[f"{screen}_b"])
            context.sides.pop(f"{screen} all-cell", None)
    context.name = f"E4 | {cell} | {variant}" + (" | draw-level B2" if draw_level_b else "")
    context.meta = dict(base.meta)
    context.meta["kept_train"] = np.asarray(arrays["VCC_kept"])[order_all[data.split("train")]]
    return context


def thinned_context(dt_cell: str, k: int, thinned: bool, index: int) -> transfer.Context:
    """A DT cell's label and B0 to B4 with draw-k features: VCC thinned or not, the screen unthinned (Section 10.1, item 3b)."""
    base_cell = e1cells.CELLS[dt_cell]["base"]
    screen = e1cells.CELLS[dt_cell]["screens"][0]
    context = e1b.build_context(dt_cell, index, {})
    matched = feature_arrays(base_cell, f"draw {k}")
    data = e1b.DATA["cells"][dt_cell]
    if thinned:
        with np.load(DATA["thinned"][f"{dt_cell} | draw {k}"], allow_pickle=False) as store:
            vcc_x, keys = store["VCC_x"], store["VCC_keys"]
    else:
        vcc_x, keys = matched["VCC_x"], matched["VCC_keys"]
    position = {kk: i for i, kk in enumerate(np.asarray(keys).tolist())}
    order_all = np.array([position["|".join(r)] for r in data.vcc[["split", "batch", "target_gene"]].astype(str).to_numpy()])
    for split, name in (("train", None), ("val", "val"), ("test", "test")):
        rows = order_all[data.split(split)]
        old = context.train if name is None else context.sides[name]
        new = transfer.Side(name=old.name, control_split=old.control_split, z=old.z, x={"main": vcc_x[rows]}, b=old.b, target_column=old.target_column)
        if name is None:
            context.train = new
        else:
            context.sides[name] = new
    old = context.sides[screen]
    context.sides[screen] = transfer.Side(name=screen, control_split=screen, z=old.z, x={"main": matched[f"{screen}_x"]}, b=old.b,
                                          target_column=old.target_column)
    context.name = f"E4 | {dt_cell} | draw {k} | {'thinned' if thinned else 'unthinned'} VCC features"
    return context


def disjoint_context(cell: str, index: int) -> transfer.Context:
    """Half B's features, half A's label (Section 11.2, item 5), on the kept rows and constructs."""
    context = matched_context(cell, "cell-disjoint", index)
    data = e1b.DATA["cells"][cell]
    halves = halves_labels(cell)["vcc_halves.csv"]
    spec = e1cells.CELLS[cell]
    version = "pooled control halves" if spec["features"] == "pooled" else "deciding: 350 controls from each batch half"
    part = halves[(halves["level"] == 10) & (halves["version"] == version) & (halves["half"] == "A")]
    label = part.set_index(["split", "batch", "target_gene"])["y"]
    train = data.vcc[data.split("train")]
    keys = list(zip(train["split"], train["batch"], train["target_gene"]))
    values = np.array([label.get(k, np.nan) for k in keys], dtype=np.float64)
    kept = context.meta["kept_train"] & np.isfinite(values)
    rows = np.flatnonzero(kept)
    context.train = transfer.Side(name="train", control_split="train", z=context.train.z[rows], x={"main": context.train.x["main"][rows]},
                                  b=context.train.b[rows], target_column=context.train.target_column[rows])
    context.labels = {"half A": values[rows]}
    context.meta = {"genes": context.meta["genes"][rows], "context_index": index, "n_guides": context.meta["n_guides"][rows]}
    context.sides = {k: v for k, v in context.sides.items() if k in ("RPE1", "K562-essential")}
    context.name = f"E4 | {cell} | cell-disjoint"
    return context


# --------------------------------------------------------------------------- tasks


def task_list(refits: int) -> list[tuple[dict, FitSpec]]:
    out = []

    def add(description, label, models, heads_seeds, variant="all training rows"):
        for r in models:
            for head, seeds in heads_seeds:
                for seed in seeds:
                    out.append((description, FitSpec(description["name"], label, r, head, seed, variant)))

    every_head = (("forest", SEEDS), ("boosting", SEEDS), ("ridge", (0,)))
    for cell in CELLS:
        for k in REPLICATES:
            d = {"kind": "matched", "cell": cell, "variant": f"draw {k}", "name": f"E4 | {cell} | draw {k}"}
            add(d, "rebuilt", MATCHED_MODELS, every_head)
            add(d, "rebuilt", R.POSITIVE, (("monotone", (0,)),))
            d_b = {"kind": "draw-level B2", "cell": cell, "variant": f"draw {k}", "name": f"E4 | {cell} | draw {k} | draw-level B2"}
            add(d_b, "rebuilt", ("[C1; B2]",), (("forest", SEEDS),))
            add(d, "existing", CORNER_MODELS, (("forest", SEEDS),))
            add(d, "source", SOURCE_MODELS, (("forest", SEEDS),))
            if k == 1:
                for r in range(refits):
                    add(d, "rebuilt", CORNER_MODELS, (("forest", (0,)),), variant=f"refit {r}")
            for screen in e1cells.CELLS[cell]["screens"]:
                d2 = {"kind": "size factor", "cell": cell, "variant": f"shared {screen} {k}", "screen": screen,
                      "name": f"E4 | {cell} | shared {screen} {k}"}
                label = f"rebuilt:G1-SU-{screen}" if cell == "G1" else "rebuilt"
                add(d2, label, SIZE_FACTOR_MODELS, (("forest", SEEDS),))
        d = {"kind": "disjoint", "cell": cell, "name": f"E4 | {cell} | cell-disjoint"}
        add(d, "half A", CORNER_MODELS, (("forest", SEEDS),))
        d = {"kind": "corner", "cell": cell, "name": f"E4 | {cell} | all-cell heads on matched sides"}
        add(d, "rebuilt", CORNER_MODELS, (("forest", SEEDS),))
    if DATA.get("thinned"):
        for base in ("G5", "G6"):
            for screen in e1cells.e1inputs.SCREENS:
                dt_cell = f"{base}-DT-{screen}"
                for k in REPLICATES:
                    for thinned in (True, False):
                        d = {"kind": "thinned", "cell": dt_cell, "k": k, "thinned": thinned,
                             "name": f"E4 | {dt_cell} | draw {k} | {'thinned' if thinned else 'unthinned'} VCC features"}
                        add(d, "rebuilt", SIZE_FACTOR_MODELS, (("forest", SEEDS),))
    keys = [s.key() for _, s in out]
    if len(keys) != len(set(keys)):
        raise RuntimeError("a fit is listed twice")
    return out


def run_list(refits: int, tier: int) -> list[tuple[dict, FitSpec]]:
    """The listed fits up to the tier, without the fits the amendment drops (Section 16, items 1 and 3)."""
    out = []
    for d, s in task_list(refits):
        t = fit_tiers.e4_tier(d, s)
        if t is not None and t <= tier:
            out.append((d, s))
    return out


def smoke_subset(item) -> bool:
    """Head seed 0, G5 and its depth-thinned cells, feature draw 1 (E1b's smoke fits cover G5)."""
    d, s = item
    if d["kind"] in ("disjoint", "corner"):
        first = True
    elif "k" in d:
        first = d["k"] == 1
    else:
        first = d["variant"].split()[-1] == "1"
    return s.seed == 0 and (d["cell"] == "G5" or d["cell"].startswith("G5-DT")) and first


def fitting_code() -> dict:
    """Digests of the pipeline modules the fits import (bookkeeping modules excluded)."""
    out = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith("bioinf2027.") and name.split(".")[-1] not in ("config", "provenance", "runlog") and getattr(module, "__file__", None):
            path = Path(module.__file__)
            out[path.relative_to(REPO).as_posix()] = sha256(path)
    return out


def refuse_if_inputs_differ(folder: Path, protocol: dict) -> list:
    """Compare the run with every earlier tier's protocol; raise if one recorded other inputs, heads, seeds or code."""
    current = json.loads(json.dumps(strict_json(fit_tiers.comparable(protocol))))
    compared = []
    for path in sorted(folder.glob("protocol_tier*.json")):
        earlier = fit_tiers.comparable(json.loads(path.read_text()))
        differing = fit_tiers.differences(earlier, current)
        if differing:
            raise SystemExit("%s recorded other %s; its predictions are not reused and nothing is fitted"
                             % (path.name, ", ".join(differing)))
        compared.append(path.name)
    return compared


def context_for(description: dict, index: int) -> transfer.Context:
    key = description["name"]
    if key not in DATA["contexts"]:
        kind = description["kind"]
        cell = description["cell"]
        if kind == "matched":
            context = matched_context(cell, description["variant"], index)
        elif kind == "draw-level B2":
            context = matched_context(cell, description["variant"], index, draw_level_b=True)
        elif kind == "size factor":
            screen = description["screen"]
            foreign = {f"rebuilt:G1-SU-{screen}": f"G1-SU-{screen}"} if cell == "G1" else {}
            context = matched_context(cell, description["variant"], index, foreign)
            context.sides = {k: v for k, v in context.sides.items() if k in ("val", "test", screen)}
        elif kind == "disjoint":
            context = disjoint_context(cell, index)
        elif kind == "thinned":
            context = thinned_context(cell, description["k"], description["thinned"], index)
        elif kind == "corner":
            context = matched_context(cell, "draw 1", index, all_cell_extra=True)
            for k in (2, 3):
                extra = matched_context(cell, f"draw {k}", index, all_cell_extra=True)
                for screen in e1cells.CELLS[cell]["screens"]:
                    context.sides[f"{screen} matched draw {k}"] = extra.sides[f"{screen} matched"]
            for screen in e1cells.CELLS[cell]["screens"]:
                context.sides[f"{screen} matched draw 1"] = context.sides.pop(f"{screen} matched")
            # the cache-built all-cell external z with E1b's external x (Section 15, items 2 and 3)
            cache = feature_arrays("external", "all-cell")
            for screen in e1cells.CELLS[cell]["screens"]:
                if not np.array_equal(cache[f"{screen}_constructs"], e1b.DATA["screens"][screen].constructs):
                    raise RuntimeError("the cache-built %s features list the constructs in another order" % screen)
                old = context.sides[screen]
                context.sides[f"{screen} cache-built z"] = transfer.Side(name=f"{screen} cache-built z", control_split=screen, z=cache[f"{screen}_z"],
                                                                         x=old.x, b=old.b, target_column=old.target_column)
            context.name = description["name"]
        else:
            raise ValueError(kind)
        context.name = description["name"]
        DATA["contexts"][key] = context
    return DATA["contexts"][key]


def worker(description: dict, spec: FitSpec, index: int) -> dict:
    context = context_for(description, index)
    folder = predictions_root() / spec.file_name()[:2]
    return transfer.run_fit(context, spec, folder, REFIT_SEED, {})


def cost(item) -> tuple:
    description, spec = item
    rows = e1b.DATA["train_rows"].get(description["cell"], 3000)
    columns = {"z": 768, "x": 2000, "[x; m]": 2004, "x-": 2000, "[x; m]-": 2004, "C1": 768, "C2": 768, "C3": 768,
               "[C1; B2]": 769, "[z; B2]": 769}.get(spec.representation, 5)
    weight = {"forest": 1.0, "boosting": 0.05, "ridge": 0.02, "monotone": 0.001}[spec.head]
    tier = 0 if description["cell"] in e1cells.PRIMARY and description["kind"] == "matched" and spec.label == "rebuilt" else 1
    return (tier, -rows * columns * weight)


def recorded_prices(tasks: list) -> list[float]:
    """Each fit's recorded price in seconds, as progress_status.py prices E4: E1b's recorded seconds per training row
    and column of the head, times the cell's training rows (3,000 for a cell outside E1's) and the columns."""
    import progress_status as ps

    rows = ps.train_rows()
    model = ps.pricing(ps.e1b_finished(), rows)
    return [ps.price(d["cell"], s.head, s.representation, rows.get(d["cell"], 3000), model, by_kind=False) for d, s in tasks]


def smallest_first(tasks: list, prices: list[float]) -> list:
    """The fits in ascending order of price; fits of equal price keep their order."""
    return [tasks[i] for i in sorted(range(len(tasks)), key=lambda i: prices[i])]


def run_pool(tasks: list, index: dict, started: float) -> tuple[list, list]:
    """Every fit in a pool of WORKERS processes forked after the inputs are loaded."""
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        pending = {pool.submit(worker, d, s, index[d["name"]]): s for d, s in tasks}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as error:  # recorded and reported; the run continues
                    failed.append({"fit": spec.key(), "error": repr(error)})
                    print("[e4] FAILED %s: %r" % (spec.key(), error), flush=True)
                    continue
                if len(done) % 50 == 0 or not pending:
                    print("[e4] %d of %d done, %d failed, %.0f s" % (len(done), len(tasks), len(failed), time.time() - started), flush=True)
    return done, failed


def slot_worker(task) -> dict:
    description, spec, i = task
    return worker(description, spec, i)


def run_fits(tasks: list, index: dict, started: float) -> tuple[list, list, int | None]:
    """The fits, by the runner chosen (--slots or the pool); with --slots also the number shed and queued again."""
    if SLOTS:
        killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
        listed = [(d, s, index[d["name"]]) for d, s in tasks]
        return fit_slots.run_slots(listed, slot_worker, lambda t: t[1].key(), lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS), "e4", started,
                                   every=50, killed=killed)
    done, failed = run_pool(tasks, index, started)
    return done, failed, None


# --------------------------------------------------------------------------- main


def main(mode: str, tier: int | None) -> None:
    started = time.time()
    features_record = load_features()
    cells = list(CELLS) + [f"G1-SU-{s}" for s in e1cells.e1inputs.SCREENS] + [f"{b}-DT-{s}" for b in ("G5", "G6") for s in e1cells.e1inputs.SCREENS]
    inputs = e1b.load_inputs(cells)
    refits = 1 if SMOKE else inputs["refits"]
    e1b.DATA["train_rows"] = {cell: int(e1b.DATA["cells"][cell].split("train").sum()) for cell in cells}
    if mode == "plan":
        listed = task_list(refits)
        tiers = pd.Series([fit_tiers.e4_tier(d, s) for d, s in listed]).fillna(0).astype(int)
        print("[e4] %d listed fits; per tier (0: dropped by the amendment): %s" % (len(listed), tiers.value_counts().sort_index().to_dict()), flush=True)
        return
    tasks = run_list(refits, tier)
    if SMOKE:
        tasks = [t for t in tasks if smoke_subset(t)]
    tasks = sorted(tasks, key=cost)
    if SMALLEST_FIRST:
        tasks = smallest_first(tasks, recorded_prices(tasks))
    print("[e4] tier %d: %d fits; %s" % (tier, len(tasks), pd.Series([s.head for _, s in tasks]).value_counts().to_dict()), flush=True)
    DATA["contexts"] = {}
    names = sorted({d["name"] for d, _ in tasks})
    index = {name: i for i, name in enumerate(names)}
    folder = predictions_root().parent if SMOKE else OUT
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"plan": PLAN, "fits": len(tasks), "refits": refits, "inputs": {**inputs, "features": features_record},
                "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING, "ridge grid": list(heads.RIDGE_GRID), "monotone": heads.MONOTONE},
                "seeds": {"heads": list(SEEDS), "refits": REFIT_SEED, "feature replicates": "label draws 1 to 3 (20260801 to 20260803)"},
                "predictions": config.portable(predictions_root()), "tier": tier, "smoke": SMOKE,
                "tiers": "bioinf2027/plans/E4.md, Section 16; bioinf2027/scripts/fit_tiers.py", "fitting_code": fitting_code(),
                "per_tier": {str(k): int(v) for k, v in pd.Series([fit_tiers.e4_tier(d, s) for d, s in tasks]).value_counts().sort_index().items()}}
    if SLOTS:
        protocol["runner"] = {"runner": "one process per fit (fit_slots.py)",
                              config.portable(Path(fit_slots.__file__).resolve()): sha256(Path(fit_slots.__file__))}
    if SMALLEST_FIRST:
        protocol["order"] = "smallest first: ascending recorded price (progress_status.py's price of E4)"
    protocol["earlier_tiers_compared"] = refuse_if_inputs_differ(folder, protocol)
    protocol_path = folder / f"protocol_tier{tier}.json"
    protocol_path.write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    done, failed, shed = run_fits(tasks, index, started)
    manifest = {"protocol": file_record(protocol_path), "tier": tier, "fits": len(tasks), "completed": len(done), "failed": failed,
                "reused": sum(bool(r.get("reused")) for r in done),
                "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done}, "seconds": round(time.time() - started, 1)}
    if SLOTS:
        manifest["fits_shed_and_queued_again"] = shed
    (folder / f"predictions_manifest_tier{tier}.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[e4] %d completed, %d failed" % (len(done), len(failed)), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    TIER, arguments = fit_tiers.parse_tier(sys.argv[1:])
    WORKERS, arguments = fit_tiers.parse_workers(arguments, WORKERS)
    for flag in ("--slots", "--smallest-first"):  # change no value
        if flag in arguments:
            arguments.remove(flag)
            if flag == "--slots":
                SLOTS = True
            else:
                SMALLEST_FIRST = True
    if "--workers-file" in arguments:  # changes no value
        i = arguments.index("--workers-file")
        WORKERS_FILE = config.storage_root() / arguments[i + 1]
        arguments = arguments[:i] + arguments[i + 2:]
    if arguments == ["--plan"]:
        main("plan", TIER)
    elif TIER is None:
        raise SystemExit("a fit run takes --tier 1, 2 or 3 (bioinf2027/plans/E4.md, Section 16)")
    elif arguments == ["--smoke"]:
        SMOKE = True
        WORKERS = 3  # beside the running jobs; never reported
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
        main("fit", TIER)
    elif arguments:
        raise SystemExit("unknown arguments: %s" % arguments)
    else:
        main("fit", TIER)
