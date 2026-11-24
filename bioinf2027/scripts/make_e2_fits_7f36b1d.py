"""E2, tests (i) and (ii): the fits of the new arms in E1b's harness (bioinf2027/plans/E2.md, Sections 4, 5 and 16).

For each encoder e (scGPT 512, Stack 1,600, PCA-50 and PCA-768) and each of E1b's cells G5 and G6, E1b's context is
built unchanged and each side's z is replaced by e's features (VCC rows by (split, batch, target gene); external
constructs by construct; PCA as x W). Representations (E2.md, Section 5): e, C1 (Gaussian of e's width, realisation
0), C3 (e with its rows shuffled, E1b's permutation), [B0; e], [B0; C1], [B0; C3] and N0 (the size-matched null; VCC
sides only; for PCA-50 the features `pca50_null` of make_e2_features.py, E2.md Section 17); in G6 [e; B2] and
[C1; B2] (the positive control at e's width); for PCA-50 also C2-50 (a random projection of x of width 50); for
PCA-768 only e and [B0; e]; Geneformer's N0. Labels existing and rebuilt; forest
and boosting with head seeds 0 to 2, ridge with seed 0; all training rows; E1b's head settings. Fitting follows
`make_e1b_attribution.fit` step by step. No module under bioinf2027/src/ changes.

Check E0, before any new fit: through this script's code, E1b's z forest seed 0 (G5, existing) and C1 boosting seed 0
(G6, existing) are refitted with Geneformer's z and compared exactly with E1b's saved files.
Predictions under $IGA_STORAGE_ROOT/bioinf2027/E2/predictions/; manifest and protocol in bioinf2027/results/E2/fits/.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, heads, transfer  # noqa: E402
from bioinf2027 import representations as rep  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import make_e1b_fits as e1f  # noqa: E402
import make_e1b_stats as e1s  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "fits"
PLAN = "bioinf2027/plans/E2.md, Sections 4, 5 and 16"
CELLS = tuple(e1cells.PRIMARY)
LABELS = ("existing", "rebuilt")
HEAD_SEEDS = (("forest", (0, 1, 2)), ("boosting", (0, 1, 2)), ("ridge", (0,)))
ENCODERS = {"scgpt": 512, "stack": 1600, "pca50": 50, "pca768": 768}
WORKERS = 6
WORKERS_FILE: Path | None = None  # --workers-file <path under the data root>: the worker count, read as fits finish (changes no value)
SMOKE = False
DATA: dict = {}


class Stop(RuntimeError):
    """A check failed: nothing new is fitted (hard limit 9)."""


def predictions_root() -> Path:
    base = config.storage_root() / "bioinf2027" / ("scratch/E2_fits_smoke" if SMOKE else "E2") / "predictions"
    return base


def out_dir() -> Path:
    return config.storage_root() / "bioinf2027" / "scratch" / "E2_fits_smoke" if SMOKE else OUT


# --------------------------------------------------------------------------- features on the sides


def load_features(wanted=("scgpt", "stack")) -> dict:
    record = json.loads((REPO / "bioinf2027" / "results" / "E2" / "features.json").read_text())["outputs"]
    out = {}
    for name in tuple(w for w in wanted if w in ("scgpt", "stack")) + ("geneformer_null", "pca") + (("pca50_null",) if "pca50" in wanted else ()):
        rec = record[name]
        path = config.storage_root() / rec["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != rec["sha256"]:
            raise Stop("the %s features do not match their record" % name)
        with np.load(path, allow_pickle=False) as z:
            out[name] = {k: z[k] for k in z.files}
    if not record["pca"]["check"]["passed"]:
        raise Stop("the PCA check of Section 3 has not passed; the PCA arm would be computed from the cells (not implemented)")
    return out


def vcc_keys(data: e1cells.CellData, split: str) -> list[str]:
    v = data.vcc[data.split(split)]
    return [f"{a}|{b}|{c}" for a, b, c in zip(v["split"], v["batch"].astype(str), v["target_gene"].astype(str))]


NULL_SOURCE = {"geneformer": "geneformer_null", "pca50": "pca50_null"}  # N0 read from a file of its own


def encoder_sides(cell: str, encoder: str, null: bool = False) -> dict:
    """The context's sides with z replaced by the encoder's features (or its size-matched null)."""
    context = DATA["contexts"][cell]
    data = e1f.DATA["cells"][cell]
    F = DATA["features"]
    out = {}
    for name, side in [("train", context.train)] + list(context.sides.items()):
        if encoder.startswith("pca") and not null:
            width = ENCODERS[encoder]
            z = (side.x["main"].astype(np.float64) @ F["pca"]["W"][:, :width].astype(np.float64)).astype(np.float32)
        elif name in ("train", "val", "test"):
            source = F[NULL_SOURCE[encoder]] if encoder in NULL_SOURCE else F[encoder]
            index = {k: i for i, k in enumerate(source["keys"].tolist())}
            keys = vcc_keys(data, name)
            missing = [k for k in keys if k not in index]
            if missing:
                raise Stop("%s: %d VCC rows of %s have no %s feature" % (cell, len(missing), name, encoder))
            values = source["null" if encoder in NULL_SOURCE else ("vcc_null" if null else "vcc")]
            z = values[[index[k] for k in keys]]
        else:
            if null or encoder == "geneformer":
                z = np.full((side.n, side.z.shape[1] if encoder == "geneformer" else ENCODERS[encoder]), np.nan, dtype=np.float32)
            else:
                constructs = F[encoder][f"{name}_constructs"].tolist()
                index = {c: i for i, c in enumerate(constructs)}
                own = DATA["screens"][name]
                z = F[encoder][f"{name}_arm"][[index[c] for c in own]]
        out[name] = dataclasses.replace(side, z=np.ascontiguousarray(z, dtype=np.float32))
    return out


# --------------------------------------------------------------------------- representations

BASE = {"e": "z", "C3": "C3", "N0": "z"}


def representations(encoder: str) -> list[str]:
    if encoder == "pca768":
        return ["e", "[B0; e]"]
    out = ["e", "C1", "C3", "[B0; e]", "[B0; C1]", "[B0; C3]", "N0"]  # N0 of PCA-50: E2.md, Section 17
    if encoder == "pca50":
        out.append("C2-50")
    return out


class Matrix:
    def __init__(self, spec: FitSpec, encoder: str, sides: dict, rows: np.ndarray) -> None:
        self.spec, self.encoder, self.sides, self.rows = spec, encoder, sides, rows
        r = spec.representation
        core = r.replace("[B0; ", "").replace("; B2]", "").replace("[", "").replace("]", "")
        self.core = core
        self.width = ENCODERS.get(encoder, 768)
        train = sides["train"]
        if core in ("e", "N0", "C3"):
            self.blocks = transfer.Blocks(FitSpec(spec.context, spec.label, BASE[core], spec.head, spec.seed, spec.variant, 0), train, rows)
        elif core == "C1":
            self.scaler = rep.BlockScaler(self._c1(train)[rows])
        elif core == "C2-50":
            self.projection = rep.c2_projection(0, out_dim=50)
            self.scaler = rep.BlockScaler(rep.c2_features(train.x["main"], self.projection)[rows])
        else:
            raise ValueError(r)

    def _c1(self, side: transfer.Side) -> np.ndarray:
        return rep.c1_features(side.control_total, side.control_split, 0, dim=self.width)

    def block(self, side: transfer.Side) -> np.ndarray:
        if self.core in ("e", "N0", "C3"):
            return self.blocks.matrix(side)
        if self.core == "C1":
            return self.scaler(self._c1(side))
        return self.scaler(rep.c2_features(side.x["main"], self.projection))

    def __call__(self, side: transfer.Side) -> np.ndarray:
        parts = [self.block(side)]
        r = self.spec.representation
        if r.startswith("[B0; "):
            parts.insert(0, side.b[:, [0]])
        if r.endswith("; B2]"):
            parts.append(side.b[:, [2]])
        return np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 else parts[0]


def representation_name(encoder: str, r: str) -> str:
    return r.replace("e", encoder) if r in ("e", "[B0; e]", "[e; B2]") else f"{r}[{encoder}]"


def fit(cell: str, encoder: str, spec: FitSpec, r: str) -> tuple[dict, dict]:
    context = DATA["contexts"][cell]
    rows = transfer.training_rows(context, spec, e1f.REFIT_SEED)
    y = context.labels[spec.label][rows]
    sides = DATA["sides"][(cell, encoder, r == "N0")]
    matrix = Matrix(dataclasses.replace(spec, representation=r), encoder, sides, rows)
    start = time.time()
    record, out = {}, {}
    with heads.single_thread():
        x_train = matrix(sides["train"])[rows]
        if spec.head == "ridge":
            val_label = context.labels[spec.label + "@val"]
            mask = np.isfinite(val_label)
            model, scaler, choice = heads.fit_ridge_by_validation(x_train, y, matrix(sides["val"])[mask], val_label[mask])
            record["ridge"] = choice

            def predict(m):
                return model.predict(scaler.transform(np.asarray(m, dtype=np.float64)))
        else:
            model = transfer._head(spec).fit(x_train, y)

            def predict(m):
                return model.predict(m)
        for name in context.sides:
            m = matrix(sides[name])
            values = np.full(m.shape[0], np.nan)
            usable = np.isfinite(m).all(axis=1)
            if usable.any():
                values[usable] = predict(m[usable])
            out[name] = values
    record["seconds"] = time.time() - start
    record["training_rows"] = int(rows.size)
    return out, record


def save(spec: FitSpec, out: dict, record: dict) -> dict:
    folder = predictions_root() / spec.context.replace(" | ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / spec.file_name()
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, spec=np.array(json.dumps(spec.as_dict(), sort_keys=True)), record=np.array(json.dumps(record, sort_keys=True)),
             **{name.replace("|", "__"): values for name, values in out.items()})
    os.replace(temporary, path)
    return {"spec": spec.as_dict(), "file": file_record(path), "seconds": record["seconds"]}


def worker(task) -> dict:
    cell, encoder, r, spec = task
    path = predictions_root() / spec.context.replace(" | ", "_") / spec.file_name()
    if path.exists():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == json.dumps(spec.as_dict(), sort_keys=True):
                return {"spec": spec.as_dict(), "file": file_record(path), "reused": True}
    out, record = fit(cell, encoder, spec, r)
    return {**save(spec, out, record), "reused": False}


def tasks(encoders: list[str]) -> list:
    out = []
    for encoder in encoders:
        reps = representations(encoder)
        for cell in CELLS:
            extra = ["[e; B2]", "[C1; B2]"] if (cell == "G6" and encoder != "pca768") else []
            for r in reps + extra:
                for label in LABELS:
                    for head, seeds in HEAD_SEEDS:
                        for seed in (seeds[:1] if SMOKE else seeds):
                            spec = FitSpec(f"E2 | {encoder} | {cell}", label, representation_name(encoder, r), head, seed)
                            out.append((cell, encoder, r, spec))
    weight = {"forest": 1.0, "boosting": 0.05, "ridge": 0.02}
    return sorted(out, key=lambda t: -weight[t[3].head] * ENCODERS.get(t[1], 768))


def geneformer_null_tasks() -> list:
    out = []
    for cell in CELLS:
        for label in LABELS:
            for head, seeds in HEAD_SEEDS:
                for seed in (seeds[:1] if SMOKE else seeds):
                    out.append((cell, "geneformer", "N0", FitSpec(f"E2 | geneformer | {cell}", label, "N0[geneformer]", head, seed)))
    return out


# --------------------------------------------------------------------------- check E0


def check_e0() -> list:
    """E1b's z forest (G5) and C1 boosting (G6) refitted through this code with Geneformer's z, compared exactly."""
    results = []
    e1b = e1s.Predictions(e1f.OUT / "predictions_manifest.json")
    for cell, r, head, base in (("G5", "e", "forest", "z"), ("G6", "C1", "boosting", "C1")):
        context = DATA["contexts"][cell]
        sides = {"train": context.train, **context.sides}
        spec = FitSpec(cell, "existing", base, head, 0)
        rows = transfer.training_rows(context, spec, e1f.REFIT_SEED)
        matrix = Matrix(dataclasses.replace(spec, representation=r), "geneformer", sides, rows)
        y = context.labels["existing"][rows]
        with heads.single_thread():
            model = transfer._head(spec).fit(matrix(sides["train"])[rows], y)
            saved = e1b.get(cell, "existing", base, head, 0)
            equal = {}
            for name in context.sides:
                m = matrix(sides[name])
                values = np.full(m.shape[0], np.nan)
                ok = np.isfinite(m).all(axis=1)
                values[ok] = model.predict(m[ok])
                equal[name] = bool(np.array_equal(values, saved[name], equal_nan=True))
        results.append({"fit": spec.key(), "sides_equal": equal, "passed": all(equal.values())})
        if not all(equal.values()):
            raise Stop("check E0 failed: %s" % json.dumps(results))
    return results


def worker_limit() -> int:
    """Fits in progress at once: the integer in --workers-file when it holds one, else --workers. Changes no value: each
    fit is a function of its specification and of the inputs loaded before the first worker starts."""
    if WORKERS_FILE is not None:
        try:
            return max(1, int(WORKERS_FILE.read_text().strip()))
        except (OSError, ValueError):
            pass
    return WORKERS


def resize(pools: list, limit: int, more: bool) -> list:
    """Adds a pool of workers when the limit rises and fits remain; closes the newest idle pools that the limit no longer
    needs. Every pool forks from this process after the inputs are loaded, so every worker holds the same inputs."""
    capacity = sum(p["size"] for p in pools)
    if more and capacity < limit:
        size = limit - capacity
        pools = pools + [{"pool": ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("fork")), "size": size, "busy": 0}]
        print("[e2 fits] workers: %d (a pool of %d added)" % (limit, size), flush=True)
    while len(pools) > 1 and pools[-1]["busy"] == 0 and sum(p["size"] for p in pools) - pools[-1]["size"] >= limit:
        pools[-1]["pool"].shutdown(wait=True)
        print("[e2 fits] workers: %d (a pool of %d closed)" % (limit, pools[-1]["size"]), flush=True)
        pools = pools[:-1]
    return pools


def run_tasks(listed: list, function, name_of, started: float) -> tuple[list, list]:
    """Runs every task once, submitted in the listed order, with at most worker_limit() in progress (read again after
    each wait); returns the results and the failures (each named by name_of)."""
    done, failed = [], []
    queue, pending, pools = list(listed), {}, []
    try:
        while queue or pending:
            limit = worker_limit()
            pools = resize(pools, limit, bool(queue))
            for pool in pools:
                while queue and pool["busy"] < pool["size"] and len(pending) < limit:
                    task = queue.pop(0)
                    pending[pool["pool"].submit(function, task)] = (task, pool)
                    pool["busy"] += 1
            finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                task, pool = pending.pop(future)
                pool["busy"] -= 1
                try:
                    done.append(future.result())
                except Exception as error:
                    failed.append({"fit": name_of(task), "error": repr(error)})
                    print("[e2 fits] FAILED %s: %r" % (name_of(task), error), flush=True)
                    continue
                if len(done) % 25 == 0 or not (pending or queue):
                    print("[e2 fits] %d of %d done, %d failed, %.0f s" % (len(done), len(listed), len(failed), time.time() - started), flush=True)
    finally:
        for pool in pools:
            pool["pool"].shutdown(wait=True, cancel_futures=True)
    return done, failed


def run(encoders: list[str], include_geneformer_null: bool) -> None:
    started = time.time()
    inputs = e1f.load_inputs(list(CELLS))
    order = list(e1cells.CELLS)
    DATA["contexts"] = {cell: e1f.build_context(cell, order.index(cell), {}) for cell in CELLS}
    DATA["screens"] = {s: e1f.DATA["screens"][s].constructs.tolist() for s in e1f.DATA["screens"]}
    DATA["features"] = load_features(tuple(encoders))
    real_heads = (copy.deepcopy(heads.FOREST), copy.deepcopy(heads.BOOSTING))
    checks = {"E0": check_e0()}
    print("[e2 fits] check E0 passed", flush=True)
    if SMOKE:
        heads.FOREST["n_estimators"] = 5
        heads.BOOSTING["max_iter"] = 20
    DATA["sides"] = {}
    for encoder in encoders:
        for cell in CELLS:
            DATA["sides"][(cell, encoder, False)] = encoder_sides(cell, encoder)
            if "N0" in representations(encoder):
                DATA["sides"][(cell, encoder, True)] = encoder_sides(cell, encoder, null=True)
    if include_geneformer_null:
        for cell in CELLS:
            DATA["sides"][(cell, "geneformer", True)] = encoder_sides(cell, "geneformer", null=True)
    listed = tasks(encoders) + (geneformer_null_tasks() if include_geneformer_null else [])
    folder = out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"plan": PLAN, "fits": len(listed), "encoders": {e: ENCODERS[e] for e in encoders}, "geneformer_null": include_geneformer_null,
                "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING, "ridge grid": list(heads.RIDGE_GRID)},
                "check_E0_heads": {"forest": real_heads[0], "boosting": real_heads[1]}, "checks": checks,
                "inputs": {**inputs, "features": file_record(REPO / "bioinf2027" / "results" / "E2" / "features.json"),
                           "e1b_predictions": file_record(e1f.OUT / "predictions_manifest.json")},
                "fitting_code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (transfer, heads, rep)},
                "script": {config.portable(Path(__file__)): sha256(Path(__file__))}, "smoke": SMOKE}
    name = "_".join(encoders) + ("_gfnull" if include_geneformer_null else "")
    (folder / f"protocol_{name}.json").write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    done, failed = run_tasks(listed, worker, lambda t: t[3].key(), started)
    manifest = {"protocol": file_record(folder / f"protocol_{name}.json"), "fits": len(listed), "completed": len(done), "failed": failed,
                "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done}, "seconds": round(time.time() - started, 1)}
    (folder / f"predictions_manifest_{name}.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[e2 fits] %d completed, %d failed, %.0f s" % (len(done), len(failed), time.time() - started), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
    gf = "--geneformer-null" in arguments
    if gf:
        arguments.remove("--geneformer-null")
    for flag in ("--workers", "--workers-file"):  # change no value
        if flag in arguments:
            i = arguments.index(flag)
            if flag == "--workers":
                WORKERS = int(arguments[i + 1])
            else:
                WORKERS_FILE = config.storage_root() / arguments[i + 1]
            arguments = arguments[:i] + arguments[i + 2:]
    if not all(a in ENCODERS for a in arguments) or not (arguments or gf):
        raise SystemExit("usage: make_e2_fits.py [scgpt] [stack] [pca50] [pca768] [--geneformer-null] [--smoke] [--workers N] [--workers-file PATH-UNDER-DATA-ROOT]")
    run(arguments, gf)
