"""E2, test (iii): E6's T for the new arms (bioinf2027/plans/E2.md, Sections 5 and 15; E6's design unchanged).

E6's harness (`make_e6_fits`): its folds (seed 20271204), endpoint 27/788 (the rebuilt label), constructs with at least
27 cells, both directions, five gene-disjoint folds, forest with head seeds 0 to 2. Each screen's z is replaced by the
encoder's construct features (`z_override`, as E6 does for D2's z-). Representations: e, C1 of e's width (each
construct's Gaussian row fixed across folds through E6's `control_rows`, realisation 0), [e; B2] and [C1; B2]: 120
fits per encoder; PCA-768 e only. scGPT runs this test by the author's O4 decision (gate G4/G5, item 5).

Check E6-0, before any new fit: through this script's code, E6's z forest seed 0 (RPE1 to K562-essential, fold 0) and
C1 forest seed 0 are refitted with E6's own z and compared exactly with E6's saved predictions.
Predictions under $IGA_STORAGE_ROOT/bioinf2027/E2/predictions_e6/; manifest and protocol in bioinf2027/results/E2/fits/.

--slots runs each fit in a process of its own (fit_slots.py), so that the memory rules can shed one worker at a time
(the author's change of 2026-11-15 to how E2 meets the spikes, bioinf2027/GATE.md); --workers sets the number of fits
in progress and --workers-file (a path under the data root) a file whose integer replaces it as fits finish. A fit is
a function of its specification and of the inputs loaded before the first worker starts, so neither changes a value.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

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

from bioinf2027 import config, heads, transfer  # noqa: E402
from bioinf2027 import representations as rep  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import fit_slots  # noqa: E402
import make_e2_fits as e2f  # noqa: E402
import make_e6_fits as e6f  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E2" / "fits"
PLAN = "bioinf2027/plans/E2.md, Sections 5 and 15 (test iii)"
WIDTHS = {"scgpt": 512, "stack": 1600, "pca50": 50, "pca768": 768}
REPRESENTATIONS = ("e", "C1", "[e; B2]", "[C1; B2]")
SEEDS = (0, 1, 2)
WORKERS = 6
WORKERS_FILE: Path | None = None  # --workers-file <path under the data root>: the worker count, read as fits finish (changes no value)
SLOTS = False  # --slots: one process per fit, so that one worker can be shed at a time (fit_slots.py; changes no value)
SMOKE = False
DATA: dict = {}


class Stop(RuntimeError):
    pass


def predictions_root() -> Path:
    return config.storage_root() / "bioinf2027" / ("scratch/E2_fits_smoke" if SMOKE else "E2") / "predictions_e6"


def overrides(encoder: str) -> dict:
    """Each screen's construct features of the encoder, in E6's construct order."""
    F = DATA["features"]
    out = {}
    for screen in e6f.SCREENS:
        constructs = e6f.DATA["screens"][screen].constructs.tolist()
        if encoder.startswith("pca"):
            x, _ = e6f.DATA[("x", screen)]["primary"]
            out[screen] = (x.astype(np.float64) @ F["pca"]["W"][:, :WIDTHS[encoder]].astype(np.float64)).astype(np.float32)
        else:
            index = {c: i for i, c in enumerate(F[encoder][f"{screen}_constructs"].tolist())}
            out[screen] = F[encoder][f"{screen}_arm"][[index[c] for c in constructs]].astype(np.float32)
    return out


class Matrix:
    def __init__(self, r: str, width: int, train: transfer.Side, rows: np.ndarray, spec: FitSpec) -> None:
        self.r, self.width = r, width
        core = r.replace("; B2]", "").replace("[", "")
        self.core = core
        if core == "e":
            self.blocks = transfer.Blocks(dataclasses.replace(spec, representation="z"), train, rows)
        else:
            self.scaler = rep.BlockScaler(self._c1(train)[rows])

    def _c1(self, side: transfer.Side) -> np.ndarray:
        full = rep.c1_features(side.control_total, side.control_split, 0, dim=self.width)
        return full if side.control_rows is None else full[side.control_rows]

    def __call__(self, side: transfer.Side) -> np.ndarray:
        block = self.blocks.matrix(side) if self.core == "e" else self.scaler(self._c1(side))
        if self.r.endswith("; B2]"):
            return np.concatenate([block, side.b[:, [2]]], axis=1).astype(np.float32)
        return block


def fit(context: transfer.Context, spec: FitSpec, r: str, width: int) -> tuple[dict, dict]:
    rows = transfer.training_rows(context, spec, e6f.REFIT_SEED)
    y = context.labels[spec.label][rows]
    matrix = Matrix(r, width, context.train, rows, spec)
    start = time.time()
    out = {}
    with heads.single_thread():
        model = transfer._head(spec).fit(matrix(context.train)[rows], y)
        for name, side in context.sides.items():
            m = matrix(side)
            values = np.full(m.shape[0], np.nan)
            ok = np.isfinite(m).all(axis=1)
            if ok.any():
                values[ok] = model.predict(m[ok])
            out[name] = values
    return out, {"seconds": time.time() - start, "training_rows": int(rows.size)}


def context_of(encoder: str, source: str, k: int) -> transfer.Context:
    key = (encoder, source, k)
    if key not in DATA["contexts"]:
        override = None if encoder == "geneformer" else DATA["overrides"][encoder]
        DATA["contexts"][key] = e6f.build_context(source, e6f.PRIMARY, f"fold {k}", index=e6f.stable_index(f"E2 | {encoder} | {source} | {k}"),
                                                  z_override=override)
    return DATA["contexts"][key]


def worker(task) -> dict:
    encoder, source, k, r, spec = task
    folder = predictions_root() / encoder
    path = folder / spec.file_name()
    if path.exists():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == json.dumps(spec.as_dict(), sort_keys=True):
                return {"spec": spec.as_dict(), "file": file_record(path), "reused": True}
    out, record = fit(context_of(encoder, source, k), spec, r, WIDTHS[encoder])
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, spec=np.array(json.dumps(spec.as_dict(), sort_keys=True)), record=np.array(json.dumps(record, sort_keys=True)),
             **{n.replace("|", "__"): v for n, v in out.items()})
    os.replace(temporary, path)
    return {"spec": spec.as_dict(), "file": file_record(path), "reused": False}


def tasks(encoders: list[str]) -> list:
    out = []
    for encoder in encoders:
        reps = ("e",) if encoder == "pca768" else REPRESENTATIONS
        for source in e6f.SCREENS:
            for k in range(5):
                for r in reps:
                    for seed in (SEEDS[:1] if SMOKE else SEEDS):
                        name = e2f.representation_name(encoder, r)
                        context = f"E2 | E6 | {encoder} | {source}>{e6f.other(source)} | fold {k}"
                        out.append((encoder, source, k, r, FitSpec(context, "rebuilt", name, "forest", seed)))
    return sorted(out, key=lambda t: -WIDTHS[t[0]])


def check_e6_0() -> list:
    """E6's z and C1 forest fits (seed 0, RPE1 to K562-essential, fold 0) refitted through this code, compared exactly."""
    manifest = json.loads((REPO / "bioinf2027" / "results" / "E6" / "predictions_manifest_tier1.json").read_text())
    results = []
    context = context_of("geneformer", "RPE1", 0)
    for r, base in (("e", "z"), ("C1", "C1")):
        spec = FitSpec(context.name, "rebuilt", base, "forest", 0)
        record = manifest["files"].get(spec.key())
        if record is None:
            raise Stop("check E6-0: E6 holds no %s" % spec.key())
        path = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != record["sha256"]:
            raise Stop("check E6-0: %s does not match E6's manifest" % spec.key())
        _, _, saved = transfer.load_predictions(path)
        out, _ = fit(context, spec, r, 768)
        equal = {n: bool(np.array_equal(v, saved[n], equal_nan=True)) for n, v in out.items() if n in saved}
        results.append({"fit": spec.key(), "sides_equal": equal, "passed": bool(equal) and all(equal.values())})
        if not results[-1]["passed"]:
            raise Stop("check E6-0 failed: %s" % json.dumps(results))
    return results


def run_pool(listed: list, started: float) -> tuple[list, list]:
    """Every fit in a pool of WORKERS processes forked after the inputs are loaded."""
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("fork")) as pool:
        pending = {pool.submit(worker, t): t for t in listed}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                t = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as error:
                    failed.append({"fit": t[4].key(), "error": repr(error)})
                    print("[e2 e6] FAILED %s: %r" % (t[4].key(), error), flush=True)
                    continue
                if len(done) % 20 == 0 or not pending:
                    print("[e2 e6] %d of %d done, %d failed, %.0f s" % (len(done), len(listed), len(failed), time.time() - started), flush=True)
    return done, failed


def run_listed(listed: list, started: float) -> tuple[list, list, int | None]:
    """The fits, by the runner chosen (--slots or the pool); with --slots also the number shed and queued again."""
    if SLOTS:
        killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
        return fit_slots.run_slots(listed, worker, lambda t: t[4].key(), lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS), "e2 e6", started,
                                   every=20, killed=killed)
    done, failed = run_pool(listed, started)
    return done, failed, None


def run(encoders: list[str]) -> None:
    started = time.time()
    inputs = e6f.load_inputs()
    DATA["contexts"] = {}
    e2f.DATA["features"] = DATA["features"] = e2f.load_features(tuple(encoders))
    checks = {"E6-0": check_e6_0()}
    print("[e2 e6] check E6-0 passed", flush=True)
    if SMOKE:
        heads.FOREST["n_estimators"] = 5
    DATA["overrides"] = {e: overrides(e) for e in encoders}
    for e in encoders:
        for s in e6f.SCREENS:
            for k in range(5):
                context_of(e, s, k)
    listed = tasks(encoders)
    OUT.mkdir(parents=True, exist_ok=True)
    name = "e6_" + "_".join(encoders) + ("_smoke" if SMOKE else "")
    protocol = {"plan": PLAN, "fits": len(listed), "encoders": {e: WIDTHS[e] for e in encoders}, "checks": checks, "inputs": inputs,
                "heads": {"forest": heads.FOREST}, "folds_seed": e6f.FOLD_SEED, "endpoint": e6f.PRIMARY, "label": "rebuilt",
                "fitting_code": {config.portable(Path(m.__file__)): sha256(Path(m.__file__)) for m in (transfer, heads, rep)},
                "script": {config.portable(Path(__file__)): sha256(Path(__file__))}, "smoke": SMOKE}
    if SLOTS:
        protocol["runner"] = "one process per fit (fit_slots.py)"
        protocol["script"].update({config.portable(Path(fit_slots.__file__).resolve()): sha256(Path(fit_slots.__file__))})
    (OUT / f"protocol_{name}.json").write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    done, failed, shed = run_listed(listed, started)
    manifest = {"protocol": file_record(OUT / f"protocol_{name}.json"), "fits": len(listed), "completed": len(done), "failed": failed,
                "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done}, "seconds": round(time.time() - started, 1)}
    if SLOTS:
        manifest["fits_shed_and_queued_again"] = shed
    (OUT / f"predictions_manifest_{name}.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[e2 e6] %d completed, %d failed, %.0f s" % (len(done), len(failed), time.time() - started), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
    if "--slots" in arguments:  # changes no value
        SLOTS = True
        arguments.remove("--slots")
    for flag in ("--workers", "--workers-file"):  # change no value
        if flag in arguments:
            i = arguments.index(flag)
            if flag == "--workers":
                WORKERS = int(arguments[i + 1])
            else:
                WORKERS_FILE = config.storage_root() / arguments[i + 1]
            arguments = arguments[:i] + arguments[i + 2:]
    if not arguments or not all(a in WIDTHS for a in arguments):
        raise SystemExit("usage: make_e2_fits_e6.py {scgpt|stack|pca50|pca768} ... [--smoke] [--workers N] [--workers-file PATH-UNDER-DATA-ROOT] [--slots]")
    run(arguments)
