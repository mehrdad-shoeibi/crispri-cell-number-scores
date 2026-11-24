"""E2 (d): the external size-matched null, post hoc (bioinf2027/plans/E2_null_external.md; gate G3, item 5 (d), the
author's answer of 2026-11-16, bioinf2027/GATE.md). Three stages, each run on its own:

--features  N0x for Geneformer, scGPT, Stack and PCA-50 on each Replogle screen: for construct t (E1b's external
            table order) the mean of the encoder's per-cell vectors over n_t control cells of the screen (E2's part
            <screen>_control), drawn with numpy.random.default_rng([20271131, i_s, t]) (i_s 0 for RPE1, 1 for
            K562-essential; without replacement unless n_t exceeds the control cells), minus the mean over all of the
            screen's control cells; n_t the construct's eligible perturbed cells, counted as
            make_e2_features.external_features counts them. The same drawn cells for every encoder. Per-cell vectors:
            scGPT and Stack from E2's embeddings; PCA-50 the cell's scores on the first 50 axes of E2's PCA basis of its
            normalised expression on the 2,000 features (the E4 cache), as make_e2_features.pca50_null scores VCC's
            controls; Geneformer from its cached per-cell embeddings (the stores D2 read), by the cell's row in the
            screen's file. Output: $IGA_STORAGE_ROOT/bioinf2027/E2/null_external/features.npz and
            bioinf2027/results/E2/null_external/features.json. Checks: every control cell has a vector for every
            encoder; no value is NaN except for a construct without cells (as e's external feature).
--fits      24 forest fits through make_e2_fits.py's code (contexts, sides, Matrix, head): representation N0x[<e>],
            trained on the VCC null features of (i) (N0 of the encoder; N0_z for Geneformer) on the rebuilt label,
            head seeds 0 to 2, cells G5 and G6, the external sides holding N0x. Check, in each fit's process: its VCC
            validation and test predictions equal those of the saved N0 fit of the same specification, bit for bit,
            or the fit fails. One process per fit (fit_slots.py; --workers N, --workers-file under the data root, read
            as fits finish; the newest fit is shed and queued again when the number falls). Predictions under
            $IGA_STORAGE_ROOT/bioinf2027/E2/predictions_null_external/; manifest and protocol in
            bioinf2027/results/E2/null_external/. --smoke: PCA-50 only (6 fits, the real heads), to
            $IGA_STORAGE_ROOT/bioinf2027/scratch/E2_null_external_smoke/.
--stats     E1b's joint external draws (10,000 replicates, seeds 20260901 and 20260902), the Spearman with the screen's
            rebuilt label, the mean over head seeds inside each replicate (make_e2_stats.rho_ext): T_N0 = rho(N0x) -
            rho(C1) and e - N0x = rho(e) - rho(N0x), paired; with T, rho(e) and rho(C1) reproduced to 1e-12 from
            bioinf2027/results/E2/stats/external.csv, or nothing is written. The reading of the plan note, Section 5.
            Outputs: bioinf2027/results/E2/null_external/null_external.csv, readings.csv and stats.json.
Every value is post hoc and unadjusted, and not part of E2's family.

--label existing (with --fits or --stats): follow-up (4) of bioinf2027/plans/E2_null_external_followup.md (gate G7,
            item 2): the same 24 fits and statistics with the training label switched to the existing (all-cells) VCC
            label, each fit checked against the saved N0 fit on that label. The null features are (d)'s, unchanged.
            Predictions under $IGA_STORAGE_ROOT/bioinf2027/E2/predictions_null_external_existing/; protocol, manifest
            and statistics in bioinf2027/results/E2/null_external_followup/existing_label/; smoke to
            $IGA_STORAGE_ROOT/bioinf2027/scratch/E2_null_external_existing_smoke/. With the default label (rebuilt)
            every path and output is as before.
"""
from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"

import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, features  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import e2cells  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PLAN = "bioinf2027/plans/E2_null_external.md"
RESULTS = REPO / "bioinf2027" / "results" / "E2" / "null_external"
FEATURES_JSON = RESULTS / "features.json"  # the null features of (d), the same for every label
FOLLOWUP = REPO / "bioinf2027" / "results" / "E2" / "null_external_followup"
FOLLOWUP_PLAN = "bioinf2027/plans/E2_null_external_followup.md"
LABELS = ("rebuilt", "existing")
ENCODERS = ("geneformer", "scgpt", "stack", "pca50")
SEED = 20271131
SCREEN_INDEX = {"RPE1": 0, "K562-essential": 1}
CELLS = ("G5", "G6")
LABEL = "rebuilt"
SEEDS = (0, 1, 2)
POST_HOC = "post hoc, unadjusted, not part of E2's family (gate G3, item 5 (d); bioinf2027/plans/E2_null_external.md)"
TOLERANCE = 1e-12
SMOKE = False
WORKERS = 4
WORKERS_FILE: Path | None = None


class Stop(RuntimeError):
    """A check failed: nothing is written (hard limit 9)."""


def data_dir() -> Path:
    """The null features (the same for a smoke run: they are fixed by the seed)."""
    return config.storage_root() / "bioinf2027" / "E2" / "null_external"


def suffix() -> str:
    """'' for (d)'s label, '_<label>' for a follow-up label."""
    return "" if LABEL == "rebuilt" else f"_{LABEL}"


def outputs_dir() -> Path:
    """(d)'s results folder for the rebuilt label; the follow-up's for another label."""
    return RESULTS if LABEL == "rebuilt" else FOLLOWUP / f"{LABEL}_label"


def results_dir() -> Path:
    """The fits' protocol and manifest: a smoke run's under scratch."""
    return config.storage_root() / "bioinf2027" / "scratch" / f"E2_null_external{suffix()}_smoke" if SMOKE else outputs_dir()


def predictions_dir() -> Path:
    return config.storage_root() / "bioinf2027" / (f"scratch/E2_null_external{suffix()}_smoke" if SMOKE else "E2") / f"predictions_null_external{suffix()}"


def plan() -> str:
    return PLAN if LABEL == "rebuilt" else FOLLOWUP_PLAN


def status() -> str:
    if LABEL == "rebuilt":
        return POST_HOC
    return "post hoc, unadjusted, not part of E2's family (gate G7, item 2, follow-up (4); %s)" % FOLLOWUP_PLAN


def name_of(encoder: str) -> str:
    return f"N0x[{encoder}]"


# --------------------------------------------------------------------------- stage 1: the features


def construct_counts(screen: str) -> tuple[np.ndarray, np.ndarray]:
    """E1b's external constructs of the screen (table order) and their eligible perturbed cells, as external_features counts them."""
    cache = e2cells._cache(f"screen_{screen}")
    table = features.read_endpoint(config.screen_paths()[screen]["endpoint"])
    order = np.argsort(cache["pert_code"], kind="stable")
    bounds = np.searchsorted(cache["pert_code"][order], np.arange(cache["constructs"].size + 1))
    by_construct = {c: i for i, c in enumerate(cache["constructs"].tolist())}
    constructs = table["construct"].astype(str).to_numpy()
    n = np.array([bounds[by_construct[c] + 1] - bounds[by_construct[c]] for c in constructs.tolist()], dtype=np.int64)
    return constructs, n


def control_vectors(encoder: str, screen: str, P: dict) -> np.ndarray:
    """The encoder's vector of every control cell of the screen, in the order of E2's part <screen>_control."""
    part = P[f"{screen}_control"]
    folder = config.storage_root() / "bioinf2027" / "E2" / "embeddings"
    if encoder == "scgpt":
        v = np.load(folder / "scgpt" / f"{screen}_control.npy", allow_pickle=False)
        rows = np.load(folder / "scgpt" / f"{screen}_control_rows.npy", allow_pickle=False)
        if not np.array_equal(rows, part["rows"]) or v.shape[0] != rows.size:
            raise Stop("scGPT's %s_control does not cover its part" % screen)
        return v.astype(np.float64)
    if encoder == "stack":
        out = np.full((part["rows"].size, 0), np.nan)
        filled = np.zeros(part["rows"].size, dtype=bool)
        for path in sorted((folder / "stack" / "units").glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                sel = z["part"] == f"{screen}_control"
                if not sel.any():
                    continue
                if out.shape[1] == 0:
                    out = np.full((part["rows"].size, z["vectors"].shape[1]), np.nan)
                out[z["index"][sel]] = z["vectors"][sel]
                filled[z["index"][sel]] = True
        if not filled.all():
            raise Stop("Stack's units do not cover %s_control" % screen)
        return out
    if encoder == "pca50":
        import make_e2_features as e2feat

        e4, lab = e2feat._e4(f"screen_{screen}"), e2cells._cache(f"screen_{screen}")
        if not (np.array_equal(e4["nt_rows"], lab["nt_rows"]) and np.array_equal(lab["nt_rows"], part["rows"])):
            raise Stop("the E4 cache, the E1a cache and the control part list other %s control cells" % screen)
        recorded = json.loads((REPO / "bioinf2027" / "results" / "E2" / "features.json").read_text())["outputs"]["pca"]
        basis = e2feat.out_dir() / "pca_basis.npz"
        if sha256(basis) != recorded["sha256"]:
            raise Stop("pca_basis.npz does not match features.json")
        with np.load(basis, allow_pickle=False) as store:
            W = store["W"][:, :50].astype(np.float64)
        return e2feat._values(e4["nt_counts"], lab["nt_total"]) @ W
    if encoder == "geneformer":
        import make_d2_embeddings as d2emb
        import make_d2_units as d2units

        store = d2units.external_store(screen)
        nt_store = store[store["role_in_store"] == "nt"].drop_duplicates("orig_obs_idx").set_index("orig_obs_idx")
        found = nt_store.reindex(part["rows"])
        if found["cached_file"].isna().any():
            raise Stop("%s: a control cell has no cached Geneformer embedding" % screen)
        out = np.empty((part["rows"].size, 768), dtype=np.float64)
        for (store_name, name), group in found.reset_index().groupby(["store", "cached_file"]):
            array = np.load(d2emb.cached_path(store_name, name), mmap_mode="r")
            out[group.index.to_numpy()] = np.asarray(array[group["cached_row"].to_numpy(dtype=np.int64)], dtype=np.float64)
        return out
    raise Stop("unknown encoder %s" % encoder)


def draws(screen: str, n: np.ndarray, controls: int) -> list:
    out = []
    for t, k in enumerate(n.tolist()):
        out.append(None if k == 0 else np.random.default_rng([SEED, SCREEN_INDEX[screen], t]).choice(controls, size=k, replace=k > controls))
    return out


def run_features() -> None:
    started = time.time()
    P = e2cells.parts()
    arrays, record = {}, {"plan": PLAN, "seed": SEED, "screens": {}, "encoders": {}}
    for screen in e2cells.SCREENS:
        constructs, n = construct_counts(screen)
        controls = P[f"{screen}_control"]["rows"].size
        picks = draws(screen, n, controls)
        arrays[f"{screen}__constructs"] = np.array(constructs.tolist(), dtype=str)
        arrays[f"{screen}__n"] = n
        record["screens"][screen] = {"constructs": int(constructs.size), "constructs_without_cells": int((n == 0).sum()), "control_cells": int(controls),
                                     "constructs_drawn_with_replacement": int((n > controls).sum()), "median_n": float(np.median(n[n > 0])),
                                     "largest_n": int(n.max())}
        for encoder in ENCODERS:
            v = control_vectors(encoder, screen, P)
            if not np.isfinite(v).all():
                raise Stop("%s, %s: a control cell's vector is not finite" % (encoder, screen))
            mean = v.mean(axis=0)
            null = np.full((constructs.size, v.shape[1]), np.nan)
            for t, pick in enumerate(picks):
                if pick is not None:
                    null[t] = v[pick].mean(axis=0) - mean
            if np.isnan(null[n > 0]).any():
                raise Stop("%s, %s: a null feature is NaN" % (encoder, screen))
            arrays[f"{encoder}__{screen}__null"] = null.astype(np.float32)
            record["encoders"].setdefault(encoder, {})[screen] = {"width": int(v.shape[1]), "control_cells": int(v.shape[0])}
            print("[e2 null external] %s %s: %d constructs, width %d, %.0f s" % (encoder, screen, constructs.size, v.shape[1], time.time() - started), flush=True)
    folder = data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "features.npz"
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)
    record.update({"output": file_record(path), "code": code_digests(), "seconds": round(time.time() - started, 1)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "features.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 null external] features written, %.0f s" % (time.time() - started), flush=True)


def load_null() -> dict:
    record = json.loads(FEATURES_JSON.read_text())
    path = config.storage_root() / record["output"]["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["output"]["sha256"]:
        raise Stop("features.npz does not match its record")
    with np.load(path, allow_pickle=False) as store:
        return {k: store[k] for k in store.files}


# --------------------------------------------------------------------------- stage 2: the fits


def saved_n0(mf, encoder: str, cell: str, seed: int) -> dict:
    import make_e1b_stats as e1s

    if "saved" not in DATA:
        DATA["saved"] = [e1s.Predictions(p) for p in sorted(mf.OUT.glob("predictions_manifest_*.json")) if not p.name.endswith("_smoke.json")
                         and not p.name.startswith("predictions_manifest_e6")]
    for source in DATA["saved"]:
        values = source.get(f"E2 | {encoder} | {cell}", LABEL, mf.representation_name(encoder, "N0"), "forest", seed)
        if values is not None:
            return values
    raise Stop("no saved N0 fit of %s, %s, seed %d" % (encoder, cell, seed))


DATA: dict = {}


def fit_worker(task) -> dict:
    import make_e2_fits as mf

    cell, encoder, _, spec = task
    path = mf.predictions_root() / spec.context.replace(" | ", "_") / spec.file_name()
    if path.exists():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == json.dumps(spec.as_dict(), sort_keys=True):
                return {"spec": spec.as_dict(), "file": file_record(path), "reused": True, "vcc_equal": True}
    out, record = mf.fit(cell, encoder, spec, "N0")
    saved = saved_n0(mf, encoder, cell, spec.seed)
    equal = {s: bool(np.array_equal(out[s], saved[s], equal_nan=True)) for s in ("val", "test")}
    if not all(equal.values()):
        raise Stop("%s: the VCC predictions differ from the saved N0 fit's: %s" % (spec.key(), equal))
    record["vcc_predictions_equal_to_the_saved_n0_fit"] = equal
    return {**mf.save(spec, out, record), "reused": False, "vcc_equal": True}


def run_fits() -> None:
    import fit_slots
    import make_e1b_fits as e1f
    import make_e2_fits as mf

    started = time.time()
    root = predictions_dir()
    mf.predictions_root = lambda: root  # this script's predictions apart from E2's
    encoders = ("pca50",) if SMOKE else ENCODERS
    inputs = e1f.load_inputs(list(mf.CELLS))
    order = list(e1cells.CELLS)
    mf.DATA["contexts"] = {cell: e1f.build_context(cell, order.index(cell), {}) for cell in mf.CELLS}
    mf.DATA["screens"] = {s: e1f.DATA["screens"][s].constructs.tolist() for s in e1f.DATA["screens"]}
    mf.DATA["features"] = mf.load_features(("scgpt", "stack", "pca50"))
    null = load_null()
    mf.DATA["sides"] = {}
    for encoder in encoders:
        for cell in CELLS:
            sides = mf.encoder_sides(cell, encoder, null=True)
            for screen in e2cells.SCREENS:
                index = {c: i for i, c in enumerate(null[f"{screen}__constructs"].tolist())}
                own = mf.DATA["screens"][screen]
                missing = [c for c in own if c not in index]
                if missing:
                    raise Stop("%s: %d constructs of the context have no null feature" % (screen, len(missing)))
                z = null[f"{encoder}__{screen}__null"][[index[c] for c in own]]
                if z.shape[1] != sides[screen].z.shape[1]:
                    raise Stop("%s, %s: null width %d, side width %d" % (encoder, screen, z.shape[1], sides[screen].z.shape[1]))
                sides[screen] = dataclasses.replace(sides[screen], z=np.ascontiguousarray(z, dtype=np.float32))
            mf.DATA["sides"][(cell, encoder, True)] = sides
    listed = [(cell, encoder, "N0", FitSpec(f"E2 | {encoder} | {cell}", LABEL, name_of(encoder), "forest", seed))
              for encoder in sorted(encoders, key=lambda e: -mf.ENCODERS.get(e, 768)) for cell in CELLS for seed in SEEDS]
    print("[e2 null external] %d fits, inputs loaded, %.0f s" % (len(listed), time.time() - started), flush=True)
    folder = results_dir()
    folder.mkdir(parents=True, exist_ok=True)
    protocol = {"plan": plan(), "fits": len(listed), "encoders": list(encoders), "label": LABEL, "head": "forest", "seeds": list(SEEDS),
                "heads": {"forest": mf.heads.FOREST}, "inputs": {**inputs, "null_features": file_record(FEATURES_JSON),
                                                                  "e2_features": file_record(REPO / "bioinf2027" / "results" / "E2" / "features.json")},
                "check": "each fit's VCC validation and test predictions equal those of the saved N0 fit of the same specification, bit for bit",
                "runner": "one process per fit (fit_slots.py)", "code": code_digests(), "smoke": SMOKE}
    (folder / "protocol.json").write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    killed = WORKERS_FILE.with_name(WORKERS_FILE.name + ".killed") if WORKERS_FILE is not None else None
    done, failed, shed = fit_slots.run_slots(listed, fit_worker, lambda t: t[3].key(), lambda: fit_slots.limit_from(WORKERS_FILE, WORKERS),
                                             "e2 null external", started, killed=killed)
    manifest = {"protocol": file_record(folder / "protocol.json"), "fits": len(listed), "completed": len(done), "failed": failed,
                "vcc_predictions_equal_to_the_saved_n0_fits": all(r.get("vcc_equal") for r in done) and not failed,
                "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done}, "fits_shed_and_queued_again": shed,
                "seconds": round(time.time() - started, 1)}
    (folder / "predictions_manifest.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[e2 null external] %d completed, %d failed, %.0f s" % (len(done), len(failed), time.time() - started), flush=True)
    if failed:
        raise SystemExit(1)


# --------------------------------------------------------------------------- stage 3: the statistics


READING = {  # (state of T_N0, state of e - N0x) -> reading (plan note, Section 5)
    ("below zero", "contains zero"): "the negative T is reproduced by a feature that carries only the construct's cell count",
    ("below zero", "below zero"): "a feature of cell count alone falls below C1, and e falls further below it: cell count does not account for all of T",
    ("below zero", "above zero"): "a feature of cell count alone falls below C1 by more than e does",
    ("contains zero", "below zero"): "not reproduced: cell count alone does not give a negative T, and e falls below its size-matched null",
    ("above zero", "below zero"): "not reproduced: cell count alone does not give a negative T, and e falls below its size-matched null",
    ("contains zero", "contains zero"): "inconclusive: neither the null's margin nor e's distance from it is resolved",
    ("above zero", "contains zero"): "not reproduced: cell count alone gives no negative T",
    ("above zero", "above zero"): "not reproduced: cell count alone gives no negative T",
    ("contains zero", "above zero"): "not reproduced, and e exceeds its null",
}

READING_EXISTING = {  # state of T_N0 on the existing label -> reading (bioinf2027/plans/E2_null_external_followup.md, Section 3)
    "below zero": "with a head trained on the all-cells label, a feature that carries only the construct's cell count ranks the external "
                  "constructs below C1",
    "contains zero": "with a head trained on the all-cells label, a feature that carries only the construct's cell count gives no external "
                     "ranking below C1",
    "above zero": "with a head trained on the all-cells label, a feature that carries only the construct's cell count ranks the external "
                  "constructs above C1",
}


def vcc_margin(vcc: pd.DataFrame, encoder: str, cell: str) -> str:
    """The null's in-distribution margin on LABEL in the cell (vcc.csv, forest, N0 - C1), with its line, as a qualifier."""
    rows = vcc[(vcc.encoder == encoder) & (vcc.cell == cell) & (vcc.label == LABEL) & (vcc["head"] == "forest") & (vcc.quantity == "N0 - C1")]
    if len(rows) != 1:
        raise Stop("%d rows of vcc.csv for the null margin of %s, %s, %s" % (len(rows), encoder, cell, LABEL))
    r = rows.iloc[0]
    line = int(rows.index[0]) + 2
    shown = "shown" if r["r2_lo"] > 0 else "not shown"
    return "the null's in-distribution margin over C1 on the %s label in %s is %s: %+.3f [%+.3f, %+.3f] (bioinf2027/results/E2/stats/vcc.csv:%d)" % (
        LABEL, cell, shown, r["r2"], r["r2_lo"], r["r2_hi"], line)


def run_stats() -> None:
    import make_e2_stats as st

    e1s, att = st.e1s, st.att
    started = time.time()
    e1s.WORKERS = WORKERS
    st.DATA["missing"] = []
    st.DATA["order_marker"] = st.order_marker()
    predictions = st.load_predictions()
    inputs = st.load_cells()
    manifest_path = outputs_dir() / "predictions_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["failed"] or manifest["completed"] != manifest["fits"] or not manifest["vcc_predictions_equal_to_the_saved_n0_fits"]:
        raise Stop("the fits of (d) are not complete or their check did not pass")
    lookup = st.DATA["lookup"]
    names = {**st.e2_names(), **{name_of(e): e for e in ENCODERS}}
    st.DATA["lookup"] = st.Lookup(lookup.e1b, lookup.e2 + [e1s.Predictions(manifest_path)], names)
    e1s.DATA["predictions"] = st.DATA["lookup"]
    scorer = e1s.Scorer()
    e1s.STATE["scorer"] = scorer
    wanted = []
    for encoder in ENCODERS:
        for cell in CELLS:
            for screen in st.SCREENS:
                for representation in (st.roles(encoder)["e"], st.roles(encoder)["C1"], name_of(encoder)):
                    endpoint = (e1s.DATA["cells"][cell].external_source[screen], "y")
                    e1s.request_object(scorer, screen, endpoint, "all", cell, LABEL, representation, "forest", e1s.ALL_ROWS, 0,
                                       st.wanted_seeds(representation, "forest"))
                    wanted.append((encoder, cell, screen, representation))
    scorer.run()
    print("[e2 null external] scored %d objects, %.0f s" % (len(wanted), time.time() - started), flush=True)
    external = pd.read_csv(REPO / "bioinf2027" / "results" / "E2" / "stats" / "external.csv", float_precision="round_trip")
    family = pd.read_csv(REPO / "bioinf2027" / "results" / "E2" / "stats" / "family.csv", float_precision="round_trip")
    vcc = pd.read_csv(REPO / "bioinf2027" / "results" / "E2" / "stats" / "vcc.csv", float_precision="round_trip")
    rows, readings, compared, largest, failed = [], [], 0, 0.0, []
    for encoder in ENCODERS:
        r = st.roles(encoder)
        for cell in CELLS:
            for screen in st.SCREENS:
                rho = {k: st.rho_ext(cell, screen, LABEL, n) for k, n in (("e", r["e"]), ("C1", r["C1"]), ("N0x", name_of(encoder)))}
                if any(v is None for v in rho.values()):
                    raise Stop("%s, %s, %s: a planned fit is missing" % (encoder, cell, screen))
                values = {f"rho({r['e']})": rho["e"], f"rho({r['C1']})": rho["C1"], f"rho({name_of(encoder)})": rho["N0x"],
                          "T": st.minus(rho["e"], rho["C1"]), "T_N0": st.minus(rho["N0x"], rho["C1"]), "e - N0x": st.minus(rho["e"], rho["N0x"])}
                summaries = {q: st.summary(v) for q, v in values.items()}
                for quantity in ("T", f"rho({r['e']})", f"rho({r['C1']})"):
                    part = external[(external.encoder == encoder) & (external.cell == cell) & (external.screen == screen) & (external.label == LABEL)
                                    & (external["head"] == "forest") & (external.quantity == quantity)]
                    if len(part) != 1:
                        raise Stop("%d saved rows of external.csv for %s, %s, %s, %s" % (len(part), encoder, cell, screen, quantity))
                    for k in ("estimate", "lo", "hi"):
                        mine, theirs = float(summaries[quantity][k]), float(part.iloc[0][k])
                        compared += 1
                        largest = max(largest, abs(mine - theirs))
                        if abs(mine - theirs) > TOLERANCE:
                            failed.append({"where": (encoder, cell, screen, quantity, k), "mine": mine, "saved": theirs})
                for quantity, s in summaries.items():
                    rows.append({"encoder": encoder, "width": st.width(encoder), "cell": cell, "screen": screen, "label": LABEL, "head": "forest",
                                 "quantity": quantity, **s, "state": att.state(s["lo"], s["hi"]) if not quantity.startswith("rho(") else "",
                                 "role": status(), "markers": st.markers(encoder, "Replogle")})
                holm = family[(family.encoder == encoder) & (family.cell == cell) & (family.screen == screen) & (family.statistic == "T")]
                if LABEL == "rebuilt":
                    state_t = holm.iloc[0]["state"] if len(holm) == 1 else "not in E2's family (Geneformer: E1b's z)"
                else:
                    state_t = "not in E2's family (the existing label)"
                key = (att.state(summaries["T_N0"]["lo"], summaries["T_N0"]["hi"]), att.state(summaries["e - N0x"]["lo"], summaries["e - N0x"]["hi"]))
                reading = READING.get(key, "undefined interval") if LABEL == "rebuilt" else READING_EXISTING.get(key[0], "undefined interval")
                qualifiers = POST_HOC + "; the precondition: T of (ii) is not read for claims (family.csv, why_not_read)" if LABEL == "rebuilt" else (
                    status() + "; " + vcc_margin(vcc, encoder, cell) + "; no statement on why the external values are negative (gate G7, item 6)")
                readings.append({"encoder": encoder, "cell": cell, "screen": screen, "T": summaries["T"]["estimate"], "T_state_in_E2_family": state_t,
                                 "T_N0": summaries["T_N0"]["estimate"], "T_N0_lo": summaries["T_N0"]["lo"], "T_N0_hi": summaries["T_N0"]["hi"],
                                 "e_minus_N0x": summaries["e - N0x"]["estimate"], "e_minus_N0x_lo": summaries["e - N0x"]["lo"],
                                 "e_minus_N0x_hi": summaries["e - N0x"]["hi"], "reading": reading,
                                 "qualifiers": qualifiers + ("; " + st.markers(encoder, "Replogle") if st.markers(encoder, "Replogle") else "")})
    if failed or st.DATA["missing"]:
        raise Stop("a check failed: %s" % json.dumps({"reproduction": failed[:10], "missing": st.DATA["missing"]}))
    folder = outputs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / "null_external.csv", index=False)
    pd.DataFrame(readings).to_csv(folder / "readings.csv", index=False)
    record = {"plan": plan(), "status": status(), "settings": {"replicates": e1s.B, "external_seeds": e1s.EXT_SEED, "label": LABEL, "head": "forest",
                                                            "head_seeds": list(SEEDS), "null_seed": SEED},
              "check_reproduction_of_external_csv": {"values_compared": compared, "largest_difference": largest, "tolerance": TOLERANCE},
              "external_replicate_index_sha256": {s: e1s.STATE["draws"][s]["draw"].index_sha256 for s in st.SCREENS},
              "inputs": {**inputs, "predictions": predictions, "null_predictions": file_record(manifest_path),
                         "null_features": file_record(FEATURES_JSON),
                         "external_csv": file_record(REPO / "bioinf2027" / "results" / "E2" / "stats" / "external.csv"),
                         "family_csv": file_record(REPO / "bioinf2027" / "results" / "E2" / "stats" / "family.csv")},
              "code": code_digests(), "outputs": {n: file_record(folder / n) for n in ("null_external.csv", "readings.csv")},
              "seconds": round(time.time() - started, 1)}
    (folder / "stats.json").write_text(json.dumps(strict_json(record), indent=1, allow_nan=False) + "\n")
    print("[e2 null external] statistics written, %d saved values reproduced (largest difference %.2e), %.0f s"
          % (compared, largest, time.time() - started), flush=True)


def code_digests() -> dict:
    roots = [REPO / "bioinf2027" / "scripts", REPO / "bioinf2027" / "src"]
    out = {}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and any(Path(path).resolve().is_relative_to(r) for r in roots):
            out[config.portable(Path(path).resolve())] = sha256(Path(path))
    return dict(sorted(out.items()))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--smoke" in arguments:
        SMOKE = True
        arguments.remove("--smoke")
    for flag in ("--workers", "--workers-file", "--label"):
        if flag in arguments:
            i = arguments.index(flag)
            if flag == "--workers":
                WORKERS = int(arguments[i + 1])
            elif flag == "--label":
                LABEL = arguments[i + 1]
                if LABEL not in LABELS:
                    raise SystemExit("--label must be one of %s" % ", ".join(LABELS))
            else:
                WORKERS_FILE = config.storage_root() / arguments[i + 1]
            arguments = arguments[:i] + arguments[i + 2:]
    stages = {"--features": run_features, "--fits": run_fits, "--stats": run_stats}
    if len(arguments) != 1 or arguments[0] not in stages or (SMOKE and arguments[0] != "--fits") or (LABEL != "rebuilt" and arguments[0] == "--features"):
        raise SystemExit("usage: make_e2_null_external.py {--features | --fits [--smoke] [--workers N] [--workers-file PATH] [--label L] | --stats [--workers N] [--label L]}")
    stages[arguments[0]]()
