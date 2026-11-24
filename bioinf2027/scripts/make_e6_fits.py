"""E6, every head fit: transfer between the two Replogle screens with gene-disjoint folds.

Design: bioinf2027/plans/E6.md, Sections 3, 4 and 6 as amended by Sections 13 and 14, with the details of
Section 15. Inputs: the E1a external tables of G1, G2, G5 and G6 (labels and B0 to B4, refused unless they
match their manifests), the external feature inputs of bioinf2027/src/bioinf2027/e1inputs.py (the stored
z, the full-pass delta), and, for the harness check, E1b's saved predictions. No statistic is computed
here; bioinf2027/scripts/make_e6_stats.py reads the saved predictions.

Fit sets: for each direction (source screen S, target screen T), each endpoint and each of five folds of
target genes (the union of both screens' eligible target genes in sorted Ensembl order, permuted with
seed 20271204, fold = position mod 5), the models are fitted on the S constructs whose gene is outside
the fold and predict the T constructs whose gene is in it; the whole-screen fit uses every S construct.

Tiers (bioinf2027/plans/E6.md, Section 16; bioinf2027/scripts/fit_tiers.py): --tier N runs every listed
fit up to tier N (R(z-), in tier 1, once D2's arms exist). Each run writes protocol_tier<N>.json and
predictions_manifest_tier<N>.json; before fitting, it compares its inputs, heads, seeds, refits, the
seeds of the 700-control endpoints and the fitting code with every earlier tier's protocol and refuses
to run (and so to reuse an earlier tier's predictions) if any differs.

Options: --check runs the harness check (stop rule 4) and the timing probe and exits; --plan prints the
task counts per tier; --tier N (required for a fit run); --workers N (default 12; changes no prediction); --smoke fits a small subset with 5 trees into
the scratch folder, never reported, with --tier N as well.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioinf2027 import config, e1cells, e1inputs, features, heads, representations as rep, transfer  # noqa: E402
from bioinf2027.provenance import file_record, sha256, strict_json  # noqa: E402
from bioinf2027.transfer import FitSpec  # noqa: E402

import fit_tiers  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "bioinf2027" / "results" / "E6"
PLAN = "bioinf2027/plans/E6.md, Sections 3, 4 and 6 as amended by Sections 13 to 15"
SCREENS = e1inputs.SCREENS
WORKERS = 12
SEEDS = (0, 1, 2)
FOLD_SEED = 20271204
FOLD_SENSITIVITY = (20271207, 20271208)
SIZE_SEED = 20271205
REFIT_SEED = 20271206
RIDGE_INNER_SEED = 20271201
DRAWS = 20
REFITS = 50
REFITS_CUT = 20
REFIT_LIMIT_HOURS = 48.0
CPU_LIMIT_HOURS = 200.0
ENDPOINTS = {"27/788": "G1", "10/788": "G2", "27/700": "G5", "10/700": "G6"}
PRIMARY = "27/788"
MATCHED_GENES = {"27/788": 98, "10/788": 140}
LABELS = {"rebuilt": "y", "mean": "y_mean"}
R = transfer
MODELS = R.FAMILY + R.MASKED + R.CONTROLS + R.PIPELINE + R.TRIVIAL_MODELS + ("B1",)
FAMILY_REPEAT = R.FAMILY + R.MASKED + ("C1",)
REFIT_MODELS = R.FAMILY + ("C1",)
SIZE_MODELS = R.FAMILY + ("C1", "TR")
SMOKE = False

DATA: dict = {}


def predictions_root() -> Path:
    if SMOKE:
        return config.storage_root() / "bioinf2027" / "scratch" / "E6_fits_smoke" / "predictions"
    return config.storage_root() / "bioinf2027" / "E6" / "predictions"


# --------------------------------------------------------------------------- inputs


def load_inputs() -> dict:
    DATA["tables"] = {}
    for name, cell in ENDPOINTS.items():
        tables = e1cells.load_tables(cell)
        DATA["tables"][name] = {s: tables[s] for s in SCREENS}
    base = DATA["tables"][PRIMARY]
    symbols = features.hvg_symbols()
    position = {s: i for i, s in enumerate(symbols.tolist())}
    DATA["screens"] = {}
    present = {}
    for screen in SCREENS:
        inputs = e1inputs.load_screen_inputs(screen, base[screen])
        for name in ENDPOINTS:
            if not np.array_equal(DATA["tables"][name][screen]["construct"].astype(str).to_numpy(), inputs.constructs):
                raise RuntimeError("the %s tables list different %s constructs" % (name, screen))
        native_symbols = set(e1inputs.native_symbols(screen).tolist())
        present[screen] = np.array([s in native_symbols for s in symbols.tolist()])
        DATA["screens"][screen] = inputs
    both = present["RPE1"] & present["K562-essential"]
    DATA["features_mapped_in_both"] = int(both.sum())
    for screen in SCREENS:
        inputs = DATA["screens"][screen]
        primary = inputs.x["full_pass"].copy()
        primary[:, ~both] = 0.0
        target = np.array([position.get(s, -1) for s in base[screen]["target_symbol"].astype(str)], dtype=np.int64)
        target_primary = np.where((target >= 0) & both[np.maximum(target, 0)], target, -1)
        DATA[("x", screen)] = {"primary": (primary, target_primary), "e1b": (inputs.x["full_pass"], target)}
    shared, ensembl = shared_gene_inputs()
    for screen in SCREENS:
        DATA[("x", screen)]["shared genes"] = shared[screen]
    DATA["shared_genes"] = ensembl
    DATA["e4"] = load_e4_features(both)
    genes = sorted(set(base["RPE1"]["target_ensg"].astype(str)) | set(base["K562-essential"]["target_ensg"].astype(str)))
    DATA["genes"] = np.array(genes)
    DATA["folds"] = {seed: fold_of(genes, seed) for seed in (FOLD_SEED,) + FOLD_SENSITIVITY}
    return {"features_mapped_in_both_screens": DATA["features_mapped_in_both"], "e4_features": DATA["e4"].get("record"), "shared_genes": int(len(ensembl)),
            "target_genes_in_the_union": len(genes),
            "endpoint_manifests": {c: file_record(REPO / "bioinf2027" / "endpoints" / c / "manifest.json") for c in sorted(set(ENDPOINTS.values()))}}


def load_e4_features(both: np.ndarray) -> dict:
    """G1's external features from E4's builder (bioinf2027/results/E4/features.json), if they exist and passed their checks."""
    record_path = REPO / "bioinf2027" / "results" / "E4" / "features.json"
    if not record_path.exists():
        return {}
    record = json.loads(record_path.read_text())
    if not record.get("checks_passed"):
        return {}
    out = {}
    for variant in ("draw 1", "draw 2", "draw 3", "cell-disjoint"):
        entry = record["features"][f"G1 | {variant}"]
        path = config.storage_root() / entry["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
        if sha256(path) != entry["sha256"]:
            raise RuntimeError("the E4 feature file of G1 %s differs from its record" % variant)
        with np.load(path, allow_pickle=False) as store:
            for screen in SCREENS:
                if not np.array_equal(store[f"{screen}_constructs"], DATA["screens"][screen].constructs):
                    raise RuntimeError("E4's %s constructs are in another order" % screen)
                x = store[f"{screen}_x"].copy()
                x[:, ~both] = 0.0
                out[(screen, variant)] = (x, store[f"{screen}_z"], store[f"{screen}_kept"])
    out["record"] = file_record(record_path)
    return out


def fold_of(genes: list, seed: int) -> dict:
    order = np.random.default_rng(seed).permutation(len(genes))
    return {genes[i]: position % 5 for position, i in enumerate(order)}


def shared_gene_inputs() -> tuple[dict, list]:
    """The native delta over every gene both screens measure, matched by Ensembl identifier, sorted; clipped to +/-10."""
    import h5py

    from bioinf2027 import singlecell

    ids = {}
    for screen in SCREENS:
        with h5py.File(e1inputs.paths()[f"{screen}_single_cell"], "r") as handle:
            var = handle["var"]
            index = var.attrs["_index"]
            ids[screen] = singlecell.read_annotation(var, index.decode() if isinstance(index, bytes) else str(index))
    shared = sorted(set(ids["RPE1"].tolist()) & set(ids["K562-essential"].tolist()))
    out = {}
    base = DATA["tables"][PRIMARY]
    for screen in SCREENS:
        column = {g: i for i, g in enumerate(ids[screen].tolist())}
        native = DATA["screens"][screen].x["native"]
        x = native[:, [column[g] for g in shared]]
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(np.float32)
        position = {g: i for i, g in enumerate(shared)}
        target = np.array([position.get(g, -1) for g in base[screen]["target_ensg"].astype(str)], dtype=np.int64)
        out[screen] = (x, target)
    return out, shared


# --------------------------------------------------------------------------- contexts


def side(screen: str, rows: np.ndarray, endpoint: str, basis: str, name: str, z_override=None) -> transfer.Side:
    x, target = DATA[("x", screen)][basis]
    table = DATA["tables"][endpoint][screen]
    z_full = DATA["screens"][screen].z if z_override is None else z_override
    return transfer.Side(
        name=name,
        control_split=screen,
        z=z_full[rows],
        x={"main": x[rows]},
        b=table[list(e1cells.TRIVIAL)].to_numpy(dtype=np.float64)[rows],
        target_column=target[rows],
        control_rows=np.asarray(rows, dtype=np.int64),
        z_full=z_full,
    )


def context_name(source: str, endpoint: str, fit_set: str, basis: str = "primary", assignment: int = FOLD_SEED, draw: int | None = None,
                 matched: str | None = None) -> str:
    parts = ["E6", f"{source}>{other(source)}", endpoint, fit_set, basis]
    if assignment != FOLD_SEED:
        parts.append(f"folds {assignment}")
    if draw is not None:
        parts.append(f"size-matched draw {draw}")
    if matched is not None:
        parts.append(f"E4 features {matched}")
    return " | ".join(parts)


def stable_index(name: str) -> int:
    """A fixed index per context for the refit generator, independent of the order of the tasks."""
    import hashlib

    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


def other(screen: str) -> str:
    return SCREENS[1] if screen == SCREENS[0] else SCREENS[0]


def matched_side(screen: str, rows: np.ndarray, variant: str, name: str) -> transfer.Side:
    """A side whose x and z come from E4's builder for G1 (Section 6, items 10 and 11)."""
    x, z, _ = DATA["e4"][(screen, variant)]
    _, target = DATA[("x", screen)]["primary"]
    table = DATA["tables"][PRIMARY][screen]
    return transfer.Side(name=name, control_split=screen, z=z[rows], x={"main": x[rows]},
                         b=table[list(e1cells.TRIVIAL)].to_numpy(dtype=np.float64)[rows], target_column=target[rows],
                         control_rows=np.asarray(rows, dtype=np.int64), z_full=z)


def build_context(source: str, endpoint: str, fit_set: str, basis: str = "primary", assignment: int = FOLD_SEED, draw: int | None = None,
                  index: int = 0, z_override: dict | None = None, matched: str | None = None) -> transfer.Context:
    target = other(source)
    folds = DATA["folds"][assignment]
    s_genes = DATA["tables"][endpoint][source]["target_ensg"].astype(str).to_numpy()
    t_genes = DATA["tables"][endpoint][target]["target_ensg"].astype(str).to_numpy()
    if fit_set == "whole screen":
        train_rows = np.arange(s_genes.size)
        test_rows = np.arange(t_genes.size)
    else:
        k = int(fit_set.split()[-1])
        train_rows = np.flatnonzero(np.array([folds[g] != k for g in s_genes]))
        test_rows = np.flatnonzero(np.array([folds[g] == k for g in t_genes]))
        if draw is not None:
            k_genes = np.array(sorted(set(s_genes[train_rows].tolist())))
            rng = np.random.default_rng([SIZE_SEED, draw, k])
            chosen = set(k_genes[rng.choice(k_genes.size, MATCHED_GENES[endpoint], replace=False)].tolist())
            train_rows = train_rows[np.isin(s_genes[train_rows], list(chosen))]
    zs = None if z_override is None else z_override[source]
    zt = None if z_override is None else z_override[target]
    if matched is not None:
        train = matched_side(source, train_rows, matched, "train")
        sides = {target: matched_side(target, test_rows, matched, target)}
    else:
        train = side(source, train_rows, endpoint, basis, "train", zs)
        sides = {target: side(target, test_rows, endpoint, basis, target, zt)}
        primary_fold = endpoint == PRIMARY and basis == "primary" and assignment == FOLD_SEED and draw is None and fit_set != "whole screen"
        if primary_fold and (target, "cell-disjoint") in DATA["e4"] and z_override is None:
            kept = DATA["e4"][(target, "cell-disjoint")][2]
            sides[f"{target} cell-disjoint"] = matched_side(target, test_rows[kept[test_rows]], "cell-disjoint", f"{target} cell-disjoint")
    labels = {}
    for name, column in LABELS.items():
        labels[name] = DATA["tables"][endpoint][source][column].to_numpy(dtype=np.float64)[train_rows]
    return transfer.Context(
        name=context_name(source, endpoint, fit_set, basis, assignment, draw),
        train=train,
        labels=labels,
        sides=sides,
        meta={"genes": s_genes[train_rows], "context_index": index, "n_guides": np.ones(train_rows.size), "test_rows": test_rows,
              "train_rows": train_rows},
    )


# --------------------------------------------------------------------------- ridge with gene-grouped inner folds


def ridge_cv_fit(context: transfer.Context, spec: FitSpec, folder: Path) -> dict:
    """Ridge on standardised inputs, penalty by R-squared in five inner folds grouped by target gene (seed 20271201)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    path = folder / spec.file_name()
    spec_text = json.dumps(spec.as_dict(), sort_keys=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as store:
            if str(store["spec"]) == spec_text:
                return {"spec": spec.as_dict(), "file": file_record(path), "reused": True}
    rows = transfer.training_rows(context, spec, REFIT_SEED)
    y = context.labels[spec.label][rows]
    blocks = transfer.Blocks(spec, context.train, rows)
    x = blocks.matrix(context.train)[rows].astype(np.float64)
    genes = context.meta["genes"][rows]
    unique = np.array(sorted(set(genes.tolist())))
    inner = dict(zip(unique[np.random.default_rng(RIDGE_INNER_SEED).permutation(unique.size)], np.arange(unique.size) % 5))
    folds = np.array([inner[g] for g in genes])
    scores = {}
    with heads.single_thread():
        for alpha in heads.RIDGE_GRID:
            sse = sst = 0.0
            for f in range(5):
                test = folds == f
                scaler = StandardScaler().fit(x[~test])
                model = Ridge(alpha=alpha).fit(scaler.transform(x[~test]), y[~test])
                pred = model.predict(scaler.transform(x[test]))
                sse += float(((y[test] - pred) ** 2).sum())
                sst += float(((y[test] - y[test].mean()) ** 2).sum())
            scores[float(alpha)] = 1.0 - sse / sst if sst > 0 else float("nan")
        best = max([a for a in scores if np.isfinite(scores[a])], key=lambda a: (scores[a], -a))
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=best).fit(scaler.transform(x), y)
        out = {}
        for name, s in context.sides.items():
            out[name] = model.predict(scaler.transform(blocks.matrix(s).astype(np.float64)))
    record = {"ridge": {"inner_cv_r2": scores, "chosen": best, "at_grid_edge": best in (min(heads.RIDGE_GRID), max(heads.RIDGE_GRID))},
              "training_rows": int(rows.size)}
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, spec=np.array(spec_text), record=np.array(json.dumps(record, sort_keys=True)), **out)
    os.replace(temporary, path)
    return {"spec": spec.as_dict(), "file": file_record(path), "reused": False}


# --------------------------------------------------------------------------- tasks


def task_list(refits: int, seeds_700: tuple) -> list[tuple[dict, FitSpec]]:
    """(context description, fit specification) for every fit."""
    out = []

    def add(description, label, models, heads_seeds, variant="all training rows"):
        name = context_name(**description)
        for r in models:
            for head, seeds in heads_seeds:
                for seed in seeds:
                    out.append((description, FitSpec(name, label, r, head, seed, variant)))

    fit_sets = [f"fold {k}" for k in range(5)] + ["whole screen"]
    for source in SCREENS:
        for endpoint in ENDPOINTS:
            seeds = SEEDS if endpoint in ("27/788", "10/788") else seeds_700
            for fit_set in fit_sets:
                d = {"source": source, "endpoint": endpoint, "fit_set": fit_set}
                add(d, "rebuilt", MODELS, (("forest", seeds), ("boosting", seeds), ("ridge-cv", (0,))))
                add(d, "rebuilt", R.POSITIVE, (("monotone", (0,)),))
                if endpoint == PRIMARY and fit_set != "whole screen":
                    add(d, "mean", FAMILY_REPEAT, (("forest", SEEDS),))
                    for r in range(refits):
                        add(d, "rebuilt", REFIT_MODELS, (("forest", (0,)),), variant=f"refit {r}")
            if endpoint == PRIMARY:
                for basis in ("e1b", "shared genes"):
                    for k in range(5):
                        d = {"source": source, "endpoint": endpoint, "fit_set": f"fold {k}", "basis": basis}
                        models = ("x", "m", "[x; m]", "x-", "m-", "[x; m]-") + (("C2",) if basis == "e1b" else ())
                        add(d, "rebuilt", models, (("forest", SEEDS),))
                for assignment in FOLD_SENSITIVITY:
                    for k in range(5):
                        d = {"source": source, "endpoint": endpoint, "fit_set": f"fold {k}", "assignment": assignment}
                        add(d, "rebuilt", REFIT_MODELS, (("forest", (0,)),))
            if endpoint == PRIMARY and DATA.get("e4"):
                for variant in ("draw 1", "draw 2", "draw 3"):
                    for k in range(5):
                        d = {"source": source, "endpoint": endpoint, "fit_set": f"fold {k}", "matched": variant}
                        add(d, "rebuilt", REFIT_MODELS, (("forest", (0,)),))
            if endpoint in MATCHED_GENES:
                for k in range(5):
                    for draw in range(DRAWS):
                        d = {"source": source, "endpoint": endpoint, "fit_set": f"fold {k}", "draw": draw}
                        add(d, "rebuilt", SIZE_MODELS, (("forest", SEEDS),))
    keys = [s.key() for _, s in out]
    if len(keys) != len(set(keys)):
        raise RuntimeError("a fit is listed twice")
    return out


def z_minus_tasks() -> list[tuple[dict, FitSpec]]:
    """R(z-) of the family (Section 4.1): z- of D2 in place of the stored z, once D2's arms exist."""
    record = REPO / "bioinf2027" / "results" / "D2" / "aggregate.json"
    if not record.exists():
        return []
    out = []
    for source in SCREENS:
        for k in range(5):
            d = {"source": source, "endpoint": PRIMARY, "fit_set": f"fold {k}", "z_minus": True}
            name = context_name(source, PRIMARY, f"fold {k}") + " | z- of D2"
            for seed in SEEDS:
                out.append((d, FitSpec(name, "rebuilt", "z", "forest", seed)))
    return out


def d2_arms() -> dict:
    record = json.loads((REPO / "bioinf2027" / "results" / "D2" / "aggregate.json").read_text())
    path = config.storage_root() / record["arms"]["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(path) != record["arms"]["sha256"]:
        raise RuntimeError("the D2 arms differ from their record")
    with np.load(path, allow_pickle=False) as store:
        out = {}
        for screen in SCREENS:
            if not np.array_equal(store[f"{screen}_constructs"], DATA["screens"][screen].constructs):
                raise RuntimeError("the D2 arms list the %s constructs in another order" % screen)
            out[screen] = store[f"{screen}_z_minus"]
    return out


def shared_c2_tasks() -> list[tuple[dict, FitSpec]]:
    """C2 of the shared-gene basis: the shared x through a projection of its own size, fitted as a z block."""
    out = []
    for source in SCREENS:
        for k in range(5):
            d = {"source": source, "endpoint": PRIMARY, "fit_set": f"fold {k}", "basis": "shared genes", "c2": True}
            name = context_name(source, PRIMARY, f"fold {k}", "shared genes") + " | C2 as z"
            for seed in SEEDS:
                out.append((d, FitSpec(name, "rebuilt", "z", "forest", seed)))
    return out


# --------------------------------------------------------------------------- workers


def context_for(description: dict, index: int) -> transfer.Context:
    key = json.dumps(description, sort_keys=True)
    if key not in DATA["contexts"]:
        d = dict(description)
        c2 = d.pop("c2", False)
        z_minus = d.pop("z_minus", False)
        override = None
        if z_minus:
            override = d2_arms()
        if c2:
            override = {}
            for screen in SCREENS:
                x, _ = DATA[("x", screen)]["shared genes"]
                override[screen] = rep.c2_features(x, rep.c2_projection(x_dim=x.shape[1]))
        context = build_context(index=stable_index(context_name(**{k: v for k, v in d.items()})), z_override=override, **d)
        if c2:
            context.name = context.name + " | C2 as z"
        if z_minus:
            context.name = context.name + " | z- of D2"
        DATA["contexts"][key] = context
    return DATA["contexts"][key]


def worker(description: dict, spec: FitSpec, index: int) -> dict:
    context = context_for(description, index)
    folder = predictions_root() / spec.file_name()[:2]
    if spec.head == "ridge-cv":
        return ridge_cv_fit(context, spec, folder)
    return transfer.run_fit(context, spec, folder, REFIT_SEED, {})


def cost(item) -> tuple:
    description, spec = item
    columns = {"z": 768, "x": 2000, "[x; m]": 2004, "x-": 2000, "[x; m]-": 2004, "C1": 768, "C2": 768, "C3": 768,
               "[C1; B2]": 769, "[z; B2]": 769, "[TR; z]": 771, "[TR; x]": 2003, "[TR; [x; m]]": 2007}.get(spec.representation, 5)
    weight = {"forest": 1.0, "boosting": 0.05, "ridge-cv": 0.1, "monotone": 0.001}[spec.head]
    rows = 300 if "draw" in description else 1500
    primary = description["endpoint"] == PRIMARY and spec.variant == "all training rows" and "draw" not in description
    return (0 if primary else 1, -rows * columns * weight)


# --------------------------------------------------------------------------- harness check and timing


def harness_check() -> dict:
    """Stop rule 4: the E6 harness reproduces E1b's saved RPE1 predictions of one fit to 1e-12."""
    import make_e1b_fits as e1b

    manifest = json.loads((REPO / "bioinf2027" / "results" / "E1" / "predictions_manifest.json").read_text())
    spec = FitSpec("G1", "rebuilt", "x", "forest", 0)
    record = manifest["files"][spec.key()]
    saved = config.storage_root() / record["path"].split("$IGA_STORAGE_ROOT/", 1)[1]
    if sha256(saved) != record["sha256"]:
        raise RuntimeError("E1b's saved prediction differs from its manifest")
    e1b.load_inputs(["G1", "G2"])
    context = e1b.build_context("G1", list(e1cells.CELLS).index("G1"), {})
    folder = config.storage_root() / "bioinf2027" / "scratch" / "E6_harness_check"
    target = folder / spec.file_name()
    if target.exists():
        target.unlink()
    transfer.run_fit(context, spec, folder, e1b.REFIT_SEED, e1b.EXTERNAL_BASES)
    _, _, ours = transfer.load_predictions(target)
    _, _, theirs = transfer.load_predictions(saved)
    difference = float(np.abs(ours["RPE1"] - theirs["RPE1"]).max())
    return {"fit": spec.as_dict(), "e1b_file": record, "max_abs_difference": difference, "tolerance": 1e-12, "passed": bool(difference <= 1e-12)}


def timing_probe() -> dict:
    """One forest fit per representation of the primary endpoint's first fold, both directions; projects the CPU hours."""
    out = []
    for source in SCREENS:
        context = build_context(source, PRIMARY, "fold 0")
        for r in ("z", "x", "[x; m]", "C1", "m", "TR"):
            spec = FitSpec(context.name, "rebuilt", r, "forest", 0)
            start = time.time()
            rows = transfer.training_rows(context, spec, REFIT_SEED)
            blocks = transfer.Blocks(spec, context.train, rows)
            with heads.single_thread():
                heads.forest(0).fit(blocks.matrix(context.train)[rows], context.labels["rebuilt"][rows])
            out.append({"source": source, "representation": r, "seconds": time.time() - start, "training_rows": int(rows.size)})
    seconds = {(o["source"], o["representation"]): o["seconds"] for o in out}
    per_rep = {r: np.mean([seconds[(s, r)] for s in SCREENS]) for r in ("z", "x", "[x; m]", "C1", "m", "TR")}
    return {"fits": out, "seconds_per_representation": per_rep}


def project(tasks, per_rep: dict) -> float:
    big = {"x": per_rep["x"], "[x; m]": per_rep["[x; m]"], "x-": per_rep["x"], "[x; m]-": per_rep["[x; m]"], "[TR; x]": per_rep["x"],
           "[TR; [x; m]]": per_rep["[x; m]"], "z": per_rep["z"], "C1": per_rep["C1"], "C2": per_rep["C1"], "C3": per_rep["z"],
           "[C1; B2]": per_rep["C1"], "[z; B2]": per_rep["z"], "[TR; z]": per_rep["z"]}
    total = 0.0
    for description, spec in tasks:
        base = big.get(spec.representation, per_rep["m"])
        factor = {"forest": 1.0, "boosting": 0.05, "ridge-cv": 0.1, "monotone": 0.01}[spec.head]
        scale = 0.1 if "draw" in description else 1.0
        total += base * factor * scale
    return total / 3600.0


# --------------------------------------------------------------------------- main


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


def main(mode: str, tier: int | None = None) -> None:
    started = time.time()
    inputs = load_inputs()
    DATA["contexts"] = {}
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "check":
        check = harness_check()
        timing = timing_probe()
        tasks = task_list(REFITS, SEEDS)
        refit_tasks = [t for t in tasks if t[1].variant.startswith("refit ")]
        refit_hours = project(refit_tasks, timing["seconds_per_representation"])
        refits = REFITS if refit_hours <= REFIT_LIMIT_HOURS else REFITS_CUT
        tasks = task_list(refits, SEEDS) + shared_c2_tasks()
        hours = project(tasks, timing["seconds_per_representation"])
        seeds_700 = SEEDS if hours <= CPU_LIMIT_HOURS else (0,)
        if seeds_700 != SEEDS:
            hours = project(task_list(refits, seeds_700) + shared_c2_tasks(), timing["seconds_per_representation"])
        report = {"plan": PLAN, "inputs": inputs, "harness_check": check, "timing_probe": timing,
                  "projected_cpu_hours_for_50_refits": round(refit_hours, 1), "refits": refits,
                  "projected_cpu_hours": round(hours, 1), "seeds_of_the_700_control_endpoints": list(seeds_700),
                  "rules": {"refits": "Section 6, item 9: 50 refits, cut to 20 if they project more than 48 hours of CPU",
                            "seeds": "Section 10: if the run projects more than 200 hours of CPU, the 700-control endpoints run with head seed 0 only"},
                  "seconds": round(time.time() - started, 1)}
        (OUT / "checks.json").write_text(json.dumps(strict_json(report), indent=1, allow_nan=False) + "\n")
        print("[e6] harness %s (largest %.3g); refits %d; seeds of the 700-control endpoints %s; projected %.0f CPU hours" % (
            check["passed"], check["max_abs_difference"], refits, list(seeds_700), hours), flush=True)
        return
    if SMOKE:
        refits, seeds_700 = 1, (0,)
    else:
        checks = json.loads((OUT / "checks.json").read_text())
        if not checks["harness_check"]["passed"]:
            raise SystemExit("the E6 harness check has not passed (bioinf2027/results/E6/checks.json); no fit is run")
        refits, seeds_700 = int(checks["refits"]), tuple(checks["seeds_of_the_700_control_endpoints"])
    listed = task_list(refits, seeds_700) + shared_c2_tasks() + z_minus_tasks()
    if mode == "plan":
        tiers = pd.Series([fit_tiers.e6_tier(d, s) for d, s in listed])
        print("[e6] %d listed fits; per tier: %s" % (len(listed), tiers.value_counts().sort_index().to_dict()), flush=True)
        return
    tasks = [(d, s) for d, s in listed if fit_tiers.e6_tier(d, s) <= tier]
    if SMOKE:
        def small(item):
            description, spec = item
            return (spec.seed == 0 and description["source"] == "K562-essential" and description.get("draw") in (None, 0)
                    and description["fit_set"] in ("fold 0", "whole screen"))
        tasks = [t for t in tasks if small(t)]
    tasks = sorted(tasks, key=cost)
    print("[e6] tier %d: %d fits; %s" % (tier, len(tasks), pd.Series([s.head for _, s in tasks]).value_counts().to_dict()), flush=True)
    protocol = {"plan": PLAN, "fits": len(tasks), "inputs": {**inputs, "checks": None if SMOKE else file_record(OUT / "checks.json"),
                                                           "e4_features": DATA["e4"].get("record")},
                "seeds": {"heads": list(SEEDS), "folds": FOLD_SEED, "fold sensitivity": list(FOLD_SENSITIVITY), "size-matched draws": SIZE_SEED,
                          "refits": REFIT_SEED, "ridge inner folds": RIDGE_INNER_SEED},
                "refits": refits, "seeds_of_the_700_control_endpoints": list(seeds_700), "heads": {"forest": heads.FOREST, "boosting": heads.BOOSTING,
                "ridge grid": list(heads.RIDGE_GRID), "monotone": heads.MONOTONE}, "predictions": config.portable(predictions_root()),
                "tier": tier, "smoke": SMOKE, "tiers": "bioinf2027/plans/E6.md, Section 16; bioinf2027/scripts/fit_tiers.py",
                "fitting_code": fitting_code(),
                "per_tier": {str(k): int(v) for k, v in pd.Series([fit_tiers.e6_tier(d, s) for d, s in tasks]).value_counts().sort_index().items()}}
    if z_minus_tasks():
        protocol["inputs"]["d2_aggregate"] = file_record(REPO / "bioinf2027" / "results" / "D2" / "aggregate.json")
    folder = predictions_root().parent if SMOKE else OUT
    folder.mkdir(parents=True, exist_ok=True)
    protocol["earlier_tiers_compared"] = refuse_if_inputs_differ(folder, protocol)
    protocol_path = folder / f"protocol_tier{tier}.json"
    protocol_path.write_text(json.dumps(strict_json(protocol), indent=1, allow_nan=False) + "\n")
    context = multiprocessing.get_context("fork")
    done, failed = [], []
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context) as pool:
        pending = {pool.submit(worker, d, s, 0): s for d, s in tasks}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = pending.pop(future)
                try:
                    done.append(future.result())
                except Exception as error:  # recorded and reported; the run continues
                    failed.append({"fit": spec.key(), "error": repr(error)})
                    print("[e6] FAILED %s: %r" % (spec.key(), error), flush=True)
                    continue
                if len(done) % 100 == 0 or not pending:
                    print("[e6] %d of %d done, %d failed, %.0f s" % (len(done), len(tasks), len(failed), time.time() - started), flush=True)
    manifest = {"protocol": file_record(protocol_path), "tier": tier, "fits": len(tasks), "completed": len(done), "failed": failed,
                "reused": sum(bool(r.get("reused")) for r in done),
                "files": {FitSpec(**r["spec"]).key(): r["file"] for r in done}, "seconds": round(time.time() - started, 1)}
    (folder / f"predictions_manifest_tier{tier}.json").write_text(json.dumps(strict_json(manifest), indent=1, allow_nan=False) + "\n")
    print("[e6] %d completed, %d failed" % (len(done), len(failed)), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    TIER, arguments = fit_tiers.parse_tier(sys.argv[1:])
    WORKERS, arguments = fit_tiers.parse_workers(arguments, WORKERS)
    if arguments == ["--check"]:
        main("check")
    elif arguments == ["--plan"]:
        main("plan")
    elif TIER is None:
        raise SystemExit("a fit run takes --tier 1, 2 or 3 (bioinf2027/plans/E6.md, Section 16)")
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
